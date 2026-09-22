"""Backtest engine: execution timing, fees, sizing, assumptions."""

from __future__ import annotations

import polars as pl
import pytest

from quant_lab.backtest.assumptions import effective_size_fraction, validate_assumptions
from quant_lab.backtest.engine import run_backtest
from quant_lab.config import ExecutionConfig
from quant_lab.strategies.base import Strategy


class OneShot(Strategy):
    name = "oneshot"
    default_params = {"entry_at": 10, "exit_at": 20}

    def signals(self, data: pl.DataFrame) -> pl.DataFrame:
        idx = pl.int_range(0, data.height)
        return data.with_columns(
            (idx == self.params["entry_at"]).alias("entries"),
            (idx == self.params["exit_at"]).alias("exits"),
        )


def _clean_exec(**overrides) -> ExecutionConfig:
    base = dict(
        allow_short=False,
        execution_delay_bars=1,
        fees=0.0,
        slippage=0.0,
        position_size=1.0,
        leverage=1.0,
        max_position_size=1.0,
    )
    base.update(overrides)
    return ExecutionConfig(**base)


def test_execution_delay_and_price(bars):
    result = run_backtest(bars, OneShot(), timeframe="1m", execution=_clean_exec())
    assert result.trades.shape[0] == 1
    trade = result.trades.iloc[0]
    # signal at bar 10 -> fill at bar 11 open
    assert trade["entry_time"] == bars["timestamp"][11]
    assert trade["entry_price"] == pytest.approx(bars["open"][11])
    assert trade["exit_price"] == pytest.approx(bars["open"][21])


def test_fees_invariant(bars):
    exec_cfg = _clean_exec(fees=0.001, slippage=0.0)
    result = run_backtest(bars, OneShot(), timeframe="1m", execution=exec_cfg)
    m = result.metrics
    assert m["fees_paid"] > 0
    assert m["net_pnl"] == pytest.approx(m["gross_pnl"] - m["fees_paid"] - m["funding_paid"])


def test_zero_fees_configured_explicitly(bars):
    result = run_backtest(bars, OneShot(), timeframe="1m", execution=_clean_exec())
    assert result.metrics["fees_paid"] == pytest.approx(0.0)


def test_validate_assumptions_rejects_bad_values():
    with pytest.raises(ValueError):
        validate_assumptions(ExecutionConfig(fees=1.5))
    with pytest.raises(ValueError):
        validate_assumptions(ExecutionConfig(initial_capital=0))
    with pytest.raises(ValueError):
        validate_assumptions(ExecutionConfig(intrabar_policy="nonsense"))


def test_effective_size_fraction_caps():
    cfg = ExecutionConfig(position_size=1.0, leverage=2.0, max_position_size=1.0)
    assert effective_size_fraction(cfg) == 1.0
    cfg2 = ExecutionConfig(position_size=0.5, leverage=1.0, max_position_size=1.0)
    assert effective_size_fraction(cfg2) == 0.5
    assert effective_size_fraction(cfg2, {"position_size": 0.2}) == 0.2


def test_delay_zero_fills_at_close(bars):
    result = run_backtest(
        bars, OneShot(), timeframe="1m", execution=_clean_exec(execution_delay_bars=0)
    )
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(bars["close"][10])
