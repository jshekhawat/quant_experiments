"""Deterministic end-to-end test.

synthetic data -> strategy -> backtest -> known expected result.

If a future change alters the calculation, these assertions fail.
"""

from __future__ import annotations

import polars as pl
import pytest

from quant_lab.backtest.engine import run_backtest
from quant_lab.config import ExecutionConfig
from quant_lab.features.core import ema
from quant_lab.strategies.base import Strategy
from quant_lab.synthetic import synthetic_bars


class EmaCross(Strategy):
    name = "ema_cross"
    default_params = {"span": 10}

    def signals(self, data: pl.DataFrame) -> pl.DataFrame:
        e = data.select(ema(data, span=self.params["span"]).alias("e"))["e"]
        close = data["close"]
        return data.with_columns(
            (close > e).fill_null(False).alias("entries"),
            (close < e).fill_null(False).alias("exits"),
        )


EXPECTED = {
    "total_return": -0.0200720161451039,
    "max_drawdown": -0.06765986852287054,
    "trade_count": 41,
    "fees_paid": 3199.2019598715606,
    "net_pnl": -2007.2016145103844,
    "win_rate": 0.2682926829268293,
}


def test_end_to_end_known_result():
    df = synthetic_bars(500, seed=99)
    result = run_backtest(
        df,
        EmaCross(),
        symbol="TESTUSDT",
        timeframe="1m",
        dataset_id="synthetic/test/bars/TESTUSDT/1m",
        execution=ExecutionConfig(
            allow_short=False, execution_delay_bars=1, fees=0.0004, slippage=0.0002
        ),
    )
    m = result.metrics
    assert m["total_return"] == pytest.approx(EXPECTED["total_return"], rel=1e-9)
    assert m["max_drawdown"] == pytest.approx(EXPECTED["max_drawdown"], rel=1e-9)
    assert m["trade_count"] == EXPECTED["trade_count"]
    assert m["fees_paid"] == pytest.approx(EXPECTED["fees_paid"], rel=1e-9)
    assert m["net_pnl"] == pytest.approx(EXPECTED["net_pnl"], rel=1e-9)
    assert m["win_rate"] == pytest.approx(EXPECTED["win_rate"], rel=1e-9)
    # invariant across the whole pipeline
    assert m["net_pnl"] == pytest.approx(m["gross_pnl"] - m["fees_paid"] - m["funding_paid"])
