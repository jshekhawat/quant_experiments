"""Regime-gated backtesting.

Applies a regime gate to a base strategy's signals and runs the VectorBT engine
on both the gated and ungated signal sets, so the effect of the gate can be read
directly off the performance metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from ..backtest.engine import BacktestResult, run_backtest
from ..config import Config, ExecutionConfig
from ..data.schemas import DatasetType
from ..data.storage import DatasetKey
from ..strategies.base import PrecomputedSignals, Strategy
from .confluence import LabelConfig, prepare_labelled
from .regimes import RegimeConfig, gate_signals


@dataclass
class RegimeBacktest:
    ungated: BacktestResult
    gated: BacktestResult
    allowed_regimes: list[str]
    regime_counts: pl.DataFrame
    config: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "allowed_regimes": self.allowed_regimes,
            "ungated_trades": self.ungated.metrics.get("trade_count"),
            "gated_trades": self.gated.metrics.get("trade_count"),
            "ungated_total_return": self.ungated.metrics.get("total_return"),
            "gated_total_return": self.gated.metrics.get("total_return"),
            "ungated_sharpe": self.ungated.metrics.get("sharpe"),
            "gated_sharpe": self.gated.metrics.get("sharpe"),
        }


def run_regime_backtest(
    config: Config,
    strategy: Strategy,
    *,
    symbol: str,
    timeframe: str,
    market_type: str = "futures",
    start: str,
    end: str,
    allowed_regimes: Sequence[str],
    regime_config: RegimeConfig | None = None,
    execution: ExecutionConfig | None = None,
    source_timeframe: str = "1m",
    signal_col: str = "entries",
) -> RegimeBacktest:
    """Run the base strategy ungated and gated to ``allowed_regimes``."""
    regime_config = regime_config or RegimeConfig()
    frame = prepare_labelled(
        config,
        strategy,
        symbol=symbol,
        timeframe=timeframe,
        market_type=market_type,
        start=start,
        end=end,
        label=LabelConfig(),
        signal_col=signal_col,
        source_timeframe=source_timeframe,
        regime_config=regime_config,
    )
    key = DatasetKey("binance", market_type, DatasetType.BARS, symbol, timeframe)
    dataset_id = key.dataset_id

    gated_frame = gate_signals(frame, allowed_regimes, columns=(signal_col,))

    ungated = run_backtest(
        frame, PrecomputedSignals(), symbol=symbol, timeframe=timeframe,
        dataset_id=dataset_id, execution=execution, start=start, end=end,
    )
    gated = run_backtest(
        gated_frame, PrecomputedSignals(), symbol=symbol, timeframe=timeframe,
        dataset_id=dataset_id, execution=execution, start=start, end=end,
    )

    counts = (
        frame.filter(pl.col(signal_col).fill_null(False))
        .group_by("regime")
        .agg(pl.len().alias("signals"))
        .sort("signals", descending=True)
    )
    return RegimeBacktest(
        ungated=ungated,
        gated=gated,
        allowed_regimes=list(allowed_regimes),
        regime_counts=counts,
        config=regime_config.to_dict(),
    )
