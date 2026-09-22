"""Parameter sweeps.

The workhorse :func:`run_sweep` reuses the strategy/backtest engine for any
strategy and produces a tidy result matrix. :func:`vectorbt_sweep` demonstrates
VectorBT's native parameterization for the common single-parameter case, where
all combinations are simulated in one portfolio object.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ..backtest.engine import run_backtest
from ..config import ExecutionConfig
from ..strategies.base import Strategy

METRIC_COLUMNS = [
    "total_return",
    "cagr",
    "max_drawdown",
    "sharpe",
    "sortino",
    "calmar",
    "volatility",
    "win_rate",
    "profit_factor",
    "expectancy",
    "trade_count",
    "turnover",
    "fees_paid",
]


def parameter_grid(
    strategy_cls: type[Strategy],
    overrides: dict[str, list[Any]] | None = None,
) -> dict[str, list[Any]]:
    grid = {k: list(v) for k, v in strategy_cls.param_grid.items()}
    if overrides:
        grid.update({k: list(v) for k, v in overrides.items()})
    return grid


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[k] for k in keys))]


def run_sweep(
    data: pl.DataFrame,
    strategy_cls: type[Strategy],
    grid: dict[str, list[Any]],
    *,
    symbol: str = "UNKNOWN",
    timeframe: str = "1m",
    dataset_id: str = "",
    execution: ExecutionConfig | None = None,
    progress: bool = False,
) -> pl.DataFrame:
    """Run every parameter combination and return one row of metrics per combo."""
    combos = expand_grid(grid)
    rows: list[dict[str, Any]] = []
    iterator = combos
    if progress:
        try:
            from rich.progress import track

            iterator = track(combos, description="sweeping")
        except Exception:  # noqa: BLE001
            iterator = combos
    for params in iterator:
        strategy = strategy_cls(**params)
        result = run_backtest(
            data,
            strategy,
            symbol=symbol,
            timeframe=timeframe,
            dataset_id=dataset_id,
            execution=execution,
        )
        row: dict[str, Any] = dict(params)
        for col in METRIC_COLUMNS:
            row[col] = result.metrics.get(col)
        rows.append(row)
    return pl.DataFrame(rows)


def vectorbt_sweep(
    data: pl.DataFrame,
    strategy: Strategy,
    param: str,
    values: list[Any],
    *,
    symbol: str = "UNKNOWN",
    timeframe: str = "1m",
    execution: ExecutionConfig | None = None,
) -> pl.DataFrame:
    """Native VectorBT parameterization for one parameter.

    Signals for each parameter value become columns of a single 2-D
    ``Portfolio.from_signals`` call, which VectorBT simulates at once.
    """
    import vectorbt as vbt

    from ..timeframes import pandas_freq

    exec_cfg = execution or ExecutionConfig()
    base_params = dict(strategy.params)
    frames: list[pl.DataFrame] = []
    for value in values:
        params = {**base_params, param: value}
        s = type(strategy)(**params)
        frames.append(s.prepare(data))

    pdf = data.to_pandas()
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], utc=True)
    idx = pd.DatetimeIndex(pdf["timestamp"])

    def stack(col: str) -> pd.DataFrame:
        return pd.DataFrame(
            {str(v): frames[i].select(col).to_series().to_numpy().astype(bool) for i, v in enumerate(values)},
            index=idx,
        )

    entries = stack("entries")
    exits = stack("exits")
    close = pd.DataFrame(
        np.repeat(pdf["close"].to_numpy()[:, None], len(values), axis=1),
        index=idx,
        columns=entries.columns,
    )
    try:
        freq = pandas_freq(timeframe)
    except ValueError:
        freq = None

    portfolio = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        direction="longonly",
        init_cash=float(exec_cfg.initial_capital),
        fees=float(exec_cfg.fees),
        slippage=float(exec_cfg.slippage),
        size=exec_cfg.position_size,
        size_type="percent",
        freq=freq,
    )
    stats = portfolio.stats()
    rows = []
    for value in values:
        label = str(value)
        row = {param: value}
        for metric, key in [
            ("total_return", "Total Return [%]"),
            ("max_drawdown", "Max Drawdown [%]"),
            ("sharpe", "Sharpe Ratio"),
            ("trade_count", "Total Trades"),
            ("win_rate", "Win Rate [%]"),
            ("profit_factor", "Profit Factor"),
        ]:
            if key in stats.index:
                raw = stats.loc[key]
                val = float(raw[label]) if hasattr(raw, "__getitem__") else float(raw)
                if "return" in metric or "drawdown" in metric or "rate" in metric:
                    val = val / 100.0
                row[metric] = val
        rows.append(row)
    return pl.DataFrame(rows)


def best_by(results: pl.DataFrame, metric: str, *, maximize: bool = True) -> dict[str, Any]:
    if metric not in results.columns or results.height == 0:
        raise KeyError(f"metric {metric!r} not available")
    idx = results[metric].arg_max() if maximize else results[metric].arg_min()
    return results.row(int(idx), named=True)
