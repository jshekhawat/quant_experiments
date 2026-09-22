"""VectorBT-backed vectorized backtest engine.

This is the primary research engine: fast hypothesis testing, portfolio
simulation and large parameter sweeps. It is deliberately the *only* place
that depends on VectorBT. Strategies and features never do.

Timing contract (documented, enforced here):

* **signal timestamp**   = close of bar ``t`` (strategies only see data <= t)
* **order timestamp**    = bar ``t + execution_delay_bars``
* **fill timestamp**     = open of ``t + execution_delay_bars`` (market order,
  adjusted by per-side slippage)
* **position activation** = the same bar as the fill

With ``execution_delay_bars = 0`` fills happen at the close of bar ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import vectorbt as vbt

from ..config import Config, ExecutionConfig
from ..config import get_config as _get_config
from ..strategies.base import Strategy
from ..timeframes import pandas_freq
from ..timeframes import periods_per_year as tf_periods_per_year
from .assumptions import effective_size_fraction, validate_assumptions
from .metrics import compute_metrics, empty_trades

TRADE_RENAME = {
    "Exit Trade Id": "trade_id",
    "Trade Id": "trade_id",
    "Direction": "direction",
    "Size": "size",
    "Entry Timestamp": "entry_time",
    "Avg Entry Price": "entry_price",
    "Exit Timestamp": "exit_time",
    "Avg Exit Price": "exit_price",
    "PnL": "pnl",
    "Return": "return",
    "Status": "status",
}


@dataclass
class BacktestResult:
    name: str
    params: dict[str, Any]
    symbol: str
    timeframe: str
    dataset_id: str
    start: str
    end: str
    assumptions: dict[str, Any]
    metrics: dict[str, Any]
    equity: pd.Series
    returns: pd.Series
    drawdown: pd.Series
    trades: pd.DataFrame
    exposure: pd.Series
    signal_frame: pl.DataFrame
    portfolio: Any = field(default=None, repr=False)
    periods_per_year: float = 365.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary (no engine objects, no plot data)."""
        return {
            "strategy": self.name,
            "params": self.params,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "dataset_id": self.dataset_id,
            "start": self.start,
            "end": self.end,
            "assumptions": self.assumptions,
            "metrics": self.metrics,
            "periods_per_year": self.periods_per_year,
        }


def _shift_signals(sf: pl.DataFrame, delay: int) -> dict[str, np.ndarray]:
    cols = ["entries", "exits", "short_entries", "short_exits"]
    out: dict[str, np.ndarray] = {}
    for col in cols:
        expr = pl.col(col).cast(pl.Boolean)
        if delay > 0:
            expr = expr.shift(delay)
        series = sf.select(expr.fill_null(False).alias(col)).to_series()
        out[col] = series.to_numpy().astype(bool)
    return out


def _funding_costs(
    asset_value: pd.Series,
    funding_rate: pd.Series | None,
    *,
    flat_rate: float,
    bars_per_interval: int,
) -> pd.Series:
    """Signed funding cost per bar (positive = paid by the position)."""
    if funding_rate is not None and not funding_rate.empty:
        rate = funding_rate.reindex(asset_value.index).astype(float)
        # Only charge at funding timestamps (non-null, non-zero) or forward-fill
        # is the caller's choice; here we use as provided.
        rate = rate.fillna(0.0)
    elif flat_rate:
        rate = pd.Series(0.0, index=asset_value.index)
        if bars_per_interval > 0:
            rate.iloc[::bars_per_interval] = flat_rate
    else:
        return pd.Series(0.0, index=asset_value.index)
    return asset_value.astype(float) * rate


def _trades_to_frame(records: pd.DataFrame) -> pd.DataFrame:
    if records is None or records.empty:
        return empty_trades()
    df = records.copy()
    entry_fees = df["Entry Fees"] if "Entry Fees" in df.columns else 0.0
    exit_fees = df["Exit Fees"] if "Exit Fees" in df.columns else 0.0
    df["fees"] = pd.to_numeric(entry_fees, errors="coerce") + pd.to_numeric(exit_fees, errors="coerce")
    df = df.rename(columns=TRADE_RENAME)
    # Some VectorBT versions expose both "Trade Id" and "Exit Trade Id".
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    out = pd.DataFrame(index=df.index)
    for col in empty_trades().columns:
        out[col] = df[col] if col in df.columns else np.nan
    if "entry_time" in out.columns and "exit_time" in out.columns:
        out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True)
        out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True)
        out["holding_period"] = out["exit_time"] - out["entry_time"]
    else:
        out["holding_period"] = pd.NaT
    return out.reset_index(drop=True)


def run_backtest(
    data: pl.DataFrame,
    strategy: Strategy,
    *,
    symbol: str = "UNKNOWN",
    timeframe: str = "1m",
    dataset_id: str = "",
    execution: ExecutionConfig | None = None,
    config: Config | None = None,
    start: str | None = None,
    end: str | None = None,
) -> BacktestResult:
    """Run a single vectorized backtest for one instrument."""
    base = config or _get_config()
    exec_cfg = execution or base.execution
    validate_assumptions(exec_cfg)

    if data.height == 0:
        raise ValueError("cannot backtest an empty dataset")

    sf = strategy.prepare(data)
    pdf = sf.to_pandas()
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], utc=True)
    pdf = pdf.set_index("timestamp").sort_index()

    delay = int(exec_cfg.execution_delay_bars)
    sig = _shift_signals(sf, delay)
    idx = pdf.index
    entries = pd.Series(sig["entries"], index=idx)
    exits = pd.Series(sig["exits"], index=idx)
    short_entries = pd.Series(sig["short_entries"], index=idx)
    short_exits = pd.Series(sig["short_exits"], index=idx)

    close = pdf["close"].astype(float)
    exec_price = pdf["open"].astype(float) if delay >= 1 else close
    size_fraction = effective_size_fraction(exec_cfg, strategy.portfolio(data))

    try:
        freq = pandas_freq(timeframe)
    except ValueError:
        freq = None
    ppy = tf_periods_per_year(timeframe) if _valid_timeframe(timeframe) else base.backtest.annualization

    signal_kwargs: dict[str, Any] = {
        "close": close,
        "entries": entries,
        "exits": exits,
        "price": exec_price,
        "size": size_fraction,
        "size_type": "percent",
        "init_cash": float(exec_cfg.initial_capital),
        "fees": float(exec_cfg.fees),
        "slippage": float(exec_cfg.slippage),
        "freq": freq,
    }
    if exec_cfg.allow_short:
        signal_kwargs["short_entries"] = short_entries
        signal_kwargs["short_exits"] = short_exits
    else:
        signal_kwargs["direction"] = "longonly"

    portfolio = vbt.Portfolio.from_signals(**signal_kwargs)

    equity = portfolio.value().astype(float)
    asset_value = portfolio.asset_value().astype(float)

    funding_series = None
    if "funding_rate" in pdf.columns:
        funding_series = pdf["funding_rate"].astype(float)
    bars_per_interval = 0
    try:
        from ..timeframes import timeframe_seconds

        bars_per_interval = max(1, int(8 * 3600 / timeframe_seconds(timeframe)))
    except ValueError:
        bars_per_interval = 0
    funding_cost = _funding_costs(
        asset_value,
        funding_series,
        flat_rate=float(exec_cfg.funding),
        bars_per_interval=bars_per_interval,
    )
    funding_paid = float(funding_cost.sum())
    equity_adj = equity - funding_cost.cumsum()
    returns = equity_adj.pct_change().fillna(0.0)
    drawdown = equity_adj / equity_adj.cummax() - 1.0
    exposure = (asset_value / equity.replace(0, np.nan)).fillna(0.0)

    trades = _trades_to_frame(portfolio.trades.records_readable)
    fees_paid = _sum_fees(portfolio, trades)

    metrics = compute_metrics(
        equity_adj,
        initial_capital=float(exec_cfg.initial_capital),
        trades=trades,
        periods_per_year=ppy,
        fees_paid=fees_paid,
        funding_paid=funding_paid,
    )

    return BacktestResult(
        name=strategy.name,
        params=dict(strategy.params),
        symbol=symbol,
        timeframe=timeframe,
        dataset_id=dataset_id,
        start=start or str(idx[0]),
        end=end or str(idx[-1]),
        assumptions=exec_cfg.to_dict(),
        metrics=metrics,
        equity=equity_adj,
        returns=returns,
        drawdown=drawdown,
        trades=trades,
        exposure=exposure,
        signal_frame=sf,
        portfolio=portfolio,
        periods_per_year=ppy,
    )


def _valid_timeframe(timeframe: str) -> bool:
    try:
        tf_periods_per_year(timeframe)
        return True
    except ValueError:
        return False


def _sum_fees(portfolio: vbt.Portfolio, trades: pd.DataFrame) -> float:
    try:
        orders = portfolio.orders.records_readable
        if "Fees" in orders.columns:
            return float(orders["Fees"].sum())
    except Exception:  # noqa: BLE001 - best effort, fall back below
        pass
    if "fees" in trades.columns and trades["fees"].notna().any():
        return float(trades["fees"].fillna(0.0).sum())
    return 0.0
