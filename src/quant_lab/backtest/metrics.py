"""Performance metrics.

Computed directly from an equity curve and a canonical trade table so that the
numbers do not depend on a particular backtesting engine. QuantStats is used
separately for richer reports, not here.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADE_COLUMNS = [
    "trade_id",
    "direction",
    "entry_time",
    "exit_time",
    "size",
    "entry_price",
    "exit_price",
    "pnl",
    "return",
    "fees",
    "holding_period",
]


def empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=TRADE_COLUMNS)


def compute_metrics(
    equity: pd.Series,
    *,
    initial_capital: float,
    trades: pd.DataFrame,
    periods_per_year: float,
    fees_paid: float = 0.0,
    funding_paid: float = 0.0,
) -> dict[str, float | int | None]:
    equity = equity.astype(float).dropna()
    metrics: dict[str, float | int | None] = {}
    if equity.empty:
        return _empty_metrics()

    returns = equity.pct_change().dropna()
    final = float(equity.iloc[-1])
    total_return = final / float(initial_capital) - 1.0

    n = len(equity)
    years = n / periods_per_year if periods_per_year else float("nan")
    cagr = (final / initial_capital) ** (1.0 / years) - 1.0 if years and years > 0 else None

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())

    vol = float(returns.std(ddof=1)) * math.sqrt(periods_per_year) if len(returns) > 1 else 0.0
    sharpe = _safe_ratio(float(returns.mean()) * periods_per_year, vol)
    downside = returns[returns < 0]
    downside_vol = (
        float(downside.std(ddof=1)) * math.sqrt(periods_per_year) if len(downside) > 1 else 0.0
    )
    sortino = _safe_ratio(float(returns.mean()) * periods_per_year, downside_vol)
    calmar = _safe_ratio(cagr, abs(max_dd)) if cagr is not None and max_dd != 0 else None

    trade_stats = _trade_stats(trades)
    net_pnl = final - initial_capital
    gross_pnl = net_pnl + fees_paid + funding_paid

    notional = 0.0
    if not trades.empty and {"size", "entry_price", "exit_price"}.issubset(trades.columns):
        notional = float(
            (trades["size"].abs() * trades["entry_price"]).sum()
            + (trades["size"].abs() * trades["exit_price"]).sum()
        )
    turnover = notional / initial_capital if initial_capital else 0.0

    metrics.update(
        {
            "initial_capital": float(initial_capital),
            "final_equity": final,
            "total_return": total_return,
            "cagr": cagr,
            "net_pnl": net_pnl,
            "gross_pnl": gross_pnl,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "volatility": vol,
            "win_rate": trade_stats["win_rate"],
            "profit_factor": trade_stats["profit_factor"],
            "expectancy": trade_stats["expectancy"],
            "avg_win": trade_stats["avg_win"],
            "avg_loss": trade_stats["avg_loss"],
            "trade_count": trade_stats["trade_count"],
            "avg_holding_period": trade_stats["avg_holding_period"],
            "turnover": turnover,
            "fees_paid": float(fees_paid),
            "funding_paid": float(funding_paid),
        }
    )
    return metrics


def _trade_stats(trades: pd.DataFrame) -> dict[str, float | int | None]:
    if trades is None or trades.empty:
        return {
            "win_rate": None,
            "profit_factor": None,
            "expectancy": None,
            "avg_win": None,
            "avg_loss": None,
            "trade_count": 0,
            "avg_holding_period": None,
        }
    pnl = trades["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = float(wins.sum())
    gross_loss = float(losses.sum())
    holding = trades["holding_period"].dropna() if "holding_period" in trades else pd.Series(dtype="timedelta64[ns]")
    avg_holding = float(holding.dt.total_seconds().mean()) if len(holding) else None
    return {
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else None,
        "profit_factor": (gross_win / abs(gross_loss)) if gross_loss < 0 else None,
        "expectancy": float(pnl.mean()) if len(pnl) else None,
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "trade_count": int(len(pnl)),
        "avg_holding_period": avg_holding,
    }


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0 or not np.isfinite(denominator):
        return None
    value = numerator / denominator
    return float(value) if np.isfinite(value) else None


def _empty_metrics() -> dict[str, float | int | None]:
    keys = [
        "initial_capital", "final_equity", "total_return", "cagr", "net_pnl",
        "gross_pnl", "max_drawdown", "sharpe", "sortino", "calmar", "volatility",
        "win_rate", "profit_factor", "expectancy", "avg_win", "avg_loss",
        "trade_count", "avg_holding_period", "turnover", "fees_paid", "funding_paid",
    ]
    return {k: None for k in keys}
