"""Statistical robustness utilities.

The purpose is to judge whether a result is robust, not to find the single
highest historical return. These helpers expose the underlying distributions
rather than collapsing them into an opaque "strategy quality score".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.metrics import compute_metrics


@dataclass(slots=True)
class Distribution:
    name: str
    values: np.ndarray

    def summary(self) -> dict[str, float]:
        v = self.values[np.isfinite(self.values)]
        if v.size == 0:
            return {}
        return {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "p05": float(np.percentile(v, 5)),
            "p25": float(np.percentile(v, 25)),
            "median": float(np.median(v)),
            "p75": float(np.percentile(v, 75)),
            "p95": float(np.percentile(v, 95)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "prob_negative": float(np.mean(v < 0)),
        }


@dataclass(slots=True)
class RobustnessReport:
    distributions: dict[str, Distribution] = field(default_factory=dict)

    def add(self, name: str, values: np.ndarray) -> None:
        self.distributions[name] = Distribution(name, np.asarray(values, dtype=float))

    def summary(self) -> dict[str, dict[str, float]]:
        return {name: d.summary() for name, d in self.distributions.items()}

    def format(self) -> str:
        lines = ["Robustness distributions:"]
        for name, d in self.distributions.items():
            s = d.summary()
            if not s:
                continue
            lines.append(
                f"  {name:<28} mean={s['mean']:.4g} p05={s['p05']:.4g} "
                f"median={s['median']:.4g} p95={s['p95']:.4g} "
                f"P(<0)={s['prob_negative']:.2%}"
            )
        return "\n".join(lines)


def _block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if block <= 1:
        return rng.integers(0, n, size=n)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    return idx[:n]


def bootstrap_returns(
    returns: pd.Series | np.ndarray,
    *,
    n: int = 1000,
    block: int = 1,
    periods_per_year: float = 365.0,
    seed: int = 0,
) -> RobustnessReport:
    """Bootstrap the return series and report the distribution of outcomes."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        raise ValueError("need at least two finite returns")
    rng = np.random.default_rng(seed)
    totals = np.empty(n)
    sharpes = np.empty(n)
    max_dds = np.empty(n)
    for i in range(n):
        sample = r[_block_bootstrap_indices(r.size, block, rng)]
        equity = np.cumprod(1.0 + sample)
        totals[i] = equity[-1] - 1.0
        std = sample.std(ddof=1)
        sharpes[i] = (sample.mean() / std) * np.sqrt(periods_per_year) if std > 0 else 0.0
        peak = np.maximum.accumulate(equity)
        max_dds[i] = float(np.min(equity / peak - 1.0))
    report = RobustnessReport()
    report.add("bootstrap_total_return", totals)
    report.add("bootstrap_sharpe", sharpes)
    report.add("bootstrap_max_drawdown", max_dds)
    return report


def monte_carlo_trades(
    trade_pnl: pd.Series | np.ndarray,
    *,
    n: int = 1000,
    initial_capital: float = 100_000.0,
    seed: int = 0,
) -> RobustnessReport:
    """Reshuffle trade PnL order to study path dependence."""
    pnl = np.asarray(trade_pnl, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    if pnl.size < 2:
        raise ValueError("need at least two trades")
    rng = np.random.default_rng(seed)
    finals = np.empty(n)
    max_dds = np.empty(n)
    for i in range(n):
        shuffled = rng.permutation(pnl)
        equity = initial_capital + np.cumsum(shuffled)
        equity = np.concatenate([[initial_capital], equity])
        finals[i] = equity[-1]
        peak = np.maximum.accumulate(equity)
        max_dds[i] = float(np.min(equity / peak - 1.0))
    report = RobustnessReport()
    report.add("mc_final_equity", finals)
    report.add("mc_total_pnl", finals - initial_capital)
    report.add("mc_max_drawdown", max_dds)
    return report


def parameter_sensitivity(results: pd.DataFrame, param: str, metric: str) -> pd.DataFrame:
    """Summarize a metric across the values of one parameter.

    ``results`` is a flat table with one row per parameter combination and a
    column named ``param`` plus the metric columns.
    """
    if param not in results.columns:
        raise KeyError(f"{param!r} not in results columns")
    if metric not in results.columns:
        raise KeyError(f"{metric!r} not in results columns")
    grouped = results.groupby(param)[metric]
    return grouped.agg(["mean", "std", "min", "max", "count"]).reset_index()


def drawdown_distribution(returns: pd.Series | np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return (equity / peak - 1.0).astype(float)


def return_distribution(returns: pd.Series | np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[np.isfinite(r)].astype(float)


def metrics_from_returns(
    returns: pd.Series,
    *,
    initial_capital: float = 100_000.0,
    periods_per_year: float = 365.0,
) -> dict:
    equity = initial_capital * (1.0 + returns.fillna(0.0)).cumprod()
    return compute_metrics(
        equity,
        initial_capital=initial_capital,
        trades=pd.DataFrame(columns=["pnl"]),
        periods_per_year=periods_per_year,
    )
