"""Human-readable performance reports and plots.

QuantStats is used where it is useful (tearsheet HTML); the core summary is
generated explicitly so that execution assumptions and the experiment
identifier are always displayed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.assumptions import FIELD_HELP
from ..backtest.engine import BacktestResult


def _pct(x: Any, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def _num(x: Any, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _money(x: Any) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:,.2f}"


def _duration(seconds: Any) -> str:
    if seconds is None or (isinstance(seconds, float) and not np.isfinite(seconds)):
        return "n/a"
    seconds = float(seconds)
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


def format_summary(result: BacktestResult, experiment_id: str | None = None) -> str:
    m = result.metrics
    lines: list[str] = []
    lines.append("=" * 68)
    lines.append(f"Backtest summary: {result.name} on {result.symbol} ({result.timeframe})")
    lines.append("=" * 68)
    if experiment_id:
        lines.append(f"experiment id : {experiment_id}")
    lines.append(f"dataset       : {result.dataset_id}")
    lines.append(f"data range    : {result.start} -> {result.end}")
    lines.append(f"parameters    : {json.dumps(result.params, sort_keys=True)}")
    lines.append("")
    lines.append("Performance")
    lines.append(f"  total return        : {_pct(m.get('total_return'))}")
    lines.append(f"  CAGR                : {_pct(m.get('cagr'))}")
    lines.append(f"  net PnL             : {_money(m.get('net_pnl'))}")
    lines.append(f"  gross PnL           : {_money(m.get('gross_pnl'))}")
    lines.append(f"  max drawdown        : {_pct(m.get('max_drawdown'))}")
    lines.append(f"  volatility (ann.)   : {_pct(m.get('volatility'))}")
    lines.append(f"  Sharpe              : {_num(m.get('sharpe'))}")
    lines.append(f"  Sortino             : {_num(m.get('sortino'))}")
    lines.append(f"  Calmar              : {_num(m.get('calmar'))}")
    lines.append("")
    lines.append("Trades")
    lines.append(f"  trade count         : {m.get('trade_count')}")
    lines.append(f"  win rate            : {_pct(m.get('win_rate'))}")
    lines.append(f"  profit factor       : {_num(m.get('profit_factor'))}")
    lines.append(f"  expectancy          : {_money(m.get('expectancy'))}")
    lines.append(f"  average win         : {_money(m.get('avg_win'))}")
    lines.append(f"  average loss        : {_money(m.get('avg_loss'))}")
    lines.append(f"  avg holding period  : {_duration(m.get('avg_holding_period'))}")
    lines.append(f"  turnover            : {_num(m.get('turnover'))}x")
    lines.append(f"  fees paid           : {_money(m.get('fees_paid'))}")
    lines.append(f"  funding paid        : {_money(m.get('funding_paid'))}")
    lines.append("")
    lines.append("Execution assumptions")
    width = max(len(k) for k in result.assumptions)
    for k, v in result.assumptions.items():
        lines.append(f"  {k:<{width}} : {v}    # {FIELD_HELP.get(k, '')}")
    lines.append("=" * 68)
    return "\n".join(lines)


@dataclass(slots=True)
class ReportPaths:
    summary_md: Path
    summary_json: Path
    plots: dict[str, Path]
    tearsheet: Path | None = None


def save_report(
    result: BacktestResult,
    out_dir: Path,
    *,
    experiment_id: str | None = None,
    make_plots: bool = True,
    make_tearsheet: bool = True,
) -> ReportPaths:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_md = out_dir / "summary.md"
    summary_md.write_text(format_summary(result, experiment_id), encoding="utf-8")

    summary_json = out_dir / "summary.json"
    payload = result.to_dict()
    payload["experiment_id"] = experiment_id
    summary_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    plots: dict[str, Path] = {}
    if make_plots:
        plots = make_plots_bundle(result, out_dir)

    tearsheet: Path | None = None
    if make_tearsheet:
        tearsheet = make_quantstats_tearsheet(result, out_dir / "tearsheet.html")

    return ReportPaths(summary_md, summary_json, plots, tearsheet)


def make_plots_bundle(result: BacktestResult, out_dir: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"

    out: dict[str, Path] = {}
    equity = result.equity
    drawdown = result.drawdown

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(equity.index, equity.values, lw=1.2, color="#1f77b4")
    ax.set_title(f"Equity curve — {result.name} {result.symbol}")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out["equity"] = _save_fig(fig, out_dir / "equity.png")

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(drawdown.index, drawdown.values * 100, 0, color="#d62728", alpha=0.5)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out["drawdown"] = _save_fig(fig, out_dir / "drawdown.png")

    monthly = _monthly_returns(equity)
    if not monthly.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        monthly.plot(kind="bar", ax=ax, color=np.where(monthly.values >= 0, "#2ca02c", "#d62728"))
        ax.set_title("Monthly returns")
        ax.set_ylabel("Return")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        out["monthly_returns"] = _save_fig(fig, out_dir / "monthly_returns.png")

    if not result.trades.empty:
        pnl = result.trades["pnl"].astype(float).dropna()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(pnl, bins=50, color="#1f77b4", alpha=0.8)
        ax.axvline(0, color="black", lw=1)
        ax.set_title("Trade PnL distribution")
        ax.set_xlabel("PnL")
        fig.tight_layout()
        out["trade_pnl"] = _save_fig(fig, out_dir / "trade_pnl_distribution.png")

    rets = result.returns.dropna()
    if not rets.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(rets, bins=80, color="#7f7f7f", alpha=0.8)
        ax.set_title("Bar return distribution")
        ax.set_xlabel("Return")
        fig.tight_layout()
        out["returns"] = _save_fig(fig, out_dir / "returns_distribution.png")

    return out


def _save_fig(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _monthly_returns(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype=float)
    monthly = equity.resample("ME").last()
    return monthly.pct_change().dropna()


def make_quantstats_tearsheet(result: BacktestResult, path: Path) -> Path | None:
    """Best-effort QuantStats HTML tearsheet. Returns None if unavailable."""
    try:
        import quantstats as qs
    except Exception:  # noqa: BLE001
        return None
    returns = result.returns.copy()
    returns = returns[~returns.index.duplicated(keep="first")].dropna()
    if len(returns) < 30:
        return None
    try:
        qs.reports.html(
            returns,
            output=str(path),
            title=f"{result.name} {result.symbol} {result.timeframe}",
        )
        return path
    except Exception:  # noqa: BLE001 - reporting must never break a backtest
        return None
