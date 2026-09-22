"""Backtesting engines (VectorBT primary, NautilusTrader optional)."""

from .assumptions import (
    describe,
    effective_size_fraction,
    format_assumptions,
    validate_assumptions,
)
from .engine import BacktestResult, run_backtest
from .metrics import compute_metrics

__all__ = [
    "BacktestResult",
    "run_backtest",
    "compute_metrics",
    "validate_assumptions",
    "effective_size_fraction",
    "format_assumptions",
    "describe",
]
