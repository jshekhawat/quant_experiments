"""Optimization: parameter sweeps (VectorBT) and Optuna search."""

from .sweep import (
    METRIC_COLUMNS,
    best_by,
    expand_grid,
    parameter_grid,
    run_sweep,
    vectorbt_sweep,
)

__all__ = [
    "run_sweep",
    "vectorbt_sweep",
    "parameter_grid",
    "expand_grid",
    "best_by",
    "METRIC_COLUMNS",
]


def __getattr__(name: str):
    if name in {"optimize", "OptimizationResult"}:
        from . import optuna_opt

        return getattr(optuna_opt, name)
    raise AttributeError(name)
