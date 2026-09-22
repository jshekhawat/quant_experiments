"""Optuna-based parameter search for when grid sweeps are not enough.

Optuna is an optional dependency: ``uv sync --extra optimization``. The lab
works fine without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from ..backtest.engine import run_backtest
from ..config import ExecutionConfig
from ..strategies.base import Strategy


@dataclass
class OptimizationResult:
    best_params: dict[str, Any]
    best_value: float
    metric: str
    n_trials: int
    trials: list[dict[str, Any]] = field(default_factory=list)

    def to_polars(self) -> pl.DataFrame:
        return pl.DataFrame(self.trials) if self.trials else pl.DataFrame()


def _suggest(trial, name: str, spec: Any) -> Any:
    if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], (int, float, str)):
        return trial.suggest_categorical(name, list(spec))
    if isinstance(spec, tuple):
        kind = spec[0]
        if kind == "int":
            return trial.suggest_int(name, int(spec[1]), int(spec[2]), step=int(spec[3]) if len(spec) > 3 else 1)
        if kind == "float":
            log = bool(spec[3]) if len(spec) > 3 else False
            return trial.suggest_float(name, float(spec[1]), float(spec[2]), log=log)
        if kind == "categorical":
            return trial.suggest_categorical(name, list(spec[1]))
    raise ValueError(f"unsupported search-space spec for {name!r}: {spec!r}")


def optimize(
    data: pl.DataFrame,
    strategy_cls: type[Strategy],
    search_space: dict[str, Any],
    *,
    metric: str = "sharpe",
    direction: str = "maximize",
    n_trials: int = 50,
    timeout: float | None = None,
    seed: int = 0,
    symbol: str = "UNKNOWN",
    timeframe: str = "1m",
    dataset_id: str = "",
    execution: ExecutionConfig | None = None,
) -> OptimizationResult:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Optuna is not installed. Install it with `uv sync --extra optimization`."
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials_log: list[dict[str, Any]] = []

    def objective(trial) -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in search_space.items()}
        strategy = strategy_cls(**params)
        result = run_backtest(
            data,
            strategy,
            symbol=symbol,
            timeframe=timeframe,
            dataset_id=dataset_id,
            execution=execution,
        )
        value = result.metrics.get(metric)
        trials_log.append({**params, metric: value, "trial": trial.number})
        if value is None:
            return float("-inf") if direction == "maximize" else float("inf")
        return float(value)

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return OptimizationResult(
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        metric=metric,
        n_trials=len(study.trials),
        trials=trials_log,
    )
