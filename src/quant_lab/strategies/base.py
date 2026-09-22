"""Deliberately simple strategy abstraction.

A strategy declares its parameters, optionally precomputes features, and emits
four boolean signal columns:

    entries        long entry
    exits          long exit
    short_entries  short entry
    short_exits    short exit

Signals are evaluated using information available at the **close** of bar ``t``
and are executed later by the engine according to
``ExecutionConfig.execution_delay_bars``. Strategies must never look ahead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import polars as pl

from ..features.base import FeatureSpec, compute_features

# Canonical signal column names produced by every strategy.
SIGNAL_COLUMNS = ("entries", "exits", "short_entries", "short_exits")


class Strategy(ABC):
    """Base class. Subclasses set ``name`` and implement :meth:`signals`."""

    name: ClassVar[str] = "strategy"
    #: parameter name -> default value
    default_params: ClassVar[dict[str, Any]] = {}
    #: parameter name -> list of values, for parameter sweeps
    param_grid: ClassVar[dict[str, list[Any]]] = {}
    #: features this strategy needs, used by the default :meth:`features`
    feature_specs: ClassVar[list[Any]] = []

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise TypeError(f"{self.name}: unknown parameters {sorted(unknown)}")
        self.params: dict[str, Any] = {**self.default_params, **params}

    # --- interface -------------------------------------------------------
    def features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Attach feature columns. Override for anything non-trivial."""
        if not self.feature_specs:
            return data
        return compute_features(data, _coerce_specs(self.feature_specs, self.params))

    @abstractmethod
    def signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Return ``data`` with the four signal boolean columns."""

    def portfolio(self, data: pl.DataFrame) -> dict[str, Any] | None:
        """Optional per-strategy execution overrides (e.g. size, fees).

        Returning ``None`` means "use the global execution assumptions".
        """
        return None

    # --- helpers ---------------------------------------------------------
    def prepare(self, data: pl.DataFrame) -> pl.DataFrame:
        """Run features then signals and validate the output."""
        featured = self.features(data)
        out = self.signals(featured)
        return _normalize_signals(out)

    def describe(self) -> dict[str, Any]:
        return {"strategy": self.name, "params": dict(self.params)}


def _coerce_specs(specs: list[Any], params: dict[str, Any]) -> list[FeatureSpec]:
    """Turn class-level specs into concrete FeatureSpec values.

    A spec entry may be:
      * a string feature name
      * a ``(name, params)`` tuple
      * a dict as accepted by ``parse_specs``
      * a callable ``params -> FeatureSpec``
    """
    from ..features.base import parse_specs

    resolved: list[Any] = []
    for spec in specs:
        if callable(spec):
            resolved.append(spec(params))
        else:
            resolved.append(spec)
    return parse_specs(resolved)


def _normalize_signals(df: pl.DataFrame) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for col in SIGNAL_COLUMNS:
        if col in df.columns:
            exprs.append(
                pl.when(pl.col(col).is_null())
                .then(pl.lit(False))
                .otherwise(pl.col(col).cast(pl.Boolean))
                .alias(col)
            )
        else:
            exprs.append(pl.lit(False).alias(col))
    return df.with_columns(exprs)


# --- registry ----------------------------------------------------------------
_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(cls: type[Strategy]) -> type[Strategy]:
    name = cls.name
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"strategy {name!r} already registered")
    _REGISTRY[name] = cls
    return cls


def get_strategy(name: str) -> type[Strategy]:
    _ensure_builtins()
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def build_strategy(name: str, **params: Any) -> Strategy:
    return get_strategy(name)(**params)


def available_strategies() -> list[str]:
    _ensure_builtins()
    return sorted(_REGISTRY)


def _ensure_builtins() -> None:
    # Import the bundled strategies so they register themselves.
    from . import bollinger_breakout as _bb  # noqa: F401
    from . import range_breakout_hold as _rbh  # noqa: F401
    from . import squeeze_release as _sq  # noqa: F401
    from . import volatility_breakout as _vb  # noqa: F401


StrategyFactory = Callable[..., Strategy]


class PrecomputedSignals(Strategy):
    """Strategy that returns an already-computed signal frame.

    Used for gated or externally generated signals: ``data`` is expected to
    already contain the canonical signal columns, so :meth:`signals` is a no-op.
    """

    name = "precomputed"
    default_params: ClassVar[dict[str, Any]] = {}

    def signals(self, data: pl.DataFrame) -> pl.DataFrame:
        return data
