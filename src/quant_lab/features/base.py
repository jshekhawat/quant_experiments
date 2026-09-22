"""Feature registry.

Features are functions ``(frame, **params) -> pl.Expr`` operating on a bar
frame. Registering one is a one-liner and requires no change to the backtest
engine:

    @feature("ema")
    def ema(frame, *, window: int = 20):
        return pl.col("close").ewm_mean(span=window, adjust=False)

Every registered feature must be temporally causal: it may only use the
current and past bars. Use trailing windows and explicit ``shift`` — never
centered windows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import polars as pl

FeatureFn = Callable[..., pl.Expr]

_REGISTRY: dict[str, FeatureFn] = {}


def feature(name: str) -> Callable[[FeatureFn], FeatureFn]:
    def decorator(fn: FeatureFn) -> FeatureFn:
        if name in _REGISTRY and _REGISTRY[name] is not fn:
            raise ValueError(f"feature {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_feature(name: str) -> FeatureFn:
    if name not in _REGISTRY:
        raise KeyError(f"unknown feature {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_features() -> list[str]:
    return sorted(_REGISTRY)


@dataclass(slots=True)
class FeatureSpec:
    """A request for one feature column."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    alias: str | None = None

    @property
    def output(self) -> str:
        if self.alias:
            return self.alias
        if self.params:
            return f"{self.name}_{'_'.join(str(v) for v in self.params.values())}"
        return self.name

    def expr(self, frame: pl.DataFrame) -> pl.Expr:
        return get_feature(self.name)(frame, **self.params).alias(self.output)


def compute_features(frame: pl.DataFrame, specs: list[FeatureSpec]) -> pl.DataFrame:
    """Return ``frame`` with the requested feature columns appended."""
    if not specs:
        return frame
    exprs = [spec.expr(frame) for spec in specs]
    return frame.with_columns(exprs)


def parse_specs(raw: list[Any]) -> list[FeatureSpec]:
    """Accept flexible spec forms from config files / CLI.

    * ``"ema"`` -> FeatureSpec("ema")
    * ``{"name": "ema", "window": 100}``
    * ``{"feature": "rsi", "params": {"period": 14}, "alias": "rsi14"}``
    * a ready :class:`FeatureSpec`
    """
    out: list[FeatureSpec] = []
    for item in raw:
        if isinstance(item, FeatureSpec):
            out.append(item)
        elif isinstance(item, str):
            out.append(FeatureSpec(item))
        elif isinstance(item, dict):
            if "feature" in item:
                out.append(
                    FeatureSpec(
                        name=item["feature"],
                        params=item.get("params", {}),
                        alias=item.get("alias"),
                    )
                )
            else:
                data = dict(item)
                name = data.pop("name")
                alias = data.pop("alias", None)
                params = data.pop("params", data)
                out.append(FeatureSpec(name=name, params=params, alias=alias))
        else:
            raise TypeError(f"cannot parse feature spec: {item!r}")
    return out
