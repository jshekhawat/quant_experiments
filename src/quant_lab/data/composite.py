"""Composite cross-venue datasets.

Every composite has an explicit methodology recorded alongside the data:
venues included, weighting, timestamp alignment, missing-data treatment. We
never silently combine instruments whose contract specifications differ — the
caller must supply compatible inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl


@dataclass(slots=True)
class CompositeDataset:
    name: str
    frame: pl.DataFrame
    methodology: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        lines = [f"Composite dataset: {self.name}"]
        for k, v in self.methodology.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def _align(
    frames: dict[str, pl.DataFrame],
    value_col: str,
    *,
    alignment: str = "outer",
    fill_limit: int | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    venues = list(frames)
    out: pl.DataFrame | None = None
    for venue, df in frames.items():
        sub = df.select(pl.col("timestamp"), pl.col(value_col).alias(venue)).sort("timestamp")
        out = sub if out is None else out.join(sub, on="timestamp", how=alignment, coalesce=True)
    assert out is not None
    out = out.sort("timestamp")
    if fill_limit is not None:
        out = out.with_columns(
            [pl.col(v).forward_fill(limit=fill_limit).alias(v) for v in venues]
        )
    return out, venues


def _weights(venues: list[str], weights: dict[str, float] | None) -> dict[str, float]:
    if weights is None:
        return {v: 1.0 / len(venues) for v in venues}
    total = sum(weights.get(v, 0.0) for v in venues)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return {v: weights.get(v, 0.0) / total for v in venues}


def composite_metric(
    frames: dict[str, pl.DataFrame],
    value_col: str,
    *,
    name: str,
    weights: dict[str, float] | None = None,
    alignment: str = "outer",
    fill_limit: int | None = None,
    aggregate: str = "weighted_mean",
) -> CompositeDataset:
    aligned, venues = _align(frames, value_col, alignment=alignment, fill_limit=fill_limit)
    w = _weights(venues, weights)
    if aggregate == "weighted_mean":
        expr = pl.sum_horizontal([pl.col(v).fill_null(0.0) * w[v] for v in venues])
    elif aggregate == "sum":
        expr = pl.sum_horizontal([pl.col(v).fill_null(0.0) for v in venues])
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}")
    frame = aligned.select("timestamp", expr.alias(value_col))
    methodology = {
        "venues": venues,
        "value": value_col,
        "aggregate": aggregate,
        "weights": w,
        "timestamp_alignment": alignment,
        "missing_data": f"forward_fill(limit={fill_limit})" if fill_limit else "left as null / zero-filled in aggregate",
    }
    return CompositeDataset(name=name, frame=frame, methodology=methodology)


def composite_funding(
    frames: dict[str, pl.DataFrame],
    *,
    weights: dict[str, float] | None = None,
    alignment: str = "outer",
    fill_limit: int | None = None,
) -> CompositeDataset:
    """Weighted mean funding rate across venues."""
    return composite_metric(
        frames,
        "funding_rate",
        name="composite_funding",
        weights=weights,
        alignment=alignment,
        fill_limit=fill_limit,
        aggregate="weighted_mean",
    )


def composite_volume(
    frames: dict[str, pl.DataFrame],
    *,
    weights: dict[str, float] | None = None,
    alignment: str = "outer",
    fill_limit: int | None = None,
) -> CompositeDataset:
    """Total (or weighted) quote volume across venues."""
    return composite_metric(
        frames,
        "quote_volume",
        name="composite_volume",
        weights=weights,
        alignment=alignment,
        fill_limit=fill_limit,
        aggregate="sum",
    )


def composite_open_interest(
    frames: dict[str, pl.DataFrame],
    *,
    weights: dict[str, float] | None = None,
    alignment: str = "outer",
    fill_limit: int | None = None,
) -> CompositeDataset:
    return composite_metric(
        frames,
        "open_interest",
        name="composite_open_interest",
        weights=weights,
        alignment=alignment,
        fill_limit=fill_limit,
        aggregate="sum",
    )


def cross_venue_spread(
    frames: dict[str, pl.DataFrame],
    *,
    value_col: str = "close",
    alignment: str = "inner",
) -> CompositeDataset:
    """Max-min relative spread of a price across venues, per timestamp."""
    aligned, venues = _align(frames, value_col, alignment=alignment)
    lo = pl.min_horizontal([pl.col(v) for v in venues])
    hi = pl.max_horizontal([pl.col(v) for v in venues])
    mid = (hi + lo) / 2.0
    spread = pl.when(mid > 0).then((hi - lo) / mid).otherwise(None)
    frame = aligned.select("timestamp", spread.alias("cross_venue_spread"))
    methodology = {
        "venues": venues,
        "value": value_col,
        "definition": "(max - min) / midpoint across venues",
        "timestamp_alignment": alignment,
        "missing_data": "inner join; only timestamps present on all venues",
    }
    return CompositeDataset("cross_venue_spread", frame, methodology)
