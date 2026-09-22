"""Resample canonical bars to a higher timeframe.

OHLCV aggregation follows the standard convention:

* ``open``  = first bar's open
* ``high``  = max high
* ``low``   = min low
* ``close`` = last bar's close
* ``volume``, ``quote_volume``, ``trade_count``, ``taker_buy_*`` = sum

``timestamp`` is the **open** time of the bucket (``label="left"``) so the
backtest timing contract still holds. Resampling is lossy: it cannot recover
the intrabar path.
"""

from __future__ import annotations

import polars as pl

from ..timeframes import timeframe_seconds
from .schemas import BAR_SCHEMA

_AGG: list[pl.Expr] = [
    pl.col("open").first().alias("open"),
    pl.col("high").max().alias("high"),
    pl.col("low").min().alias("low"),
    pl.col("close").last().alias("close"),
    pl.col("volume").sum().alias("volume"),
    pl.col("quote_volume").sum().alias("quote_volume"),
    pl.col("trade_count").sum().alias("trade_count"),
    pl.col("taker_buy_volume").sum().alias("taker_buy_volume"),
    pl.col("taker_buy_quote_volume").sum().alias("taker_buy_quote_volume"),
]


def resample_bars(
    df: pl.DataFrame,
    *,
    timeframe: str,
    source_timeframe: str = "1m",
    drop_incomplete: bool = True,
    closed: str = "left",
    label: str = "left",
) -> pl.DataFrame:
    """Aggregate bars from ``source_timeframe`` into ``timeframe``.

    When ``drop_incomplete`` is true, trailing buckets that do not contain a
    full ``timeframe / source_timeframe`` bars are removed (avoids a partial
    final candle). Leading partial buckets are kept only if full.
    """
    if df.height == 0:
        return df
    if timeframe_seconds(timeframe) < timeframe_seconds(source_timeframe):
        raise ValueError("resample_bars only aggregates to a higher timeframe")
    if timeframe_seconds(timeframe) == timeframe_seconds(source_timeframe):
        return df.sort("timestamp")

    ordered = df.sort("timestamp")
    out = ordered.group_by_dynamic(
        "timestamp", every=timeframe, closed=closed, label=label
    ).agg(_AGG)

    if drop_incomplete:
        expected = timeframe_seconds(timeframe) // timeframe_seconds(source_timeframe)
        counts = ordered.group_by_dynamic(
            "timestamp", every=timeframe, closed=closed, label=label
        ).agg(pl.len().alias("_count"))
        out = (
            out.join(counts, on="timestamp", how="left")
            .filter(pl.col("_count") >= expected)
            .drop("_count")
        )

    return out.select([c for c in BAR_SCHEMA if c in out.columns]).sort("timestamp")
