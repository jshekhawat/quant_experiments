"""Market-structure features derived from flow and derivatives data.

These expect the relevant canonical columns to be present on the frame (for
example ``taker_buy_volume`` for volume delta, or an ``open_interest`` column
joined from the open-interest dataset).
"""

from __future__ import annotations

import polars as pl

from .base import feature
from .core import safe_div


def _volume_delta_expr() -> pl.Expr:
    # Taker buy volume minus taker sell volume, expressed from total volume.
    return 2.0 * pl.col("taker_buy_volume") - pl.col("volume")


@feature("volume_delta")
def volume_delta(frame: pl.DataFrame) -> pl.Expr:
    return _volume_delta_expr()


@feature("cvd")
def cvd(frame: pl.DataFrame) -> pl.Expr:
    """Cumulative volume delta (net taker flow) over the whole series."""
    return _volume_delta_expr().cum_sum()


@feature("cvd_change")
def cvd_change(frame: pl.DataFrame, *, window: int = 20) -> pl.Expr:
    delta = _volume_delta_expr()
    return delta.rolling_sum(window_size=window, min_periods=window)


@feature("open_interest_change")
def open_interest_change(frame: pl.DataFrame, *, period: int = 1, column: str = "open_interest") -> pl.Expr:
    return pl.col(column).pct_change(n=period)


@feature("open_interest_zscore")
def open_interest_zscore(frame: pl.DataFrame, *, window: int = 96, column: str = "open_interest") -> pl.Expr:
    mean = pl.col(column).rolling_mean(window_size=window, min_periods=window)
    std = pl.col(column).rolling_std(window_size=window, min_periods=window)
    return safe_div(pl.col(column) - mean, std)


@feature("funding_rate")
def funding_rate(frame: pl.DataFrame, *, column: str = "funding_rate") -> pl.Expr:
    return pl.col(column)


@feature("funding_cumulative")
def funding_cumulative(frame: pl.DataFrame, *, window: int = 8, column: str = "funding_rate") -> pl.Expr:
    return pl.col(column).rolling_sum(window_size=window, min_periods=window)


@feature("basis")
def basis(
    frame: pl.DataFrame,
    *,
    futures_column: str = "close",
    index_column: str = "index_close",
) -> pl.Expr:
    return safe_div(pl.col(futures_column) - pl.col(index_column), pl.col(index_column))


@feature("premium")
def premium(frame: pl.DataFrame, *, mark_column: str = "mark_close", index_column: str = "index_close") -> pl.Expr:
    return safe_div(pl.col(mark_column) - pl.col(index_column), pl.col(index_column))


@feature("liquidation_intensity")
def liquidation_intensity(
    frame: pl.DataFrame,
    *,
    window: int = 20,
    notional_column: str = "liquidation_notional",
    volume_column: str = "quote_volume",
) -> pl.Expr:
    liq = pl.col(notional_column).rolling_sum(window_size=window, min_periods=1)
    vol = pl.col(volume_column).rolling_sum(window_size=window, min_periods=window)
    return safe_div(liq, vol)


@feature("relative_volume")
def relative_volume(frame: pl.DataFrame, *, window: int = 20, column: str = "volume") -> pl.Expr:
    mean = pl.col(column).rolling_mean(window_size=window, min_periods=window)
    return safe_div(pl.col(column), mean)


@feature("range_expansion")
def range_expansion(frame: pl.DataFrame, *, window: int = 20) -> pl.Expr:
    rng = pl.col("high") - pl.col("low")
    mean = rng.rolling_mean(window_size=window, min_periods=window)
    return safe_div(rng, mean)


@feature("trade_imbalance")
def trade_imbalance(frame: pl.DataFrame) -> pl.Expr:
    """Taker buy volume as a fraction of total volume in [0, 1]."""
    return safe_div(pl.col("taker_buy_volume"), pl.col("volume"))
