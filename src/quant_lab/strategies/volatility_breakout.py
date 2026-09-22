"""Reference strategy: volatility breakout.

Long entry requires all of:

* ``close > EMA(ema_span)``
* the bar's move exceeds a volatility threshold:
  ``close - prev_close > atr_multiplier * ATR(atr_period)``
* volume expansion: ``volume > volume_multiplier * mean(volume, volume_window)``

Optionally mirrored for shorts. All parameters are configurable.

This strategy exists only to demonstrate the full pipeline; it is not claimed
to be profitable.
"""

from __future__ import annotations

import polars as pl

from ..features.core import atr as atr_expr
from ..features.core import ema as ema_expr
from ..features.core import safe_div
from .base import Strategy, register_strategy


@register_strategy
class VolatilityBreakout(Strategy):
    name = "volatility_breakout"
    default_params = {
        "ema_span": 100,
        "atr_period": 14,
        "atr_multiplier": 1.5,
        "volume_window": 20,
        "volume_multiplier": 1.5,
        "allow_short": False,
        "exit_on_ema_cross": True,
    }
    param_grid = {
        "ema_span": [50, 100, 150, 200],
        "atr_period": [10, 14, 20],
        "volume_multiplier": [1.0, 1.25, 1.5, 2.0],
    }

    def features(self, data: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        return data.with_columns(
            ema_expr(data, span=p["ema_span"]).alias("ema"),
            atr_expr(data, period=p["atr_period"]).alias("atr"),
            pl.col("volume")
            .rolling_mean(window_size=p["volume_window"], min_periods=p["volume_window"])
            .alias("volume_mean"),
        ).with_columns(
            safe_div(pl.col("volume"), pl.col("volume_mean")).alias("relative_volume")
        )

    def signals(self, data: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        prev_close = pl.col("close").shift(1)
        abs_range = (pl.col("close") - prev_close).abs()
        moved = abs_range > p["atr_multiplier"] * pl.col("atr")
        volume_ok = pl.col("relative_volume") > p["volume_multiplier"]
        above = pl.col("close") > pl.col("ema")
        below = pl.col("close") < pl.col("ema")

        long_entry = above & moved & volume_ok
        long_exit = below if p["exit_on_ema_cross"] else pl.lit(False)

        if p["allow_short"]:
            short_entry = below & moved & volume_ok
            short_exit = above if p["exit_on_ema_cross"] else pl.lit(False)
        else:
            short_entry = pl.lit(False)
            short_exit = pl.lit(False)

        return data.with_columns(
            long_entry.fill_null(False).alias("entries"),
            long_exit.fill_null(False).alias("exits"),
            short_entry.fill_null(False).alias("short_entries"),
            short_exit.fill_null(False).alias("short_exits"),
        )
