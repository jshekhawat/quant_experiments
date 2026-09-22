"""Bollinger Band breakout strategy.

Classic momentum variant of Bollinger Bands:

* **long entry** when the close crosses *above* the upper band
  ``mid + num_std * stdev(close, window)``;
* **exit** when the close crosses back *below* the middle band (default), or
  below the lower band if ``exit_on_mid`` is disabled;
* optional EMA trend filter (``trend_ema``, disabled by default);
* optional mirrored shorts.

Crossing conditions (rather than "close is above the band") prevent pyramiding
into an already-open position. All windows are trailing, so the strategy is
causal.
"""

from __future__ import annotations

import polars as pl

from ..features.core import ema as ema_expr
from .base import Strategy, register_strategy


@register_strategy
class BollingerBreakout(Strategy):
    name = "bollinger_breakout"
    default_params = {
        "window": 20,
        "num_std": 2.0,
        "exit_on_mid": True,
        "trend_ema": 0,
        "allow_short": False,
    }
    param_grid = {
        "window": [10, 20, 50],
        "num_std": [1.5, 2.0, 2.5],
    }

    def features(self, data: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        window, num_std = p["window"], p["num_std"]
        mid = pl.col("close").rolling_mean(window_size=window, min_periods=window)
        std = pl.col("close").rolling_std(window_size=window, min_periods=window)
        out = data.with_columns(
            mid.alias("bb_mid"),
            (mid + num_std * std).alias("bb_upper"),
            (mid - num_std * std).alias("bb_lower"),
        )
        if p["trend_ema"]:
            out = out.with_columns(ema_expr(data, span=p["trend_ema"]).alias("trend_ema"))
        return out

    def signals(self, data: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        close = pl.col("close")
        prev_close = close.shift(1)

        upper, mid, lower = pl.col("bb_upper"), pl.col("bb_mid"), pl.col("bb_lower")
        cross_up = (close > upper) & (prev_close <= upper.shift(1))

        exit_level = mid if p["exit_on_mid"] else lower
        cross_dn = (close < exit_level) & (prev_close >= exit_level.shift(1))

        trend_ok_long = pl.lit(True)
        trend_ok_short = pl.lit(True)
        if p["trend_ema"]:
            trend_ok_long = close > pl.col("trend_ema")
            trend_ok_short = close < pl.col("trend_ema")

        long_entry = cross_up & trend_ok_long
        long_exit = cross_dn

        if p["allow_short"]:
            cross_dn_band = (close < lower) & (prev_close >= lower.shift(1))
            short_entry = cross_dn_band & trend_ok_short
            short_exit = (close > mid) & (prev_close <= mid.shift(1))
        else:
            short_entry = pl.lit(False)
            short_exit = pl.lit(False)

        return data.with_columns(
            long_entry.fill_null(False).alias("entries"),
            long_exit.fill_null(False).alias("exits"),
            short_entry.fill_null(False).alias("short_entries"),
            short_exit.fill_null(False).alias("short_exits"),
        )
