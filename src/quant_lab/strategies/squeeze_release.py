"""Squeeze-release breakout strategy.

A Bollinger breakout that is only taken when the bands have recently been
*compressed* (a volatility squeeze) and are now expanding. The idea: a band
cross after a quiet period is more likely to be a genuine range expansion than
a cross during an already-volatile market.

All parameters configurable; windows are trailing.
"""

from __future__ import annotations

import polars as pl

from ..features.core import ema as ema_expr
from ..features.core import safe_div
from .base import Strategy, register_strategy


@register_strategy
class SqueezeRelease(Strategy):
    name = "squeeze_release"
    default_params = {
        "window": 20,
        "num_std": 2.0,
        "bandwidth_window": 240,
        "squeeze_z": -0.5,
        "squeeze_lookback": 96,
        "require_expansion": True,
        "trend_ema": 0,
        "allow_short": False,
    }
    param_grid = {
        "num_std": [1.5, 2.0, 2.5],
        "squeeze_z": [-1.0, -0.5, 0.0],
        "squeeze_lookback": [48, 96, 192],
    }

    def features(self, data: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        w, n = p["window"], p["num_std"]
        mid = pl.col("close").rolling_mean(window_size=w, min_periods=w)
        std = pl.col("close").rolling_std(window_size=w, min_periods=w)
        out = data.with_columns(
            mid.alias("bb_mid"),
            (mid + n * std).alias("bb_upper"),
            (mid - n * std).alias("bb_lower"),
        ).with_columns(
            safe_div(pl.col("bb_upper") - pl.col("bb_lower"), pl.col("bb_mid")).alias("bandwidth")
        )
        bw_win = p["bandwidth_window"]
        out = out.with_columns(
            safe_div(
                pl.col("bandwidth")
                - pl.col("bandwidth").rolling_mean(window_size=bw_win, min_periods=bw_win),
                pl.col("bandwidth").rolling_std(window_size=bw_win, min_periods=bw_win),
            ).alias("bandwidth_z"),
            (pl.col("bandwidth") > pl.col("bandwidth").shift(1)).alias("bandwidth_rising"),
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
        cross_dn_band = (close < lower) & (prev_close >= lower.shift(1))

        squeezed_recently = (
            (pl.col("bandwidth_z") <= p["squeeze_z"])
            .cast(pl.Int8)
            .rolling_max(window_size=p["squeeze_lookback"], min_periods=1)
            .fill_null(0)
            > 0
        )
        expansion = pl.col("bandwidth_rising") if p["require_expansion"] else pl.lit(True)

        long_entry = cross_up & squeezed_recently & expansion
        long_exit = close < mid

        if p["allow_short"]:
            short_entry = cross_dn_band & squeezed_recently & expansion
            short_exit = close > mid
        else:
            short_entry = pl.lit(False)
            short_exit = pl.lit(False)

        return data.with_columns(
            long_entry.fill_null(False).alias("entries"),
            long_exit.fill_null(False).alias("exits"),
            short_entry.fill_null(False).alias("short_entries"),
            short_exit.fill_null(False).alias("short_exits"),
        )
