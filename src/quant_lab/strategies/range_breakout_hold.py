"""Range breakout with a hold requirement.

Enter long when price closes above the prior ``lookback``-bar high and then
*holds* above that level for ``hold_bars`` consecutive closes (a fresh hold, so
it fires once). Exit when price closes below the prior ``exit_lookback``-bar
low, or below the trend EMA if configured.

Requiring a hold filters out single-bar spikes that immediately fail — the
behaviour the excursion study showed dominates raw breakouts.
"""

from __future__ import annotations

import polars as pl

from ..features.core import ema as ema_expr
from .base import Strategy, register_strategy


@register_strategy
class RangeBreakoutHold(Strategy):
    name = "range_breakout_hold"
    default_params = {
        "lookback": 60,
        "hold_bars": 2,
        "exit_lookback": 30,
        "trend_ema": 200,
        "allow_short": False,
    }
    param_grid = {
        "lookback": [30, 60, 120],
        "hold_bars": [1, 2, 3],
        "exit_lookback": [10, 30, 60],
    }

    def features(self, data: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        out = data.with_columns(
            pl.col("high")
            .rolling_max(window_size=p["lookback"], min_periods=p["lookback"])
            .shift(1)
            .alias("prior_high"),
            pl.col("low")
            .rolling_min(window_size=p["lookback"], min_periods=p["lookback"])
            .shift(1)
            .alias("prior_low"),
            pl.col("low")
            .rolling_min(window_size=p["exit_lookback"], min_periods=p["exit_lookback"])
            .shift(1)
            .alias("exit_low"),
            pl.col("high")
            .rolling_max(window_size=p["exit_lookback"], min_periods=p["exit_lookback"])
            .shift(1)
            .alias("exit_high"),
        )
        if p["trend_ema"]:
            out = out.with_columns(ema_expr(data, span=p["trend_ema"]).alias("trend_ema"))
        return out

    def signals(self, data: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        close = pl.col("close")
        hold = p["hold_bars"]

        above = (close > pl.col("prior_high")).cast(pl.Int8)
        below = (close < pl.col("prior_low")).cast(pl.Int8)
        hold_up = (
            above.rolling_min(window_size=hold, min_periods=hold).fill_null(0) == 1
        )
        hold_dn = (
            below.rolling_min(window_size=hold, min_periods=hold).fill_null(0) == 1
        )

        trend_ok_long = pl.lit(True)
        trend_ok_short = pl.lit(True)
        if p["trend_ema"]:
            trend_ok_long = close > pl.col("trend_ema")
            trend_ok_short = close < pl.col("trend_ema")

        long_entry = hold_up & (~hold_up.shift(1).fill_null(False)) & trend_ok_long
        long_exit = close < pl.col("exit_low")

        if p["allow_short"]:
            short_entry = hold_dn & (~hold_dn.shift(1).fill_null(False)) & trend_ok_short
            short_exit = close > pl.col("exit_high")
        else:
            short_entry = pl.lit(False)
            short_exit = pl.lit(False)

        return data.with_columns(
            long_entry.fill_null(False).alias("entries"),
            long_exit.fill_null(False).alias("exits"),
            short_entry.fill_null(False).alias("short_entries"),
            short_exit.fill_null(False).alias("short_exits"),
        )
