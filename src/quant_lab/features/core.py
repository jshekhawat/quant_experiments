"""Basic, reusable indicator features.

All functions return a lazy :class:`polars.Expr` so they compose and so that
Polars can push them through the query plan. Every window is trailing
(backward-looking), which is the first line of defence against lookahead bias.
"""

from __future__ import annotations

import math

import polars as pl

from .base import feature


def safe_div(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator == 0).then(None).otherwise(numerator / denominator)


@feature("sma")
def sma(frame: pl.DataFrame, *, window: int = 20, column: str = "close") -> pl.Expr:
    return pl.col(column).rolling_mean(window_size=window, min_periods=window)


@feature("ema")
def ema(frame: pl.DataFrame, *, span: int = 20, column: str = "close") -> pl.Expr:
    return pl.col(column).ewm_mean(span=span, adjust=False, min_periods=span)


@feature("true_range")
def true_range(frame: pl.DataFrame) -> pl.Expr:
    prev_close = pl.col("close").shift(1)
    return pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )


@feature("atr")
def atr(frame: pl.DataFrame, *, period: int = 14) -> pl.Expr:
    tr = true_range(frame)
    return tr.ewm_mean(alpha=1.0 / period, adjust=False, min_periods=period)


@feature("rsi")
def rsi(frame: pl.DataFrame, *, period: int = 14, column: str = "close") -> pl.Expr:
    delta = pl.col(column).diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gain.ewm_mean(alpha=1.0 / period, adjust=False, min_periods=period)
    avg_loss = loss.ewm_mean(alpha=1.0 / period, adjust=False, min_periods=period)
    rs = safe_div(avg_gain, avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


@feature("vwap")
def vwap(frame: pl.DataFrame, *, window: int = 20) -> pl.Expr:
    typical = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    pv = (typical * pl.col("volume")).rolling_sum(window_size=window, min_periods=window)
    vol = pl.col("volume").rolling_sum(window_size=window, min_periods=window)
    return safe_div(pv, vol)


@feature("returns")
def returns(
    frame: pl.DataFrame, *, period: int = 1, column: str = "close", log: bool = False
) -> pl.Expr:
    if log:
        return (pl.col(column) / pl.col(column).shift(period)).log()
    return pl.col(column).pct_change(n=period)


@feature("rolling_volume")
def rolling_volume(frame: pl.DataFrame, *, window: int = 20, column: str = "volume") -> pl.Expr:
    return pl.col(column).rolling_mean(window_size=window, min_periods=window)


@feature("realized_volatility")
def realized_volatility(
    frame: pl.DataFrame,
    *,
    window: int = 20,
    period: int = 1,
    column: str = "close",
    annualization: float | None = None,
) -> pl.Expr:
    r = returns(frame, period=period, column=column, log=True)
    vol = r.rolling_std(window_size=window, min_periods=window)
    if annualization:
        vol = vol * math.sqrt(annualization)
    return vol


@feature("zscore")
def zscore(frame: pl.DataFrame, *, window: int = 20, column: str = "close") -> pl.Expr:
    mean = pl.col(column).rolling_mean(window_size=window, min_periods=window)
    std = pl.col(column).rolling_std(window_size=window, min_periods=window)
    return safe_div(pl.col(column) - mean, std)


@feature("roc")
def roc(frame: pl.DataFrame, *, period: int = 1, column: str = "close") -> pl.Expr:
    return pl.col(column).pct_change(n=period)


@feature("stdev")
def stdev(frame: pl.DataFrame, *, window: int = 20, column: str = "close") -> pl.Expr:
    return pl.col(column).rolling_std(window_size=window, min_periods=window)


@feature("rolling_max")
def rolling_max(frame: pl.DataFrame, *, window: int = 20, column: str = "close") -> pl.Expr:
    return pl.col(column).rolling_max(window_size=window, min_periods=window)


@feature("rolling_min")
def rolling_min(frame: pl.DataFrame, *, window: int = 20, column: str = "close") -> pl.Expr:
    return pl.col(column).rolling_min(window_size=window, min_periods=window)


@feature("shift")
def shift(frame: pl.DataFrame, *, periods: int = 1, column: str = "close") -> pl.Expr:
    return pl.col(column).shift(periods)


@feature("momentum")
def momentum(frame: pl.DataFrame, *, window: int = 20, column: str = "close") -> pl.Expr:
    return pl.col(column) - pl.col(column).shift(window)


@feature("price_range")
def price_range(frame: pl.DataFrame) -> pl.Expr:
    """Bar range relative to close (a simple realized-range measure)."""
    return safe_div(pl.col("high") - pl.col("low"), pl.col("close"))


def _bollinger_mid_expr(column: str, window: int) -> pl.Expr:
    return pl.col(column).rolling_mean(window_size=window, min_periods=window)


def _bollinger_std_expr(column: str, window: int) -> pl.Expr:
    return pl.col(column).rolling_std(window_size=window, min_periods=window)


@feature("bollinger_mid")
def bollinger_mid(frame: pl.DataFrame, *, window: int = 20, column: str = "close") -> pl.Expr:
    return _bollinger_mid_expr(column, window)


@feature("bollinger_upper")
def bollinger_upper(
    frame: pl.DataFrame, *, window: int = 20, num_std: float = 2.0, column: str = "close"
) -> pl.Expr:
    return _bollinger_mid_expr(column, window) + num_std * _bollinger_std_expr(column, window)


@feature("bollinger_lower")
def bollinger_lower(
    frame: pl.DataFrame, *, window: int = 20, num_std: float = 2.0, column: str = "close"
) -> pl.Expr:
    return _bollinger_mid_expr(column, window) - num_std * _bollinger_std_expr(column, window)


@feature("bollinger_bandwidth")
def bollinger_bandwidth(
    frame: pl.DataFrame, *, window: int = 20, num_std: float = 2.0, column: str = "close"
) -> pl.Expr:
    mid = _bollinger_mid_expr(column, window)
    width = 2.0 * num_std * _bollinger_std_expr(column, window)
    return safe_div(width, mid)


@feature("bollinger_percent_b")
def bollinger_percent_b(
    frame: pl.DataFrame, *, window: int = 20, num_std: float = 2.0, column: str = "close"
) -> pl.Expr:
    mid = _bollinger_mid_expr(column, window)
    std = _bollinger_std_expr(column, window)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return safe_div(pl.col(column) - lower, upper - lower)
