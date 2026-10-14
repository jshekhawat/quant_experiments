"""Bollinger-basis SMA(20) versus EMA crossover signals.

A fixed **SMA(20)** (the Bollinger middle line) is crossed by an EMA of period
``n``. A bullish cross is the EMA moving from below to above the SMA; a bearish
cross is the mirror. This module provides:

* Bollinger band construction with the repo's established convention (mid =
  rolling mean over the window; bands = mid ± ``num_std`` × rolling **sample**
  standard deviation, ``ddof=1``; bandwidth = ``(upper − lower) / mid``);
* actual crossing-event detection (not a persistent condition);
* target-position construction for the long-only / short-only / long-short
  variants, with optional causal entry-gating or hold-gating filters;
* causal Bollinger-bandwidth / price-position filters.

Backtesting itself reuses :mod:`quant_lab.analytics.ema_backtest`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from . import ema_backtest as eb

# EMA periods evaluated against the fixed SMA(20).
EMA_PERIODS: tuple[int, ...] = (5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200)
SIDES: tuple[str, ...] = ("long", "short", "both")
FORWARD_HORIZONS: dict[str, int] = {"1h": 1, "4h": 4, "8h": 8, "12h": 12, "24h": 24, "48h": 48, "72h": 72}
BOLLINGER_WINDOW = 20
BOLLINGER_NUM_STD = 2.0


def _ffill(a: np.ndarray) -> np.ndarray:
    idx = np.where(~np.isnan(a), np.arange(a.size), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


def bollinger(
    close: np.ndarray, *, window: int = BOLLINGER_WINDOW, num_std: float = BOLLINGER_NUM_STD
) -> dict[str, np.ndarray]:
    """Bollinger mid/upper/lower/bandwidth using the repo's std convention (ddof=1)."""
    s = pl.Series("close", np.asarray(close, dtype=float))
    mid = s.rolling_mean(window_size=window, min_periods=window)
    sd = s.rolling_std(window_size=window, min_periods=window)  # polars default ddof=1
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    width = (upper - lower) / mid
    return {
        "mid": mid.to_numpy(),
        "upper": upper.to_numpy(),
        "lower": lower.to_numpy(),
        "width": width.to_numpy(),
        "std": sd.to_numpy(),
    }


def cross_sign(ema: np.ndarray, sma: np.ndarray) -> np.ndarray:
    """``+1`` while EMA > SMA, ``-1`` while EMA < SMA, ``0`` during warm-up."""
    valid = np.isfinite(ema) & np.isfinite(sma)
    sign = np.zeros(ema.size, dtype=np.int8)
    sign[valid & (ema > sma)] = 1
    sign[valid & (ema < sma)] = -1
    return sign


def cross_events(ema: np.ndarray, sma: np.ndarray) -> np.ndarray:
    """``+1`` on a bullish cross bar, ``-1`` on a bearish cross, else ``0``.

    The first bar on which the sign becomes available also counts as a cross.
    """
    sign = cross_sign(ema, sma)
    valid = sign != 0
    changed = np.zeros(sign.size, dtype=bool)
    changed[1:] = (sign[1:] != sign[:-1]) & (sign[1:] != 0)
    changed |= valid & ~np.concatenate([[False], valid[:-1]])
    ev = np.zeros(sign.size, dtype=np.int8)
    ev[changed] = sign[changed]
    return ev


def cross_positions(
    ema: np.ndarray,
    sma: np.ndarray,
    *,
    side: str = "both",
    allow_long: np.ndarray | None = None,
    allow_short: np.ndarray | None = None,
    mode: str = "none",
) -> np.ndarray:
    """Target position for a crossover system.

    ``mode``:

    * ``none`` — raw crossover state (hold until the opposite cross);
    * ``hold`` — hold only while the filter allows the current side (position is
      flattened when the filter blocks it and re-opens if the filter allows);
    * ``entry`` — a cross only opens a position if the filter allowed it **at the
      cross bar**; a blocked cross leaves the book flat until the next cross.
    """
    sign = cross_sign(ema, sma)
    if side == "long":
        base = np.where(sign == 1, 1, 0).astype(np.int8)
    elif side == "short":
        base = np.where(sign == -1, -1, 0).astype(np.int8)
    elif side == "both":
        base = sign.copy()
    else:
        raise ValueError(f"unknown side {side!r}")

    if mode == "none":
        return base

    al = np.ones(sign.size, dtype=bool) if allow_long is None else np.asarray(allow_long, dtype=bool)
    as_ = np.ones(sign.size, dtype=bool) if allow_short is None else np.asarray(allow_short, dtype=bool)

    if mode == "hold":
        allowed = ((base == 1) & al) | ((base == -1) & as_)
        return np.where(allowed, base, 0).astype(np.int8)

    if mode == "entry":
        changed = np.zeros(sign.size, dtype=bool)
        changed[1:] = (sign[1:] != sign[:-1]) & (sign[1:] != 0)
        changed |= (sign != 0) & ~np.concatenate([[False], (sign != 0)[:-1]])
        if side == "long":
            resolved = np.where(changed, np.where((sign == 1) & al, 1, 0), np.nan)
        elif side == "short":
            resolved = np.where(changed, np.where((sign == -1) & as_, -1, 0), np.nan)
        else:
            resolved = np.where(
                changed,
                np.where((sign == 1) & al, 1, np.where((sign == -1) & as_, -1, 0)),
                np.nan,
            )
        return np.nan_to_num(_ffill(resolved)).astype(np.int8)

    raise ValueError(f"unknown mode {mode!r}")


# --- filters -----------------------------------------------------------------

FILTER_KINDS: tuple[str, ...] = (
    "bw_high",
    "bw_low",
    "bw_expanding",
    "bw_contracting",
    "squeeze_breakout",
    "price_above_sma",  # for longs; shorts use price_below
)


def bandwidth_filters(
    bb_width: np.ndarray,
    close: np.ndarray,
    sma: np.ndarray,
    *,
    lookback: int = 168,
    squeeze_q: float = 0.2,
    squeeze_span: int = 24,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Causal (allow_long, allow_short) arrays for each Experiment-C filter.

    All thresholds are trailing: rolling medians/quantiles use only past bars.
    """
    w = pd.Series(bb_width, dtype=float)
    minp = max(5, lookback // 4)
    roll_med = w.rolling(lookback, min_periods=minp).median()
    roll_q = w.rolling(lookback, min_periods=minp).quantile(squeeze_q)
    expanding = w.diff() > 0
    contracting = w.diff() < 0
    was_squeezed = (w < roll_q).rolling(squeeze_span, min_periods=1).max().fillna(0).astype(bool)
    squeeze_breakout = was_squeezed & expanding

    high = (w > roll_med).fillna(False).to_numpy()
    low = (w < roll_med).fillna(False).to_numpy()
    expand = expanding.fillna(False).to_numpy()
    contract = contracting.fillna(False).to_numpy()
    squeeze = squeeze_breakout.fillna(False).to_numpy()
    above = np.isfinite(sma) & (close > sma)
    below = np.isfinite(sma) & (close < sma)
    ones = np.ones(w.size, dtype=bool)

    return {
        "bw_high": (high, high),
        "bw_low": (low, low),
        "bw_expanding": (expand, expand),
        "bw_contracting": (contract, contract),
        "squeeze_breakout": (squeeze, squeeze),
        "price_above_sma": (above, below),  # longs need above, shorts need below
        "none": (ones, ones),
    }


# --- bundle ------------------------------------------------------------------


@dataclass(slots=True)
class CrossBundle:
    symbol: str
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    funding: np.ndarray
    ema: dict[int, np.ndarray]
    sma: np.ndarray
    bb_mid: np.ndarray
    bb_upper: np.ndarray
    bb_lower: np.ndarray
    bb_width: np.ndarray

    def target(self, period: int, side: str = "both", **kw) -> np.ndarray:
        return cross_positions(self.ema[period], self.sma, side=side, **kw)

    def events(self, period: int) -> np.ndarray:
        return cross_events(self.ema[period], self.sma)


def load_cross_bundle(
    symbol: str,
    *,
    start: str,
    end: str,
    periods: tuple[int, ...] = EMA_PERIODS,
    timeframe: str = "1h",
    source_timeframe: str = "1m",
    market: str = "futures",
    exchange: str = "binance",
    window: int = BOLLINGER_WINDOW,
    num_std: float = BOLLINGER_NUM_STD,
    root: Path | None = None,
    with_funding: bool = True,
) -> CrossBundle | None:
    """Load 1h bars + EMAs (reusing the engine loader) and add SMA(20)/Bollinger."""
    base = eb.load_symbol(
        symbol, start=start, end=end, spans=set(periods), timeframe=timeframe,
        source_timeframe=source_timeframe, market=market, exchange=exchange,
        root=root, with_funding=with_funding,
    )
    if base is None:
        return None
    bands = bollinger(base.close, window=window, num_std=num_std)
    return CrossBundle(
        symbol=base.symbol, timestamps=base.timestamps, open=base.open, high=base.high,
        low=base.low, close=base.close, funding=base.funding, ema=base.ema,
        sma=bands["mid"], bb_mid=bands["mid"], bb_upper=bands["upper"],
        bb_lower=bands["lower"], bb_width=bands["width"],
    )


# --- diagnostics -------------------------------------------------------------


def forward_returns(close: np.ndarray, horizons: dict[str, int] | None = None) -> dict[str, np.ndarray]:
    horizons = horizons or FORWARD_HORIZONS
    return {
        name: np.concatenate([close[h:] / close[:-h] - 1.0, np.full(h, np.nan)])
        for name, h in horizons.items()
    }


__all__ = [
    "BOLLINGER_NUM_STD",
    "BOLLINGER_WINDOW",
    "CrossBundle",
    "EMA_PERIODS",
    "FILTER_KINDS",
    "FORWARD_HORIZONS",
    "SIDES",
    "bandwidth_filters",
    "bollinger",
    "cross_events",
    "cross_positions",
    "cross_sign",
    "forward_returns",
    "load_cross_bundle",
]
