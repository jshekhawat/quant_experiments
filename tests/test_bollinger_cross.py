"""Tests for the Bollinger-basis SMA/EMA crossover signals (analytics.bollinger_cross)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.analytics import bollinger_cross as bc


def test_bollinger_matches_rolling_convention():
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, 300))
    bands = bc.bollinger(close, window=20, num_std=2.0)
    s = pd.Series(close)
    mid = s.rolling(20, min_periods=20).mean().to_numpy()
    sd = s.rolling(20, min_periods=20).std(ddof=1).to_numpy()
    assert np.allclose(bands["mid"], mid, equal_nan=True)
    assert np.allclose(bands["upper"] - bands["lower"], 4.0 * sd, equal_nan=True)
    assert np.allclose(bands["width"], (bands["upper"] - bands["lower"]) / bands["mid"], equal_nan=True)


def test_cross_sign_and_events():
    ema = np.array([np.nan, np.nan, 2.0, 3.0, -1.0, -1.0, -1.0, 0.5, 0.5])
    sma = np.zeros_like(ema)
    sign = bc.cross_sign(ema, sma)
    assert list(sign) == [0, 0, 1, 1, -1, -1, -1, 1, 1]
    ev = bc.cross_events(ema, sma)
    # first valid bar counts as a cross; then sign flips
    assert list(ev) == [0, 0, 1, 0, -1, 0, 0, 1, 0]


def test_cross_positions_sides_and_modes():
    ema = np.array([1.0, -1.0, -1.0, 1.0, 1.0])
    sma = np.zeros_like(ema)
    both = bc.cross_positions(ema, sma, side="both")
    assert list(both) == [1, -1, -1, 1, 1]
    long = bc.cross_positions(ema, sma, side="long")
    assert list(long) == [1, 0, 0, 1, 1]
    short = bc.cross_positions(ema, sma, side="short")
    assert list(short) == [0, -1, -1, 0, 0]

    # entry gating: block all longs -> long book stays flat
    block = np.zeros(5, dtype=bool)
    assert list(bc.cross_positions(ema, sma, side="long", allow_long=block, mode="entry")) == [0, 0, 0, 0, 0]
    # hold gating: same for long-only
    assert list(bc.cross_positions(ema, sma, side="long", allow_long=block, mode="hold")) == [0, 0, 0, 0, 0]
    # entry gate that blocks only the first cross still enters on the second
    allow = np.array([False, True, True, True, True])
    assert list(bc.cross_positions(ema, sma, side="long", allow_long=allow, mode="entry")) == [0, 0, 0, 1, 1]


def test_bandwidth_filters_are_causal():
    rng = np.random.default_rng(1)
    width = np.abs(rng.normal(0.05, 0.01, 500))
    close = 100 + np.cumsum(rng.normal(0, 1, 500))
    sma = pd.Series(close).rolling(20, min_periods=20).mean().to_numpy()
    filters = bc.bandwidth_filters(width, close, sma, lookback=100)
    assert set(["bw_high", "bw_low", "bw_expanding", "bw_contracting", "squeeze_breakout"]).issubset(filters)

    width2 = width.copy()
    width2[400:] *= 5.0
    f2 = bc.bandwidth_filters(width2, close, sma, lookback=100)
    for key in ("bw_high", "bw_low", "bw_expanding", "bw_contracting"):
        a, b = filters[key][0], f2[key][0]
        assert np.array_equal(a[:400], b[:400])  # future changes cannot alter past allowances


def test_forward_returns():
    close = np.array([100.0, 110.0, 121.0, 133.1])
    fwd = bc.forward_returns(close, {"1h": 1, "2h": 2})
    assert np.isclose(fwd["1h"][0], 0.10) and np.isclose(fwd["1h"][1], 0.10)
    assert np.isnan(fwd["1h"][-1])
    assert np.isclose(fwd["2h"][0], 0.21) and np.isnan(fwd["2h"][-1]) and np.isnan(fwd["2h"][-2])
