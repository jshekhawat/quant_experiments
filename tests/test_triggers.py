"""Tests for the squeeze-release and range-breakout-hold triggers."""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_lab.strategies.base import SIGNAL_COLUMNS
from quant_lab.strategies.range_breakout_hold import RangeBreakoutHold
from quant_lab.strategies.squeeze_release import SqueezeRelease


def _causal(strategy, bars):
    cutoff = 400
    full = strategy.prepare(bars)
    part = strategy.prepare(bars.head(cutoff))
    for col in SIGNAL_COLUMNS:
        assert np.array_equal(
            full[col].head(cutoff).to_numpy(), part[col].to_numpy()
        ), f"{col} leaks future information"


def test_squeeze_release_columns_and_causality(bars):
    out = SqueezeRelease().prepare(bars)
    for col in SIGNAL_COLUMNS:
        assert col in out.columns
        assert out[col].dtype == pl.Boolean
    _causal(SqueezeRelease(), bars)


def test_range_breakout_hold_columns_and_causality(bars):
    out = RangeBreakoutHold().prepare(bars)
    for col in SIGNAL_COLUMNS:
        assert col in out.columns
    _causal(RangeBreakoutHold(), bars)


def _rising_breakout_frame() -> pl.DataFrame:
    n = 90
    close = [100.0] * 60 + [100.0 + i for i in range(1, n - 60 + 1)]
    high = [c + 0.1 for c in close]
    low = [c - 0.1 for c in close]
    ts = pl.datetime_range(
        pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 2, 12), interval="1m", eager=True
    ).head(n)
    return pl.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1.0] * n,
            "quote_volume": close,
            "trade_count": [1] * n,
            "taker_buy_volume": [0.5] * n,
            "taker_buy_quote_volume": [c * 0.5 for c in close],
        }
    )


def test_range_breakout_hold_triggers_after_hold():
    frame = _rising_breakout_frame()
    out = RangeBreakoutHold(lookback=60, hold_bars=2, exit_lookback=30, trend_ema=0).prepare(frame)
    assert out["entries"].any()
    idx = int(out.with_row_index("i").filter(pl.col("entries"))["i"][0])
    # entry occurs at the second consecutive close above the prior range high
    assert idx == 61


def test_range_breakout_hold_single_bar_is_filtered():
    """hold_bars=1 fires on the first breakout bar; hold_bars=2 is stricter."""
    frame = _rising_breakout_frame()
    h1 = RangeBreakoutHold(lookback=60, hold_bars=1, exit_lookback=30, trend_ema=0).prepare(frame)
    h2 = RangeBreakoutHold(lookback=60, hold_bars=2, exit_lookback=30, trend_ema=0).prepare(frame)
    assert int(h1["entries"].sum()) >= int(h2["entries"].sum())
    idx1 = int(h1.with_row_index("i").filter(pl.col("entries"))["i"][0])
    assert idx1 == 60
