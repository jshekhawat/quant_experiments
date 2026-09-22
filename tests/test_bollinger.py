"""Bollinger feature and breakout strategy tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from quant_lab.features.core import bollinger_lower, bollinger_mid, bollinger_upper
from quant_lab.strategies.base import SIGNAL_COLUMNS
from quant_lab.strategies.bollinger_breakout import BollingerBreakout


def test_bollinger_matches_pandas(bars):
    out = bars.with_columns(
        bollinger_mid(bars, window=20).alias("mid"),
        bollinger_upper(bars, window=20, num_std=2.0).alias("upper"),
        bollinger_lower(bars, window=20, num_std=2.0).alias("lower"),
    )
    close = pd.Series(bars["close"].to_numpy())
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    mask = mid.notna().to_numpy()
    assert np.allclose(out["mid"].to_numpy()[mask], mid.to_numpy()[mask])
    assert np.allclose(out["upper"].to_numpy()[mask], (mid + 2 * sd).to_numpy()[mask])
    assert np.allclose(out["lower"].to_numpy()[mask], (mid - 2 * sd).to_numpy()[mask])


def test_bollinger_band_ordering(bars):
    out = bars.with_columns(
        bollinger_upper(bars, window=20).alias("upper"),
        bollinger_lower(bars, window=20).alias("lower"),
    ).drop_nulls(["upper", "lower"])
    assert (out["upper"] >= out["lower"]).all()


def test_breakout_signals_are_causal(bars):
    cutoff = 400
    full = BollingerBreakout().prepare(bars)
    part = BollingerBreakout().prepare(bars.head(cutoff))
    for col in SIGNAL_COLUMNS:
        assert np.array_equal(
            full[col].head(cutoff).to_numpy(), part[col].to_numpy()
        ), f"{col} leaks future information"


def test_breakout_triggers_on_upper_band_cross():
    n = 40
    close = [100.0] * 20 + [100.0 + i for i in range(1, n - 20 + 1)]
    open_ = close
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    ts = pl.datetime_range(
        pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 1, 1), interval="1m", eager=True
    ).head(n)
    frame = pl.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
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
    out = BollingerBreakout(window=10, num_std=1.0).prepare(frame)
    assert out["entries"].any()
    # first entry is on a close above the prior-bar upper band
    idx = int(out.with_row_index("i").filter(pl.col("entries"))["i"][0])
    assert out["close"][idx] > out["bb_upper"][idx - 1]
