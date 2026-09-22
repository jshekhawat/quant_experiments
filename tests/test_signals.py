"""Strategy signal tests."""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_lab.strategies.base import SIGNAL_COLUMNS
from quant_lab.strategies.volatility_breakout import VolatilityBreakout


def test_signal_columns_present(bars):
    out = VolatilityBreakout().prepare(bars)
    for col in SIGNAL_COLUMNS:
        assert col in out.columns
        assert out[col].dtype == pl.Boolean


def test_signals_are_causal(bars):
    cutoff = 400
    full = VolatilityBreakout().prepare(bars)
    part = VolatilityBreakout().prepare(bars.head(cutoff))
    for col in SIGNAL_COLUMNS:
        a = full[col].head(cutoff).to_numpy()
        b = part[col].to_numpy()
        assert np.array_equal(a, b), f"{col} changes when future data is appended"


def _breakout_frame() -> pl.DataFrame:
    n = 60
    close = [100.0] * n
    close[40] = 130.0  # sharp move up
    close[41:] = [130.0] * (n - 41)
    open_ = [100.0] * n
    open_[40] = 100.0
    high = [c + 1 for c in close]
    low = [c - 1 for c in close]
    volume = [10.0] * n
    volume[40] = 100.0  # volume spike
    ts = pl.datetime_range(
        pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 1, 1), interval="1m", eager=True
    ).head(n)
    return pl.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": [v * c for v, c in zip(volume, close, strict=True)],
            "trade_count": [1] * n,
            "taker_buy_volume": [v / 2 for v in volume],
            "taker_buy_quote_volume": [v / 2 * c for v, c in zip(volume, close, strict=True)],
        }
    )


def test_breakout_triggers_entry():
    frame = _breakout_frame()
    strategy = VolatilityBreakout(
        ema_span=10, atr_period=5, atr_multiplier=0.5, volume_window=10, volume_multiplier=1.0
    )
    out = strategy.prepare(frame)
    assert bool(out["entries"][40]) is True


def test_unknown_parameter_rejected():
    import pytest

    with pytest.raises(TypeError):
        VolatilityBreakout(nonexistent=1)
