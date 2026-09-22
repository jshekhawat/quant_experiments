"""Bar resampling tests."""

from __future__ import annotations

from quant_lab.data.resample import resample_bars
from quant_lab.synthetic import synthetic_bars


def test_resample_1m_to_1h_ohlcv():
    bars = synthetic_bars(120, seed=51)
    h1 = resample_bars(bars, timeframe="1h", source_timeframe="1m")
    assert h1.height == 2
    first = h1.row(0, named=True)
    assert first["open"] == bars["open"][0]
    assert first["high"] == bars["high"][:60].max()
    assert first["low"] == bars["low"][:60].min()
    assert first["close"] == bars["close"][59]
    assert first["volume"] == bars["volume"][:60].sum()
    assert first["quote_volume"] == bars["quote_volume"][:60].sum()
    assert first["trade_count"] == bars["trade_count"][:60].sum()
    assert first["taker_buy_volume"] == bars["taker_buy_volume"][:60].sum()
    assert first["timestamp"] == bars["timestamp"][0]


def test_resample_15m_bucket_count():
    bars = synthetic_bars(120, seed=52)
    out = resample_bars(bars, timeframe="15m", source_timeframe="1m")
    assert out.height == 8


def test_drop_incomplete_bucket():
    bars = synthetic_bars(90, seed=53)  # one full hour + 30 minutes
    dropped = resample_bars(bars, timeframe="1h", source_timeframe="1m", drop_incomplete=True)
    kept = resample_bars(bars, timeframe="1h", source_timeframe="1m", drop_incomplete=False)
    assert dropped.height == 1
    assert kept.height == 2


def test_resample_same_timeframe_is_noop():
    bars = synthetic_bars(30, seed=54)
    out = resample_bars(bars, timeframe="1m", source_timeframe="1m")
    assert out.height == bars.height


def test_resample_column_order():
    from quant_lab.data.schemas import BAR_SCHEMA

    bars = synthetic_bars(60, seed=55)
    out = resample_bars(bars, timeframe="1h", source_timeframe="1m")
    assert out.columns == list(BAR_SCHEMA)
