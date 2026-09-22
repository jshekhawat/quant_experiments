"""Data validation tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from quant_lab.data.schemas import BAR_SCHEMA, DatasetType
from quant_lab.data.storage import DatasetKey
from quant_lab.data.validate import validate
from quant_lab.synthetic import synthetic_bars

KEY = DatasetKey("synthetic", "test", DatasetType.BARS, "TESTUSDT", "1m")


def test_clean_data_passes():
    report = validate(synthetic_bars(300, seed=11), KEY, timeframe="1m")
    assert report.passed
    assert report.n_errors == 0


def test_duplicate_and_non_monotonic():
    bars = synthetic_bars(10, seed=12)
    dup = pl.concat([bars, bars.head(1)])
    report = validate(dup, KEY, timeframe="1m")
    codes = {i.code for i in report.issues}
    assert "duplicate_timestamps" in codes

    reversed_ = bars.reverse()
    report2 = validate(reversed_, KEY, timeframe="1m")
    assert "non_monotonic_timestamps" in {i.code for i in report2.issues}


def test_invalid_prices_and_volumes():
    bars = synthetic_bars(20, seed=13)
    bad = bars.with_columns(
        pl.when(pl.int_range(0, bars.height) == 5).then(-1.0).otherwise(pl.col("close")).alias("close"),
        pl.when(pl.int_range(0, bars.height) == 6).then(pl.col("low") - 5).otherwise(pl.col("high")).alias("high"),
        pl.when(pl.int_range(0, bars.height) == 7).then(-3.0).otherwise(pl.col("volume")).alias("volume"),
    )
    report = validate(bad, KEY, timeframe="1m")
    codes = {i.code for i in report.issues}
    assert "invalid_price_close" in codes
    assert "high_below_low" in codes
    assert "negative_volume" in codes
    assert not report.passed


def test_missing_timestamps_detected():
    bars = synthetic_bars(60, seed=14)
    gapped = bars.filter(pl.int_range(0, bars.height) != 30)
    report = validate(gapped, KEY, timeframe="1m")
    assert "missing_timestamps" in {i.code for i in report.issues}


def test_naive_timestamps_flagged():
    start = datetime(2024, 1, 1)
    schema = dict(BAR_SCHEMA)
    schema["timestamp"] = pl.Datetime("us")  # naive on purpose
    df = pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=i) for i in range(5)],
            "open": [1.0] * 5,
            "high": [2.0] * 5,
            "low": [0.5] * 5,
            "close": [1.5] * 5,
            "volume": [1.0] * 5,
            "quote_volume": [1.5] * 5,
            "trade_count": [1] * 5,
            "taker_buy_volume": [0.5] * 5,
            "taker_buy_quote_volume": [0.75] * 5,
        },
        schema=schema,
    )
    assert df["timestamp"].dtype.time_zone is None
    report = validate(df, KEY, timeframe="1m")
    assert "naive_timestamps" in {i.code for i in report.issues}
