"""Composite cross-venue dataset tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.data.composite import (
    composite_funding,
    composite_volume,
    cross_venue_spread,
)
from quant_lab.data.schemas import FUNDING_SCHEMA, DatasetType, cast_canonical


def _funding(rates: list[float], symbol: str) -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    df = pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=8 * i) for i in range(len(rates))],
            "symbol": [symbol] * len(rates),
            "funding_rate": rates,
            "funding_interval_hours": [8] * len(rates),
        },
        schema=FUNDING_SCHEMA,
    )
    return cast_canonical(df, DatasetType.FUNDING)


def test_composite_funding_equal_weight():
    a = _funding([0.0001, 0.0002], "BTCUSDT")
    b = _funding([0.0003, 0.0004], "BTCUSDT")
    comp = composite_funding({"venueA": a, "venueB": b})
    assert comp.frame["funding_rate"].to_list() == pytest.approx([0.0002, 0.0003])
    assert comp.methodology["venues"] == ["venueA", "venueB"]
    assert comp.methodology["aggregate"] == "weighted_mean"
    assert "timestamp_alignment" in comp.methodology


def test_composite_funding_custom_weights():
    a = _funding([0.0002], "BTCUSDT")
    b = _funding([0.0000], "BTCUSDT")
    comp = composite_funding({"venueA": a, "venueB": b}, weights={"venueA": 3.0, "venueB": 1.0})
    assert comp.frame["funding_rate"][0] == pytest.approx(0.00015)


def test_cross_venue_spread():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    a = pl.DataFrame({"timestamp": [start], "close": [100.0]})
    b = pl.DataFrame({"timestamp": [start], "close": [102.0]})
    spread = cross_venue_spread({"venueA": a, "venueB": b})
    assert spread.frame["cross_venue_spread"][0] == pytest.approx((102 - 100) / 101)


def test_composite_volume_is_sum():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    a = pl.DataFrame({"timestamp": [start], "quote_volume": [10.0]})
    b = pl.DataFrame({"timestamp": [start], "quote_volume": [5.0]})
    comp = composite_volume({"a": a, "b": b})
    assert comp.frame["quote_volume"][0] == pytest.approx(15.0)
