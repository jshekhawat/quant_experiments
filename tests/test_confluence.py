"""Confluence filter evaluation tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.analytics.confluence import (
    Filter,
    LabelConfig,
    add_confluence_features,
    combine_filters,
    evaluate_confluence,
    label_signals,
)


def _labelled_frame() -> pl.DataFrame:
    # 4 signals; 2 are "good". Feature x selects signals 0 and 2.
    return pl.DataFrame(
        {
            "entries": [True, True, True, True, False],
            "is_good": [True, False, True, False, False],
            "excursion_to_reversion_pct": [0.01, 0.001, 0.02, 0.0005, 0.0],
            "time_to_reversion_seconds": [7200.0, 120.0, 10800.0, 60.0, 60.0],
            "x": [1, 0, 1, 0, 0],
        }
    )


def test_evaluate_confluence_baseline_and_lift():
    table = evaluate_confluence(
        _labelled_frame(), [Filter("x=1", pl.col("x") == 1, ("x",))]
    )
    rows = {r["filter"]: r for r in table.iter_rows(named=True)}
    assert rows["BASELINE (all signals)"]["n_signals"] == 4
    assert rows["BASELINE (all signals)"]["p_good"] == 0.5
    assert rows["x=1"]["n_signals"] == 2
    assert rows["x=1"]["p_good"] == 1.0
    assert rows["x=1"]["lift"] == 2.0


def test_missing_feature_filter_is_skipped():
    table = evaluate_confluence(
        _labelled_frame(), [Filter("nope", pl.col("does_not_exist") > 0, ("does_not_exist",))]
    )
    assert table.height == 1  # only the baseline row


def test_combine_filters_requires_union():
    a = Filter("a", pl.col("x") > 0, ("x",))
    b = Filter("y", pl.col("y") > 0, ("y",))
    combo = combine_filters([a, b])
    assert set(combo.requires) == {"x", "y"}
    assert "a" in combo.name and "y" in combo.name


def test_add_confluence_features(bars):
    out = add_confluence_features(bars)
    for col in ("relative_volume", "taker_ratio", "cvd_change_20", "ema_200", "close_location"):
        assert col in out.columns
    # trailing window -> first 19 relative-volume values are null
    assert out["relative_volume"].head(19).null_count() == 19
    assert out["relative_volume"].drop_nulls().min() > 0


def _long_run_frame() -> pl.DataFrame:
    """Signal at bar 0; rises 10% over 50 bars, reverts to 99 at bar 120."""
    n = 200
    close = [100.0] * n
    for i in range(1, 51):
        close[i] = 100.0 + i * 0.2  # up to 110
    for i in range(51, 120):
        close[i] = 110.0
    for i in range(120, n):
        close[i] = 99.0
    high = [c + 0.1 for c in close]
    low = [min(c, 100.0) - 0.1 for c in close]
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=i) for i in range(n)],
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1.0] * n,
            "quote_volume": close,
            "trade_count": [1] * n,
            "taker_buy_volume": [0.5] * n,
            "taker_buy_quote_volume": [c * 0.5 for c in close],
            "entries": [True] + [False] * (n - 1),
            "exits": [False] * n,
            "short_entries": [False] * n,
            "short_exits": [False] * n,
        }
    )


def test_label_signals_good_outcome():
    labelled = label_signals(_long_run_frame(), label=LabelConfig(horizon_bars=1000))
    row = labelled.filter(pl.col("entries")).row(0, named=True)
    assert row["is_good"] is True
    assert row["time_to_reversion_seconds"] > 3600


def test_label_signals_fast_reversion_is_not_good():
    frame = _long_run_frame().with_columns(
        pl.when(pl.int_range(0, 200) >= 3).then(99.0).otherwise(pl.col("close")).alias("close")
    )
    labelled = label_signals(frame, label=LabelConfig(horizon_bars=1000))
    row = labelled.filter(pl.col("entries")).row(0, named=True)
    assert row["is_good"] is False
