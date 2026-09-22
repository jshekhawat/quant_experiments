"""Post-signal excursion analysis tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.analytics.excursions import analyze_excursions


def _frame() -> pl.DataFrame:
    # Signal at bar 0 (close 100). Path rises to 110 (max excursion 10) then
    # closes back to 99. Bar 2 wicks to 99 but closes at 104.
    close = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 105.0, 99.0, 98.0]
    high = [100.0, 103.0, 105.0, 107.0, 109.0, 110.0, 106.0, 100.0, 99.0]
    low = [100.0, 100.5, 99.0, 105.0, 107.0, 109.0, 104.0, 98.0, 97.0]
    n = len(close)
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
            "atr": [4.0] * n,
            "entries": [True] + [False] * (n - 1),
            "exits": [False] * n,
            "short_entries": [False] * n,
            "short_exits": [False] * n,
        }
    )


def test_pct_excursion_timing():
    result = analyze_excursions(
        _frame(), mode="pct_excursion", targets=(0.5, 0.9), horizon_bars=100
    )
    event = result.events.row(0, named=True)
    assert event["max_favourable_move"] == 10.0
    assert event["time_to_max_bars"] == 5
    assert event["hit_50pct"] is True
    assert event["time_to_50pct_bars"] == 2  # level 105 reached at bar 2
    assert event["hit_90pct"] is True
    assert event["time_to_90pct_bars"] == 4  # level 109 reached at bar 4


def test_atr_mode_timing():
    result = analyze_excursions(
        _frame(), mode="atr", targets=(0.5, 0.9), horizon_bars=100, atr_col="atr"
    )
    event = result.events.row(0, named=True)
    assert event["time_to_50pct_bars"] == 1  # level 102 (0.5*4)
    assert event["time_to_90pct_bars"] == 2  # level 103.6 (0.9*4)


def test_pct_signal_never_reached():
    result = analyze_excursions(
        _frame(), mode="pct_signal", targets=(0.5, 0.9), horizon_bars=100
    )
    event = result.events.row(0, named=True)
    assert event["hit_50pct"] is False
    assert event["hit_90pct"] is False


def test_reversion_basis_close_vs_touch():
    close_basis = analyze_excursions(
        _frame(), mode="pct_excursion", targets=(0.5,), horizon_bars=100,
        reversion_basis="close",
    ).events.row(0, named=True)
    touch_basis = analyze_excursions(
        _frame(), mode="pct_excursion", targets=(0.5,), horizon_bars=100,
        reversion_basis="touch",
    ).events.row(0, named=True)
    # close basis reverts at bar 7 (close 99); touch basis at bar 2 (low 99)
    assert close_basis["reverted"] is True
    assert close_basis["time_to_reversion_bars"] == 7
    assert touch_basis["time_to_reversion_bars"] == 2


def test_horizon_censoring():
    # +6% level (106) is reached at bar 3 (high 107), beyond the 2-bar horizon.
    result = analyze_excursions(
        _frame(), mode="pct_signal", targets=(0.06,), horizon_bars=2
    )
    event = result.events.row(0, named=True)
    assert event["hit_6pct"] is False
    # with a long enough horizon it is reached
    result2 = analyze_excursions(
        _frame(), mode="pct_signal", targets=(0.06,), horizon_bars=100
    )
    assert result2.events.row(0, named=True)["hit_6pct"] is True


def test_no_signals():
    frame = _frame().with_columns(pl.lit(False).alias("entries"))
    result = analyze_excursions(frame, mode="pct_excursion")
    assert result.events.is_empty()
    assert result.config["n_signals"] == 0


def test_reversion_only_has_no_milestones():
    result = analyze_excursions(
        _frame(), targets=(), horizon_bars=100, reversion_basis="close"
    )
    assert "hit_50pct" not in result.events.columns
    assert result.summary.height == 1
    assert result.summary["milestone"][0] == "reversion_to_signal"
    assert not result.reversion_cdf.is_empty()
    fractions = result.reversion_cdf["fraction"].to_list()
    assert fractions == sorted(fractions)  # cumulative
    assert fractions[-1] == 1.0


def test_markdown_includes_cdf():
    result = analyze_excursions(_frame(), targets=(), horizon_bars=100)
    md = result.to_markdown()
    assert "Time to reversion" in md
    assert "reversion_to_signal" in md


def test_excursion_to_reversion_recorded():
    result = analyze_excursions(
        _frame(), targets=(), horizon_bars=100, reversion_basis="close"
    )
    event = result.events.row(0, named=True)
    # peak high (110) occurs at bar 5, before the close-basis reversion at bar 7
    assert event["excursion_to_reversion"] == 10.0
    assert event["excursion_to_reversion_pct"] == 0.10


def test_excursion_by_reversion_bucket():
    from quant_lab.analytics.excursions import excursion_by_reversion_bucket

    events = pl.DataFrame(
        {
            "time_to_reversion_seconds": [60.0, 60.0, 300.0, 10_000.0, None],
            "excursion_to_reversion": [5.0, 3.0, 8.0, 20.0, 50.0],
            "excursion_to_reversion_pct": [0.05, 0.03, 0.08, 0.20, 0.50],
        }
    )
    table = excursion_by_reversion_bucket(events)
    by = {row["bucket"]: row for row in table.iter_rows(named=True)}
    assert by["<=1m"]["n"] == 2
    assert by["<=1m"]["mean_value"] == 4.0
    assert by["<=1m"]["min_value"] == 3.0
    assert by["<=1m"]["max_value"] == 5.0
    assert by["1m-5m"]["n"] == 1
    assert by["1h-4h"]["n"] == 1
    assert by["not_reverted_7d"]["n"] == 1
    assert by["not_reverted_7d"]["mean_value"] == 50.0
