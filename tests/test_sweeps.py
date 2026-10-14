"""H2 failed sweeps: level causality, detection, cooldown, alignment."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from quant_lab.analytics.sweeps import (
    add_outcomes,
    aligned_flow,
    detect_sweeps,
    prior_range_levels,
)
from quant_lab.data.schemas import BAR_SCHEMA
from quant_lab.synthetic import synthetic_bars


def _bars(rows: list[tuple[float, float, float]]) -> pl.DataFrame:
    """Build canonical bars from (high, low, close); other columns are filler."""
    ts = [datetime(2026, 3, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(len(rows))]
    return pl.DataFrame(
        {
            "timestamp": ts,
            "open": [c for _, _, c in rows],
            "high": [h for h, _, _ in rows],
            "low": [lo for _, lo, _ in rows],
            "close": [c for _, _, c in rows],
            "volume": [1.0] * len(rows),
            "quote_volume": [10.0] * len(rows),
            "trade_count": [1] * len(rows),
            "taker_buy_volume": [0.5] * len(rows),
            "taker_buy_quote_volume": [5.0] * len(rows),
        },
        schema=BAR_SCHEMA,
    )


def test_prior_range_level_excludes_current_bar():
    bars = synthetic_bars(100, seed=1)
    out = prior_range_levels(bars, lookback=5)
    # prior_high at t must equal max(high[t-5..t-1]) and never see high[t].
    highs = bars["high"].to_numpy()
    for t in range(6, 100):
        assert abs(out["prior_high"][t] - highs[t - 5 : t].max()) < 1e-12
    # Shrinking the current bar's high cannot change its own prior level.
    bumped = bars.with_columns(
        pl.when(pl.int_range(pl.len()) == 50).then(pl.col("high") + 100.0).otherwise(pl.col("high")).alias("high")
    )
    out2 = prior_range_levels(bumped, lookback=5)
    assert abs(out2["prior_high"][50] - out["prior_high"][50]) < 1e-12


def test_detect_sweeps_direction():
    rows = [
        (9.5, 9.0, 9.2),
        (10.0, 9.3, 9.8),
        (9.8, 9.4, 9.6),
        (11.0, 9.4, 9.5),  # upside sweep: wick 11 > prior high 10, close 9.5 < 10
        (9.6, 9.4, 9.5),
        (9.8, 8.5, 9.6),  # downside sweep: wick 8.5 < prior low ~9.4, close 9.6 > 9.4
    ]
    out = detect_sweeps(_bars(rows), lookback=3, cooldown=0)
    assert out["upside_sweep"][3]
    assert out["sweep_dir"][3] == -1.0
    assert out["downside_sweep"][5]
    assert out["sweep_dir"][5] == 1.0
    assert not out["sweep"][4]


def test_cooldown_deduplicates():
    bars = synthetic_bars(300, seed=2)
    # Force a sweep-like flag every bar, then check cooldown spacing.
    forced = bars.with_columns(pl.lit(True).alias("_force"))
    out = detect_sweeps(forced, lookback=5, cooldown=24)
    idx = out.with_row_index("i").filter(pl.col("sweep"))["i"].to_list()
    if len(idx) >= 2:
        assert all(b - a > 24 for a, b in zip(idx, idx[1:], strict=False))


def test_aligned_flow_alignment():
    bars = synthetic_bars(300, seed=5)
    out = aligned_flow(
        detect_sweeps(bars, lookback=6, cooldown=0), flow_window=1, act_window=5, norm_window=40
    )
    df = out.filter(pl.col("sweep_dir") != 0).drop_nulls(["fz", "af_z"])
    assert df.height > 0
    # af_z must equal the signed-flow z-score aligned to the sweep direction.
    lhs = df["af_z"].to_numpy()
    rhs = (-df["sweep_dir"] * df["fz"]).to_numpy()
    assert np.allclose(lhs, rhs)


def test_outcomes_alignment_and_delay():
    bars = synthetic_bars(300, seed=3)
    sig = add_outcomes(detect_sweeps(bars, lookback=6, cooldown=0), {"4h": 4})
    close = bars["close"].to_numpy()
    d = sig["d"].to_numpy()
    for i in range(len(close) - 5):
        expected = d[i] * (close[i + 5] / close[i + 1] - 1.0)
        if np.isfinite(expected):
            assert abs(sig["y_4h"][i] - expected) < 1e-12
    assert sig["y_4h"].to_list()[-5:] == [None] * 5


def test_detection_is_deterministic():
    bars = synthetic_bars(400, seed=4)
    a = detect_sweeps(bars, lookback=10, cooldown=12)
    b = detect_sweeps(bars, lookback=10, cooldown=12)
    assert a["sweep"].equals(b["sweep"])
    assert a["sweep_dir"].equals(b["sweep_dir"])
