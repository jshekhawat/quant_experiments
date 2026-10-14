"""H3 breakout continuation: detection, flow alignment, failure outcomes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from quant_lab.analytics.breakouts import (
    add_outcomes,
    detect_breakouts,
    flow_alignment,
)
from quant_lab.data.schemas import BAR_SCHEMA
from quant_lab.synthetic import synthetic_bars


def _bars(rows: list[tuple[float, float, float]]) -> pl.DataFrame:
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


def test_detect_breakouts_direction_and_displacement():
    rows = [
        (9.5, 9.0, 9.2),
        (10.0, 9.3, 9.8),
        (9.8, 9.4, 9.6),
        (10.5, 9.4, 10.4),  # close 10.4 > prior high 10 -> bull, level 10
        (9.6, 9.3, 9.4),
        (9.5, 8.4, 8.5),  # close 8.5 < prior low ~9.3 -> bear
    ]
    out = detect_breakouts(_bars(rows), lookback=3, cooldown=0)
    assert out["bull"][3]
    assert out["bdir"][3] == 1.0
    assert abs(out["break_disp"][3] - (10.4 - 10.0) / 10.0) < 1e-12
    assert out["bdir"][5] == -1.0


def test_breakout_level_excludes_current_bar():
    bars = synthetic_bars(200, seed=1)
    out = detect_breakouts(bars, lookback=5, cooldown=0)
    # A breakout is only ever measured against strictly prior highs/lows.
    highs = bars["high"].to_numpy()
    closes = bars["close"].to_numpy()
    for t in range(6, 200):
        if closes[t] > np.max(highs[t - 5 : t]):
            assert out["bdir"][t] == 1.0


def test_flow_alignment_identity():
    bars = synthetic_bars(400, seed=2)
    out = flow_alignment(
        detect_breakouts(bars, lookback=6, cooldown=0), flow_window=2, act_window=5, norm_window=40
    )
    df = out.filter(pl.col("bdir") != 0).drop_nulls(["fz", "flow_align_z"])
    assert df.height > 0
    lhs = df["flow_align_z"].to_numpy()
    rhs = (df["bdir"] * df["fz"]).to_numpy()
    assert np.allclose(lhs, rhs)


def test_outcomes_and_failure_flag():
    rows = [
        (9.5, 9.0, 9.2),
        (10.0, 9.3, 9.8),
        (9.8, 9.4, 9.6),
        (10.5, 9.4, 10.4),  # bull breakout, level 10
        (10.0, 9.5, 9.9),  # back below level -> failure
        (9.9, 9.4, 9.8),
        (9.8, 9.3, 9.7),
        (9.7, 9.2, 9.6),
        (9.6, 9.1, 9.5),
    ]
    out = add_outcomes(detect_breakouts(_bars(rows), lookback=3, cooldown=0), {"4h": 4})
    assert out["fail_4h"][3] == 1.0
    close = np.array([c for _, _, c in rows])
    expected = out["d"][3] * (close[3 + 5] / close[3 + 1] - 1.0)
    assert abs(out["y_4h"][3] - expected) < 1e-12


def test_cooldown_deduplicates():
    bars = synthetic_bars(400, seed=3)
    out = detect_breakouts(bars, lookback=5, cooldown=24)
    idx = out.with_row_index("i").filter(pl.col("breakout"))["i"].to_list()
    assert all(b - a > 24 for a, b in zip(idx, idx[1:], strict=False))


def test_detection_deterministic():
    bars = synthetic_bars(500, seed=4)
    a = detect_breakouts(bars, lookback=10, cooldown=12)
    b = detect_breakouts(bars, lookback=10, cooldown=12)
    assert a["breakout"].equals(b["breakout"])
    assert a["bdir"].equals(b["bdir"])
