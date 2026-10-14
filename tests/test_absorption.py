"""H1 price-absorption: signal formulas, causality, alignment and estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.analytics.absorption import (
    absorption_regression,
    absorption_signal,
    add_outcomes,
    matched_spread,
    normalized_flow,
    rolling_impact_model,
)
from quant_lab.synthetic import synthetic_bars


def test_normalized_flow_definition():
    bars = synthetic_bars(200, seed=1)
    out = normalized_flow(bars, adv_window=20)
    qd = 2.0 * bars["taker_buy_quote_volume"] - bars["quote_volume"]
    assert (out["qd"] - qd).abs().max() < 1e-9
    adv = bars["quote_volume"].rolling_mean(window_size=20, min_periods=20)
    expected_f = qd / adv
    got = out["f"].to_numpy()
    exp = expected_f.to_numpy()
    mask = np.isfinite(exp)
    assert np.allclose(got[mask], exp[mask])


def test_rolling_impact_model_matches_ols():
    bars = synthetic_bars(400, seed=2)
    out = rolling_impact_model(normalized_flow(bars, adv_window=10), window=30, min_periods=30)
    f = out["f"].to_numpy()
    r = out["r"].to_numpy()
    i = 200  # a fully-warmed row
    win = slice(i - 29, i + 1)
    slope, intercept = np.polyfit(f[win], r[win], 1)
    assert abs(out["beta"][i] - slope) < 1e-9
    assert abs(out["alpha"][i] - intercept) < 1e-9
    assert abs(out["s"][i] - (r[i] - (intercept + slope * f[i]))) < 1e-9


def test_absorption_is_causal():
    bars = synthetic_bars(1200, seed=3)
    full = absorption_signal(bars, adv_window=10, impact_window=60)
    part = absorption_signal(bars.head(800), adv_window=10, impact_window=60)
    a = full["A"].head(800).to_numpy()
    b = part["A"].to_numpy()
    mask = np.isfinite(a) & np.isfinite(b)
    assert mask.sum() > 0
    assert np.allclose(a[mask], b[mask])


def test_outcomes_alignment_and_direction():
    bars = synthetic_bars(300, seed=4)
    sig = add_outcomes(absorption_signal(bars, adv_window=10, impact_window=40), {"4h": 4})
    close = bars["close"].to_numpy()
    d = sig["d"].to_numpy()
    for i in range(len(close) - 5):
        expected = d[i] * (close[i + 5] / close[i + 1] - 1.0)
        if np.isfinite(expected):
            assert abs(sig["y_4h"][i] - expected) < 1e-12
    # one-bar execution delay: last five rows have no forward return
    assert sig["y_4h"].to_list()[-5:] == [None] * 5


def _matched_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for t in range(400):
        for s in range(4):
            vol = s % 2
            for absorbed in (0, 1):
                a = 2.0 if absorbed else -2.0
                base = 0.01 if absorbed else -0.01
                rows.append(
                    {
                        "timestamp": pd.Timestamp("2026-03-01", tz="UTC") + pd.Timedelta(hours=t),
                        "symbol": f"S{s}",
                        "A": a + rng.normal(scale=0.1),
                        "strong_flow": True,
                        "y_4h": base + rng.normal(scale=0.001),
                        "realized_vol": 0.01 + 0.001 * vol,
                        "past_ret": 0.0,
                        "abs_f": 1.0,
                        "regime": "up_lowvol" if vol else "range_highvol",
                    }
                )
    return pd.DataFrame(rows)


def test_matched_spread_recovers_effect():
    df = _matched_frame()
    spread = matched_spread(df, "y_4h")
    assert spread > 0.015  # ~0.02 true difference


def test_absorption_regression_recovers_coefficient():
    rng = np.random.default_rng(1)
    n = 5000
    a = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "A": a,
            "y_4h": 0.5 * a + rng.normal(scale=0.1, size=n),
            "past_ret": rng.normal(scale=0.01, size=n),
            "realized_vol": np.abs(rng.normal(scale=0.01, size=n)),
            "abs_f": np.abs(rng.normal(size=n)),
            "regime": rng.choice(["up_lowvol", "range_highvol"], size=n),
        }
    )
    assert abs(absorption_regression(df, "y_4h") - 0.5) < 0.02


def test_impact_model_is_deterministic():
    bars = synthetic_bars(300, seed=9)
    a = absorption_signal(bars, adv_window=10, impact_window=50)
    b = absorption_signal(bars, adv_window=10, impact_window=50)
    assert a["A"].equals(b["A"])
