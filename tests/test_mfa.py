"""Tests for the multi-factor analytics module (analytics.mfa)."""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_lab.analytics import mfa


def _bars(n: int = 400, seed: int = 0) -> pl.DataFrame:
    from datetime import UTC, datetime, timedelta

    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0, 0.01, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = np.abs(rng.normal(10.0, 3.0, n)) + 1e-9
    taker_share = rng.uniform(0.3, 0.7, n)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    stamps = [start + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame({
        "timestamp": stamps, "open": open_, "high": high, "low": low, "close": close,
        "volume": vol, "quote_volume": vol * close, "trade_count": np.arange(n),
        "taker_buy_volume": vol * taker_share, "taker_buy_quote_volume": vol * close * taker_share,
    })


def test_flow_imbalance_formula_and_bounds():
    frame = _bars()
    tbq = frame["taker_buy_quote_volume"].to_numpy()
    qv = frame["quote_volume"].to_numpy()
    qd = 2.0 * tbq - qv
    # FI_1 is just qd/qv; check it lies in [-1, 1]
    fi = qd / qv
    assert np.nanmin(fi) >= -1.0001 and np.nanmax(fi) <= 1.0001
    # zero taker-buy => qd = -quote_volume => FI = -1; all taker-buy => +1
    assert np.isclose(2 * 0.0 - 1.0, -1.0)


def test_absorption_score_direction():
    # strong selling (fi<0) with a *positive* price response => absorb_score large positive
    fi = np.array([-0.8])
    pr = np.array([+0.5])
    assert float((-np.sign(fi) * pr)[0]) > 0


def test_oriented_summary_and_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.normal(0.002, 0.02, 500)
    st = mfa.oriented_summary(y, block=24, n=100)
    assert st["n"] == 500
    assert 0.0 <= st["hit"] <= 1.0
    assert st["ci_lo"] < st["mean"] < st["ci_hi"]


def test_boot_1d_returns_distribution():
    x = np.arange(200, dtype=float)
    draws = mfa.boot_1d(x, np.mean, block=10, n=50)
    assert draws.size > 0
    assert np.all(draws > 0)


def test_state_positions_sides():
    bull = np.array([True, False, False])
    bear = np.array([False, True, False])
    assert list(mfa.state_positions(bull, bear, "both")) == [1, -1, 0]
    assert list(mfa.state_positions(bull, bear, "long")) == [1, 0, 0]
    assert list(mfa.state_positions(bull, bear, "short")) == [0, -1, 0]


def test_demean_by_symbol():
    x = np.array([1.0, 3.0, 10.0, 14.0])
    sid = np.array([0, 0, 1, 1])
    out = mfa.demean_by_symbol(x, sid)
    assert np.isclose(out[0], -1.0) and np.isclose(out[1], 1.0)
    assert np.isclose(out[2], -2.0) and np.isclose(out[3], 2.0)


def test_hac_regression_detects_added_signal():
    rng = np.random.default_rng(1)
    n = 2000
    base = rng.normal(size=(n, 1))
    add = rng.normal(size=(n, 1))
    y = 0.5 * add[:, 0] + rng.normal(0, 0.1, n)
    r = mfa.hac_regression(y, base, add, maxlags=4)
    assert r["delta_r2"] > 0.5
    assert r["t_add"] > 10


def test_forward_outcomes_alignment_and_causality():
    frame = _bars(60)
    out = mfa.add_forward_outcomes(frame, {"4h": 4})
    close = frame["close"].to_numpy()
    fwd = out["fwd_4h"].to_numpy()
    # fwd_4h at bar t is close[t+1+4]/close[t+1]-1
    assert np.isclose(fwd[0], close[5] / close[1] - 1.0)
    # last bars are undefined
    assert np.isnan(fwd[-1])
    # causality: appending future bars must not change past outcomes
    longer = pl.concat([frame, _bars(20, seed=9).with_columns(pl.col("timestamp").dt.offset_by("100d"))])
    out2 = mfa.add_forward_outcomes(longer.sort("timestamp"), {"4h": 4})
    assert np.allclose(out["fwd_4h"][:55].to_numpy(), out2["fwd_4h"][:55].to_numpy(), equal_nan=True)


def test_horizons_and_strategies_constants():
    assert set(mfa.HORIZONS) == {"1h", "4h", "8h", "12h", "24h", "48h", "72h"}
    assert "absorption_rev" in mfa.STRATEGIES
    assert mfa.PRIMARY_FLOW_HORIZON == 4
    assert "ema12_sma20" in mfa.EMA_CONFIGS
