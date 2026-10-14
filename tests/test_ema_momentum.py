"""Tests for the EMA-momentum features, signals and analysis (analytics.ema_momentum)."""

from __future__ import annotations

import numpy as np

from quant_lab.analytics import ema_momentum as em


def test_ema_roc_formula():
    ma = np.array([100.0, 110.0, 121.0, 133.1])
    roc = em.ema_roc(ma, 1)
    assert np.isnan(roc[0])
    assert np.isclose(roc[1], 0.10)          # (110/100 - 1) / 1
    assert np.isclose(roc[3], 0.10)          # (133.1/121 - 1) / 1


def test_acceleration_is_difference_of_roc():
    ma = np.array([100.0, 110.0, 121.0, 133.1])
    roc = em.ema_roc(ma, 1)
    acc = em.ema_acceleration(roc)
    assert np.isnan(acc[1])
    assert np.isclose(acc[2], roc[2] - roc[1])


def test_normalized_spread_and_velocity():
    fast = np.array([10.0, 12.0, 15.0, 19.0])
    slow = np.array([10.0, 10.0, 10.0, 10.0])
    atr = np.full(4, 2.0)
    d = em.normalized_spread(fast, slow, atr)
    assert np.isclose(d[0], 0.0)
    assert np.isclose(d[1], 1.0)
    v = em.spread_velocity(d, 1)
    assert np.isnan(v[0]) and np.isclose(v[1], 1.0)
    # a spread moving toward zero (negative -> less negative) has positive velocity
    d2 = np.array([-2.0, -1.0, 0.0])
    assert em.spread_velocity(d2, 1)[2] > 0


def test_wilder_atr_nonnegative_and_causal():
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, 200))
    high = close + 1.0
    low = close - 1.0
    atr = em.wilder_atr(high, low, close, period=14)
    assert np.isnan(atr[:13]).all()
    assert (atr[13:] >= 0).all()
    # future bars cannot change past ATR
    atr2 = em.wilder_atr(high, low, close, period=14)
    high2 = high.copy()
    high2[150:] += 50.0
    atr3 = em.wilder_atr(high2, low, close, period=14)
    assert np.allclose(atr2[:150], atr3[:150], equal_nan=True)


def test_sign_events():
    x = np.array([-1.0, -0.5, 0.5, 0.2, -0.3])
    ev = em.sign_events(x)
    assert list(ev) == [0, 0, 1, 0, -1]


def test_latch_positions_sides():
    bull = np.array([False, True, False, False])
    bear = np.array([False, False, False, True])
    both = em.latch_positions(bull, bear, "both")
    assert list(both) == [0, 1, 1, -1]
    assert list(em.latch_positions(bull, bear, "long")) == [0, 1, 1, 0]
    assert list(em.latch_positions(bull, bear, "short")) == [0, 0, 0, -1]


def test_confirmation_events_window():
    early = np.zeros(10, dtype=bool)
    early[2] = True
    confirm = np.zeros(10, dtype=bool)
    confirm[6] = True
    out = em.confirmation_events(early, confirm, window=5)
    assert out[6]
    # outside the window -> no confirmation
    out2 = em.confirmation_events(early, confirm, window=2)
    assert not out2.any()


def _feature_set(*, roc, accel, vel, sign):
    cfg = em.config_by_name(em.PRIMARY_CONFIG)
    return em.FeatureSet(
        config=cfg,
        roc={1: np.asarray(roc, dtype=float)},
        accel={1: np.asarray(accel, dtype=float)},
        spread=np.asarray(sign, dtype=float),   # magnitude irrelevant to signal flags
        spread_vel={1: np.asarray(vel, dtype=float)},
        sign=np.asarray(sign, dtype=np.int8),
    )


def test_build_signals_conditions():
    # deceleration + convergence: still below the slow average, ROC negative but
    # rising, spread velocity crossing positive (spread shrinking towards zero)
    sig = em.build_signals(_feature_set(
        roc=[-0.02, -0.015], accel=[0.0, 0.005], vel=[-0.001, 0.001], sign=[-1, -1]))
    assert sig.decel_bull[1]
    assert sig.converge_bull[1]
    assert not sig.reversal_bull[1]
    # reversal: ROC crosses above 0 while still below the slow average;
    # combined additionally needs a positive spread velocity
    sig2 = em.build_signals(_feature_set(
        roc=[-0.02, 0.01], accel=[0.0, 0.03], vel=[0.0, 0.002], sign=[-1, -1]))
    assert sig2.reversal_bull[1]
    assert sig2.combined_bull[1]


def test_analyze_lead_time_and_false_turn():
    n = 40
    close = np.linspace(100.0, 110.0, n)
    high = close + 0.5
    low = close - 0.5
    atr = np.full(n, 1.0)
    # fast rises gradually and crosses the slow average near the end, so there
    # is a window where the fast EMA is still below the slow EMA but its ROC > 0
    fast = np.concatenate([np.full(20, 99.0), np.linspace(99.5, 111.0, 20)])
    slow = np.full(n, 105.0)
    ts = np.arange(n).astype("datetime64[ns]")
    cfg = em.config_by_name(em.PRIMARY_CONFIG)
    fs = em.FeatureSet(
        config=cfg, roc={1: em.ema_roc(fast, 1)}, accel={1: em.ema_acceleration(em.ema_roc(fast, 1))},
        spread=em.normalized_spread(fast, slow, atr),
        spread_vel={1: em.spread_velocity(em.normalized_spread(fast, slow, atr), 1)},
        sign=np.where(fast > slow, 1, -1).astype(np.int8),
    )
    bundle = em.MomentumBundle(
        symbol="T", timestamps=ts, open=close, high=high, low=low, close=close,
        funding=np.zeros(n), atr=atr, sma={20: slow}, ema={12: fast}, features={cfg.name: fs},
        signals={1: {cfg.name: em.build_signals(fs)}},
    )
    events = em.analyze_lead_time(bundle, cfg.name)
    assert not events.empty
    # the bullish crossover near the end must be reachable with non-negative lead
    rev = events[events.signal == "reversal_bull"]
    assert not rev.empty
    assert (rev["lead_same_h"].dropna() >= 0).all()
    assert rev["false_turn_0.5"].notna().all()


def test_atr_stop_flattens_after_adverse_move():
    close = np.array([100.0, 90.0, 80.0, 79.0, 130.0])
    high = close + 0.5
    low = close - 0.5
    atr = np.full(5, 1.0)
    target = np.ones(5, dtype=np.int8)
    out = em.apply_atr_stop(high, low, close, atr, target, mult=2.0)
    assert out[0] == 1
    # after a >2 ATR fall, the long is flattened
    assert out[3] == 0


def test_min_hold_suppresses_flips():
    target = np.array([1, -1, -1, 1, 1, -1], dtype=np.int8)
    out = em.apply_min_hold(target, 3)
    # the flip at bar 1 is suppressed; position stays long until a change >=3 bars later
    assert out[1] == 1
    assert out[2] == 1
