"""Tests for the EMA-crossover backtest engine (analytics.ema_backtest)."""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_lab.analytics.ema_backtest import (
    desired_positions,
    extract_trades,
    returns_metrics,
    simulate,
)
from quant_lab.features.core import ema as ema_expr


def _ema(close: np.ndarray, span: int) -> np.ndarray:
    frame = pl.DataFrame({"close": close})
    return frame.select(ema_expr(frame, span=span)).to_series().to_numpy().astype(float)


def test_buy_and_hold_identity():
    n = 200
    close = np.linspace(100, 150, n)
    open_ = close * 0.999
    ts = np.arange(n).astype("datetime64[ns]")
    res = simulate(ts, open_, close, np.ones(n, dtype=np.int8), cost_per_side=0.0)
    equity = float(np.cumprod(1 + res.returns)[-1])
    # held from bar 1's open to the end, marked at close
    assert np.isclose(equity, close[-1] / open_[1], rtol=1e-9)


def test_costs_reduce_return_monotonically():
    rng = np.random.default_rng(0)
    n = 2000
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    ts = np.arange(n).astype("datetime64[ns]")
    target = np.where(np.arange(n) % 7 < 4, 1, -1).astype(np.int8)
    rets = [
        returns_metrics(simulate(ts, open_, close, target, cost_per_side=c).returns, ppy=8760.0)["total_return"]
        for c in (0.0, 0.0005, 0.001, 0.002)
    ]
    assert rets[0] >= rets[1] >= rets[2] >= rets[3]


def test_execution_contract_and_no_lookahead():
    n = 300
    close = 100 + np.sin(np.arange(n) / 9.0) * 8
    open_ = close
    ts = np.arange(n).astype("datetime64[ns]")
    ef, es, e200 = _ema(close, 5), _ema(close, 20), _ema(close, 50)
    t1 = desired_positions(close, ef, es, e200, "both_trend_regime")

    # execution: position during bar t is the target decided at t-1
    res = simulate(ts, open_, close, t1, cost_per_side=0.0)
    assert np.array_equal(res.positions[1:], t1[:-1])
    assert res.positions[0] == 0

    # causality: perturbing only the future cannot change past positions
    close2 = close.copy()
    close2[200:] *= 3.0
    t2 = desired_positions(close2, ef, es, e200, "both_trend_regime")
    assert np.array_equal(t1[:200], t2[:200])


def test_trend_entry_vs_regime_semantics():
    n = 60
    close = np.full(n, 100.0)
    close[30:] = 90.0
    ema_fast = np.full(n, 110.0)
    ema_fast[25:] = 95.0            # fast crosses below slow at bar 25
    ema_slow = np.full(n, 100.0)
    ema200 = np.full(n, 95.0)
    regime = desired_positions(close, ema_fast, ema_slow, ema200, "both_trend_regime")
    entry = desired_positions(close, ema_fast, ema_slow, ema200, "both_trend_entry")

    # regime: long while price>EMA200, flat when the filter blocks the short,
    # short once price falls below EMA200
    assert regime[10] == 1
    assert regime[27] == 0
    assert regime[35] == -1
    # entry: the cross at 25 was blocked, so the book stays flat thereafter
    assert entry[35] == 0


def test_extract_trades_reconciles_with_returns():
    n = 500
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.0005, n))
    ts = np.arange(n).astype("datetime64[ns]")
    target = np.where((np.arange(n) // 30) % 2 == 0, 1, -1).astype(np.int8)
    res = simulate(ts, open_, close, target, cost_per_side=0.0005)
    trades = extract_trades(ts, open_, close, res.positions, cost_per_side=0.0005)
    assert len(trades) == res.n_trades
    # trade returns and per-bar returns should agree in aggregate
    assert abs(trades["ret"].sum() - res.returns.sum()) < 0.6
