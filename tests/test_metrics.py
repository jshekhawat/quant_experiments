"""Metric calculation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_lab.backtest.metrics import compute_metrics, empty_trades


def test_known_equity_metrics():
    equity = pd.Series([100.0, 110.0, 99.0, 108.0])
    metrics = compute_metrics(
        equity, initial_capital=100.0, trades=empty_trades(), periods_per_year=1.0
    )
    assert metrics["total_return"] == pytest.approx(0.08)
    assert metrics["max_drawdown"] == pytest.approx(-0.1)
    assert metrics["net_pnl"] == pytest.approx(8.0)
    assert metrics["trade_count"] == 0


def test_trade_stats():
    trades = pd.DataFrame(
        {
            "pnl": [10.0, -5.0, 20.0, -5.0],
            "holding_period": pd.to_timedelta([60, 60, 60, 60], unit="s"),
        }
    )
    equity = pd.Series([100.0, 120.0])
    metrics = compute_metrics(
        equity, initial_capital=100.0, trades=trades, periods_per_year=1.0
    )
    assert metrics["trade_count"] == 4
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["profit_factor"] == pytest.approx(30.0 / 10.0)
    assert metrics["expectancy"] == pytest.approx(5.0)
    assert metrics["avg_holding_period"] == pytest.approx(60.0)


def test_empty_metrics():
    metrics = compute_metrics(
        pd.Series(dtype=float), initial_capital=100.0, trades=empty_trades(), periods_per_year=1.0
    )
    assert metrics["total_return"] is None
