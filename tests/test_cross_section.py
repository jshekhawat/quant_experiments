"""Cross-sectional research tests."""

from __future__ import annotations

from quant_lab.features.cross_section import (
    correlation_matrix,
    cross_sectional_momentum,
    market_breadth,
    relative_strength,
    wide_close,
    wide_returns,
)
from quant_lab.synthetic import synthetic_bars


def _frames():
    return {
        "AAAUSDT": synthetic_bars(300, seed=31, start_price=100.0, symbol="AAAUSDT"),
        "BBBUSDT": synthetic_bars(300, seed=32, start_price=50.0, symbol="BBBUSDT"),
        "CCCUSDT": synthetic_bars(300, seed=33, start_price=10.0, symbol="CCCUSDT"),
    }


def test_wide_close_and_returns():
    wide = wide_close(_frames())
    assert set(wide.columns) == {"timestamp", "AAAUSDT", "BBBUSDT", "CCCUSDT"}
    assert wide.height == 300
    rets = wide_returns(wide)
    assert "AAAUSDT" in rets.columns


def test_correlation_matrix():
    rets = wide_returns(wide_close(_frames()))
    corr = correlation_matrix(rets)
    assert "symbol" in corr.columns
    assert corr.height == 3


def test_cross_sectional_momentum_ranks():
    wide = wide_close(_frames())
    mom = cross_sectional_momentum(wide, window=20)
    # ranks are 1..3 per row where all present
    row = mom.drop_nulls().row(0, named=True)
    values = sorted(v for k, v in row.items() if k != "timestamp")
    assert values == [1, 2, 3]


def test_market_breadth_bounds():
    breadth = market_breadth(wide_close(_frames()), window=20).drop_nulls()
    assert breadth["breadth"].min() >= 0
    assert breadth["breadth"].max() <= 1


def test_relative_strength():
    rs = relative_strength(wide_close(_frames()), "AAAUSDT", window=10)
    assert "BBBUSDT" in rs.columns
