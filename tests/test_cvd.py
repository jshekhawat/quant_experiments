"""CVD reconstruction and universe-selection tests."""

from __future__ import annotations

import polars as pl

from quant_lab.data.cvd import add_cvd, write_cvd_partitions
from quant_lab.data.universe import load_universe, rank_by_volume, save_universe
from quant_lab.synthetic import synthetic_bars


def test_volume_delta_and_cvd_match_definition():
    bars = synthetic_bars(64, seed=1)
    out = add_cvd(bars, symbol="TESTUSDT", market="futures")

    expected_delta = 2.0 * bars["taker_buy_volume"] - bars["volume"]
    assert out["volume_delta"].to_list() == expected_delta.to_list()
    assert out["cvd"].to_list() == expected_delta.cum_sum().to_list()
    assert out["symbol"][0] == "TESTUSDT"
    assert out["market"][0] == "futures"


def test_add_cvd_is_causal():
    """Truncating the series must not change past CVD values."""
    bars = synthetic_bars(120, seed=2)
    full = add_cvd(bars, symbol="X", market="futures")
    part = add_cvd(bars.head(50), symbol="X", market="futures")
    assert full["cvd"].head(50).equals(part["cvd"])


def test_cvd_continuous_across_month_partitions(tmp_path):
    # 200 one-minute bars starting late on Jan 31 spill into February.
    bars = synthetic_bars(200, seed=3, start="2026-01-31T22:30:00")
    cvd = add_cvd(bars, symbol="XUSDT", market="futures")

    paths = write_cvd_partitions(cvd, tmp_path, interval="1m")
    assert len(paths) == 2  # January + February partitions

    reread = pl.read_parquet([str(p) for p in paths]).sort("timestamp")
    assert reread["cvd"].equals(cvd["cvd"])

    # February continues from where January stopped, it does not reset to zero.
    jan = pl.read_parquet(str(tmp_path / "cvd/futures/XUSDT/1m/2026/01.parquet"))
    feb = pl.read_parquet(str(tmp_path / "cvd/futures/XUSDT/1m/2026/02.parquet"))
    assert feb["cvd"][0] == jan["cvd"][-1] + feb["volume_delta"][0]


def test_rank_by_volume_filters_and_sorts():
    tickers = [
        {"symbol": "BTCUSDT", "quoteVolume": "1000"},
        {"symbol": "ETHUSDT", "quoteVolume": "2000"},
        {"symbol": "USDCUSDT", "quoteVolume": "5000"},  # stable/stable, excluded
        {"symbol": "ABCUSDC", "quoteVolume": "9000"},  # wrong quote asset
        {"symbol": "OLDUSDT_250101", "quoteVolume": "9000"},  # delivery contract
        {"symbol": "DOGEUSDT", "quoteVolume": "500"},
    ]
    assert rank_by_volume(tickers, n=2) == ["ETHUSDT", "BTCUSDT"]
    assert rank_by_volume(tickers, n=3, exclude_stables=False)[0] == "USDCUSDT"


def test_universe_roundtrip(tmp_path):
    universe = {"futures": ["BTCUSDT", "ETHUSDT"], "spot": ["SOLUSDT"]}
    path = save_universe(tmp_path / "u.json", universe, n=2)
    assert load_universe(path) == universe
