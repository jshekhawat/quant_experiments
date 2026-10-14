"""Universe selection tests: crypto filtering and cross-market intersection."""

from __future__ import annotations

from quant_lab.data.universe import (
    cross_market_universe,
    is_non_crypto,
    rank_by_volume,
)


def test_is_non_crypto():
    assert is_non_crypto("NVDAUSDT")  # tokenised equity perp
    assert is_non_crypto("XAUTUSDT")  # tokenised gold
    assert is_non_crypto("NVDABUSDT")  # tokenised-equity spot symbol (trailing B)
    assert not is_non_crypto("METUSDT")  # Meteora, a crypto token
    assert not is_non_crypto("BTCUSDT")
    assert not is_non_crypto("ARBUSDT")  # ends in B but is crypto


def test_rank_by_volume_excludes_non_crypto():
    tickers = [
        {"symbol": "BTCUSDT", "quoteVolume": "1000"},
        {"symbol": "NVDAUSDT", "quoteVolume": "9000"},
        {"symbol": "ETHUSDT", "quoteVolume": "2000"},
        {"symbol": "XAUTUSDT", "quoteVolume": "8000"},
    ]
    assert rank_by_volume(tickers, n=2, exclude_non_crypto=True) == ["ETHUSDT", "BTCUSDT"]


def test_cross_market_universe_intersection_and_crypto():
    universe = {
        "spot": ["BTCUSDT", "ETHUSDT", "XAUTUSDT", "OPUSDT"],
        "futures": ["BTCUSDT", "ETHUSDT", "XAUTUSDT", "ARBUSDT"],
    }
    # OPUSDT is spot-only, ARBUSDT is perp-only, both dropped by the intersection.
    assert cross_market_universe(universe) == ["BTCUSDT", "ETHUSDT", "XAUTUSDT"]
    assert cross_market_universe(universe, crypto_only=True) == ["BTCUSDT", "ETHUSDT"]
