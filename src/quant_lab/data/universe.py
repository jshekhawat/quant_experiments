"""Liquid-symbol universe selection.

The bulk archive host (``data.binance.vision``) carries no liquidity ranking, so
this asks Binance's public REST ticker for rolling 24h quote volume and ranks by
it. The result is an ordinary, ordered symbol list that can be cached to JSON and
reused offline — the historical archives for a symbol never change when it drops
out of today's top N.

Stable/stable and fiat pairs are excluded by default: they can top the volume
table while carrying no meaningful taker flow, so they are useless for CVD.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

TICKER_URLS: dict[str, str] = {
    "futures": "https://fapi.binance.com/fapi/v1/ticker/24hr",
    "futures_um": "https://fapi.binance.com/fapi/v1/ticker/24hr",
    "um": "https://fapi.binance.com/fapi/v1/ticker/24hr",
    "spot": "https://api.binance.com/api/v3/ticker/24hr",
}

# Quote assets that are themselves stablecoins or fiat. A pair whose base is in
# this set (USDCUSDT, EURUSDT, ...) has no tradeable directional flow.
STABLE_BASES: frozenset[str] = frozenset(
    {
        # stablecoins
        "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "USDS", "PYUSD",
        "USD1", "RLUSD", "USDD", "GUSD", "USTC", "SUSD", "USDX", "XUSD", "USDY",
        "USDF", "USDR", "DUSD", "MIM", "FRAX", "LUSD", "VAI", "USDB",
        # fiat
        "EUR", "EURI", "AEUR", "TRY", "BRL", "ARS", "JPY", "GBP", "AUD", "BIDR",
        "IDRT", "RUB", "UAH", "NGN", "ZAR", "PLN", "RON", "CZK", "MXN", "COP",
        "PEN", "BKRW",
    }
)


# Non-crypto instruments Binance lists as perps and (sometimes) spot tokens:
# tokenised equities, ETFs, and metals/commodities. They have no independent
# crypto spot-vs-perp order-flow relationship, so crypto-only studies drop them.
NON_CRYPTO_BASES: frozenset[str] = frozenset(
    {
        # metals / energy
        "XAU", "XAG", "XAUT", "PAXG", "GLD", "SLV", "CL", "BZ", "WTI", "NATGAS",
        "COPPER", "BRENT", "USO", "UNG",
        # equities
        "AAPL", "AMD", "AMZN", "AVGO", "CBRS", "CRCL", "GOOGL", "INTC", "META",
        "MRVL", "MSTR", "MSFT", "MU", "MUU", "NBIS", "NVDA", "ORCL", "PLTR",
        "SAMSUNG", "SKHY", "SKHYNIX", "SNXX", "SNDK", "SPCX", "TSLA", "AAOI",
        "LITE",
        # ETFs / indices
        "SPY", "QQQ", "TQQQ", "SQQQ", "SOXL", "SOXS", "EWY", "KORU", "ARKK",
    }
)


def is_non_crypto(symbol: str, quote: str = "USDT") -> bool:
    """True for tokenised equities/ETFs/commodities masquerading as ``*USDT``."""
    base = _base_asset(symbol, quote)
    if base in NON_CRYPTO_BASES:
        return True
    # Tokenised-equity spot symbols carry a trailing 'B' (e.g. NVDABUSDT).
    if base.endswith("B") and base[:-1] in NON_CRYPTO_BASES:
        return True
    return False


def fetch_tickers(
    market: str = "futures",
    *,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch the live 24h ticker for one market."""
    url = TICKER_URLS.get(market.lower())
    if url is None:
        raise ValueError(f"unsupported market {market!r}; choose one of {sorted(TICKER_URLS)}")
    http = session or requests
    response = http.get(url, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError(f"unexpected ticker payload for {market!r}")
    return data


def _base_asset(symbol: str, quote: str) -> str:
    return symbol[: -len(quote)] if symbol.endswith(quote) else symbol


def rank_by_volume(
    tickers: list[dict[str, Any]],
    *,
    n: int = 100,
    quote: str = "USDT",
    exclude_stables: bool = True,
    exclude_non_crypto: bool = False,
) -> list[str]:
    """Rank tickers by 24h quote volume, returning the top ``n`` symbols."""
    ranked: list[tuple[str, float]] = []
    for row in tickers:
        symbol = str(row.get("symbol", ""))
        # Delivery contracts (``BTCUSDT_250627``) are not what "futures" means here.
        if not symbol.endswith(quote) or "_" in symbol:
            continue
        if exclude_stables and _base_asset(symbol, quote) in STABLE_BASES:
            continue
        if exclude_non_crypto and is_non_crypto(symbol, quote):
            continue
        try:
            volume = float(row.get("quoteVolume", 0.0))
        except (TypeError, ValueError):
            continue
        ranked.append((symbol, volume))
    # Ties broken by symbol for determinism.
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [symbol for symbol, _ in ranked[:n]]


def fetch_top_symbols(
    market: str = "futures",
    n: int = 100,
    *,
    quote: str = "USDT",
    exclude_stables: bool = True,
    exclude_non_crypto: bool = False,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[str]:
    """Top ``n`` symbols by 24h quote volume for a single market."""
    tickers = fetch_tickers(market, timeout=timeout, session=session)
    return rank_by_volume(
        tickers,
        n=n,
        quote=quote,
        exclude_stables=exclude_stables,
        exclude_non_crypto=exclude_non_crypto,
    )


def fetch_universe(
    n: int = 100,
    *,
    markets: tuple[str, ...] | list[str] = ("futures", "spot"),
    quote: str = "USDT",
    exclude_stables: bool = True,
    exclude_non_crypto: bool = False,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> dict[str, list[str]]:
    """Top ``n`` symbols per market, keyed by market name."""
    return {
        market: fetch_top_symbols(
            market,
            n,
            quote=quote,
            exclude_stables=exclude_stables,
            exclude_non_crypto=exclude_non_crypto,
            timeout=timeout,
            session=session,
        )
        for market in markets
    }


def cross_market_universe(
    universe: dict[str, list[str]],
    *,
    markets: tuple[str, ...] | list[str] = ("spot", "futures"),
    crypto_only: bool = False,
    quote: str = "USDT",
) -> list[str]:
    """Symbols listed in **every** requested market, in the first market's order.

    A spot-vs-perp flow study is only defined where both legs exist, so the
    intersection is the working universe.
    """
    sets = [set(universe.get(market, [])) for market in markets]
    common = set.intersection(*sets) if sets else set()
    if crypto_only:
        common = {symbol for symbol in common if not is_non_crypto(symbol, quote)}
    order = universe.get(markets[0], []) if markets else []
    # Fall back to sorted order if the anchor market is absent.
    return [s for s in order if s in common] or sorted(common)


def save_universe(
    path: Path | str,
    universe: dict[str, list[str]],
    *,
    n: int | None = None,
    quote: str = "USDT",
    rank_by: str = "quoteVolume_24h",
) -> Path:
    """Persist a universe to JSON with provenance."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "rank_by": rank_by,
        "quote": quote,
        "n": n if n is not None else max((len(v) for v in universe.values()), default=0),
        "universe": universe,
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def load_universe(path: Path | str) -> dict[str, list[str]]:
    """Load a cached universe file (accepts either the wrapped or bare form)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    universe = data.get("universe", data) if isinstance(data, dict) else data
    if not isinstance(universe, dict):
        raise ValueError(f"{path} is not a universe file")
    return {
        str(market): [str(symbol) for symbol in symbols]
        for market, symbols in universe.items()
    }


__all__ = [
    "NON_CRYPTO_BASES",
    "STABLE_BASES",
    "TICKER_URLS",
    "cross_market_universe",
    "fetch_tickers",
    "fetch_top_symbols",
    "fetch_universe",
    "is_non_crypto",
    "load_universe",
    "rank_by_volume",
    "save_universe",
]
