"""Deterministic synthetic market data for tests and examples.

The generator is fully reproducible from a seed. It exists so that the whole
pipeline (features -> strategy -> backtest -> known result) can be tested
without touching the network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from .data.schemas import BAR_SCHEMA, DatasetType
from .data.storage import DatasetKey


def synthetic_bars(
    n: int = 500,
    *,
    seed: int = 7,
    start: str = "2024-01-01T00:00:00",
    timeframe_seconds: int = 60,
    start_price: float = 100.0,
    drift: float = 0.0,
    volatility: float = 0.002,
    volume: float = 10.0,
    symbol: str = "TESTUSDT",
) -> pl.DataFrame:
    """Generate ``n`` canonical 1-bar-per-``timeframe_seconds`` bars."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, volatility, size=n)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(0, volatility, size=n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = np.abs(rng.normal(volume, volume * 0.3, size=n)) + 1e-9
    taker_share = rng.uniform(0.3, 0.7, size=n)
    quote_volume = vol * close
    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC)
    timestamps = [start_dt + timedelta(seconds=timeframe_seconds * i) for i in range(n)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "quote_volume": quote_volume,
            "trade_count": rng.integers(1, 500, size=n).astype(np.int64),
            "taker_buy_volume": vol * taker_share,
            "taker_buy_quote_volume": quote_volume * taker_share,
        },
        schema=BAR_SCHEMA,
    )


def synthetic_key(symbol: str = "TESTUSDT", timeframe: str = "1m") -> DatasetKey:
    return DatasetKey(
        exchange="synthetic",
        market_type="test",
        dataset=DatasetType.BARS,
        symbol=symbol,
        timeframe=timeframe,
    )
