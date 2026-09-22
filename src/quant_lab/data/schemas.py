"""Canonical, exchange-independent market-data schemas.

Every normalized dataset in the lab conforms to one of these schemas. Source
adapters are responsible for translating exchange-specific raw formats into
these columns. Strategies only ever consume canonical data.

Timestamps are always UTC, timezone-aware, microsecond resolution.
"""

from __future__ import annotations

from enum import StrEnum

import polars as pl

UTC = "UTC"
TIME_UNIT = "us"


def ts_dtype() -> pl.DataType:
    return pl.Datetime(TIME_UNIT, UTC)


class DatasetType(StrEnum):
    """Logical dataset domains. Kept separate rather than one giant table."""

    BARS = "bars"
    TRADES = "trades"
    AGG_TRADES = "agg_trades"
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"
    LIQUIDATION = "liquidation"
    METRICS = "metrics"
    MARK_PRICE_BARS = "mark_price_bars"
    INDEX_PRICE_BARS = "index_price_bars"
    PREMIUM_INDEX_BARS = "premium_index_bars"

    @property
    def is_bar_like(self) -> bool:
        return self in {
            DatasetType.BARS,
            DatasetType.MARK_PRICE_BARS,
            DatasetType.INDEX_PRICE_BARS,
            DatasetType.PREMIUM_INDEX_BARS,
        }


# --- Canonical schemas -------------------------------------------------------
# These are ordered: the order is the canonical physical column order.

BAR_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": ts_dtype(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "quote_volume": pl.Float64(),
    "trade_count": pl.Int64(),
    "taker_buy_volume": pl.Float64(),
    "taker_buy_quote_volume": pl.Float64(),
}

TRADE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": ts_dtype(),
    "trade_id": pl.Int64(),
    "price": pl.Float64(),
    "quantity": pl.Float64(),
    "quote_quantity": pl.Float64(),
    "side": pl.Utf8(),
}

AGG_TRADE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": ts_dtype(),
    "agg_trade_id": pl.Int64(),
    "price": pl.Float64(),
    "quantity": pl.Float64(),
    "first_trade_id": pl.Int64(),
    "last_trade_id": pl.Int64(),
    "side": pl.Utf8(),
}

FUNDING_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": ts_dtype(),
    "symbol": pl.Utf8(),
    "funding_rate": pl.Float64(),
    "funding_interval_hours": pl.Int64(),
}

OPEN_INTEREST_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": ts_dtype(),
    "symbol": pl.Utf8(),
    "open_interest": pl.Float64(),
    "open_interest_value": pl.Float64(),
}

LIQUIDATION_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": ts_dtype(),
    "symbol": pl.Utf8(),
    "side": pl.Utf8(),
    "price": pl.Float64(),
    "quantity": pl.Float64(),
    "notional": pl.Float64(),
}

METRICS_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": ts_dtype(),
    "symbol": pl.Utf8(),
    "open_interest": pl.Float64(),
    "open_interest_value": pl.Float64(),
    "toptrader_account_long_short_ratio": pl.Float64(),
    "toptrader_position_long_short_ratio": pl.Float64(),
    "global_account_long_short_ratio": pl.Float64(),
    "taker_long_short_vol_ratio": pl.Float64(),
}

SCHEMAS: dict[DatasetType, dict[str, pl.DataType]] = {
    DatasetType.BARS: BAR_SCHEMA,
    DatasetType.TRADES: TRADE_SCHEMA,
    DatasetType.AGG_TRADES: AGG_TRADE_SCHEMA,
    DatasetType.FUNDING: FUNDING_SCHEMA,
    DatasetType.OPEN_INTEREST: OPEN_INTEREST_SCHEMA,
    DatasetType.LIQUIDATION: LIQUIDATION_SCHEMA,
    DatasetType.METRICS: METRICS_SCHEMA,
    DatasetType.MARK_PRICE_BARS: BAR_SCHEMA,
    DatasetType.INDEX_PRICE_BARS: BAR_SCHEMA,
    DatasetType.PREMIUM_INDEX_BARS: BAR_SCHEMA,
}


def schema_for(dataset: DatasetType) -> dict[str, pl.DataType]:
    return SCHEMAS[dataset]


def empty_frame(dataset: DatasetType) -> pl.DataFrame:
    return pl.DataFrame(schema=schema_for(dataset))


def cast_canonical(df: pl.DataFrame, dataset: DatasetType) -> pl.DataFrame:
    """Select and cast columns to the canonical schema for ``dataset``.

    Missing optional columns are created as nulls; extra columns are dropped.
    """
    schema = schema_for(dataset)
    exprs: list[pl.Expr] = []
    for name, dtype in schema.items():
        if name in df.columns:
            if dtype == pl.Utf8():
                exprs.append(pl.col(name).cast(pl.Utf8, strict=False).alias(name))
            elif dtype == pl.Int64():
                exprs.append(pl.col(name).cast(pl.Int64, strict=False).alias(name))
            else:
                exprs.append(pl.col(name).cast(pl.Float64, strict=False).alias(name))
        else:
            exprs.append(pl.lit(None).cast(dtype).alias(name))
    out = df.select(exprs)
    # timestamp is the only datetime; guarantee UTC typing.
    out = out.with_columns(pl.col("timestamp").cast(ts_dtype()))
    return out
