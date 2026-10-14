"""Recreate cumulative volume delta (CVD) from normalized bars.

Binance klines carry ``taker_buy_volume`` (and the quote equivalent), so net
taker flow is an exact per-bar reconstruction that needs no tick data:

    volume_delta = 2 * taker_buy_volume - volume
    cvd          = cumsum(volume_delta)

The cumulative sum is anchored at the first bar of the requested range and is
continuous across month partitions, so a reader sees one running line per symbol
rather than disconnected monthly segments. Spot and futures are kept as separate
series — they are different instruments and are never summed.

Output layout (mirrors the normalized store)::

    {derived}/cvd/{market_type}/{symbol}/{interval}/{year}/{month:02d}.parquet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ..config import Config, get_config
from .build import build_dataset
from .schemas import DatasetType
from .storage import DatasetKey, read_dataset

log = logging.getLogger(__name__)

# Stable physical column order for CVD partitions.
CVD_COLUMNS = (
    "timestamp",
    "symbol",
    "market",
    "volume_delta",
    "cvd",
    "quote_volume_delta",
    "cvd_quote",
)


def volume_delta_expr() -> pl.Expr:
    """Net taker flow in base units (buy taker volume minus sell taker volume)."""
    return 2.0 * pl.col("taker_buy_volume") - pl.col("volume")


def quote_volume_delta_expr() -> pl.Expr:
    """Net taker flow in quote units; comparable across price levels."""
    return 2.0 * pl.col("taker_buy_quote_volume") - pl.col("quote_volume")


def add_cvd(
    frame: pl.DataFrame,
    *,
    symbol: str | None = None,
    market: str | None = None,
) -> pl.DataFrame:
    """Sort by time and append base/quote volume delta and their cumulative sums."""
    if frame.is_empty():
        return frame
    out = frame.sort("timestamp").with_columns(
        volume_delta_expr().alias("volume_delta"),
        quote_volume_delta_expr().alias("quote_volume_delta"),
    )
    out = out.with_columns(
        pl.col("volume_delta").cum_sum().alias("cvd"),
        pl.col("quote_volume_delta").cum_sum().alias("cvd_quote"),
    )
    if symbol is not None:
        out = out.with_columns(pl.lit(symbol).alias("symbol"))
    if market is not None:
        out = out.with_columns(pl.lit(market).alias("market"))
    return out.select([column for column in CVD_COLUMNS if column in out.columns])


def cvd_dir(root: Path, market: str, symbol: str, interval: str = "1m") -> Path:
    return Path(root) / "cvd" / market / symbol / interval


def cvd_partition_path(
    root: Path,
    market: str,
    symbol: str,
    year: int,
    month: int,
    interval: str = "1m",
) -> Path:
    return cvd_dir(root, market, symbol, interval) / f"{year:04d}" / f"{month:02d}.parquet"


def write_cvd_partitions(
    frame: pl.DataFrame,
    root: Path,
    *,
    interval: str = "1m",
) -> list[Path]:
    """Split a continuous CVD frame into monthly partitions, written atomically."""
    if frame.is_empty():
        return []
    tagged = frame.sort("timestamp").with_columns(
        pl.col("timestamp").dt.year().alias("_year"),
        pl.col("timestamp").dt.month().alias("_month"),
    )
    written: list[Path] = []
    for keys, part in tagged.group_by(["_year", "_month"], maintain_order=True):
        year, month = int(keys[0]), int(keys[1])
        part = part.drop(["_year", "_month"]).sort("timestamp")
        market = str(part["market"][0])
        symbol = str(part["symbol"][0])
        path = cvd_partition_path(root, market, symbol, year, month, interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        part.write_parquet(tmp, compression="zstd", statistics=True)
        tmp.replace(path)
        written.append(path)
    return written


@dataclass(slots=True)
class CvdResult:
    """Outcome of a CVD reconstruction run."""

    markets: list[str] = field(default_factory=list)
    requested: int = 0
    built: int = 0
    partitions: int = 0
    rows: int = 0
    missing: list[str] = field(default_factory=list)
    output_root: Path | None = None

    def summary(self) -> str:
        return (
            f"CVD: {self.built}/{self.requested} series built, "
            f"{self.partitions} partitions, {self.rows} rows, "
            f"{len(self.missing)} missing -> {self.output_root}"
        )


def recreate_cvd(
    *,
    symbols: dict[str, list[str]],
    start: str,
    end: str,
    interval: str = "1m",
    download: bool = True,
    config: Config | None = None,
    output_root: Path | None = None,
) -> CvdResult:
    """Build bars if needed, then write continuous CVD for every symbol/market.

    ``symbols`` maps a market type (``futures``/``spot``) to its symbol list.
    Missing archives (pre-listing, delisted, or unarchived) are recorded rather
    than treated as failures.
    """
    cfg = config or get_config()
    root = Path(output_root) if output_root is not None else cfg.paths.data_derived
    result = CvdResult(markets=list(symbols), output_root=root)

    for market, market_symbols in symbols.items():
        for symbol in market_symbols:
            result.requested += 1
            key = DatasetKey(
                exchange="binance",
                market_type=market,
                dataset=DatasetType.BARS,
                symbol=symbol,
                timeframe=interval,
            )
            try:
                build_dataset(
                    dataset=DatasetType.BARS,
                    symbol=symbol,
                    interval=interval,
                    start=start,
                    end=end,
                    market_type=market,
                    download=download,
                    config=cfg,
                )
            except FileNotFoundError:
                result.missing.append(f"{market}/{symbol}")
                continue
            except Exception as exc:  # network/parse problems should not abort the run
                log.warning("build failed for %s/%s: %s", market, symbol, exc)
                result.missing.append(f"{market}/{symbol}")
                continue

            bars = read_dataset(cfg.paths.data_normalized, key, start=start, end=end)
            if bars.is_empty():
                result.missing.append(f"{market}/{symbol}")
                continue

            frame = add_cvd(bars, symbol=symbol, market=market)
            result.partitions += len(write_cvd_partitions(frame, root, interval=interval))
            result.rows += frame.height
            result.built += 1

    return result


__all__ = [
    "CVD_COLUMNS",
    "CvdResult",
    "add_cvd",
    "cvd_dir",
    "cvd_partition_path",
    "quote_volume_delta_expr",
    "recreate_cvd",
    "volume_delta_expr",
    "write_cvd_partitions",
]
