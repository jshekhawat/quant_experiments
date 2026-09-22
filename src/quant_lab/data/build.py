"""High-level dataset build orchestration.

Ties together: download raw archives -> normalize to Parquet -> update the
catalog. Public entry point is :func:`build_dataset`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ..config import Config
from ..config import get_config as _get_config
from .catalog import Catalog, build_entry, catalog_path
from .normalize import normalize_frames, normalize_raw_files
from .schemas import DatasetType
from .sources.binance import BinanceSource
from .storage import DatasetKey, list_partitions


@dataclass(slots=True)
class BuildResult:
    key: DatasetKey
    raw_downloaded: list[Path] = field(default_factory=list)
    raw_cached: list[Path] = field(default_factory=list)
    partitions: list[Path] = field(default_factory=list)
    row_count: int = 0

    def summary(self) -> str:
        return (
            f"{self.key.dataset_id}: {len(self.raw_downloaded)} downloaded, "
            f"{len(self.raw_cached)} cached, {len(self.partitions)} partitions, "
            f"{self.row_count} rows"
        )


def make_source(config: Config | None = None, market_type: str | None = None) -> BinanceSource:
    config = config or _get_config()
    return BinanceSource(
        config.paths.data_raw,
        base_url=config.data.base_url,
        timeout=config.data.request_timeout,
        verify_checksums=config.data.verify_checksums,
        market_type=market_type or config.data.market,
    )


_PERIOD_RE = re.compile(r"(\d{4})-(\d{2})(?:-(\d{2}))?$")


def _raw_period_in_range(path: Path, start: str | None, end: str | None) -> bool:
    """Filter raw archive filenames by their trailing period (YYYY-MM[-DD])."""
    match = _PERIOD_RE.search(path.stem)
    if not match:
        return True
    year, month, day = match.group(1), match.group(2), match.group(3)
    period_start = f"{year}-{month}-{day}" if day else f"{year}-{month}-01"
    period_end = f"{year}-{month}-{day}" if day else _month_end(year, month)
    if start and period_end < start[:10]:
        return False
    if end and period_start > end[:10]:
        return False
    return True


def _month_end(year: str, month: str) -> str:
    import calendar

    last = calendar.monthrange(int(year), int(month))[1]
    return f"{year}-{month}-{last:02d}"


def _existing_raw_files(source: BinanceSource, key: DatasetKey, start: str | None, end: str | None) -> list[Path]:
    d = source.raw_dir(key.dataset, key.symbol, key.timeframe if key.timeframe != "na" else None, key.market_type)
    if not d.exists():
        return []
    files = sorted(d.glob("*.zip"))
    return [f for f in files if _raw_period_in_range(f, start, end)]


def build_dataset(
    *,
    dataset: DatasetType,
    symbol: str,
    interval: str | None = None,
    start: str,
    end: str,
    market_type: str | None = None,
    exchange: str = "binance",
    download: bool = True,
    granularity: str | None = None,
    config: Config | None = None,
) -> BuildResult:
    """Download (optionally) and normalize a dataset, then update the catalog."""
    config = config or _get_config()
    market_type = market_type or config.data.market
    key = DatasetKey(
        exchange=exchange,
        market_type=market_type,
        dataset=dataset,
        symbol=symbol,
        timeframe=interval or "na",
    )
    source = make_source(config, market_type)
    result = BuildResult(key=key)

    if dataset is DatasetType.OPEN_INTEREST:
        return _build_open_interest(
            source, key, start=start, end=end, download=download, config=config
        )

    if download:
        dl = source.download(
            market_type=market_type,
            symbol=symbol,
            dataset=dataset,
            interval=interval,
            start=start,
            end=end,
            granularity=granularity,
        )
        result.raw_downloaded = dl.downloaded
        result.raw_cached = dl.skipped
        if dl.missing:
            import logging

            logging.getLogger(__name__).warning(
                "%d archive(s) missing for %s (likely before listing or not yet published)",
                len(dl.missing),
                key.dataset_id,
            )

    raw_files = _existing_raw_files(source, key, start, end)
    if not raw_files:
        raise FileNotFoundError(
            f"no raw archives found for {key.dataset_id}; run download first"
        )

    result.partitions = normalize_raw_files(source, key, raw_files, config.paths.data_normalized)
    result.row_count = _count(result.partitions)

    catalog = Catalog.load(catalog_path(config.paths.data_catalogs))
    entry = build_entry(
        key,
        config.paths.data_normalized,
        source=source.name,
        start=f"{start}T00:00:00+00:00",
        end=f"{end}T00:00:00+00:00",
    )
    catalog.upsert(entry)
    return result


def _build_open_interest(
    source: BinanceSource,
    key: DatasetKey,
    *,
    start: str,
    end: str,
    download: bool,
    config: Config,
) -> BuildResult:
    """Open interest is derived from the daily ``metrics`` archive."""
    result = BuildResult(key=key)
    metrics_key = DatasetKey(
        exchange=key.exchange,
        market_type=key.market_type,
        dataset=DatasetType.METRICS,
        symbol=key.symbol,
        timeframe="na",
    )
    if download:
        dl = source.download(
            market_type=key.market_type,
            symbol=key.symbol,
            dataset=DatasetType.METRICS,
            interval=None,
            start=start,
            end=end,
            granularity="daily",
        )
        result.raw_downloaded = dl.downloaded
        result.raw_cached = dl.skipped

    raw_files = _existing_raw_files(source, metrics_key, start, end)
    if not raw_files:
        raise FileNotFoundError(
            f"no metrics archives for {metrics_key.dataset_id}; open interest derives from metrics"
        )
    frames: list[pl.DataFrame] = []
    for path in raw_files:
        m = source.parse_raw(path, DatasetType.METRICS, key.symbol)
        frames.append(
            m.select(
                pl.col("timestamp"),
                pl.col("symbol"),
                pl.col("open_interest"),
                pl.col("open_interest_value"),
            )
        )
    result.partitions = normalize_frames(frames, key, config.paths.data_normalized)
    result.row_count = _count(result.partitions)

    catalog = Catalog.load(catalog_path(config.paths.data_catalogs))
    catalog.upsert(
        build_entry(
            key,
            config.paths.data_normalized,
            source=source.name,
            start=f"{start}T00:00:00+00:00",
            end=f"{end}T00:00:00+00:00",
            extra={"derived_from": metrics_key.dataset_id},
        )
    )
    return result


def _count(partitions: list[Path]) -> int:
    if not partitions:
        return 0
    return int(
        pl.scan_parquet([str(p) for p in partitions])
        .select(pl.len())
        .collect()
        .item()
    )


def rebuild_catalog(config: Config | None = None) -> Catalog:
    """Rebuild the catalog from every normalized dataset on disk."""
    config = config or _get_config()
    root = config.paths.data_normalized
    catalog = Catalog(catalog_path(config.paths.data_catalogs), {})
    for exchange_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        for market_dir in sorted(p for p in exchange_dir.glob("*") if p.is_dir()):
            for dataset_dir_ in sorted(p for p in market_dir.glob("*") if p.is_dir()):
                try:
                    dataset = DatasetType(dataset_dir_.name)
                except ValueError:
                    continue
                for symbol_dir in sorted(p for p in dataset_dir_.glob("*") if p.is_dir()):
                    for tf_dir in sorted(p for p in symbol_dir.glob("*") if p.is_dir()):
                        key = DatasetKey(
                            exchange=exchange_dir.name,
                            market_type=market_dir.name,
                            dataset=dataset,
                            symbol=symbol_dir.name,
                            timeframe=tf_dir.name,
                        )
                        if not list_partitions(root, key):
                            continue
                        catalog.upsert(build_entry(key, root, source="unknown"))
    return catalog


__all__ = ["BuildResult", "build_dataset", "make_source", "rebuild_catalog"]
