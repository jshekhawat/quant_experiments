"""Raw archive -> canonical Parquet normalization.

Normalization is explicit and conservative: it casts to canonical schemas,
sorts by timestamp, and de-duplicates on natural keys. It does **not** silently
repair gaps or outliers — that is the validator's job, and it only reports.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .schemas import DatasetType, cast_canonical
from .storage import (
    DatasetKey,
    partition_path,
    write_partition,
)

# Natural (dedup) keys per dataset. Timestamp alone for point-in-time series.
PRIMARY_KEYS: dict[DatasetType, list[str]] = {
    DatasetType.BARS: ["timestamp"],
    DatasetType.MARK_PRICE_BARS: ["timestamp"],
    DatasetType.INDEX_PRICE_BARS: ["timestamp"],
    DatasetType.PREMIUM_INDEX_BARS: ["timestamp"],
    DatasetType.TRADES: ["timestamp", "trade_id"],
    DatasetType.AGG_TRADES: ["timestamp", "agg_trade_id"],
    DatasetType.FUNDING: ["timestamp"],
    DatasetType.METRICS: ["timestamp"],
    DatasetType.OPEN_INTEREST: ["timestamp"],
    DatasetType.LIQUIDATION: ["timestamp", "symbol", "side", "price", "quantity"],
}


def _dedup_keys(key: DatasetKey) -> list[str]:
    return PRIMARY_KEYS[key.dataset]


def _merge_with_existing(
    frame: pl.DataFrame, key: DatasetKey, root: Path, year: int, month: int
) -> pl.DataFrame:
    existing = partition_path(root, key, year, month)
    if existing.exists():
        prior = pl.read_parquet(existing)
        frame = pl.concat([prior, frame], how="diagonal_relaxed")
    keys = [k for k in _dedup_keys(key) if k in frame.columns]
    # Keep the last occurrence for each natural key (newest write wins).
    frame = frame.unique(subset=keys, keep="last")
    return frame.sort("timestamp")


def normalize_frames(
    frames: list[pl.DataFrame],
    key: DatasetKey,
    root: Path,
) -> list[Path]:
    """Concatenate parsed frames and write monthly partitions.

    Merges with any existing partitions so repeated/incremental runs are
    idempotent.
    """
    frames = [f for f in frames if f is not None and f.height > 0]
    if not frames:
        return []
    combined = pl.concat([cast_canonical(f, key.dataset) for f in frames], how="diagonal_relaxed")
    combined = combined.with_columns(
        pl.col("timestamp").dt.year().alias("_year"),
        pl.col("timestamp").dt.month().alias("_month"),
    )
    written: list[Path] = []
    for (year, month), group in combined.partition_by(["_year", "_month"], as_dict=True).items():
        grp = group.drop(["_year", "_month"])
        grp = _merge_with_existing(grp, key, root, int(year), int(month))
        written.append(write_partition(grp, key, root, int(year), int(month)))
    return sorted(written)


def normalize_raw_files(
    source,
    key: DatasetKey,
    raw_files: list[Path],
    root: Path,
) -> list[Path]:
    """Parse raw archives with ``source`` and normalize into Parquet."""
    frames: list[pl.DataFrame] = []
    for path in raw_files:
        try:
            frames.append(source.parse_raw(path, key.dataset, key.symbol))
        except Exception as exc:  # noqa: BLE001 - surface file context
            raise RuntimeError(f"failed to parse {path}: {exc}") from exc
    return normalize_frames(frames, key, root)
