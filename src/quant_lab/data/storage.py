"""Parquet storage layout.

Parquet is the persistent format; DuckDB is the query layer. Files are
partitioned by month to avoid creating thousands of tiny files::

    {root}/{exchange}/{market_type}/{dataset}/{symbol}/{timeframe}/{year}/{month:02d}.parquet

For non-bar datasets ``timeframe`` is ``"na"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .schemas import DatasetType, cast_canonical, schema_for


@dataclass(frozen=True, slots=True)
class DatasetKey:
    exchange: str
    market_type: str
    dataset: DatasetType
    symbol: str
    timeframe: str = "na"

    @property
    def dataset_id(self) -> str:
        return f"{self.exchange}/{self.market_type}/{self.dataset.value}/{self.symbol}/{self.timeframe}"

    def as_dict(self) -> dict[str, str]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "dataset": self.dataset.value,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> DatasetKey:
        return cls(
            exchange=d["exchange"],
            market_type=d["market_type"],
            dataset=DatasetType(d["dataset"]),
            symbol=d["symbol"],
            timeframe=d.get("timeframe", "na"),
        )


def dataset_dir(root: Path, key: DatasetKey) -> Path:
    return (
        Path(root)
        / key.exchange
        / key.market_type
        / key.dataset.value
        / key.symbol
        / key.timeframe
    )


def partition_path(root: Path, key: DatasetKey, year: int, month: int) -> Path:
    return dataset_dir(root, key) / f"{year:04d}" / f"{month:02d}.parquet"


def partition_glob(root: Path, key: DatasetKey) -> str:
    base = dataset_dir(root, key)
    return str(base / "*" / "*.parquet")


def write_partition(df: pl.DataFrame, key: DatasetKey, root: Path, year: int, month: int) -> Path:
    """Write (or overwrite) a single monthly partition atomically."""
    path = partition_path(root, key, year, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = cast_canonical(df, key.dataset)
    frame = frame.sort("timestamp")
    tmp = path.with_suffix(".parquet.tmp")
    frame.write_parquet(tmp, compression="zstd", statistics=True)
    tmp.replace(path)
    return path


def list_partitions(root: Path, key: DatasetKey) -> list[Path]:
    base = dataset_dir(root, key)
    if not base.exists():
        return []
    return sorted(base.glob("*/*.parquet"))


def read_dataset(
    root: Path,
    key: DatasetKey,
    *,
    start: str | None = None,
    end: str | None = None,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Read a full normalized dataset, optionally date-filtered.

    Uses Polars' lazy scanner over the Parquet glob so predicates are pushed
    down. Returns an empty frame with the canonical schema when nothing exists.
    """
    files = list_partitions(root, key)
    if not files:
        from .schemas import empty_frame

        return empty_frame(key.dataset)

    lf = pl.scan_parquet([str(f) for f in files])
    if start is not None:
        lf = lf.filter(pl.col("timestamp") >= pl.lit(start).str.to_datetime(time_zone="UTC"))
    if end is not None:
        lf = lf.filter(pl.col("timestamp") < pl.lit(end).str.to_datetime(time_zone="UTC"))
    if columns is not None:
        lf = lf.select(columns)
    return lf.collect()


def partition_months(paths: list[Path]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for p in paths:
        try:
            out.append((int(p.parent.name), int(p.stem)))
        except ValueError:
            continue
    return sorted(out)


__all__ = [
    "DatasetKey",
    "dataset_dir",
    "list_partitions",
    "partition_glob",
    "partition_months",
    "partition_path",
    "read_dataset",
    "schema_for",
    "write_partition",
]
