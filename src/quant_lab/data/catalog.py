"""Lightweight machine-readable data catalog.

A single JSON file under ``data/catalogs/catalog.json`` answers:

* What datasets do I have?
* Which symbols / exchanges / timeframes?
* Which date ranges?
* Which files are complete?
* Where did the data come from?

It is deliberately not a database server. DuckDB is used separately for
analytical queries.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schemas import DatasetType
from .storage import DatasetKey, list_partitions, partition_months

CATALOG_VERSION = 1


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class CatalogEntry:
    exchange: str
    market_type: str
    symbol: str
    dataset_type: str
    timeframe: str
    start: str
    end: str
    source: str
    downloaded_at: str
    schema_version: int = CATALOG_VERSION
    row_count: int = 0
    files: list[str] = field(default_factory=list)
    checksums: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> DatasetKey:
        return DatasetKey(
            exchange=self.exchange,
            market_type=self.market_type,
            dataset=DatasetType(self.dataset_type),
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CatalogEntry:
        known = {f for f in cls.__slots__}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)


class Catalog:
    def __init__(self, path: Path, entries: dict[str, CatalogEntry] | None = None):
        self.path = Path(path)
        self._entries: dict[str, CatalogEntry] = entries or {}

    # --- persistence -----------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> Catalog:
        path = Path(path)
        if not path.exists():
            return cls(path, {})
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        entries = {
            item["dataset_id"]: CatalogEntry.from_dict(item)
            for item in data.get("datasets", [])
        }
        return cls(path, entries)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "catalog_version": CATALOG_VERSION,
            "updated_at": _now_iso(),
            "datasets": [
                {"dataset_id": did, **entry.to_dict()}
                for did, entry in sorted(self._entries.items())
            ],
        }
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(tmp, self.path)

    # --- mutation --------------------------------------------------------
    def upsert(self, entry: CatalogEntry) -> None:
        self._entries[entry.key.dataset_id] = entry
        self.save()

    def remove(self, key: DatasetKey) -> None:
        self._entries.pop(key.dataset_id, None)
        self.save()

    # --- queries ---------------------------------------------------------
    def get(self, key: DatasetKey) -> CatalogEntry | None:
        return self._entries.get(key.dataset_id)

    def entries(self) -> list[CatalogEntry]:
        return list(self._entries.values())

    def filter(
        self,
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        symbol: str | None = None,
        dataset_type: str | None = None,
        timeframe: str | None = None,
    ) -> list[CatalogEntry]:
        out: list[CatalogEntry] = []
        for e in self._entries.values():
            if exchange and e.exchange != exchange:
                continue
            if market_type and e.market_type != market_type:
                continue
            if symbol and e.symbol != symbol:
                continue
            if dataset_type and e.dataset_type != dataset_type:
                continue
            if timeframe and e.timeframe != timeframe:
                continue
            out.append(e)
        return out

    def to_records(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries.values()]

    def to_polars(self):
        import polars as pl

        records = self.to_records()
        if not records:
            return pl.DataFrame(
                schema={
                    "exchange": pl.Utf8,
                    "market_type": pl.Utf8,
                    "symbol": pl.Utf8,
                    "dataset_type": pl.Utf8,
                    "timeframe": pl.Utf8,
                    "start": pl.Utf8,
                    "end": pl.Utf8,
                    "source": pl.Utf8,
                    "row_count": pl.Int64,
                }
            )
        return pl.DataFrame(records)


def build_entry(
    key: DatasetKey,
    root: Path,
    *,
    source: str,
    start: str | None = None,
    end: str | None = None,
    checksums: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> CatalogEntry:
    """Inspect the partitioned files on disk and build a catalog entry."""
    import polars as pl

    from .storage import dataset_dir

    files = list_partitions(root, key)
    row_count = 0
    first_ts = None
    last_ts = None
    for f in files:
        stats = pl.scan_parquet(str(f)).select(
            pl.len().alias("n"),
            pl.col("timestamp").min().alias("min_ts"),
            pl.col("timestamp").max().alias("max_ts"),
        ).collect()
        row_count += int(stats["n"][0])
        mn = stats["min_ts"][0]
        mx = stats["max_ts"][0]
        first_ts = mn if first_ts is None or (mn is not None and mn < first_ts) else first_ts
        last_ts = mx if last_ts is None or (mx is not None and mx > last_ts) else last_ts

    base = dataset_dir(root, key)
    rel_files = [str(f.relative_to(base.parent)) for f in files]

    def iso(x: Any) -> str | None:
        return x.isoformat() if x is not None else None

    return CatalogEntry(
        exchange=key.exchange,
        market_type=key.market_type,
        symbol=key.symbol,
        dataset_type=key.dataset.value,
        timeframe=key.timeframe,
        start=start or iso(first_ts) or "",
        end=end or iso(last_ts) or "",
        source=source,
        downloaded_at=_now_iso(),
        row_count=row_count,
        files=rel_files,
        checksums=checksums or {},
        extra=extra or {},
    )


def catalog_path(catalogs_dir: Path) -> Path:
    return Path(catalogs_dir) / "catalog.json"


def available_months(root: Path, key: DatasetKey) -> Iterable[tuple[int, int]]:
    return partition_months(list_partitions(root, key))
