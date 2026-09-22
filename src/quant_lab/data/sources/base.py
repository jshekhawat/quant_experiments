"""Data source abstraction.

Strategies never talk to an exchange. They consume canonical datasets written
by a source adapter. Adapters implement a common interface so that additional
venues (Bybit, OKX, Hyperliquid, ...) can be added without touching strategy
or backtest code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ..schemas import DatasetType
from ..storage import DatasetKey


@dataclass(slots=True)
class DownloadResult:
    key: DatasetKey
    downloaded: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def n_remote_files(self) -> int:
        return len(self.downloaded) + len(self.skipped) + len(self.missing) + len(self.failed)

    def summary(self) -> str:
        return (
            f"{self.key.dataset_id}: "
            f"{len(self.downloaded)} downloaded, {len(self.skipped)} cached, "
            f"{len(self.missing)} missing, {len(self.failed)} failed"
        )


class DataSource(ABC):
    """Common interface for venue-specific historical data adapters."""

    name: str = "source"

    @abstractmethod
    def download(
        self,
        *,
        market_type: str,
        symbol: str,
        dataset: DatasetType,
        interval: str | None,
        start: str,
        end: str,
    ) -> DownloadResult:
        """Download raw archives for a dataset into the raw data directory."""

    @abstractmethod
    def parse_raw(self, zip_path: Path, dataset: DatasetType, symbol: str) -> pl.DataFrame:
        """Parse a raw archive into a canonical Polars frame."""

    # Convenience wrappers matching the spec's suggested interface. These are
    # thin and only meaningful for supported dataset types.
    def fetch_bars(self, **kwargs) -> DownloadResult:
        return self.download(dataset=DatasetType.BARS, **kwargs)

    def fetch_trades(self, **kwargs) -> DownloadResult:
        return self.download(dataset=DatasetType.TRADES, **kwargs)

    def fetch_funding(self, **kwargs) -> DownloadResult:
        return self.download(dataset=DatasetType.FUNDING, **kwargs)

    def fetch_open_interest(self, **kwargs) -> DownloadResult:
        return self.download(dataset=DatasetType.OPEN_INTEREST, **kwargs)

    def fetch_liquidations(self, **kwargs) -> DownloadResult:
        return self.download(dataset=DatasetType.LIQUIDATION, **kwargs)
