"""Downloader URL construction, checksum and idempotency tests."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

from quant_lab.data.build import _raw_period_in_range
from quant_lab.data.schemas import DatasetType
from quant_lab.data.sources.binance import BinanceSource, _iter_days, _iter_months


def _source(tmp_path: Path, **kwargs) -> BinanceSource:
    return BinanceSource(tmp_path, base_url="https://data.binance.vision", **kwargs)


def test_archive_url_klines(tmp_path):
    src = _source(tmp_path)
    url = src.archive_url(
        market_type="futures", dataset=DatasetType.BARS, symbol="BTCUSDT",
        interval="1m", period="2024-01", granularity="monthly",
    )
    assert url == "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"


def test_archive_url_trades_and_funding(tmp_path):
    src = _source(tmp_path)
    assert src.archive_url(
        market_type="futures", dataset=DatasetType.TRADES, symbol="ETHUSDT",
        interval=None, period="2024-02", granularity="monthly",
    ).endswith("/monthly/trades/ETHUSDT/ETHUSDT-trades-2024-02.zip")
    assert src.archive_url(
        market_type="futures", dataset=DatasetType.FUNDING, symbol="ETHUSDT",
        interval=None, period="2024-02", granularity="monthly",
    ).endswith("/monthly/fundingRate/ETHUSDT/ETHUSDT-fundingRate-2024-02.zip")


def test_archive_url_metrics_daily(tmp_path):
    src = _source(tmp_path)
    assert src.archive_url(
        market_type="futures", dataset=DatasetType.METRICS, symbol="BTCUSDT",
        interval=None, period="2024-01-05", granularity="daily",
    ).endswith("/daily/metrics/BTCUSDT/BTCUSDT-metrics-2024-01-05.zip")


def test_granularity_selection(tmp_path):
    src = _source(tmp_path)
    assert src._granularity(DatasetType.BARS) == "monthly"
    assert src._granularity(DatasetType.FUNDING) == "monthly"
    assert src._granularity(DatasetType.METRICS) == "daily"


def test_period_iterators():
    months = list(_iter_months(date(2024, 1, 1), date(2024, 3, 15)))
    assert [m[0] for m in months] == ["2024-01", "2024-02", "2024-03"]
    days = list(_iter_days(date(2024, 1, 1), date(2024, 1, 3)))
    assert [d[0] for d in days] == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_raw_period_in_range():
    p = Path("BTCUSDT-1m-2024-02.zip")
    assert _raw_period_in_range(p, "2024-01-01", "2024-03-01")
    assert not _raw_period_in_range(p, "2024-03-01", "2024-04-01")
    daily = Path("BTCUSDT-metrics-2024-02-10.zip")
    assert _raw_period_in_range(daily, "2024-02-01", "2024-02-28")
    assert not _raw_period_in_range(daily, "2024-02-11", "2024-02-28")


class _FakeResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content = content
        self.status_code = status
        self.text = content.decode("utf-8", errors="replace")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("status error")

    def iter_content(self, chunk_size=1024):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]


class _FakeSession:
    def __init__(self, payload: bytes, checksum: str | None):
        self.payload = payload
        self.checksum = checksum

    def get(self, url, timeout=None, stream=False):
        if url.endswith(".CHECKSUM"):
            body = f"{self.checksum}  file.zip".encode() if self.checksum else b""
            return _FakeResponse(body, 200 if self.checksum else 404)
        return _FakeResponse(self.payload, 200)


def test_download_one_verifies_and_caches(tmp_path):
    payload = b"hello world"
    checksum = hashlib.sha256(payload).hexdigest()
    src = BinanceSource(tmp_path, session=_FakeSession(payload, checksum), verify_checksums=True)
    dest = tmp_path / "f.zip"
    status, path = src._download_one("https://example.com/f.zip", dest)
    assert status == "downloaded"
    assert path == dest and dest.read_bytes() == payload
    assert dest.with_suffix(".zip.sha256").exists()

    # second call is served from cache without re-downloading
    status2, _ = src._download_one("https://example.com/f.zip", dest)
    assert status2 == "cached"


def test_download_one_checksum_mismatch_fails(tmp_path):
    src = BinanceSource(tmp_path, session=_FakeSession(b"data", "0" * 64), verify_checksums=True)
    dest = tmp_path / "f.zip"
    status, path = src._download_one("https://example.com/f.zip", dest)
    assert status == "failed"
    assert path is None
    assert not dest.exists()


def test_unsupported_dataset_raises(tmp_path):
    import pytest

    src = _source(tmp_path)
    with pytest.raises(NotImplementedError):
        src.download(
            market_type="futures", symbol="BTCUSDT", dataset=DatasetType.LIQUIDATION,
            interval=None, start="2024-01-01", end="2024-02-01",
        )
