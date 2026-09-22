"""Normalization, partitioning and query tests."""

from __future__ import annotations

import polars as pl

from quant_lab.data.normalize import normalize_frames
from quant_lab.data.query import count_rows, query
from quant_lab.data.schemas import DatasetType
from quant_lab.data.storage import DatasetKey, read_dataset
from quant_lab.synthetic import synthetic_bars

KEY = DatasetKey("synthetic", "test", DatasetType.BARS, "TESTUSDT", "1m")


def test_normalize_writes_monthly_partitions(tmp_root):
    bars = synthetic_bars(3000, seed=3)
    paths = normalize_frames([bars], KEY, tmp_root / "normalized")
    assert paths
    # 3000 minutes from 2024-01-01 -> spans Jan 1-2 only
    assert len(paths) >= 1
    for p in paths:
        assert p.name.endswith(".parquet")


def test_normalize_is_idempotent(tmp_root):
    bars = synthetic_bars(500, seed=4)
    normalize_frames([bars], KEY, tmp_root / "normalized")
    normalize_frames([bars], KEY, tmp_root / "normalized")
    out = read_dataset(tmp_root / "normalized", KEY)
    assert out.height == 500


def test_duplicate_timestamps_deduped(tmp_root):
    bars = synthetic_bars(10, seed=5)
    doubled = pl.concat([bars, bars])
    normalize_frames([doubled], KEY, tmp_root / "normalized")
    out = read_dataset(tmp_root / "normalized", KEY)
    assert out.height == 10


def test_read_dataset_date_filter(tmp_root):
    bars = synthetic_bars(2000, seed=6)
    normalize_frames([bars], KEY, tmp_root / "normalized")
    out = read_dataset(
        tmp_root / "normalized", KEY, start="2024-01-01T05:00:00", end="2024-01-01T06:00:00"
    )
    assert out.height == 60
    assert out["timestamp"].min().isoformat().startswith("2024-01-01T05:00")


def test_duckdb_query_layer(tmp_root):
    bars = synthetic_bars(300, seed=8)
    normalize_frames([bars], KEY, tmp_root / "normalized")
    assert count_rows(tmp_root / "normalized", KEY) == 300
    agg = query(
        tmp_root / "normalized", KEY,
        select="count(*) AS n, max(close) AS hi",
    )
    assert agg["n"][0] == 300
    assert agg["hi"][0] == bars["close"].max()
