"""Raw archive parsing and normalization tests."""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl
import pytest

from quant_lab.data.schemas import DatasetType
from quant_lab.data.sources.binance import BinanceSource

KLINE_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore\n"
)
KLINE_ROW = "1704067200000,42314.00,42335.80,42289.60,42331.90,289.641,1704067259999,12256155.25625,3310,175.211,7414459.86355,0\n"
KLINE_ROW_HEADERLESS = "1704067200000,42314.00,42335.80,42289.60,42331.90,289.641,1704067259999,12256155.25625,3310,175.211,7414459.86355,0\n"


def _make_zip(path: Path, content: str, member: str = "data.csv") -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, content)
    return path


def _source(tmp_path: Path) -> BinanceSource:
    return BinanceSource(tmp_path)


def test_parse_klines_with_header(tmp_path):
    path = _make_zip(tmp_path / "k.zip", KLINE_HEADER + KLINE_ROW)
    df = _source(tmp_path).parse_raw(path, DatasetType.BARS, "BTCUSDT")
    assert df.height == 1
    assert df.columns == [
        "timestamp", "open", "high", "low", "close", "volume", "quote_volume",
        "trade_count", "taker_buy_volume", "taker_buy_quote_volume",
    ]
    assert df["timestamp"].dtype == pl.Datetime("us", "UTC")
    assert df["timestamp"][0].isoformat() == "2024-01-01T00:00:00+00:00"
    assert df["close"][0] == pytest.approx(42331.90)
    assert df["trade_count"][0] == 3310


def test_parse_klines_without_header(tmp_path):
    path = _make_zip(tmp_path / "k.zip", KLINE_ROW_HEADERLESS)
    df = _source(tmp_path).parse_raw(path, DatasetType.BARS, "BTCUSDT")
    assert df.height == 1
    assert df["open"][0] == pytest.approx(42314.00)


def test_microsecond_timestamps_detected(tmp_path):
    row = "1704067200000000,1,1,1,1,1,0,0,0,0,0,0\n"
    path = _make_zip(tmp_path / "k.zip", row)
    df = _source(tmp_path).parse_raw(path, DatasetType.BARS, "BTCUSDT")
    assert df["timestamp"][0].isoformat() == "2024-01-01T00:00:00+00:00"


def test_trade_side_mapping(tmp_path):
    content = (
        "id,price,qty,quote_qty,time,is_buyer_maker\n"
        "1,100,1,100,1704067200000,true\n"
        "2,100,1,100,1704067201000,false\n"
    )
    path = _make_zip(tmp_path / "t.zip", content)
    df = _source(tmp_path).parse_raw(path, DatasetType.TRADES, "BTCUSDT")
    assert df["side"].to_list() == ["sell", "buy"]
    assert df["trade_id"].to_list() == [1, 2]


def test_funding_parse(tmp_path):
    content = "calc_time,funding_interval_hours,last_funding_rate\n1704067200000,8,0.00037409\n"
    path = _make_zip(tmp_path / "f.zip", content)
    df = _source(tmp_path).parse_raw(path, DatasetType.FUNDING, "BTCUSDT")
    assert df["funding_rate"][0] == pytest.approx(0.00037409)
    assert df["symbol"][0] == "BTCUSDT"
