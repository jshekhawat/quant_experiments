"""Binance Public Data adapter.

Uses the bulk historical archives at https://data.binance.vision rather than
repeatedly paginating old REST history. The downloader is resumable,
idempotent, checksum-aware and safe to run repeatedly.

Raw archives are stored under::

    data/raw/binance/{market_path}/{dataset}/{symbol}/{interval}/...

Normalization to canonical Parquet happens separately in ``normalize.py``.

Notable constraints of the public archive:

* ``fundingRate`` is only published monthly.
* ``metrics`` (open interest, long/short ratios) is only published daily and
  is the reliable historical source for open interest.
* There is no reliable bulk historical liquidation dataset. Liquidation
  snapshots are intentionally *not* fabricated; see ``fetch_liquidations``.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import requests

from ..schemas import DatasetType, cast_canonical
from .base import DataSource, DownloadResult

# --- static mapping ----------------------------------------------------------

_MARKET_PATHS: dict[tuple[str, str], str] = {
    ("binance", "futures"): "futures/um",
    ("binance", "futures_um"): "futures/um",
    ("binance", "um"): "futures/um",
    ("binance", "futures_cm"): "futures/cm",
    ("binance", "coinm"): "futures/cm",
    ("binance", "spot"): "spot",
}

_DATASET_PATH: dict[DatasetType, str] = {
    DatasetType.BARS: "klines",
    DatasetType.MARK_PRICE_BARS: "markPriceKlines",
    DatasetType.INDEX_PRICE_BARS: "indexPriceKlines",
    DatasetType.PREMIUM_INDEX_BARS: "premiumIndexKlines",
    DatasetType.TRADES: "trades",
    DatasetType.AGG_TRADES: "aggTrades",
    DatasetType.FUNDING: "fundingRate",
    DatasetType.METRICS: "metrics",
}

# Which archive granularities exist for each dataset in the public archive.
_GRANULARITIES: dict[DatasetType, set[str]] = {
    DatasetType.BARS: {"monthly", "daily"},
    DatasetType.MARK_PRICE_BARS: {"monthly", "daily"},
    DatasetType.INDEX_PRICE_BARS: {"monthly", "daily"},
    DatasetType.PREMIUM_INDEX_BARS: {"monthly", "daily"},
    DatasetType.TRADES: {"monthly", "daily"},
    DatasetType.AGG_TRADES: {"monthly", "daily"},
    DatasetType.FUNDING: {"monthly"},
    DatasetType.METRICS: {"daily"},
}

# Datasets with no reliable bulk historical source. Never fabricated.
_UNSUPPORTED: dict[DatasetType, str] = {
    DatasetType.OPEN_INTEREST: (
        "Open interest is derived from the daily `metrics` archive; request "
        "DatasetType.METRICS and build the open_interest dataset from it."
    ),
    DatasetType.LIQUIDATION: (
        "Binance does not publish reliable bulk historical liquidations "
        "(the forceOrder stream is gone and liquidationSnapshot is not "
        "archived). Liquidation data is intentionally unavailable rather "
        "than fabricated."
    ),
}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _iter_months(start: date, end: date) -> Iterator[tuple[str, int, int]]:
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield f"{y:04d}-{m:02d}", y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def _iter_days(start: date, end: date) -> Iterator[tuple[str, int, int]]:
    d = start
    while d <= end:
        yield d.isoformat(), d.year, d.month
        d += timedelta(days=1)


class BinanceSource(DataSource):
    name = "binance"

    def __init__(
        self,
        raw_root: Path,
        *,
        base_url: str = "https://data.binance.vision",
        timeout: int = 120,
        verify_checksums: bool = True,
        market_type: str = "futures",
        session: requests.Session | None = None,
    ):
        self.raw_root = Path(raw_root)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_checksums = verify_checksums
        self.market_type = market_type
        self.session = session or requests.Session()

    # --- path helpers ----------------------------------------------------
    def market_path(self, market_type: str | None = None) -> str:
        key = (self.name, (market_type or self.market_type).lower())
        if key not in _MARKET_PATHS:
            raise ValueError(f"unsupported market type {market_type!r}; choose one of {_MARKET_PATHS}")
        return _MARKET_PATHS[key]

    def _raw_dir(self, market_path: str, dataset: DatasetType, symbol: str, interval: str | None) -> Path:
        return (
            self.raw_root
            / self.name
            / market_path
            / dataset.value
            / symbol
            / (interval or "na")
        )

    def raw_dir(
        self,
        dataset: DatasetType,
        symbol: str,
        interval: str | None = None,
        market_type: str | None = None,
    ) -> Path:
        """Public raw archive directory for a dataset."""
        return self._raw_dir(self.market_path(market_type), dataset, symbol, interval)

    def _filename(self, dataset: DatasetType, symbol: str, interval: str | None, period: str) -> str:
        tag = interval if dataset.is_bar_like and interval else _DATASET_PATH[dataset]
        return f"{symbol}-{tag}-{period}.zip"

    def archive_url(
        self,
        *,
        market_type: str,
        dataset: DatasetType,
        symbol: str,
        interval: str | None,
        period: str,
        granularity: str,
    ) -> str:
        market_path = self.market_path(market_type)
        fname = self._filename(dataset, symbol, interval, period)
        ds_path = _DATASET_PATH[dataset]
        if dataset.is_bar_like:
            return f"{self.base_url}/data/{market_path}/{granularity}/{ds_path}/{symbol}/{interval}/{fname}"
        return f"{self.base_url}/data/{market_path}/{granularity}/{ds_path}/{symbol}/{fname}"

    def _granularity(self, dataset: DatasetType, prefer_monthly: bool = True) -> str:
        options = _GRANULARITIES.get(dataset)
        if not options:
            raise ValueError(f"unsupported dataset for download: {dataset}")
        if prefer_monthly and "monthly" in options:
            return "monthly"
        return "daily"

    # --- download --------------------------------------------------------
    def _fetch_checksum(self, url: str) -> str | None:
        try:
            resp = self.session.get(f"{url}.CHECKSUM", timeout=self.timeout)
        except requests.RequestException:
            return None
        if resp.status_code != 200:
            return None
        token = resp.text.split()[0] if resp.text.strip() else None
        return token

    def _download_one(
        self,
        url: str,
        dest: Path,
    ) -> tuple[str, Path | None]:
        """Return (status, path). Status in {cached, downloaded, missing, failed}."""
        sidecar = dest.with_suffix(dest.suffix + ".sha256")
        expected = self._fetch_checksum(url) if self.verify_checksums else None

        if dest.exists():
            if not self.verify_checksums:
                return "cached", dest
            if sidecar.exists():
                recorded = sidecar.read_text(encoding="utf-8").split()[0]
                if _sha256(dest) == recorded and (expected is None or recorded == expected):
                    return "cached", dest
            if expected is not None and _sha256(dest) == expected:
                sidecar.write_text(f"{expected}  {dest.name}", encoding="utf-8")
                return "cached", dest
            # stale/corrupt: fall through to re-download

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with self.session.get(url, stream=True, timeout=self.timeout) as resp:
                if resp.status_code == 404:
                    return "missing", None
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
        except requests.RequestException:
            if tmp.exists():
                tmp.unlink()
            return "failed", None

        if expected is not None:
            actual = _sha256(tmp)
            if actual != expected:
                tmp.unlink()
                return "failed", None
            sidecar.write_text(f"{actual}  {dest.name}", encoding="utf-8")
        tmp.replace(dest)
        return "downloaded", dest

    def download(
        self,
        *,
        market_type: str,
        symbol: str,
        dataset: DatasetType,
        interval: str | None,
        start: str,
        end: str,
        granularity: str | None = None,
    ) -> DownloadResult:  # type: ignore[override]
        market_type = market_type or self.market_type
        result = DownloadResult(
            key=_key(self.name, market_type, dataset, symbol, interval)
        )
        if dataset in _UNSUPPORTED:
            raise NotImplementedError(f"{dataset.value}: {_UNSUPPORTED[dataset]}")

        gran = granularity or self._granularity(dataset)
        if gran not in _GRANULARITIES[dataset]:
            raise ValueError(f"{dataset.value} is not available as {gran} on Binance")

        start_d = _parse_date(start)
        end_d = _parse_date(end)
        market_path = self.market_path(market_type)
        raw_dir = self._raw_dir(market_path, dataset, symbol, interval)

        if dataset.is_bar_like and not interval:
            raise ValueError(f"{dataset.value} requires an interval (e.g. '1m')")

        iterator = _iter_months(start_d, end_d) if gran == "monthly" else _iter_days(start_d, end_d)
        for period, _y, _m in iterator:
            url = self.archive_url(
                market_type=market_type,
                dataset=dataset,
                symbol=symbol,
                interval=interval,
                period=period,
                granularity=gran,
            )
            dest = raw_dir / self._filename(dataset, symbol, interval, period)
            status, path = self._download_one(url, dest)
            if status == "cached" and path is not None:
                result.skipped.append(path)
            elif status == "downloaded" and path is not None:
                result.downloaded.append(path)
            elif status == "missing":
                result.missing.append(url)
            else:
                result.failed.append(url)
        return result

    # --- parsing ---------------------------------------------------------
    def parse_raw(self, zip_path: Path, dataset: DatasetType, symbol: str) -> pl.DataFrame:  # type: ignore[override]
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"no CSV member inside {zip_path}")
            member = members[0]
            with zf.open(member) as fh:
                head = fh.readline()
                fh.seek(0)
                has_header = not _looks_numeric(head)
                frame = pl.read_csv(
                    fh,
                    has_header=has_header,
                    infer_schema_length=2048,
                    ignore_errors=True,
                ) if has_header else pl.read_csv(
                    fh,
                    has_header=False,
                    new_columns=_POSITIONAL[dataset],
                    infer_schema_length=2048,
                    ignore_errors=True,
                )
        parsed = _PARSERS[dataset](frame, symbol)
        return cast_canonical(parsed, dataset)


# --- parsing helpers ---------------------------------------------------------

_POSITIONAL: dict[DatasetType, list[str]] = {
    DatasetType.BARS: [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ],
    DatasetType.MARK_PRICE_BARS: [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ],
    DatasetType.INDEX_PRICE_BARS: [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ],
    DatasetType.PREMIUM_INDEX_BARS: [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ],
    DatasetType.TRADES: ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"],
    DatasetType.AGG_TRADES: [
        "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
        "transact_time", "is_buyer_maker",
    ],
    DatasetType.FUNDING: ["calc_time", "funding_interval_hours", "last_funding_rate"],
    DatasetType.METRICS: [
        "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
        "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
    ],
    DatasetType.OPEN_INTEREST: [],
    DatasetType.LIQUIDATION: [],
}


def _looks_numeric(line: bytes) -> bool:
    token = line.split(b",", 1)[0].strip()
    try:
        float(token)
        return True
    except ValueError:
        return False


def _pick(frame: pl.DataFrame, *candidates: str) -> str | None:
    lowered = {c.lower(): c for c in frame.columns}
    for cand in candidates:
        if cand in frame.columns:
            return cand
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def _epoch_to_utc(col: str) -> pl.Expr:
    c = pl.col(col).cast(pl.Int64, strict=False)
    return (
        pl.when(c > 1_000_000_000_000_000)
        .then(pl.from_epoch(c, time_unit="us"))
        .when(c > 1_000_000_000_000)
        .then(pl.from_epoch(c, time_unit="ms"))
        .otherwise(pl.from_epoch(c, time_unit="s"))
        .dt.replace_time_zone("UTC")
        .alias("timestamp")
    )


def _bool_side(frame: pl.DataFrame, col: str | None) -> pl.Expr:
    if col is None:
        return pl.lit(None, dtype=pl.Utf8).alias("side")
    maker = pl.col(col)
    if frame.schema[col] == pl.Utf8:
        maker = maker.str.to_lowercase().eq("true")
    return pl.when(maker).then(pl.lit("sell")).otherwise(pl.lit("buy")).alias("side")


def _parse_bars(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    ts = _pick(frame, "open_time", "open time", "open")
    expr = [
        _epoch_to_utc(ts),
        pl.col(_pick(frame, "open")).cast(pl.Float64, strict=False).alias("open"),
        pl.col(_pick(frame, "high")).cast(pl.Float64, strict=False).alias("high"),
        pl.col(_pick(frame, "low")).cast(pl.Float64, strict=False).alias("low"),
        pl.col(_pick(frame, "close")).cast(pl.Float64, strict=False).alias("close"),
        pl.col(_pick(frame, "volume", "vol")).cast(pl.Float64, strict=False).alias("volume"),
        pl.col(_pick(frame, "quote_volume")).cast(pl.Float64, strict=False).alias("quote_volume"),
        pl.col(_pick(frame, "count", "trade_count", "number_of_trades"))
        .cast(pl.Int64, strict=False)
        .alias("trade_count"),
        pl.col(_pick(frame, "taker_buy_volume")).cast(pl.Float64, strict=False).alias("taker_buy_volume"),
        pl.col(_pick(frame, "taker_buy_quote_volume"))
        .cast(pl.Float64, strict=False)
        .alias("taker_buy_quote_volume"),
    ]
    return frame.select(expr)


def _parse_trades(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    ts = _pick(frame, "time", "transact_time", "timestamp")
    maker = _pick(frame, "is_buyer_maker", "is_maker")
    expr = [
        _epoch_to_utc(ts),
        pl.col(_pick(frame, "id", "trade_id")).cast(pl.Int64, strict=False).alias("trade_id"),
        pl.col(_pick(frame, "price")).cast(pl.Float64, strict=False).alias("price"),
        pl.col(_pick(frame, "qty", "quantity")).cast(pl.Float64, strict=False).alias("quantity"),
        pl.col(_pick(frame, "quote_qty", "quote_quantity")).cast(pl.Float64, strict=False).alias("quote_quantity"),
        _bool_side(frame, maker),
    ]
    return frame.select(expr)


def _parse_agg_trades(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    ts = _pick(frame, "transact_time", "time", "timestamp")
    maker = _pick(frame, "is_buyer_maker", "is_maker")
    expr = [
        _epoch_to_utc(ts),
        pl.col(_pick(frame, "agg_trade_id", "a")).cast(pl.Int64, strict=False).alias("agg_trade_id"),
        pl.col(_pick(frame, "price")).cast(pl.Float64, strict=False).alias("price"),
        pl.col(_pick(frame, "quantity", "qty")).cast(pl.Float64, strict=False).alias("quantity"),
        pl.col(_pick(frame, "first_trade_id")).cast(pl.Int64, strict=False).alias("first_trade_id"),
        pl.col(_pick(frame, "last_trade_id")).cast(pl.Int64, strict=False).alias("last_trade_id"),
        _bool_side(frame, maker),
    ]
    return frame.select(expr)


def _parse_funding(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    ts = _pick(frame, "calc_time", "funding_time", "timestamp")
    expr = [
        _epoch_to_utc(ts),
        pl.lit(symbol).alias("symbol"),
        pl.col(_pick(frame, "last_funding_rate", "funding_rate", "fundingRate"))
        .cast(pl.Float64, strict=False)
        .alias("funding_rate"),
        pl.col(_pick(frame, "funding_interval_hours"))
        .cast(pl.Int64, strict=False)
        .alias("funding_interval_hours"),
    ]
    if _pick(frame, "funding_interval_hours") is None:
        expr[-1] = pl.lit(None, dtype=pl.Int64).alias("funding_interval_hours")
    return frame.select(expr)


def _parse_metrics(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    ts = _pick(frame, "create_time", "timestamp", "time")
    ts_expr = (
        pl.col(ts).str.replace(" ", "T").str.to_datetime(strict=False).dt.replace_time_zone("UTC")
        if frame.schema[ts] == pl.Utf8
        else _epoch_to_utc(ts)
    )
    framesym = _pick(frame, "symbol")
    return frame.select(
        ts_expr.alias("timestamp"),
        (pl.col(framesym) if framesym else pl.lit(symbol)).alias("symbol"),
        pl.col(_pick(frame, "sum_open_interest", "open_interest"))
        .cast(pl.Float64, strict=False)
        .alias("open_interest"),
        pl.col(_pick(frame, "sum_open_interest_value", "open_interest_value"))
        .cast(pl.Float64, strict=False)
        .alias("open_interest_value"),
        pl.col(_pick(frame, "count_toptrader_long_short_ratio"))
        .cast(pl.Float64, strict=False)
        .alias("toptrader_account_long_short_ratio"),
        pl.col(_pick(frame, "sum_toptrader_long_short_ratio"))
        .cast(pl.Float64, strict=False)
        .alias("toptrader_position_long_short_ratio"),
        pl.col(_pick(frame, "count_long_short_ratio"))
        .cast(pl.Float64, strict=False)
        .alias("global_account_long_short_ratio"),
        pl.col(_pick(frame, "sum_taker_long_short_vol_ratio"))
        .cast(pl.Float64, strict=False)
        .alias("taker_long_short_vol_ratio"),
    )


_PARSERS = {
    DatasetType.BARS: _parse_bars,
    DatasetType.MARK_PRICE_BARS: _parse_bars,
    DatasetType.INDEX_PRICE_BARS: _parse_bars,
    DatasetType.PREMIUM_INDEX_BARS: _parse_bars,
    DatasetType.TRADES: _parse_trades,
    DatasetType.AGG_TRADES: _parse_agg_trades,
    DatasetType.FUNDING: _parse_funding,
    DatasetType.METRICS: _parse_metrics,
}


def _parse_date(value: str) -> date:
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def _key(exchange: str, market_type: str, dataset: DatasetType, symbol: str, interval: str | None):
    from ..storage import DatasetKey

    return DatasetKey(
        exchange=exchange,
        market_type=market_type,
        dataset=dataset,
        symbol=symbol,
        timeframe=(interval or "na"),
    )


def supported_datasets() -> dict[str, str]:
    out = {d.value: ",".join(sorted(g)) for d, g in _GRANULARITIES.items()}
    out.update({d.value: "unsupported: " + reason for d, reason in _UNSUPPORTED.items()})
    return out


__all__ = ["BinanceSource", "supported_datasets"]
