"""Data validation and data-quality reporting.

Validation **reports**; it does not repair. Any normalization or repair must be
explicit and documented elsewhere. Suspicious data is surfaced, never silently
fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from .schemas import DatasetType
from .storage import DatasetKey

SEVERITIES = ("error", "warning", "info")


@dataclass(slots=True)
class Issue:
    code: str
    severity: str
    count: int
    message: str
    examples: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "count": self.count,
            "message": self.message,
            "examples": [str(e) for e in self.examples[:10]],
        }


@dataclass(slots=True)
class DataQualityReport:
    dataset_id: str
    dataset_type: str
    row_count: int
    start: str | None
    end: str | None
    issues: list[Issue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_type": self.dataset_type,
            "row_count": self.row_count,
            "start": self.start,
            "end": self.end,
            "passed": self.passed,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "issues": [i.to_dict() for i in self.issues],
        }

    def format(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"Data quality [{status}] {self.dataset_id}",
            f"  rows={self.row_count} start={self.start} end={self.end}",
        ]
        if not self.issues:
            lines.append("  no issues found")
        for issue in self.issues:
            lines.append(
                f"  [{issue.severity.upper():<7}] {issue.code}: {issue.message} "
                f"(count={issue.count})"
            )
        return "\n".join(lines)


def _timestamp_issues(df: pl.DataFrame, timeframe: str | None) -> list[Issue]:
    issues: list[Issue] = []
    if "timestamp" not in df.columns:
        issues.append(Issue("missing_timestamp_column", "error", 1, "no timestamp column"))
        return issues
    ts = df["timestamp"]
    if ts.dtype == pl.Datetime and ts.dtype.time_zone is None:
        issues.append(
            Issue("naive_timestamps", "error", 1, "timestamps are timezone-naive, expected UTC")
        )

    diffs = ts.diff().drop_nulls()
    non_monotonic = int((diffs.dt.total_microseconds() <= 0).sum())
    if non_monotonic:
        issues.append(
            Issue(
                "non_monotonic_timestamps",
                "error",
                non_monotonic,
                "timestamps are not strictly increasing",
            )
        )

    duplicates = int(df.height - ts.n_unique())
    if duplicates:
        dup_examples = (
            df.group_by("timestamp").len().filter(pl.col("len") > 1).sort("len", descending=True).head(5)
        )
        issues.append(
            Issue(
                "duplicate_timestamps",
                "error",
                duplicates,
                "duplicate timestamps present",
                examples=dup_examples["timestamp"].to_list(),
            )
        )

    if timeframe and df.height > 1:
        from ..timeframes import timeframe_seconds

        try:
            step_us = timeframe_seconds(timeframe) * 1_000_000
        except ValueError:
            step_us = 0
        if step_us:
            span_us = int((ts.max() - ts.min()).total_seconds() * 1_000_000)
            expected = span_us // step_us + 1
            missing = expected - ts.n_unique()
            if missing > 0:
                issues.append(
                    Issue(
                        "missing_timestamps",
                        "warning",
                        int(missing),
                        f"{missing} expected {timeframe} bars missing",
                    )
                )
            gaps = diffs.dt.total_microseconds().filter(pl.Series(diffs.dt.total_microseconds()) > step_us)
            if gaps.len():
                issues.append(
                    Issue(
                        "unexpected_gaps",
                        "warning",
                        int(gaps.len()),
                        "timestamp gaps larger than the timeframe",
                    )
                )
    return issues


def _price_issues(df: pl.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            continue
        invalid = int((df[col] <= 0).sum() + df[col].is_null().sum())
        if invalid:
            issues.append(
                Issue(f"invalid_price_{col}", "error", invalid, f"non-positive or null {col}")
            )
    if {"high", "low"}.issubset(df.columns):
        bad_range = int((df["high"] < df["low"]).sum())
        if bad_range:
            issues.append(Issue("high_below_low", "error", bad_range, "high < low"))
    if {"high", "open", "close"}.issubset(df.columns):
        bad = int(((df["high"] < df["open"]) | (df["high"] < df["close"])).sum())
        if bad:
            issues.append(Issue("high_below_body", "error", bad, "high below open/close"))
    if {"low", "open", "close"}.issubset(df.columns):
        bad = int(((df["low"] > df["open"]) | (df["low"] > df["close"])).sum())
        if bad:
            issues.append(Issue("low_above_body", "error", bad, "low above open/close"))
    return issues


def _quantity_issues(df: pl.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    for col in ("volume", "quote_volume", "taker_buy_volume", "quantity", "quote_quantity"):
        if col not in df.columns:
            continue
        bad = int((df[col] < 0).sum())
        if bad:
            issues.append(Issue(f"negative_{col}", "error", bad, f"negative {col}"))
    if {"taker_buy_volume", "volume"}.issubset(df.columns):
        bad = int((df["taker_buy_volume"] > df["volume"] * 1.0000001).sum())
        if bad:
            issues.append(
                Issue(
                    "taker_buy_exceeds_volume",
                    "warning",
                    bad,
                    "taker buy volume exceeds total volume",
                )
            )
    return issues


def _outlier_issues(df: pl.DataFrame, threshold: float = 0.5) -> list[Issue]:
    if "close" not in df.columns or df.height < 3:
        return []
    returns = df["close"].pct_change().abs()
    outliers = int((returns > threshold).sum())
    if outliers:
        return [
            Issue(
                "price_outliers",
                "warning",
                outliers,
                f"|bar return| > {threshold:.0%}",
            )
        ]
    return []


def _duplicate_trade_issues(df: pl.DataFrame, dataset: DatasetType) -> list[Issue]:
    key = None
    if dataset is DatasetType.TRADES and "trade_id" in df.columns:
        key = "trade_id"
    elif dataset is DatasetType.AGG_TRADES and "agg_trade_id" in df.columns:
        key = "agg_trade_id"
    if key is None:
        return []
    dups = int(df.height - df[key].n_unique())
    if dups:
        return [Issue("duplicate_trades", "error", dups, f"duplicate {key} values")]
    return []


def validate(df: pl.DataFrame, key: DatasetKey, *, timeframe: str | None = None) -> DataQualityReport:
    """Validate a canonical dataset and return a report."""
    tf = timeframe or (key.timeframe if key.timeframe != "na" else None)
    report = DataQualityReport(
        dataset_id=key.dataset_id,
        dataset_type=key.dataset.value,
        row_count=df.height,
        start=str(df["timestamp"].min()) if "timestamp" in df.columns and df.height else None,
        end=str(df["timestamp"].max()) if "timestamp" in df.columns and df.height else None,
    )
    if df.height == 0:
        report.issues.append(Issue("empty_dataset", "warning", 0, "dataset has no rows"))
        return report

    report.issues.extend(_timestamp_issues(df, tf))
    report.issues.extend(_duplicate_trade_issues(df, key.dataset))
    if key.dataset.is_bar_like:
        report.issues.extend(_price_issues(df))
        report.issues.extend(_quantity_issues(df))
        report.issues.extend(_outlier_issues(df))
    return report


def validate_dataset(root, key: DatasetKey, *, start: str | None = None, end: str | None = None) -> DataQualityReport:
    from .storage import read_dataset

    df = read_dataset(root, key, start=start, end=end)
    return validate(df, key)
