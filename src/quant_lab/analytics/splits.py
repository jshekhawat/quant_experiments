"""Chronological dataset splitting.

Never optimize on the full sample and present it as out-of-sample evidence.
Splits are always chronological and derived from ranges that actually exist in
the data — no hardcoded calendar dates.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import polars as pl


@dataclass(slots=True)
class Split:
    name: str
    frame: pl.DataFrame

    @property
    def start(self) -> str | None:
        return str(self.frame["timestamp"].min()) if self.frame.height else None

    @property
    def end(self) -> str | None:
        return str(self.frame["timestamp"].max()) if self.frame.height else None

    @property
    def rows(self) -> int:
        return self.frame.height

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "rows": self.rows, "start": self.start, "end": self.end}


def chronological_split(
    df: pl.DataFrame,
    *,
    train: float = 0.6,
    validation: float = 0.2,
    test: float = 0.2,
) -> dict[str, Split]:
    """Split a frame into train/validation/test by row order (chronological)."""
    if abs(train + validation + test - 1.0) > 1e-9:
        raise ValueError("train + validation + test must sum to 1.0")
    if "timestamp" not in df.columns:
        raise ValueError("frame must have a timestamp column")
    ordered = df.sort("timestamp")
    n = ordered.height
    n_train = int(n * train)
    n_val = int(n * validation)
    train_df = ordered.slice(0, n_train)
    val_df = ordered.slice(n_train, n_val)
    test_df = ordered.slice(n_train + n_val, n - n_train - n_val)
    return {
        "train": Split("train", train_df),
        "validation": Split("validation", val_df),
        "test": Split("test", test_df),
    }


def split_by_dates(
    df: pl.DataFrame,
    *,
    train_end: str,
    validation_end: str | None = None,
) -> dict[str, Split]:
    """Explicit chronological split by timestamp boundaries."""
    ordered = df.sort("timestamp")
    train_df = ordered.filter(pl.col("timestamp") < pl.lit(train_end).str.to_datetime(time_zone="UTC"))
    rest = ordered.filter(pl.col("timestamp") >= pl.lit(train_end).str.to_datetime(time_zone="UTC"))
    out = {"train": Split("train", train_df)}
    if validation_end:
        val_df = rest.filter(
            pl.col("timestamp") < pl.lit(validation_end).str.to_datetime(time_zone="UTC")
        )
        test_df = rest.filter(
            pl.col("timestamp") >= pl.lit(validation_end).str.to_datetime(time_zone="UTC")
        )
        out["validation"] = Split("validation", val_df)
        out["test"] = Split("test", test_df)
    else:
        out["test"] = Split("test", rest)
    return out


def walk_forward_splits(
    df: pl.DataFrame,
    *,
    train_rows: int,
    test_rows: int,
    step_rows: int | None = None,
    anchored: bool = False,
) -> Iterator[tuple[Split, Split]]:
    """Yield rolling (train, test) windows.

    ``anchored=True`` grows the training window (anchored walk-forward);
    otherwise the window rolls forward by ``step_rows`` (default ``test_rows``).
    """
    ordered = df.sort("timestamp")
    n = ordered.height
    step = step_rows or test_rows
    if train_rows <= 0 or test_rows <= 0 or step <= 0:
        raise ValueError("train_rows, test_rows and step_rows must be positive")
    start = 0
    index = 0
    while start + train_rows + test_rows <= n:
        train_start = 0 if anchored else start
        train_df = ordered.slice(train_start, start + train_rows - train_start)
        test_df = ordered.slice(start + train_rows, test_rows)
        yield (
            Split(f"train_{index}", train_df),
            Split(f"test_{index}", test_df),
        )
        start += step
        index += 1


def split_summary(splits: dict[str, Split]) -> str:
    lines = ["Chronological splits:"]
    for name, split in splits.items():
        lines.append(f"  {name:<10} rows={split.rows:<8} {split.start} -> {split.end}")
    return "\n".join(lines)
