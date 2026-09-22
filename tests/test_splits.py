"""Chronological split tests."""

from __future__ import annotations

from quant_lab.analytics.splits import (
    chronological_split,
    split_by_dates,
    walk_forward_splits,
)
from quant_lab.synthetic import synthetic_bars


def test_chronological_split_sizes():
    bars = synthetic_bars(1000, seed=21)
    splits = chronological_split(bars, train=0.6, validation=0.2, test=0.2)
    assert splits["train"].rows == 600
    assert splits["validation"].rows == 200
    assert splits["test"].rows == 200
    # strictly chronological
    assert splits["train"].end <= splits["validation"].start
    assert splits["validation"].end <= splits["test"].start


def test_split_by_dates():
    bars = synthetic_bars(2000, seed=22)
    splits = split_by_dates(bars, train_end="2024-01-02T00:00:00", validation_end="2024-01-02T12:00:00")
    assert set(splits) == {"train", "validation", "test"}
    assert splits["train"].end < splits["validation"].start


def test_walk_forward_windows_advance():
    bars = synthetic_bars(1000, seed=23)
    windows = list(walk_forward_splits(bars, train_rows=300, test_rows=100, step_rows=100))
    assert len(windows) == 7
    train0, test0 = windows[0]
    train1, test1 = windows[1]
    assert test0.start < test1.start
    assert test0.end <= test1.start or test0.rows > 0
