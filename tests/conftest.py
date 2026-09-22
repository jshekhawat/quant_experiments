"""Shared pytest fixtures."""

from __future__ import annotations

import polars as pl
import pytest

from quant_lab.synthetic import synthetic_bars


@pytest.fixture
def bars() -> pl.DataFrame:
    return synthetic_bars(600, seed=42)


@pytest.fixture
def small_bars() -> pl.DataFrame:
    return synthetic_bars(60, seed=1)


@pytest.fixture
def tmp_root(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "normalized").mkdir()
    (tmp_path / "catalogs").mkdir()
    (tmp_path / "experiments").mkdir()
    return tmp_path
