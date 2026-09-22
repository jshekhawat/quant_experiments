"""Dashboard data-helper tests (read-only, no Streamlit runtime required)."""

from __future__ import annotations

import polars as pl
import pytest

from quant_lab.backtest.engine import run_backtest
from quant_lab.config import ExecutionConfig
from quant_lab.dashboard.data import (
    assumptions_table,
    list_summaries,
    load_bundle,
    metrics_table,
)
from quant_lab.reporting.experiments import save_experiment
from quant_lab.strategies.base import Strategy


class BuyHold(Strategy):
    name = "buy_hold"
    default_params = {}

    def signals(self, data: pl.DataFrame) -> pl.DataFrame:
        idx = pl.int_range(0, data.height)
        return data.with_columns(
            (idx == 1).alias("entries"),
            (idx == data.height - 2).alias("exits"),
        )


@pytest.fixture
def experiment_dir(bars, tmp_root):
    result = run_backtest(
        bars,
        BuyHold(),
        symbol="TESTUSDT",
        timeframe="1m",
        dataset_id="synthetic/test/bars/TESTUSDT/1m",
        execution=ExecutionConfig(allow_short=False, execution_delay_bars=1),
    )
    record = save_experiment(result, tmp_root / "experiments")
    return tmp_root / "experiments", record.experiment_id


def test_list_summaries(experiment_dir):
    experiments, exp_id = experiment_dir
    summaries = list_summaries(experiments)
    assert not summaries.is_empty()
    assert exp_id in summaries["experiment_id"].to_list()
    assert "sharpe" in summaries.columns


def test_load_bundle_is_read_only(experiment_dir):
    experiments, exp_id = experiment_dir
    bundle = load_bundle(experiments, exp_id)
    assert bundle.record.experiment_id == exp_id
    assert not bundle.equity.empty
    assert "equity" in bundle.equity.columns
    assert not bundle.trades.empty
    assert bundle.report_dir.name == exp_id


def test_tables(experiment_dir):
    experiments, exp_id = experiment_dir
    bundle = load_bundle(experiments, exp_id)
    metrics = metrics_table(bundle.record)
    assumptions = assumptions_table(bundle.record)
    assert "metric" in metrics.columns and "value" in metrics.columns
    assert "assumption" in assumptions.columns
    assert metrics.height > 5


def test_app_imports_without_running_ui():
    pytest.importorskip("streamlit")
    import quant_lab.dashboard.app as app

    assert hasattr(app, "render")
