"""Read-only data access helpers for the optional dashboard.

These functions are deliberately free of any Streamlit import so they can be
unit-tested and reused. The dashboard never writes data and never talks to an
exchange.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl

from ..backtest.engine import BacktestResult
from ..config import Config
from ..data.catalog import Catalog, catalog_path
from ..reporting.experiments import (
    ExperimentRecord,
    experiments_to_frame,
    load_experiment,
    load_result,
)


@dataclass(slots=True)
class ExperimentBundle:
    record: ExperimentRecord
    result: BacktestResult
    equity: pd.DataFrame
    trades: pd.DataFrame
    report_dir: Path
    tearsheet: Path | None


def list_summaries(experiments_dir: Path) -> pl.DataFrame:
    """One row per saved experiment with flattened metrics."""
    return experiments_to_frame(Path(experiments_dir))


def metrics_table(record: ExperimentRecord) -> pl.DataFrame:
    return pl.DataFrame(
        [{"metric": k, "value": v} for k, v in (record.metrics or {}).items()]
    )


def assumptions_table(record: ExperimentRecord) -> pl.DataFrame:
    return pl.DataFrame(
        [{"assumption": k, "value": v} for k, v in (record.execution or {}).items()]
    )


def load_bundle(experiments_dir: Path, experiment_id: str) -> ExperimentBundle:
    """Load everything needed to render one experiment (read-only)."""
    experiments_dir = Path(experiments_dir)
    record = load_experiment(experiments_dir, experiment_id)
    result = load_result(experiments_dir, experiment_id)
    base = experiments_dir / "results" / experiment_id

    equity = pd.DataFrame()
    equity_path = base / "equity.parquet"
    if equity_path.exists():
        equity = pd.read_parquet(equity_path)
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)

    trades = pd.DataFrame()
    trades_path = base / "trades.parquet"
    if trades_path.exists():
        trades = pd.read_parquet(trades_path)

    report_dir = experiments_dir / "reports" / experiment_id
    tearsheet = report_dir / "tearsheet.html"
    return ExperimentBundle(
        record=record,
        result=result,
        equity=equity,
        trades=trades,
        report_dir=report_dir,
        tearsheet=tearsheet if tearsheet.exists() else None,
    )


def catalog_table(config: Config) -> pl.DataFrame:
    catalog = Catalog.load(catalog_path(config.paths.data_catalogs))
    return catalog.to_polars()
