"""Experiment tracking.

Every experiment is reproducible and traceable back to the exact configuration
and dataset that produced it. Results are stored as machine-readable JSON plus
Parquet (trades, equity) under ``experiments/results/<experiment_id>/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..backtest.engine import BacktestResult

EXPERIMENT_VERSION = 1


def git_commit(root: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root) if root else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def new_experiment_id(strategy: str, params: dict[str, Any], dataset_id: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    payload = json.dumps({"s": strategy, "p": params, "d": dataset_id}, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{stamp}-{strategy}-{digest}"


@dataclass(slots=True)
class ExperimentRecord:
    experiment_id: str
    created_at: str
    strategy: str
    params: dict[str, Any]
    dataset_id: str
    data_range: dict[str, str]
    execution: dict[str, Any]
    metrics: dict[str, Any]
    periods_per_year: float
    git_commit: str | None = None
    random_seed: int = 0
    symbol: str = ""
    timeframe: str = ""
    command: str | None = None
    version: int = EXPERIMENT_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentRecord:
        known = set(cls.__slots__)
        return cls(**{k: v for k, v in d.items() if k in known})


def save_experiment(
    result: BacktestResult,
    experiments_dir: Path,
    *,
    experiment_id: str | None = None,
    command: str | None = None,
    random_seed: int = 0,
    root: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> ExperimentRecord:
    exp_id = experiment_id or new_experiment_id(result.name, result.params, result.dataset_id)
    record = ExperimentRecord(
        experiment_id=exp_id,
        created_at=datetime.now(UTC).isoformat(),
        strategy=result.name,
        params=result.params,
        dataset_id=result.dataset_id,
        data_range={"start": result.start, "end": result.end},
        execution=result.assumptions,
        metrics=result.metrics,
        periods_per_year=result.periods_per_year,
        git_commit=git_commit(root),
        random_seed=random_seed,
        symbol=result.symbol,
        timeframe=result.timeframe,
        command=command,
        extra=extra or {},
    )
    out_dir = Path(experiments_dir) / "results" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "result.json", record.to_dict())
    _write_parquet(out_dir / "trades.parquet", result.trades)
    equity_df = pd.DataFrame(
        {
            "timestamp": result.equity.index,
            "equity": result.equity.to_numpy(),
            "drawdown": result.drawdown.reindex(result.equity.index).to_numpy(),
            "exposure": result.exposure.reindex(result.equity.index).to_numpy(),
        }
    )
    _write_parquet(out_dir / "equity.parquet", equity_df)
    return record


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        df = pd.DataFrame(columns=df.columns if df is not None else [])
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
    except Exception:  # noqa: BLE001 - fall back to CSV if parquet fails
        df.to_csv(tmp.with_suffix(".csv"), index=False)
        return
    os.replace(tmp, path)


def load_experiment(experiments_dir: Path, experiment_id: str) -> ExperimentRecord:
    path = Path(experiments_dir) / "results" / experiment_id / "result.json"
    with path.open("r", encoding="utf-8") as fh:
        return ExperimentRecord.from_dict(json.load(fh))


def list_experiments(experiments_dir: Path) -> list[ExperimentRecord]:
    root = Path(experiments_dir) / "results"
    if not root.exists():
        return []
    records: list[ExperimentRecord] = []
    for path in sorted(root.glob("*/result.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                records.append(ExperimentRecord.from_dict(json.load(fh)))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def load_result(experiments_dir: Path, experiment_id: str):
    """Reconstruct a lightweight BacktestResult from stored experiment files."""
    import polars as pl

    from ..backtest.engine import BacktestResult

    record = load_experiment(experiments_dir, experiment_id)
    base = Path(experiments_dir) / "results" / experiment_id
    equity_path = base / "equity.parquet"
    trades_path = base / "trades.parquet"

    equity = pd.Series(dtype=float)
    drawdown = pd.Series(dtype=float)
    exposure = pd.Series(dtype=float)
    if equity_path.exists():
        eq = pd.read_parquet(equity_path)
        eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True)
        eq = eq.set_index("timestamp")
        equity = eq["equity"].astype(float)
        drawdown = eq["drawdown"].astype(float) if "drawdown" in eq else equity / equity.cummax() - 1.0
        exposure = eq["exposure"].astype(float) if "exposure" in eq else pd.Series(0.0, index=equity.index)
    returns = equity.pct_change().fillna(0.0)
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()

    return BacktestResult(
        name=record.strategy,
        params=record.params,
        symbol=record.symbol,
        timeframe=record.timeframe,
        dataset_id=record.dataset_id,
        start=record.data_range.get("start", ""),
        end=record.data_range.get("end", ""),
        assumptions=record.execution,
        metrics=record.metrics,
        equity=equity,
        returns=returns,
        drawdown=drawdown,
        trades=trades,
        exposure=exposure,
        signal_frame=pl.DataFrame(),
        portfolio=None,
        periods_per_year=record.periods_per_year,
    )


def experiments_to_frame(experiments_dir: Path):
    import polars as pl

    records = [r.to_dict() for r in list_experiments(experiments_dir)]
    if not records:
        return pl.DataFrame()
    flat = []
    for r in records:
        row = {
            "experiment_id": r["experiment_id"],
            "created_at": r["created_at"],
            "strategy": r["strategy"],
            "symbol": r["symbol"],
            "timeframe": r["timeframe"],
            "dataset_id": r["dataset_id"],
            "params": json.dumps(r["params"], sort_keys=True),
        }
        for k, v in (r.get("metrics") or {}).items():
            row[k] = v
        flat.append(row)
    return pl.DataFrame(flat)
