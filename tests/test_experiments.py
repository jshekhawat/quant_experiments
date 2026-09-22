"""Experiment serialization round-trip tests."""

from __future__ import annotations

import json

from quant_lab.backtest.engine import run_backtest
from quant_lab.config import ExecutionConfig
from quant_lab.reporting.experiments import (
    list_experiments,
    load_experiment,
    load_result,
    save_experiment,
)
from quant_lab.strategies.volatility_breakout import VolatilityBreakout


def test_experiment_round_trip(bars, tmp_root):
    result = run_backtest(
        bars,
        VolatilityBreakout(ema_span=20, atr_period=5, volume_window=10, volume_multiplier=1.0),
        symbol="TESTUSDT",
        timeframe="1m",
        dataset_id="synthetic/test/bars/TESTUSDT/1m",
        execution=ExecutionConfig(allow_short=False, execution_delay_bars=1),
    )
    record = save_experiment(result, tmp_root / "experiments", random_seed=123)
    assert record.random_seed == 123
    assert record.strategy == "volatility_breakout"

    loaded = load_experiment(tmp_root / "experiments", record.experiment_id)
    assert loaded.metrics["total_return"] == result.metrics["total_return"]
    assert loaded.params == result.params

    records = list_experiments(tmp_root / "experiments")
    assert len(records) == 1

    rebuilt = load_result(tmp_root / "experiments", record.experiment_id)
    assert rebuilt.name == result.name
    assert rebuilt.metrics["trade_count"] == result.metrics["trade_count"]
    assert not rebuilt.equity.empty


def test_result_json_is_machine_readable(bars, tmp_root):
    result = run_backtest(bars, VolatilityBreakout(ema_span=20), timeframe="1m")
    record = save_experiment(result, tmp_root / "experiments")
    path = tmp_root / "experiments" / "results" / record.experiment_id / "result.json"
    payload = json.loads(path.read_text())
    assert payload["strategy"] == "volatility_breakout"
    assert "metrics" in payload
    assert "execution" in payload
