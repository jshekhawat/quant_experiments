#!/usr/bin/env python
"""Run a single VectorBT backtest and persist the experiment.

Thin wrapper. Prefer ``quantlab backtest``.

    python scripts/run_backtest.py --strategy volatility_breakout \
        --symbol BTCUSDT --timeframe 1m --start 2024-01-01 --end 2024-04-01
"""

from __future__ import annotations

import argparse

from quant_lab.backtest.engine import run_backtest
from quant_lab.config import get_config
from quant_lab.data.schemas import DatasetType
from quant_lab.data.storage import DatasetKey, read_dataset
from quant_lab.reporting import format_summary, save_experiment, save_report
from quant_lab.strategies import build_strategy


def _parse_params(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        key, value = item.split("=", 1)
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value.lower() == "true" if value.lower() in {"true", "false"} else value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--market", default="futures")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--set", action="append", default=[], help="key=value")
    args = parser.parse_args()

    cfg = get_config()
    key = DatasetKey("binance", args.market, DatasetType.BARS, args.symbol, args.timeframe)
    df = read_dataset(cfg.paths.data_normalized, key, start=args.start, end=args.end)
    if df.height == 0:
        raise SystemExit(f"no data for {key.dataset_id}; run scripts/build_dataset.py first")

    strategy = build_strategy(args.strategy, **_parse_params(args.set))
    result = run_backtest(
        df, strategy, symbol=args.symbol, timeframe=args.timeframe,
        dataset_id=key.dataset_id, start=args.start, end=args.end,
    )
    record = save_experiment(result, cfg.paths.experiments, root=cfg.root, command="scripts/run_backtest.py")
    save_report(result, cfg.paths.experiments / "reports" / record.experiment_id, experiment_id=record.experiment_id)
    print(format_summary(result, record.experiment_id))


if __name__ == "__main__":
    main()
