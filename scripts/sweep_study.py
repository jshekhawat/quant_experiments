#!/usr/bin/env python
"""Run the H2 failed-liquidity-sweep study (exploratory, 2026 in-sample).

Frozen methodology: ``reports/cvd_failed_sweeps_study.md``. Reads perp 1m bars
already on disk — no downloads, no tick data.

    python scripts/sweep_study.py
    python scripts/sweep_study.py --timeframe 1h --lookback 48 --cooldown 24 \
        --start 2026-01-01 --end 2026-10-01 --design-end 2026-05-01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from quant_lab.analytics.sweeps import run_sweep_study
from quant_lab.config import get_config
from quant_lab.data.universe import load_universe


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--universe-file", type=Path, default=None)
    parser.add_argument("--symbols", default=None, help="Comma-separated override.")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-10-01")
    parser.add_argument("--design-end", default="2026-05-01")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--source-timeframe", default="1m")
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--cooldown", type=int, default=24)
    parser.add_argument("--flow-window", type=int, default=1)
    parser.add_argument("--primary-horizon", default="4h")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = get_config()
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        universe_file = args.universe_file or (cfg.root / "configs" / "divergence_universe.json")
        universe = load_universe(universe_file)
        symbols = universe.get("futures") or universe.get("spot") or []
    if not symbols:
        raise SystemExit("no symbols resolved")

    print(f"sweep study: {len(symbols)} symbols, {args.timeframe}, {args.start}..{args.end}")
    result = run_sweep_study(
        symbols=symbols,
        start=args.start,
        end=args.end,
        design_end=args.design_end,
        timeframe=args.timeframe,
        source_timeframe=args.source_timeframe,
        lookback=args.lookback,
        cooldown=args.cooldown,
        flow_window=args.flow_window,
        primary_horizon=args.primary_horizon,
        bootstrap=args.bootstrap,
        sweep=not args.no_sweep,
        config=cfg,
    )
    markdown = result.to_markdown()
    out = args.out or (cfg.root / "reports" / "cvd_failed_sweeps_results.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")

    artifacts = cfg.paths.experiments / "reports" / "sweeps"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "summary.json").write_text(
        json.dumps(
            {
                "symbols": result.symbols,
                "timeframe": result.timeframe,
                "primary_horizon": result.primary_horizon,
                "lookback": result.lookback,
                "cooldown": result.cooldown,
                "n_events": result.n_events,
                "design": result.design,
                "test": result.test,
                "test_ci": result.test_ci,
                "matched": result.matched,
                "regression": result.regression,
                "horizon_ic": result.horizon_ic,
                "ex_megacap_ic": result.ex_megacap_ic,
                "criteria": result.criteria,
                "verdict": result.verdict,
                "notes": result.notes,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    result.by_symbol.to_json(artifacts / "by_symbol.json", orient="records", indent=2)
    result.loo.to_json(artifacts / "leave_one_out.json", orient="records", indent=2)
    if not result.robustness.empty:
        result.robustness.to_json(artifacts / "robustness.json", orient="records", indent=2)
    if not result.panel.is_empty():
        events = result.panel.filter(pl.col("sweep"))
        events.write_parquet(artifacts / "events.parquet", compression="zstd")

    print(markdown)
    print(f"wrote {out}")
    print(f"wrote {artifacts}/summary.json (+ by_symbol, leave_one_out, robustness, events.parquet)")


if __name__ == "__main__":
    main()
