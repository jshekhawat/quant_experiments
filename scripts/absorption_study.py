#!/usr/bin/env python
"""Run the H1 price-absorption study (exploratory, 2026 in-sample).

Frozen methodology: ``reports/cvd_absorption_study.md``. Reads the perp 1m bars
already on disk — no downloads, no tick data.

    python scripts/absorption_study.py
    python scripts/absorption_study.py --timeframe 1h --primary-horizon 4h \
        --impact-window 720 --event-quantile 0.90 \
        --start 2026-01-01 --end 2026-10-01 --design-end 2026-05-01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_lab.analytics.absorption import run_absorption_study
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
    parser.add_argument("--adv-window", type=int, default=20)
    parser.add_argument("--impact-window", type=int, default=720)
    parser.add_argument("--event-quantile", type=float, default=0.90)
    parser.add_argument("--primary-horizon", default="4h")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--no-sweep", action="store_true", help="Skip the parameter robustness sweep.")
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

    print(f"absorption study: {len(symbols)} symbols, {args.timeframe}, {args.start}..{args.end}")
    result = run_absorption_study(
        symbols=symbols,
        start=args.start,
        end=args.end,
        design_end=args.design_end,
        timeframe=args.timeframe,
        source_timeframe=args.source_timeframe,
        adv_window=args.adv_window,
        impact_window=args.impact_window,
        event_quantile=args.event_quantile,
        primary_horizon=args.primary_horizon,
        bootstrap=args.bootstrap,
        sweep=not args.no_sweep,
        config=cfg,
    )
    markdown = result.to_markdown()
    out = args.out or (cfg.root / "reports" / "cvd_absorption_results.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")

    # Machine-readable outputs for follow-up analysis.
    artifacts = cfg.paths.experiments / "reports" / "absorption"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "summary.json").write_text(
        json.dumps(
            {
                "symbols": result.symbols,
                "timeframe": result.timeframe,
                "primary_horizon": result.primary_horizon,
                "impact_window": result.impact_window,
                "event_quantile": result.event_quantile,
                "design": result.design,
                "test": result.test,
                "test_ci": result.test_ci,
                "strong_flow_ic": result.strong_flow_ic,
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
        result.panel.write_parquet(artifacts / "panel.parquet", compression="zstd")

    print(markdown)
    print(f"wrote {out}")
    print(f"wrote {artifacts}/summary.json (+ by_symbol, leave_one_out, robustness, panel.parquet)")


if __name__ == "__main__":
    main()
