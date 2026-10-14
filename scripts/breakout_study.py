#!/usr/bin/env python
"""Run the H3 flow-confirmed breakout continuation study (exploratory, 2026).

Frozen methodology: ``reports/cvd_breakout_flow_study.md``. Reads perp 1m bars on
disk — no downloads, no tick data. By default uses **every available crypto perp**
(the widest no-cost universe); pass ``--include-noncrypto`` for all downloaded
symbols, or ``--symbols`` to override.

    python scripts/breakout_study.py
    python scripts/breakout_study.py --include-noncrypto
    python scripts/breakout_study.py --symbols BTCUSDT,ETHUSDT --no-sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from quant_lab.analytics.breakouts import run_breakout_study
from quant_lab.config import get_config
from quant_lab.data.universe import is_non_crypto


def _disk_symbols(root: Path, *, crypto_only: bool) -> list[str]:
    base = root / "binance" / "futures" / "bars"
    if not base.exists():
        return []
    syms = sorted(p.name for p in base.iterdir() if p.is_dir())
    return [s for s in syms if not (crypto_only and is_non_crypto(s))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated override.")
    parser.add_argument("--include-noncrypto", action="store_true")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-10-01")
    parser.add_argument("--design-end", default="2026-05-01")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--source-timeframe", default="1m")
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--cooldown", type=int, default=24)
    parser.add_argument("--flow-window", type=int, default=4)
    parser.add_argument("--primary-horizon", default="4h")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = get_config()
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _disk_symbols(cfg.paths.data_normalized, crypto_only=not args.include_noncrypto)
    if not symbols:
        raise SystemExit("no symbols resolved")

    print(f"breakout study: {len(symbols)} symbols, {args.timeframe}, {args.start}..{args.end}")
    result = run_breakout_study(
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
    suffix = "" if not args.include_noncrypto else "_all"
    out = args.out or (cfg.root / "reports" / f"cvd_breakout_flow_results{suffix}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")

    artifacts = cfg.paths.experiments / "reports" / f"breakouts{suffix}"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "summary.json").write_text(
        json.dumps(
            {
                "symbols": result.symbols,
                "timeframe": result.timeframe,
                "primary_horizon": result.primary_horizon,
                "lookback": result.lookback,
                "flow_window": result.flow_window,
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
    result.quintiles.to_json(artifacts / "quintiles.json", orient="records", indent=2)
    if not result.robustness.empty:
        result.robustness.to_json(artifacts / "robustness.json", orient="records", indent=2)
    if not result.panel.is_empty():
        result.panel.filter(pl.col("breakout")).write_parquet(
            artifacts / "events.parquet", compression="zstd"
        )

    print(markdown)
    print(f"wrote {out}")
    print(f"wrote {artifacts}/summary.json (+ by_symbol, leave_one_out, robustness, events.parquet)")


if __name__ == "__main__":
    main()
