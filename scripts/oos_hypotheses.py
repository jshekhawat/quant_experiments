#!/usr/bin/env python
"""Out-of-sample re-run of the four order-flow hypotheses (design 2025, test 2026).

Runs the frozen H-series methods with a genuine temporal split instead of the
single-regime 2026 sample: **design = 2025-01-01 → 2026-01-01, test = 2026**.
Each study is checkpointed (its own markdown + JSON), so a power/internet cut or
kill only loses the study in progress; re-running skips completed studies.

    python scripts/oos_hypotheses.py                 # all four
    python scripts/oos_hypotheses.py --no-sweep      # skip parameter sweeps (faster)
    python scripts/oos_hypotheses.py --symbols BTCUSDT,ETHUSDT --force
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pandas as pd
import polars as pl

from quant_lab.analytics import (
    run_absorption_study,
    run_breakout_study,
    run_divergence_study,
    run_sweep_study,
)
from quant_lab.config import get_config
from quant_lab.data.universe import is_non_crypto

ORDER = ("divergence", "absorption", "sweeps", "breakouts")


def _symbols_with(root: Path, market: str) -> list[str]:
    base = root / "binance" / market / "bars"
    if not base.exists():
        return []
    out = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and not is_non_crypto(p.name) and list(p.rglob("2025/*.parquet")):
            out.append(p.name)
    return out


def _summary(result) -> dict:
    out: dict = {}
    for f in dataclasses.fields(result):
        value = getattr(result, f.name)
        if isinstance(value, (pd.DataFrame, pl.DataFrame)):
            continue
        out["n_symbols" if f.name == "symbols" else f.name] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-10-01")
    parser.add_argument("--design-end", default="2026-01-01")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--symbols", default=None, help="Override all universes (comma list).")
    parser.add_argument("--force", action="store_true", help="Re-run studies even if outputs exist.")
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = get_config()
    root = cfg.paths.data_normalized
    if args.symbols:
        perp = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        cross = list(perp)
    else:
        perp = _symbols_with(root, "futures")
        cross = sorted(set(perp) & set(_symbols_with(root, "spot")))
    if not perp:
        raise SystemExit("no out-of-sample symbols resolved")

    artifacts = args.artifacts or (cfg.paths.experiments / "reports" / "oos")
    artifacts.mkdir(parents=True, exist_ok=True)
    out = args.out or (cfg.root / "reports" / "oos_hypotheses_results.md")

    print(f"OOS design {args.start}..{args.design_end} / test {args.design_end}..{args.end}")
    print(f"universe: {len(perp)} perps, {len(cross)} cross-market; sweep={not args.no_sweep}", flush=True)

    runners = {
        "divergence": lambda: run_divergence_study(
            symbols=cross, start=args.start, end=args.end, design_end=args.design_end,
            timeframe=args.timeframe, bootstrap=args.bootstrap, config=cfg,
        ),
        "absorption": lambda: run_absorption_study(
            symbols=perp, start=args.start, end=args.end, design_end=args.design_end,
            timeframe=args.timeframe, bootstrap=args.bootstrap, sweep=not args.no_sweep, config=cfg,
        ),
        "sweeps": lambda: run_sweep_study(
            symbols=perp, start=args.start, end=args.end, design_end=args.design_end,
            timeframe=args.timeframe, bootstrap=args.bootstrap, sweep=not args.no_sweep, config=cfg,
        ),
        "breakouts": lambda: run_breakout_study(
            symbols=perp, start=args.start, end=args.end, design_end=args.design_end,
            timeframe=args.timeframe, bootstrap=args.bootstrap, sweep=not args.no_sweep, config=cfg,
        ),
    }

    for name in ORDER:
        md_path = artifacts / f"{name}.md"
        js_path = artifacts / f"{name}.json"
        if md_path.exists() and js_path.exists() and not args.force:
            print(f"skip {name} (already done)", flush=True)
            continue
        print(f"running {name} ...", flush=True)
        try:
            result = runners[name]()
        except Exception as exc:  # noqa: BLE001 - resilience
            print(f"  FAILED {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        md_path.write_text(result.to_markdown(), encoding="utf-8")
        js_path.write_text(
            json.dumps(_summary(result), indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"  done {name}: verdict={result.verdict} "
              f"test_ic={result.test.get('ic', float('nan')):.4f}", flush=True)

    # Combined report from whatever per-study markdowns exist.
    parts = [
        "# Out-of-Sample Order-Flow Hypotheses — design 2025 / test 2026",
        "",
        f"- design: {args.start} → {args.design_end} · test: {args.design_end} → {args.end}",
        f"- universe: {len(perp)} crypto perps ({len(cross)} cross-market for divergence)",
        "- These are the frozen H-series methods re-run on a genuine temporal split; "
        "verdicts are provisional (2025 vs 2026 is still only two regimes).",
        "",
    ]
    for name in ORDER:
        md_path = artifacts / f"{name}.md"
        if md_path.exists():
            parts.append(md_path.read_text(encoding="utf-8"))
            parts.append("\n---\n")
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
