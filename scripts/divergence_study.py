#!/usr/bin/env python
"""Run the spot–perp CVD divergence study (exploratory, 2026 in-sample).

Frozen methodology: ``reports/spot_cvd_divergence_study.md``. This downloads
nothing — it reads the normalized bars already on disk for the top-10 crypto
cross-market universe and writes a results table.

    python scripts/divergence_study.py
    python scripts/divergence_study.py --timeframe 1h --primary-horizon 4h \
        --start 2026-01-01 --end 2026-10-01 --design-end 2026-05-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.analytics.divergence import run_divergence_study
from quant_lab.config import get_config
from quant_lab.data.universe import load_universe


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--universe-file", type=Path, default=None)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-10-01")
    parser.add_argument("--design-end", default="2026-05-01")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--source-timeframe", default="1m")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--norm-window", type=int, default=720)
    parser.add_argument("--primary-horizon", default="4h")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = get_config()
    universe_file = args.universe_file or (cfg.root / "configs" / "divergence_universe.json")
    universe = load_universe(universe_file)
    # Both markets carry the same list; use futures as the price/flow series set.
    symbols = universe.get("futures") or universe.get("spot") or []
    if not symbols:
        raise SystemExit(f"no symbols in {universe_file}")

    print(f"divergence study: {len(symbols)} symbols, {args.timeframe}, {args.start}..{args.end}")
    result = run_divergence_study(
        symbols=symbols,
        start=args.start,
        end=args.end,
        design_end=args.design_end,
        timeframe=args.timeframe,
        source_timeframe=args.source_timeframe,
        window=args.window,
        norm_window=args.norm_window,
        primary_horizon=args.primary_horizon,
        bootstrap=args.bootstrap,
        config=cfg,
    )
    markdown = result.to_markdown()
    out = args.out or (cfg.root / "reports" / "spot_cvd_divergence_results.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
