#!/usr/bin/env python
"""EMA9/EMA20 cross study on the top-10 coins, 1h.

Computes ``cross_above`` (EMA9 crosses above EMA20) and ``cross_below`` on the
top-10 liquid crypto perps, evaluates them as directional signals for the
subsequent move (hit rate, oriented forward return, bootstrap CIs, design/test
split) and measures **how long each move lasted** (cross bar → opposite cross).

    python scripts/ema_cross_study.py
    python scripts/ema_cross_study.py --symbols BTCUSDT,ETHUSDT --start 2025-01-01
    python scripts/ema_cross_study.py --live --n 10

Writes ``reports/ema_cross_top10_1h.md`` plus an events parquet under
``experiments/reports/ema_cross/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pandas as pd
import polars as pl

from quant_lab.analytics import run_ema_cross_study
from quant_lab.config import get_config
from quant_lab.data.universe import fetch_top_symbols, load_universe

CACHED_UNIVERSE = "configs/divergence_universe.json"


def _resolve_universe(args) -> list[str]:
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.live:
        try:
            return fetch_top_symbols(
                args.market, args.n, exclude_stables=True, exclude_non_crypto=True
            )
        except Exception as exc:  # noqa: BLE001 - offline fallback
            print(f"live universe fetch failed ({type(exc).__name__}); using cache", flush=True)
    cached = Path(CACHED_UNIVERSE)
    if cached.exists():
        universe = load_universe(cached)
        symbols = universe.get("futures") or next(iter(universe.values()))
        return symbols[: args.n]
    raise SystemExit("no universe resolved: pass --symbols or --live")


def _summary(result) -> dict:
    out: dict = {}
    for f in dataclasses.fields(result):
        if f.name in {"events"}:
            continue
        value = getattr(result, f.name)
        if isinstance(value, pd.DataFrame) or isinstance(value, pl.DataFrame):
            out[f.name] = value.to_dict(orient="records") if isinstance(value, pd.DataFrame) else value.to_dicts()
        else:
            out[f.name] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default=None, help="Comma list; overrides the universe.")
    parser.add_argument("--n", type=int, default=10, help="Universe size when resolving automatically.")
    parser.add_argument("--live", action="store_true", help="Fetch the universe live from Binance (default: cached).")
    parser.add_argument("--market", default="futures", choices=["futures", "spot"])
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--source-timeframe", default="1m")
    parser.add_argument("--fast", type=int, default=9)
    parser.add_argument("--slow", type=int, default=20)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-10-01")
    parser.add_argument("--design-end", default="2026-01-01")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--artifacts", type=Path, default=None)
    args = parser.parse_args()

    cfg = get_config()
    symbols = _resolve_universe(args)
    if not symbols:
        raise SystemExit("universe is empty")

    out = args.out or (cfg.root / "reports" / "ema_cross_top10_1h.md")
    artifacts = args.artifacts or (cfg.paths.experiments / "reports" / "ema_cross")

    print(f"EMA{args.fast}/{args.slow} cross study · {args.market} · {args.timeframe} · "
          f"{args.start}..{args.end} (design/test at {args.design_end})", flush=True)
    print(f"universe ({len(symbols)}): " + ", ".join(symbols), flush=True)

    result = run_ema_cross_study(
        symbols=symbols,
        start=args.start,
        end=args.end,
        design_end=args.design_end,
        timeframe=args.timeframe,
        source_timeframe=args.source_timeframe,
        fast=args.fast,
        slow=args.slow,
        bootstrap=args.bootstrap,
        market=args.market,
        config=cfg,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.to_markdown(), encoding="utf-8")

    artifacts.mkdir(parents=True, exist_ok=True)
    if not result.events.empty:
        pl.from_pandas(result.events).write_parquet(artifacts / "events.parquet")
    (artifacts / "summary.json").write_text(
        json.dumps(_summary(result), indent=2, default=str) + "\n", encoding="utf-8"
    )

    print(f"\nbars={result.n_bars:,}  cross_above={result.n_up:,}  cross_below={result.n_down:,}")
    if not result.predictive.empty:
        top = result.predictive[result.predictive["horizon"] == result.primary_horizon]
        for r in top.itertuples(index=False):
            print(f"  {r.group:>4}: n={r.n:,}  hit={r.hit_rate * 100:5.1f}%  mean_y={r.mean_y * 100:+.3f}%")
    if not result.durations.empty:
        for r in result.durations.itertuples(index=False):
            print(f"  duration {r.group:>4}: median={r.median_bars:.0f}h  mean={r.mean_bars:.1f}h  "
                  f"run={r.mean_run_pct:+.2f}%  lose={100 - r.share_pos_run * 100:.0f}%")
    print(f"\nwrote {out}")
    print(f"wrote {artifacts}/events.parquet + summary.json")


if __name__ == "__main__":
    main()
