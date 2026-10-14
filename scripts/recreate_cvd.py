#!/usr/bin/env python
"""Recreate historical cumulative volume delta (CVD) across Binance.

Downloads + normalizes 1m klines for a symbol universe, then writes continuous
CVD partitions under ``data/derived/cvd/``. Because Binance klines include
``taker_buy_volume`` this is an exact per-bar reconstruction — no tick data.

    # Top 100 liquid symbols, spot + USD-M futures, 2026 YTD
    python scripts/recreate_cvd.py --fetch-top 100 \
        --markets futures,spot --start 2026-01-01 --end 2026-10-01

    # Reuse a cached universe file (no ticker call)
    python scripts/recreate_cvd.py --universe-file configs/cvd_universe.json

    # Explicit symbols
    python scripts/recreate_cvd.py --symbols BTCUSDT,ETHUSDT

    # Bars already downloaded/normalized: skip the network step
    python scripts/recreate_cvd.py --universe-file configs/cvd_universe.json --no-download
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.config import get_config
from quant_lab.data.cvd import recreate_cvd
from quant_lab.data.universe import fetch_universe, load_universe, save_universe


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fetch-top", type=int, default=None, metavar="N", help="Rank live 24h volume and take the top N.")
    parser.add_argument("--universe-file", type=Path, default=None, help="Cached universe JSON to read.")
    parser.add_argument("--universe-out", type=Path, default=None, help="Where to write a freshly fetched universe.")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols applied to every --market.")
    parser.add_argument("--markets", default="futures,spot", help="Comma-separated markets (futures,spot).")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-10-01")
    parser.add_argument("--quote", default="USDT")
    parser.add_argument("--no-download", action="store_true", help="Normalize existing raw archives only.")
    args = parser.parse_args()

    cfg = get_config()
    markets = _split(args.markets)

    if args.symbols:
        symbols = [s.upper() for s in _split(args.symbols)]
        universe = {market: list(symbols) for market in markets}
    elif args.universe_file:
        cached = load_universe(args.universe_file)
        universe = {market: cached.get(market, []) for market in markets}
    else:
        top = args.fetch_top or 100
        universe = fetch_universe(top, markets=markets, quote=args.quote)
        out = args.universe_out or (cfg.root / "configs" / "cvd_universe.json")
        save_universe(out, universe, n=top, quote=args.quote)
        print(f"universe -> {out}")

    total = sum(len(v) for v in universe.values())
    print(f"recreating CVD for {total} series ({args.interval}, {args.start}..{args.end})")

    result = recreate_cvd(
        symbols=universe,
        start=args.start,
        end=args.end,
        interval=args.interval,
        download=not args.no_download,
        config=cfg,
    )
    print(result.summary())
    if result.missing:
        preview = ", ".join(result.missing[:20])
        suffix = " ..." if len(result.missing) > 20 else ""
        print(f"missing ({len(result.missing)}): {preview}{suffix}")


if __name__ == "__main__":
    main()
