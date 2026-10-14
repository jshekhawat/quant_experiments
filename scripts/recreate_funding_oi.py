#!/usr/bin/env python
"""Recreate funding-rate and open-interest history for Binance perps.

Funding comes from the monthly ``fundingRate`` archives. Open interest derives
from the daily ``metrics`` archives — Binance publishes no monthly metrics, so
OI costs one download per symbol per day and is slow (~5 min/symbol for a
full year means hours across a large universe).

Datasets are processed outer-to-inner so the cheap funding pass completes first.

    # Top 100 liquid perps, funding then open interest, 2026 YTD
    python scripts/recreate_funding_oi.py --fetch-top 100 \
        --datasets funding,open_interest --start 2026-01-01 --end 2026-10-01

    # Funding only (fast)
    python scripts/recreate_funding_oi.py --fetch-top 100 --datasets funding

    # Reuse a cached universe / skip the network
    python scripts/recreate_funding_oi.py --universe-file configs/cvd_universe.json --no-download
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.config import get_config
from quant_lab.data.build import build_dataset
from quant_lab.data.schemas import DatasetType
from quant_lab.data.universe import fetch_universe, load_universe, save_universe


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fetch-top", type=int, default=None, metavar="N")
    parser.add_argument("--universe-file", type=Path, default=None)
    parser.add_argument("--universe-out", type=Path, default=None)
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols.")
    parser.add_argument("--datasets", default="funding,open_interest", help="funding,open_interest,metrics")
    parser.add_argument("--markets", default="futures", help="OI/funding only exist for futures.")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-10-01")
    parser.add_argument("--quote", default="USDT")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    cfg = get_config()
    markets = _split(args.markets)
    datasets = [DatasetType(name) for name in _split(args.datasets)]

    if args.symbols:
        universe = {market: [s.upper() for s in _split(args.symbols)] for market in markets}
    elif args.universe_file:
        cached = load_universe(args.universe_file)
        universe = {market: cached.get(market, []) for market in markets}
    else:
        top = args.fetch_top or 100
        universe = fetch_universe(top, markets=markets, quote=args.quote)
        out = args.universe_out or (cfg.root / "configs" / "cvd_universe.json")
        save_universe(out, universe, n=top, quote=args.quote)
        print(f"universe -> {out}")

    # Datasets outer so all funding (fast) lands before the OI dig begins.
    total = sum(len(universe.get(market, [])) for market in markets)
    for dataset in datasets:
        built = requested = partitions = rows = 0
        missing: list[str] = []
        for market in markets:
            for symbol in universe.get(market, []):
                requested += 1
                try:
                    result = build_dataset(
                        dataset=dataset,
                        symbol=symbol,
                        interval=None,  # non-bar datasets are stored under /na/
                        start=args.start,
                        end=args.end,
                        market_type=market,
                        download=not args.no_download,
                        config=cfg,
                    )
                except FileNotFoundError:
                    missing.append(f"{market}/{symbol}")
                except Exception as exc:  # keep going on network/parse problems
                    print(f"! {dataset.value} {market}/{symbol}: {exc}", flush=True)
                    missing.append(f"{market}/{symbol}")
                else:
                    built += 1
                    partitions += len(result.partitions)
                    rows += result.row_count
                print(
                    f"  [{dataset.value}] {requested}/{total} {market}/{symbol}",
                    flush=True,
                )
        print(
            f"{dataset.value}: {built}/{requested} built, {partitions} partitions, "
            f"{rows} rows, {len(missing)} missing"
        )
        if missing:
            preview = ", ".join(missing[:10])
            print(f"  missing: {preview}{' ...' if len(missing) > 10 else ''}")


if __name__ == "__main__":
    main()
