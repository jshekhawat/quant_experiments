#!/usr/bin/env python
"""Download and normalize a dataset to canonical Parquet.

Thin wrapper around ``quant_lab.data.build.build_dataset``.
Prefer ``quantlab data build``.

    python scripts/build_dataset.py --symbol BTCUSDT --interval 1m \
        --start 2024-01-01 --end 2024-03-01
"""

from __future__ import annotations

import argparse

from quant_lab.data.build import build_dataset
from quant_lab.data.schemas import DatasetType


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--market", default="futures")
    parser.add_argument("--dataset", default="bars")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    result = build_dataset(
        dataset=DatasetType(args.dataset),
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
        market_type=args.market,
        download=not args.no_download,
    )
    print(result.summary())


if __name__ == "__main__":
    main()
