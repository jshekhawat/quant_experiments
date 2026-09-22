#!/usr/bin/env python
"""Download Binance bulk historical archives.

Thin wrapper around the Binance adapter. Prefer ``quantlab data download``.

    python scripts/download_binance.py --symbol BTCUSDT --interval 1m \
        --start 2024-01-01 --end 2024-03-01
"""

from __future__ import annotations

import argparse

from quant_lab.config import get_config
from quant_lab.data.build import make_source
from quant_lab.data.schemas import DatasetType


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--market", default="futures")
    parser.add_argument("--dataset", default="bars")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--granularity", default=None)
    args = parser.parse_args()

    cfg = get_config()
    source = make_source(cfg, args.market)
    result = source.download(
        market_type=args.market,
        symbol=args.symbol,
        dataset=DatasetType(args.dataset),
        interval=args.interval,
        start=args.start,
        end=args.end,
        granularity=args.granularity,
    )
    print(result.summary())


if __name__ == "__main__":
    main()
