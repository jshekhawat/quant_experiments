#!/usr/bin/env python
"""Post-signal excursion report for a breakout strategy.

For each signal, measures time to 50% and 90% of the maximum favourable
excursion, and time to revert (close basis) to the signal price.

    python scripts/analyze_breakouts.py --symbol BTCUSDT \
        --start 2026-07-01 --end 2026-09-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quant_lab.analytics import analyze_excursions
from quant_lab.config import get_config
from quant_lab.data.schemas import DatasetType
from quant_lab.data.storage import DatasetKey, read_dataset
from quant_lab.strategies import build_strategy
from quant_lab.timeframes import timeframe_seconds


def _parse_params(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        key, value = item.split("=", 1)
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="bollinger_breakout")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--market", default="futures")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--mode", default="pct_excursion")
    parser.add_argument("--targets", default="0.5,0.9")
    parser.add_argument("--reversion-only", action="store_true")
    parser.add_argument("--horizon-days", type=float, default=7.0)
    parser.add_argument("--reversion-basis", default="close")
    parser.add_argument("--side", default="long")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--out", default="experiments/reports/breakout_excursions")
    args = parser.parse_args()

    cfg = get_config()
    key = DatasetKey("binance", args.market, DatasetType.BARS, args.symbol, args.timeframe)
    df = read_dataset(cfg.paths.data_normalized, key, start=args.start, end=args.end)
    if df.height == 0:
        raise SystemExit(f"no data for {key.dataset_id}")

    strategy = build_strategy(args.strategy, **_parse_params(args.set))
    signals = strategy.prepare(df)
    targets = () if args.reversion_only else tuple(float(t) for t in args.targets.split(",") if t.strip())
    horizon = int(args.horizon_days * 86400 / timeframe_seconds(args.timeframe)) if args.horizon_days > 0 else None

    result = analyze_excursions(
        signals,
        signal_col="entries" if args.side == "long" else "short_entries",
        mode=args.mode,
        targets=targets,
        horizon_bars=horizon,
        side=args.side,
        reversion_basis=args.reversion_basis,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not result.events.is_empty():
        result.events.write_parquet(out / "events.parquet")
        result.events.write_csv(out / "events.csv")
    if not result.summary.is_empty():
        result.summary.write_parquet(out / "summary.parquet")
    if not result.reversion_cdf.is_empty():
        result.reversion_cdf.write_parquet(out / "reversion_cdf.parquet")
    if not result.excursion_by_bucket.is_empty():
        result.excursion_by_bucket.write_parquet(out / "excursion_by_bucket.parquet")
    (out / "summary.md").write_text(result.to_markdown(), encoding="utf-8")

    print(result.format())
    if not result.excursion_by_bucket.is_empty():
        print("\nExcursion before reversion, by reversion-speed bucket:")
        print(result.excursion_by_bucket.to_pandas().to_string(index=False))

    if not result.events.is_empty():
        hours = 3600.0
        fig, ax = plt.subplots(figsize=(10, 5))
        for col, label in [
            ("time_to_50pct_seconds", "50% of max excursion"),
            ("time_to_90pct_seconds", "90% of max excursion"),
            ("time_to_reversion_seconds", "reversion to signal (close)"),
        ]:
            if col in result.events.columns:
                values = result.events[col].drop_nulls().to_numpy() / hours
                if len(values):
                    ax.hist(values, bins=60, alpha=0.5, label=label)
        ax.set_xlabel("hours after signal")
        ax.set_ylabel("count")
        ax.set_title(f"{args.strategy} {args.symbol} {args.timeframe} — time after breakout")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "time_histogram.png", dpi=110)
        plt.close(fig)
        print(f"wrote {out}/summary.md and {out}/time_histogram.png")


if __name__ == "__main__":
    main()
