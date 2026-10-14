#!/usr/bin/env python
"""Resumable historical-data backfill for Binance bulk archives.

Designed for unreliable power/internet: every task is idempotent (the downloader
skips cached, checksum-verified archives) and progress is checkpointed to a state
file, so re-running the same command resumes where it stopped. Failures are
recorded and retried on the next run rather than aborting the whole job.

    # 2025 perp + spot 1m bars and funding for all crypto perps on disk
    python scripts/backfill_history.py --start 2025-01-01 --end 2026-01-01

    # just perp bars, one symbol, to test
    python scripts/backfill_history.py --start 2025-01-01 --end 2026-01-01 \
        --markets futures --datasets bars --symbols BTCUSDT

    # add the (slow) daily metrics/open-interest pass
    python scripts/backfill_history.py --start 2025-01-01 --end 2026-01-01 \
        --datasets open_interest

    # inspect progress without downloading
    python scripts/backfill_history.py --start 2025-01-01 --end 2026-01-01 --status

Run it repeatedly (cron @reboot + every few minutes, or systemd Restart=always):
a completed run exits immediately.

    # crontab -e
    @reboot          cd $HOME/quant && .venv/bin/python scripts/backfill_history.py --start 2025-01-01 --end 2026-01-01 >> /tmp/backfill.log 2>&1
    */10 * * * *     cd $HOME/quant && .venv/bin/python scripts/backfill_history.py --start 2025-01-01 --end 2026-01-01 >> /tmp/backfill.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from quant_lab.config import get_config
from quant_lab.data.build import build_dataset
from quant_lab.data.schemas import DatasetType
from quant_lab.data.universe import is_non_crypto

FUTURES_ONLY = {DatasetType.FUNDING, DatasetType.OPEN_INTEREST, DatasetType.METRICS}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _disk_symbols(root: Path, *, crypto_only: bool) -> list[str]:
    base = root / "binance" / "futures" / "bars"
    if not base.exists():
        return []
    syms = sorted(p.name for p in base.iterdir() if p.is_dir())
    return [s for s in syms if not (crypto_only and is_non_crypto(s))]


def _resolve_symbols(args, root: Path) -> list[str]:
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.universe_file:
        data = json.loads(Path(args.universe_file).read_text(encoding="utf-8"))
        uni = data.get("universe", data)
        syms: list[str] = []
        for value in uni.values():
            syms.extend(value)
        return sorted(set(syms))
    return _disk_symbols(root, crypto_only=not args.include_noncrypto)


def _tasks(markets, datasets, symbols):
    for market in markets:
        for dataset in datasets:
            if dataset in FUTURES_ONLY and market != "futures":
                continue
            for symbol in symbols:
                yield market, dataset, symbol


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _key(task, start: str, end: str) -> str:
    market, dataset, symbol = task
    return f"{market}/{dataset.value}/{symbol}|{start}|{end}"


def _print_status(state: dict) -> None:
    counts: dict[str, int] = {}
    for entry in state.values():
        counts[entry.get("status", "?")] = counts.get(entry.get("status", "?"), 0) + 1
    print("state summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(empty)")
    errors = [(k, v.get("error", "")) for k, v in state.items() if v.get("status") == "error"]
    for k, err in errors[-20:]:
        print(f"  ERROR {k}: {err[:140]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", required=True, help="e.g. 2025-01-01")
    parser.add_argument("--end", required=True, help="e.g. 2026-01-01")
    parser.add_argument("--markets", default="futures,spot")
    parser.add_argument(
        "--datasets", default="bars,funding",
        help="comma list of bars,funding,open_interest,metrics,mark_price_bars,index_price_bars,premium_index_bars",
    )
    parser.add_argument("--interval", default="1m", help="interval for bar-like datasets")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--universe-file", default=None)
    parser.add_argument("--include-noncrypto", action="store_true")
    parser.add_argument("--granularity", default=None, help="monthly | daily (default: auto)")
    parser.add_argument("--no-download", action="store_true", help="normalize cached archives only")
    parser.add_argument("--retries", type=int, default=3, help="attempts per task within a run")
    parser.add_argument("--retry-sleep", type=float, default=8.0, help="base backoff seconds")
    parser.add_argument("--pause-after-errors", type=int, default=5,
                        help="after N consecutive failures, sleep before continuing")
    parser.add_argument("--pause-seconds", type=float, default=120.0)
    parser.add_argument("--sleep", type=float, default=0.0, help="delay between tasks (s)")
    parser.add_argument("--max-tasks", type=int, default=0, help="stop after N tasks (0 = all)")
    parser.add_argument("--force", action="store_true", help="ignore state; re-run every task")
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--status", action="store_true", help="print progress and exit")
    args = parser.parse_args()

    cfg = get_config()
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    datasets = [DatasetType(d.strip()) for d in args.datasets.split(",") if d.strip()]
    symbols = _resolve_symbols(args, cfg.paths.data_normalized)
    if not symbols:
        raise SystemExit("no symbols resolved")

    state_path = args.state or (cfg.paths.data_catalogs / "backfill_state.json")
    state = _load_state(state_path)
    tasks = list(_tasks(markets, datasets, symbols))

    if args.status:
        print(f"{len(tasks)} tasks configured ({len(symbols)} symbols)")
        _print_status(state)
        return

    print(f"backfill {args.start}..{args.end} · {len(symbols)} symbols · "
          f"markets={markets} · datasets={[d.value for d in datasets]} · state={state_path}",
          flush=True)

    done = skipped = failed = missing = 0
    consecutive_errors = 0
    processed = 0
    for task in tasks:
        market, dataset, symbol = task
        key = _key(task, args.start, args.end)
        entry = state.get(key)
        if not args.force and entry and entry.get("status") in {"done", "missing"}:
            skipped += 1
            continue

        if args.max_tasks and processed >= args.max_tasks:
            print(f"reached --max-tasks ({args.max_tasks}); stopping", flush=True)
            break
        processed += 1

        interval = args.interval if dataset.is_bar_like else None
        last_error = ""
        ok = False
        for attempt in range(1, max(1, args.retries) + 1):
            try:
                result = build_dataset(
                    dataset=dataset,
                    symbol=symbol,
                    interval=interval,
                    start=args.start,
                    end=args.end,
                    market_type=market,
                    download=not args.no_download,
                    granularity=args.granularity,
                    config=cfg,
                )
                state[key] = {
                    "status": "done", "rows": result.row_count,
                    "partitions": len(result.partitions), "at": _now(),
                }
                done += 1
                print(f"  done  {market}/{dataset.value}/{symbol} rows={result.row_count}", flush=True)
                ok = True
                consecutive_errors = 0
                break
            except FileNotFoundError:
                state[key] = {"status": "missing", "at": _now()}
                missing += 1
                ok = True
                consecutive_errors = 0
                print(f"  miss  {market}/{dataset.value}/{symbol} (no archive)", flush=True)
                break
            except Exception as exc:  # noqa: BLE001 - resilience is the point
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max(1, args.retries):
                    time.sleep(args.retry_sleep * attempt)
        if not ok:
            state[key] = {"status": "error", "error": last_error, "at": _now()}
            failed += 1
            consecutive_errors += 1
            print(f"  FAIL  {market}/{dataset.value}/{symbol}: {last_error}", flush=True)
            if args.pause_after_errors and consecutive_errors >= args.pause_after_errors:
                print(f"  {consecutive_errors} consecutive failures — pausing "
                      f"{args.pause_seconds:.0f}s (likely a network cut)", flush=True)
                time.sleep(args.pause_seconds)
                consecutive_errors = 0

        _save_state(state_path, state)  # checkpoint after every task
        if args.sleep:
            time.sleep(args.sleep)

    _save_state(state_path, state)
    pending = sum(1 for t in tasks if _key(t, args.start, args.end) not in state)
    print(f"pass complete: {done} done, {missing} missing, {failed} failed, "
          f"{skipped} already-done, {pending} still pending", flush=True)
    if pending or failed:
        print("re-run the same command to resume.", flush=True)


if __name__ == "__main__":
    main()
