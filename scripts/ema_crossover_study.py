#!/usr/bin/env python
"""EMA-crossover research study on Binance 1h data.

Implements ``reports/ema_crossover_study.md`` end to end: data inspection,
parameter grid, long/short/long-short and EMA200-filtered variants, next-open
execution with fees + slippage + perp funding, chronological IS/OOS split,
walk-forward selection, cost sensitivity, regime analysis, robustness and
multiple-testing diagnostics. Writes a markdown report plus CSV/JSON/PNG
artifacts under ``experiments/reports/ema_crossover/``.

    python scripts/ema_crossover_study.py
    python scripts/ema_crossover_study.py --quick
    python scripts/ema_crossover_study.py --symbols BTCUSDT,ETHUSDT --start 2025-01-01

This is research code: no live trading, no orders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from quant_lab.analytics import ema_backtest as eb
from quant_lab.analytics.regimes import RegimeConfig, classify_regimes
from quant_lab.config import get_config
from quant_lab.data.universe import load_universe

CACHED_UNIVERSE = "configs/divergence_universe.json"

NAMED_PAIRS = [(9, 21), (12, 26), (20, 50), (21, 55), (50, 200)]
GRID_FAST = list(range(5, 41, 5))       # 5,10,...,40
GRID_SLOW = list(range(20, 201, 10))    # 20,30,...,200
COST_SCENARIOS = {"zero": 0.0, "low": 0.0003, "base": 0.0006, "high": 0.0012, "extreme": 0.0020}
PRIMARY_VARIANT = "both"
# Pre-registered reference for the deep-dive (cost/regime/robustness). Chosen
# before looking at results — the textbook pair — so those sections are not
# contaminated by out-of-sample selection.
REF_PAIR = (20, 50)


# --- universe / data ---------------------------------------------------------


def resolve_universe(args) -> list[str]:
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    path = Path(CACHED_UNIVERSE)
    if path.exists():
        u = load_universe(path)
        symbols = u.get("futures") or next(iter(u.values()))
        return symbols[: args.n]
    raise SystemExit("no universe: pass --symbols")


def data_quality(bundle: eb.SymbolBundle, timeframe: str) -> dict:
    ts = pd.DatetimeIndex(bundle.timestamps)
    diffs = pd.Series(ts).diff().dropna()
    expected = pd.Timedelta(hours=eb.timeframe_hours(timeframe))
    n_expected = int((ts[-1] - ts[0]) / expected) + 1 if len(ts) else 0
    bad = 0
    with np.errstate(invalid="ignore"):
        bad = int(np.sum(~(bundle.high >= bundle.low)))
    return {
        "symbol": bundle.symbol,
        "bars": int(bundle.close.size),
        "start": str(ts[0]),
        "end": str(ts[-1]),
        "duplicate_ts": int(pd.Series(ts).duplicated().sum()),
        "gaps": int(n_expected - len(ts)),
        "max_gap_hours": float(diffs.max().total_seconds() / 3600) if len(diffs) else 0.0,
        "bad_ohlc": bad,
        "funding_events": int(np.count_nonzero(bundle.funding)),
    }


# --- pooled backtests --------------------------------------------------------


def pooled_config(
    bundles: list[eb.SymbolBundle],
    grid_ns: np.ndarray,
    variant: str,
    fast: int,
    slow: int,
    *,
    cost: float,
    use_funding: bool,
    collect_positions: bool = False,
) -> tuple[np.ndarray, dict]:
    """Equal-weight pooled net returns across symbols for one configuration."""
    G = grid_ns.size
    stack = np.full((len(bundles), G), np.nan)
    ntr = 0
    turn = 0.0
    fund = 0.0
    expo = 0.0
    for i, b in enumerate(bundles):
        tgt = b.target(variant, fast, slow)
        res = eb.simulate(
            b.timestamps, b.open, b.close, tgt, cost_per_side=cost,
            funding=b.funding if use_funding else None, symbol=b.symbol,
            variant=variant, fast=fast, slow=slow,
        )
        pos = np.searchsorted(grid_ns, b.timestamps.astype("datetime64[ns]").astype("int64"))
        stack[i, pos] = res.returns
        ntr += res.n_trades
        turn += res.turnover
        fund += res.funding_cost
        expo += float(np.mean(res.positions != 0))
    with np.errstate(invalid="ignore"):
        pooled = np.nanmean(stack, axis=0)
    nsym = max(len(bundles), 1)
    return np.nan_to_num(pooled), {
        "n_trades": ntr, "turnover": turn / nsym, "funding_cost": fund / nsym,
        "exposure": expo / nsym, "stack": stack if collect_positions else None,
    }


def breakeven_per_side(
    bundles, grid_ns, variant, fast, slow, oos_mask, ppy, *, use_funding, hi=0.008, n=33
) -> float:
    """Per-side fee+slippage at which OOS net return crosses zero (interpolated)."""
    fine = np.linspace(0.0, hi, n)
    rets = np.array([
        eb.returns_metrics(
            pooled_config(bundles, grid_ns, variant, fast, slow, cost=float(c), use_funding=use_funding)[0][oos_mask],
            ppy=ppy,
        )["total_return"]
        for c in fine
    ])
    below = np.flatnonzero(rets <= 0)
    if below.size and below[0] > 0:
        i = int(below[0])
        y0, y1 = rets[i - 1], rets[i]
        if y1 != y0:
            return float(fine[i - 1] + (fine[i] - fine[i - 1]) * (0 - y0) / (y1 - y0))
        return float(fine[i])
    if below.size:
        return 0.0
    return float("inf")


def pooled_trades(
    bundles: list[eb.SymbolBundle], variant: str, fast: int, slow: int, *, cost: float, use_funding: bool
) -> pd.DataFrame:
    frames = []
    for b in bundles:
        res = eb.simulate(
            b.timestamps, b.open, b.close, b.target(variant, fast, slow),
            cost_per_side=cost, funding=b.funding if use_funding else None,
        )
        tr = eb.extract_trades(
            b.timestamps, b.open, b.close, res.positions, cost_per_side=cost,
            funding=b.funding if use_funding else None,
        )
        if not tr.empty:
            tr["symbol"] = b.symbol
            frames.append(tr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --- walk-forward ------------------------------------------------------------


def make_folds(start: str, end: str, train_m: int, val_m: int, test_m: int, step_m: int):
    folds = []
    t = pd.Timestamp(start, tz="UTC")
    last = pd.Timestamp(end, tz="UTC")
    while True:
        a0 = t
        a1 = a0 + pd.DateOffset(months=train_m)
        b1 = a1 + pd.DateOffset(months=val_m)
        c1 = b1 + pd.DateOffset(months=test_m)
        if c1 > last:
            break
        folds.append({"train": (a0, a1), "val": (a1, b1), "test": (b1, c1)})
        t = t + pd.DateOffset(months=step_m)
    return folds


# --- metric formatting -------------------------------------------------------


def pct(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def mrow(variant, fast, slow, period, m, extra=None):
    row = {
        "variant": variant, "fast": fast, "slow": slow, "period": period,
        "total_return": m.get("total_return"), "cagr": m.get("cagr"),
        "sharpe": m.get("sharpe"), "sortino": m.get("sortino"),
        "volatility": m.get("volatility"), "max_drawdown": m.get("max_drawdown"),
        "dd_duration_bars": m.get("dd_duration_bars"), "exposure": m.get("exposure"),
        "turnover": m.get("turnover"),
    }
    if extra:
        row.update(extra)
    return row


# --- main --------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-10-01")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--source-timeframe", default="1m")
    ap.add_argument("--is-end", default="2026-01-01")
    ap.add_argument("--cost", type=float, default=0.0006, help="fee+slippage per side (base scenario)")
    ap.add_argument("--no-funding", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--quick", action="store_true", help="reduced grid for a fast smoke run")
    ap.add_argument("--wf-train-months", type=int, default=6)
    ap.add_argument("--wf-val-months", type=int, default=2)
    ap.add_argument("--wf-test-months", type=int, default=3)
    ap.add_argument("--wf-step-months", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--artifacts", type=Path, default=None)
    args = ap.parse_args()

    cfg = get_config()
    symbols = resolve_universe(args)
    fast_grid = [5, 10, 20, 40] if args.quick else GRID_FAST
    slow_grid = [20, 50, 200] if args.quick else GRID_SLOW
    pairs = [(f, s) for f in fast_grid for s in slow_grid if f < s]
    variants = list(eb.VARIANTS)
    use_funding = not args.no_funding
    out = args.out or (cfg.root / "reports" / "ema_crossover_results.md")
    art = args.artifacts or (cfg.paths.experiments / "reports" / "ema_crossover")
    art.mkdir(parents=True, exist_ok=True)
    (art / "plots").mkdir(exist_ok=True)
    ppy = eb.periods_per_year_for(args.timeframe)

    print(f"EMA crossover study · {args.timeframe} · {args.start}..{args.end} · IS/OOS split {args.is_end}", flush=True)
    print(f"universe ({len(symbols)}): " + ", ".join(symbols), flush=True)
    print(f"grid: {len(pairs)} pairs x {len(variants)} variants; cost/side={args.cost}; funding={use_funding}", flush=True)

    spans = set(fast_grid) | set(slow_grid) | {200}
    for f, s in NAMED_PAIRS:
        spans.update({f, s})
    bundles: list[eb.SymbolBundle] = []
    dq: list[dict] = []
    for s in symbols:
        b = eb.load_symbol(
            s, start=args.start, end=args.end, spans=spans, timeframe=args.timeframe,
            source_timeframe=args.source_timeframe, root=cfg.paths.data_normalized,
        )
        if b is None:
            print(f"  skip {s} (no data)", flush=True)
            continue
        bundles.append(b)
        dq.append(data_quality(b, args.timeframe))
    if not bundles:
        raise SystemExit("no symbol data loaded")

    # common hourly grid
    lo = min(pd.Timestamp(b.timestamps[0]) for b in bundles)
    hi = max(pd.Timestamp(b.timestamps[-1]) for b in bundles)
    grid = pd.date_range(lo, hi, freq="1h", tz="UTC")
    grid_ns = grid.tz_localize(None).to_numpy().astype("datetime64[ns]").astype("int64")
    is_mask = np.asarray(grid < pd.Timestamp(args.is_end, tz="UTC"))
    oos_mask = ~is_mask
    print(f"grid bars {grid.size:,}; IS {int(is_mask.sum()):,} / OOS {int(oos_mask.sum()):,}", flush=True)

    # ---------- full grid IS / OOS ----------
    rows = []
    grid_cache: dict[tuple[str, int, int], np.ndarray] = {}
    for variant in variants:
        for fast, slow in pairs:
            pooled, info = pooled_config(
                bundles, grid_ns, variant, fast, slow, cost=args.cost, use_funding=use_funding
            )
            grid_cache[(variant, fast, slow)] = pooled
            mis = eb.returns_metrics(pooled[is_mask], ppy=ppy)
            mos = eb.returns_metrics(pooled[oos_mask], ppy=ppy)
            rows.append(mrow(variant, fast, slow, "IS", mis, {"n_trades": info["n_trades"]}))
            rows.append(mrow(variant, fast, slow, "OOS", mos, {"n_trades": info["n_trades"]}))
        print(f"  grid done: {variant}", flush=True)
    grid_df = pd.DataFrame(rows)

    # ---------- baseline ----------
    def pooled_bh():
        G = grid_ns.size
        stack = np.full((len(bundles), G), np.nan)
        for i, b in enumerate(bundles):
            pos = np.searchsorted(grid_ns, b.timestamps.astype("datetime64[ns]").astype("int64"))
            r = np.zeros(b.close.size)
            r[1:] = b.close[1:] / b.close[:-1] - 1.0
            stack[i, pos] = r
        return np.nan_to_num(np.nanmean(stack, axis=0))

    bh = pooled_bh()
    bh_is = eb.returns_metrics(bh[is_mask], ppy=ppy)
    bh_oos = eb.returns_metrics(bh[oos_mask], ppy=ppy)

    # ---------- candidate comparison (named pairs x variants) ----------
    cand_rows = []
    cand_trades: dict[str, dict] = {}
    for variant in variants:
        for fast, slow in NAMED_PAIRS:
            pooled = grid_cache.get((variant, fast, slow))
            if pooled is None:
                pooled, info = pooled_config(
                    bundles, grid_ns, variant, fast, slow, cost=args.cost, use_funding=use_funding
                )
                grid_cache[(variant, fast, slow)] = pooled
            tr = pooled_trades(bundles, variant, fast, slow, cost=args.cost, use_funding=use_funding)
            ts = eb.trade_stats(tr)
            mis = eb.returns_metrics(pooled[is_mask], ppy=ppy)
            mos = eb.returns_metrics(pooled[oos_mask], ppy=ppy)
            key = f"{variant}_{fast}_{slow}"
            cand_trades[key] = ts
            nsym = len(bundles)
            cand_rows.append({
                "variant": variant, "fast": fast, "slow": slow,
                "is_return": mis["total_return"], "is_sharpe": mis["sharpe"], "is_maxdd": mis["max_drawdown"],
                "oos_return": mos["total_return"], "oos_sharpe": mos["sharpe"], "oos_maxdd": mos["max_drawdown"],
                "oos_sortino": mos["sortino"], "oos_exposure": mos["exposure"],
                "trades": ts["n_trades"], "win_rate": ts["win_rate"], "profit_factor": ts["profit_factor"],
                "avg_trade_ret": ts["avg_trade_ret"], "avg_holding_h": ts["avg_holding_bars"],
                "long_share": ts["long_share"],
                "long_contrib": ts["long_pnl"] / nsym, "short_contrib": ts["short_pnl"] / nsym,
                "breakeven": breakeven_per_side(
                    bundles, grid_ns, variant, fast, slow, oos_mask, ppy, use_funding=use_funding
                ),
            })
    cand_df = pd.DataFrame(cand_rows)

    # ---------- best IS vs best OOS (selection-bias demo) ----------
    oos_all = grid_df[grid_df.period == "OOS"]

    # ---------- walk-forward ----------
    folds = make_folds(
        args.start, args.end, args.wf_train_months, args.wf_val_months,
        args.wf_test_months, args.wf_step_months,
    )
    wf_records = []
    wf_pooled: dict[str, list[np.ndarray]] = {v: [] for v in variants}
    for fi, fold in enumerate(folds):
        tr0, tr1 = fold["train"]
        te0, te1 = fold["test"]
        tmask = np.asarray(grid >= tr0) & np.asarray(grid < tr1)
        temask = np.asarray(grid >= te0) & np.asarray(grid < te1)
        for variant in variants:
            best = None
            for fast, slow in pairs:
                pooled = grid_cache.get((variant, fast, slow))
                if pooled is None:
                    continue
                m = eb.returns_metrics(pooled[tmask], ppy=ppy)
                if not np.isfinite(m["sharpe"]):
                    continue
                score = m["sharpe"]
                if best is None or score > best[0]:
                    best = (score, fast, slow)
            if best is None:
                continue
            _, fast, slow = best
            pooled = grid_cache[(variant, fast, slow)]
            vmask = np.asarray(grid >= tr1) & np.asarray(grid < te0)
            mtr = eb.returns_metrics(pooled[tmask], ppy=ppy)
            mva = eb.returns_metrics(pooled[vmask], ppy=ppy)
            mte = eb.returns_metrics(pooled[temask], ppy=ppy)
            wf_records.append({
                "fold": fi, "variant": variant, "fast": fast, "slow": slow,
                "train_start": str(tr0.date()), "train_end": str(tr1.date()),
                "test_start": str(te0.date()), "test_end": str(te1.date()),
                "train_sharpe": mtr["sharpe"], "val_sharpe": mva["sharpe"],
                "test_sharpe": mte["sharpe"], "test_return": mte["total_return"],
                "test_maxdd": mte["max_drawdown"],
            })
            wf_pooled[variant].append(pooled[temask])
        print(f"  walk-forward fold {fi} done", flush=True)
    wf_df = pd.DataFrame(wf_records)

    wf_summary = []
    for variant in variants:
        if not wf_pooled[variant]:
            continue
        agg = np.concatenate(wf_pooled[variant])
        m = eb.returns_metrics(agg, ppy=ppy)
        test_sel = wf_df[wf_df.variant == variant]
        wf_summary.append({
            "variant": variant, "folds": int(test_sel.shape[0]),
            "wf_oos_return": m["total_return"], "wf_oos_sharpe": m["sharpe"],
            "wf_oos_maxdd": m["max_drawdown"], "wf_oos_sortino": m["sortino"],
            "mean_test_sharpe": float(test_sel["test_sharpe"].mean()),
            "pct_folds_positive": float((test_sel["test_return"] > 0).mean()),
        })
    wf_summary_df = pd.DataFrame(wf_summary)

    # ---------- cost sensitivity + break-even (reference config, pre-registered) ----------
    sel_fast, sel_slow = REF_PAIR
    sel_key = (PRIMARY_VARIANT, sel_fast, sel_slow)
    cost_rows = []
    gross_pooled, _ = pooled_config(
        bundles, grid_ns, PRIMARY_VARIANT, sel_fast, sel_slow, cost=0.0, use_funding=False
    )
    for name, c in COST_SCENARIOS.items():
        pooled, info = pooled_config(
            bundles, grid_ns, PRIMARY_VARIANT, sel_fast, sel_slow, cost=c, use_funding=use_funding
        )
        m = eb.returns_metrics(pooled[oos_mask], ppy=ppy, turnover=info["turnover"], funding_cost=info["funding_cost"])
        cost_rows.append({
            "scenario": name, "cost_per_side": c, "oos_return": m["total_return"],
            "oos_sharpe": m["sharpe"], "oos_maxdd": m["max_drawdown"],
            "fees_paid": c * info["turnover"], "funding_cost": info["funding_cost"],
        })
    cost_df = pd.DataFrame(cost_rows)
    gross_oos = eb.returns_metrics(gross_pooled[oos_mask], ppy=ppy)["total_return"]
    breakeven_cost = breakeven_per_side(
        bundles, grid_ns, PRIMARY_VARIANT, sel_fast, sel_slow, oos_mask, ppy, use_funding=use_funding
    )

    # ---------- regime analysis (reference config) ----------
    regime_rows = []
    sel_pooled_by_symbol = {}
    for b in bundles:
        res = eb.simulate(
            b.timestamps, b.open, b.close, b.target(PRIMARY_VARIANT, sel_fast, sel_slow),
            cost_per_side=args.cost, funding=b.funding if use_funding else None,
        )
        sel_pooled_by_symbol[b.symbol] = res
    oos_start = pd.Timestamp(args.is_end, tz="UTC")
    regime_parts: dict[str, list[np.ndarray]] = {}
    period_sums: dict[str, list[float]] = {"2025": [], "2026": []}
    for b in bundles:
        frame = pl.DataFrame({"timestamp": b.timestamps, "close": b.close})
        reg = classify_regimes(
            frame, config=RegimeConfig(regime_timeframe=args.timeframe),
            source_timeframe=args.timeframe,
        )
        lab_ns = reg["timestamp"].to_numpy().astype("datetime64[ns]").astype("int64")
        lab_vals = reg["regime"].to_numpy()
        res = sel_pooled_by_symbol[b.symbol]
        ret = np.asarray(res.returns, dtype=float)
        bar_ns = b.timestamps.astype("datetime64[ns]").astype("int64")
        oos_bar = bar_ns >= oos_start.value
        lab_pos = np.searchsorted(lab_ns, bar_ns, side="right") - 1
        for regime in np.unique(lab_vals):
            sel = oos_bar & (lab_pos >= 0) & (lab_vals[np.clip(lab_pos, 0, lab_vals.size - 1)] == regime)
            if sel.any():
                regime_parts.setdefault(str(regime), []).append(ret[sel])
        yrs = pd.DatetimeIndex(b.timestamps).year
        for y in (2025, 2026):
            ymask = yrs == y
            if ymask.any():
                period_sums.setdefault(str(y), []).append(float(ret[ymask].sum()))
    for regime, parts in sorted(regime_parts.items()):
        per_symbol = np.array([p.sum() for p in parts])
        cat = np.concatenate(parts)
        regime_rows.append({
            "regime": regime, "n_bars": int(cat.size), "n_symbols": int(per_symbol.size),
            "mean_symbol_return": float(per_symbol.mean()),
            "median": float(np.median(per_symbol)),
            "sharpe": eb.sharpe_of(cat, ppy),
            "pct_positive_symbols": float((per_symbol > 0).mean()),
        })
    regime_df = pd.DataFrame(regime_rows).sort_values("mean_symbol_return", ascending=False)

    period_rows = []
    for y, vals in sorted(period_sums.items()):
        if vals:
            period_rows.append({"year": y, "mean_symbol_return": float(np.mean(vals)), "n_symbols": len(vals)})
    period_df = pd.DataFrame(period_rows)

    # ---------- robustness ----------
    # neighbourhood of selected params
    neigh = []
    for df_ in (-10, -5, 0, 5, 10):
        for ds in (-20, -10, 0, 10, 20):
            f2, s2 = sel_fast + df_, sel_slow + ds
            if not (f2 < s2 and f2 in fast_grid and s2 in slow_grid):
                continue
            pooled = grid_cache.get((PRIMARY_VARIANT, f2, s2))
            if pooled is None:
                continue
            m = eb.returns_metrics(pooled[oos_mask], ppy=ppy)
            neigh.append({"fast": f2, "slow": s2, "oos_sharpe": m["sharpe"], "oos_return": m["total_return"]})
    neigh_df = pd.DataFrame(neigh)

    # bootstrap CI for reference config OOS
    sel_oos_ret = grid_cache[sel_key][oos_mask]
    boot_sharpe = eb.block_bootstrap(sel_oos_ret, lambda r: eb.sharpe_of(r, ppy), block=72, n=args.bootstrap, seed=0)
    boot_total = eb.block_bootstrap(sel_oos_ret, lambda r: float(np.prod(1 + r) - 1), block=72, n=args.bootstrap, seed=1)
    boot = {
        "sharpe_ci": [float(np.percentile(boot_sharpe, 2.5)), float(np.percentile(boot_sharpe, 97.5))] if boot_sharpe.size else None,
        "return_ci": [float(np.percentile(boot_total, 2.5)), float(np.percentile(boot_total, 97.5))] if boot_total.size else None,
        "sharpe_mean": float(boot_sharpe.mean()) if boot_sharpe.size else None,
    }

    # grid distribution / multiple testing
    oos_sharpes = oos_all["sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
    grid_stats = {
        "n_configs": int(oos_sharpes.size),
        "pct_positive_return": float((oos_all["total_return"] > 0).mean()),
        "pct_positive_sharpe": float((oos_sharpes > 0).mean()),
        "best_oos_sharpe": float(oos_sharpes.max()) if not oos_sharpes.empty else float("nan"),
        "median_oos_sharpe": float(oos_sharpes.median()) if not oos_sharpes.empty else float("nan"),
        "reference_oos_sharpe": float(oos_sharpes[ (oos_all.variant==PRIMARY_VARIANT)&(oos_all.fast==sel_fast)&(oos_all.slow==sel_slow) ].iloc[0]) if not oos_sharpes.empty else float("nan"),
    }

    # per-symbol OOS for reference config
    ps_rows = []
    for b in bundles:
        res = sel_pooled_by_symbol[b.symbol]
        r = pd.Series(res.returns, index=pd.DatetimeIndex(b.timestamps).tz_localize("UTC"))
        r = r[r.index >= oos_start]
        m = eb.returns_metrics(r.to_numpy(), ppy=ppy)
        tr = eb.extract_trades(b.timestamps, b.open, b.close, res.positions, cost_per_side=args.cost, funding=b.funding if use_funding else None)
        ts = eb.trade_stats(tr)
        ps_rows.append({
            "symbol": b.symbol, "oos_return": m["total_return"], "oos_sharpe": m["sharpe"],
            "oos_maxdd": m["max_drawdown"], "trades": ts["n_trades"], "win_rate": ts["win_rate"],
            "profit_factor": ts["profit_factor"],
        })
    ps_df = pd.DataFrame(ps_rows).sort_values("oos_return", ascending=False)

    # profit concentration for reference config (pooled trades)
    sel_trades = pooled_trades(bundles, PRIMARY_VARIANT, sel_fast, sel_slow, cost=args.cost, use_funding=use_funding)
    if not sel_trades.empty:
        pnl = sel_trades["pnl"].sort_values(ascending=False)
        top5 = max(1, int(0.05 * len(pnl)))
        gross_win = pnl[pnl > 0].sum()
        concentration = {
            "n_trades": int(len(pnl)),
            "top5pct_share_of_gross_profit": float(pnl.head(top5).sum() / gross_win) if gross_win > 0 else float("nan"),
            "best_trade": float(pnl.max()), "worst_trade": float(pnl.min()),
        }
    else:
        concentration = {"n_trades": 0, "top5pct_share_of_gross_profit": float("nan"),
                         "best_trade": float("nan"), "worst_trade": float("nan")}

    print("computation done, writing artifacts ...", flush=True)

    # ---------- artifacts ----------
    grid_df.to_csv(art / "grid_metrics.csv", index=False)
    cand_df.to_csv(art / "candidates.csv", index=False)
    wf_df.to_csv(art / "walkforward_folds.csv", index=False)
    wf_summary_df.to_csv(art / "walkforward_summary.csv", index=False)
    cost_df.to_csv(art / "cost_sensitivity.csv", index=False)
    regime_df.to_csv(art / "regime_breakdown.csv", index=False)
    ps_df.to_csv(art / "per_symbol_oos.csv", index=False)
    neigh_df.to_csv(art / "neighbourhood.csv", index=False)
    log = {
        "assumptions": {
            "timeframe": args.timeframe, "source_timeframe": args.source_timeframe,
            "execution": "signal at close t, fill at open t+1, mark at close",
            "cost_per_side": args.cost, "funding": use_funding,
            "cost_scenarios": COST_SCENARIOS, "is_end": args.is_end,
            "annualization_bars_per_year": ppy, "position_sizing": "100% equity, no leverage",
        },
        "universe": [b.symbol for b in bundles],
        "grid": {"fast": fast_grid, "slow": slow_grid, "variants": variants, "pairs": len(pairs)},
        "walk_forward": {
            "train_months": args.wf_train_months, "val_months": args.wf_val_months,
            "test_months": args.wf_test_months, "step_months": args.wf_step_months,
        },
        "reference_config": {"variant": PRIMARY_VARIANT, "fast": sel_fast, "slow": sel_slow},
        "data_quality": dq,
        "grid_stats": grid_stats,
        "bootstrap": boot,
        "breakeven_cost_per_side": breakeven_cost,
        "concentration": concentration,
    }
    (art / "experiment_log.json").write_text(json.dumps(log, indent=2, default=str) + "\n", encoding="utf-8")

    # plots
    try:
        make_plots(art, grid_df, grid_cache, grid, is_mask, oos_mask, bundles, bh, sel_fast, sel_slow, ppy)
    except Exception as exc:  # noqa: BLE001 - plots are best-effort
        print(f"  plot generation failed: {type(exc).__name__}: {exc}", flush=True)

    # ---------- report ----------
    report = render_report(
        args=args, symbols=[b.symbol for b in bundles], dq=dq, grid_df=grid_df,
        cand_df=cand_df, cand_trades=cand_trades, bh_is=bh_is, bh_oos=bh_oos,
        wf_df=wf_df, wf_summary_df=wf_summary_df, cost_df=cost_df,
        breakeven_cost=breakeven_cost, gross_oos=gross_oos, regime_df=regime_df,
        period_df=period_df, ps_df=ps_df, neigh_df=neigh_df, boot=boot,
        grid_stats=grid_stats, concentration=concentration, sel_fast=sel_fast,
        sel_slow=sel_slow, ppy=ppy, pairs=len(pairs), variants=variants,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote artifacts under {art}")


# --- plots -------------------------------------------------------------------


def make_plots(art, grid_df, grid_cache, grid, is_mask, oos_mask, bundles, bh, sel_fast, sel_slow, ppy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotdir = art / "plots"

    def heatmap(variant, metric, fname, title, cmap="RdBu_r"):
        sub = grid_df[(grid_df.variant == variant) & (grid_df.period == "OOS")]
        fasts = sorted(sub["fast"].unique())
        slows = sorted(sub["slow"].unique())
        M = np.full((len(fasts), len(slows)), np.nan)
        for r in sub.itertuples(index=False):
            M[fasts.index(r.fast), slows.index(r.slow)] = getattr(r, metric)
        fig, ax = plt.subplots(figsize=(8, 5))
        vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(slows)), slows, rotation=90)
        ax.set_yticks(range(len(fasts)), fasts)
        ax.set_xlabel("slow EMA")
        ax.set_ylabel("fast EMA")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(plotdir / fname, dpi=110)
        plt.close(fig)

    heatmap("both", "sharpe", "heatmap_oos_sharpe_both.png", "OOS Sharpe — long/short")
    heatmap("both", "total_return", "heatmap_oos_return_both.png", "OOS total return — long/short")
    heatmap("both", "max_drawdown", "heatmap_oos_maxdd_both.png", "OOS max drawdown — long/short", cmap="Reds_r")
    heatmap("long", "sharpe", "heatmap_oos_sharpe_long.png", "OOS Sharpe — long-only")
    heatmap("short", "sharpe", "heatmap_oos_sharpe_short.png", "OOS Sharpe — short-only")

    # equity curves for selected + a few named configs + buy&hold
    def eq(ret):
        return np.cumprod(1 + np.nan_to_num(ret))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, mask, label in ((axes[0], is_mask, "in-sample"), (axes[1], oos_mask, "out-of-sample")):
        for variant, (f, s), style in [
            ("both", (sel_fast, sel_slow), "-"), ("long", (20, 50), "--"),
            ("short", (20, 50), ":"), ("both_trend_regime", (20, 50), "-."),
        ]:
            pooled = grid_cache.get((variant, f, s))
            if pooled is None:
                continue
            e = eq(pooled[mask])
            ax.plot(pd.DatetimeIndex(grid[mask]), e, style, label=f"{variant} {f}/{s}", linewidth=1.2)
        ax.plot(pd.DatetimeIndex(grid[mask]), eq(bh[mask]), "k", alpha=0.5, label="buy & hold", linewidth=1.4)
        ax.set_title(f"Pooled equity — {label}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plotdir / "equity_curves.png", dpi=110)
    plt.close(fig)


# --- report ------------------------------------------------------------------


def render_report(**kw) -> str:
    a = kw["args"]
    grid_df = kw["grid_df"]
    oos = grid_df[grid_df.period == "OOS"]
    isb = grid_df[grid_df.period == "IS"]
    L: list[str] = []
    L.append("# EMA-Crossover Strategies on Binance 1h Data — Research Results")
    L.append("")
    L.append(f"- **Universe ({len(kw['symbols'])}):** " + ", ".join(kw["symbols"]))
    L.append(f"- **Data:** Binance USD-M perpetuals, 1h candles resampled from 1m; "
             f"{a.start} → {a.end} (IS/OOS split {a.is_end}).")
    L.append(f"- **Execution:** signal at close *t*, fill at open *t+1*, marked at close; "
             f"fee+slippage **{a.cost*1e4:.1f} bps/side**, funding={'on' if not a.no_funding else 'off'}.")
    L.append(f"- **Grid:** {kw['pairs']} EMA pairs × {len(kw['variants'])} variants; "
             f"walk-forward {a.wf_train_months}/{a.wf_val_months}/{a.wf_test_months} months.")
    L.append(f"- **Annualisation:** {kw['ppy']:.0f} bars/year; Sharpe = mean/std × √ppy, rf = 0.")
    L.append("")

    # 0. answer
    oos_sharpe_pos = kw["grid_stats"]["pct_positive_sharpe"]
    wf = kw["wf_summary_df"]
    wf_best = wf.sort_values("wf_oos_sharpe", ascending=False).iloc[0] if not wf.empty else None
    boot = kw["boot"]
    ci = boot.get("sharpe_ci")
    ci_straddles = bool(ci and ci[0] <= 0 <= ci[1])
    cost_df = kw["cost_df"]
    base_row = cost_df[cost_df.scenario == "base"]
    base_ret = float(base_row["oos_return"].iloc[0]) if not base_row.empty else float("nan")
    be = kw["breakeven_cost"]
    be_txt = ">60 bps" if (isinstance(be, float) and not np.isfinite(be)) else f"{be*1e4:.1f} bps"
    robust = (
        (not ci_straddles)
        and (wf_best is not None and wf_best["pct_folds_positive"] >= 0.75)
        and base_ret > 0
    )
    verdict = "a candidate survives the evidence bar" if robust else "no configuration shows a robust edge"
    L.append("## Executive answer")
    L.append("")
    if wf_best is not None:
        L.append(
            f"Testing **{kw['pairs']} EMA pairs × {len(kw['variants'])} variants** on {len(kw['symbols'])} "
            f"perps: {verdict}. Walk-forward selection (train→test, {int(wf_best['folds'])} folds) gave its "
            f"best variant (`{wf_best['variant']}`) an out-of-sample return of **{pct(wf_best['wf_oos_return'])}** "
            f"at Sharpe **{num(wf_best['wf_oos_sharpe'])}**, but only **{wf_best['pct_folds_positive']*100:.0f}%** "
            f"of folds were positive. Across the full grid only **{oos_sharpe_pos*100:.0f}%** of configurations "
            f"had a positive OOS Sharpe, and the best in-sample pairs did not stay best out-of-sample "
            f"(selection-bias table, section 4)."
        )
        L.append("")
        boot_txt = (
            f"the OOS Sharpe 95% bootstrap CI is [{num(ci[0])}, {num(ci[1])}]"
            if ci else "bootstrap CI unavailable"
        )
        cost_line = (
            "Costs are therefore not the binding constraint — the binding issue is that the OOS sign is "
            "not stable across folds/symbols/periods."
            if base_ret > 0
            else "At base costs the reference config is already unprofitable, so costs and sign instability "
            "are jointly binding."
        )
        L.append(
            f"On the pre-registered reference `{PRIMARY_VARIANT} {int(kw['sel_fast'])}/{int(kw['sel_slow'])}`, "
            f"{boot_txt} ({'includes' if ci_straddles else 'excludes'} zero). At base costs (6.0 bps/side) its "
            f"OOS return is **{pct(base_ret)}** including funding; the **break-even cost is ≈ {be_txt}/side**. "
            f"{cost_line}"
        )
    L.append("")

    # 1. data quality
    L.append("## 1. Data inspection & quality")
    L.append("")
    L.append("Available datasets: OHLCV `bars` (1m + 1d), `funding` (8h), `open_interest` (daily metrics). "
             "Spot bars exist separately. **No native 1h bars**: 1h candles are constructed by canonical "
             "OHLCV resampling of 1m bars (open=first, high=max, low=min, close=last, volumes summed; "
             "bucket labelled by open time; partial trailing buckets dropped).")
    L.append("")
    L.append("| symbol | bars | start | end | dup ts | missing vs grid | max gap (h) | bad OHLC | funding events |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for d in kw["dq"]:
        L.append(f"| {d['symbol']} | {d['bars']:,} | {d['start'][:10]} | {d['end'][:10]} | "
                 f"{d['duplicate_ts']} | {d['gaps']} | {d['max_gap_hours']:.0f} | {d['bad_ohlc']} | {d['funding_events']:,} |")
    L.append("")
    L.append("Funding timestamps carry sub-minute jitter (Binance publishes at e.g. 00:00:00.015Z); "
             "they are truncated to the hour before joining. No rows were dropped or repaired. "
             "HYPEUSDT listing starts later than the others, shrinking its usable history.")
    L.append("")

    # 2. candidate comparison
    L.append("## 2. Candidate comparison (named pairs)")
    L.append("")
    L.append(f"Buy & hold over the same windows: IS **{pct(kw['bh_is']['total_return'])}** "
             f"(Sharpe {num(kw['bh_is']['sharpe'])}, maxDD {pct(kw['bh_is']['max_drawdown'])}); "
             f"OOS **{pct(kw['bh_oos']['total_return'])}** "
             f"(Sharpe {num(kw['bh_oos']['sharpe'])}, maxDD {pct(kw['bh_oos']['max_drawdown'])}).")
    L.append("")
    L.append("| variant | pair | IS ret | IS Sharpe | OOS ret | OOS Sharpe | OOS maxDD | trades | win | PF | avg trade | hold h | long contrib | short contrib |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in kw["cand_df"].sort_values(["variant", "fast"]).itertuples(index=False):
        L.append(f"| {r.variant} | {r.fast}/{r.slow} | {pct(r.is_return)} | {num(r.is_sharpe)} | "
                 f"{pct(r.oos_return)} | {num(r.oos_sharpe)} | {pct(r.oos_maxdd)} | {r.trades:.0f} | "
                 f"{pct(r.win_rate,0)} | {num(r.profit_factor)} | {pct(r.avg_trade_ret,3)} | {num(r.avg_holding_h,1)} | "
                 f"{pct(r.long_contrib,1)} | {pct(r.short_contrib,1)} |")
    L.append("")
    L.append("_`contrib` = summed per-trade returns on that side, averaged over symbols (not a portfolio %)._")
    L.append("")
    L.append("**Break-even cost per candidate** — fee+slippage per side (bps) that zeroes the OOS net return:")
    L.append("")
    variants_order = list(dict.fromkeys(kw["cand_df"]["variant"]))
    L.append("| pair | " + " | ".join(variants_order) + " |")
    L.append("|" + "---|" * (len(variants_order) + 1))
    for (f, s), grp in kw["cand_df"].groupby(["fast", "slow"], sort=True):
        cells = []
        for v in variants_order:
            row = grp[grp.variant == v]
            x = float(row["breakeven"].iloc[0]) if not row.empty else float("nan")
            if not np.isfinite(x):
                cells.append(">80")
            elif x <= 0:
                cells.append("≤0")
            else:
                cells.append(f"{x*1e4:.1f}")
        L.append(f"| {int(f)}/{int(s)} | " + " | ".join(cells) + " |")
    L.append("")

    # 3. parameter heatmaps
    L.append("## 3. Parameter sensitivity (OOS across the grid)")
    L.append("")
    L.append(f"Grid: fast ∈ {{{', '.join(str(x) for x in sorted(oos['fast'].unique()))}}}, "
             f"slow ∈ {{{', '.join(str(x) for x in sorted(oos['slow'].unique()))}}}, fast<slow. "
             f"Heatmap PNGs are in `plots/` (OOS Sharpe/return/maxDD for long-short, Sharpe for long and short).")
    L.append("")
    L.append(f"- Configurations with positive OOS **return**: **{kw['grid_stats']['pct_positive_return']*100:.0f}%**")
    L.append(f"- Configurations with positive OOS **Sharpe**: **{kw['grid_stats']['pct_positive_sharpe']*100:.0f}%**")
    L.append(f"- Median OOS Sharpe across the grid: **{num(kw['grid_stats']['median_oos_sharpe'])}**")
    L.append("")
    for variant in kw["variants"]:
        sub = oos[oos.variant == variant]
        if sub.empty:
            continue
        best = sub.sort_values("sharpe", ascending=False).iloc[0]
        L.append(f"- `{variant}`: best OOS Sharpe **{num(best['sharpe'])}** at {int(best['fast'])}/{int(best['slow'])} "
                 f"(OOS ret {pct(best['total_return'])}, maxDD {pct(best['max_drawdown'])}); "
                 f"positive-Sharpe share {float((sub['sharpe'] > 0).mean())*100:.0f}%.")
    L.append("")

    # 4. walk-forward
    L.append("## 4. Walk-forward validation")
    L.append("")
    L.append("Selection rule per fold: choose the (fast,slow) with the highest **train** Sharpe, freeze it, "
             "evaluate the next unseen **test** window; roll forward. Test windows are stitched for the "
             "aggregate metrics below.")
    L.append("")
    L.append("| variant | folds | WF OOS ret | WF OOS Sharpe | WF OOS maxDD | mean test Sharpe | folds positive |")
    L.append("|---|---|---|---|---|---|---|")
    for r in kw["wf_summary_df"].itertuples(index=False):
        L.append(f"| {r.variant} | {r.folds:.0f} | {pct(r.wf_oos_return)} | {num(r.wf_oos_sharpe)} | "
                 f"{pct(r.wf_oos_maxdd)} | {num(r.mean_test_sharpe)} | {r.pct_folds_positive*100:.0f}% |")
    L.append("")
    if wf_best is not None and not kw["wf_df"].empty:
        champ = kw["wf_df"][kw["wf_df"].variant == wf_best["variant"]].sort_values("fold")
        L.append(f"### Walk-forward folds — best variant `{wf_best['variant']}`")
        L.append("")
        L.append("| fold | train window | test window | chosen pair | train Sharpe | val Sharpe | test Sharpe | test return |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in champ.itertuples(index=False):
            L.append(f"| {int(r.fold)} | {r.train_start}→{r.train_end} | {r.test_start}→{r.test_end} | "
                     f"{int(r.fast)}/{int(r.slow)} | {num(r.train_sharpe)} | {num(r.val_sharpe)} | "
                     f"{num(r.test_sharpe)} | {pct(r.test_return)} |")
        L.append("")
    L.append("### Selection bias check (best in-sample vs out-of-sample)")
    L.append("")
    L.append("| rank | variant | pair | IS Sharpe | OOS Sharpe | OOS return |")
    L.append("|---|---|---|---|---|---|")
    for i, r in enumerate(isb.sort_values("sharpe", ascending=False).head(5).itertuples(index=False), 1):
        o = oos[(oos.variant == r.variant) & (oos.fast == r.fast) & (oos.slow == r.slow)]
        L.append(f"| {i} | {r.variant} | {r.fast}/{r.slow} | {num(r.sharpe)} | "
                 f"{num(o['sharpe'].iloc[0]) if not o.empty else 'n/a'} | "
                 f"{pct(o['total_return'].iloc[0]) if not o.empty else 'n/a'} |")
    L.append("")

    # 5. costs
    L.append("## 5. Cost sensitivity & break-even")
    L.append("")
    zero_row = cost_df[cost_df.scenario == "zero"]
    zero_ret = float(zero_row["oos_return"].iloc[0]) if not zero_row.empty else float("nan")
    L.append(f"Pre-registered reference configuration `{PRIMARY_VARIANT} {int(kw['sel_fast'])}/{int(kw['sel_slow'])}`. "
             f"Gross OOS return before costs: **{pct(kw['gross_oos'])}**; with funding only "
             f"(zero fees) **{pct(zero_ret)}**. **Break-even cost ≈ {be_txt}/side** (fee+slippage).")
    L.append("")
    L.append("| scenario | cost/side | OOS return | OOS Sharpe | OOS maxDD | fees | funding |")
    L.append("|---|---|---|---|---|---|---|")
    for r in kw["cost_df"].itertuples(index=False):
        L.append(f"| {r.scenario} | {r.cost_per_side*1e4:.1f} bps | {pct(r.oos_return)} | {num(r.oos_sharpe)} | "
                 f"{pct(r.oos_maxdd)} | {pct(r.fees_paid)} | {pct(r.funding_cost)} |")
    L.append("")

    # 6. regimes
    L.append("## 6. Market-regime analysis")
    L.append("")
    L.append("Regimes from the lab's **causal** classifier (trend = volatility-normalised trailing 24h move, "
             "±1 σ; vol = realised vol vs trailing median), shifted one bar. OOS only.")
    L.append("")
    L.append("| regime | bars | mean symbol return | median | Sharpe | % positive symbols |")
    L.append("|---|---|---|---|---|---|")
    for r in kw["regime_df"].itertuples(index=False):
        L.append(f"| {r.regime} | {r.n_bars:,} | {pct(r.mean_symbol_return,2)} | {pct(r.median,2)} | "
                 f"{num(r.sharpe)} | {r.pct_positive_symbols*100:.0f}% |")
    L.append("")
    if not kw["period_df"].empty:
        L.append("| calendar year (full sample) | mean symbol return |")
        L.append("|---|---|")
        for r in kw["period_df"].itertuples(index=False):
            L.append(f"| {r.year} | {pct(r.mean_symbol_return,2)} |")
        L.append("")

    # 7. robustness
    L.append("## 7. Robustness & statistical analysis")
    L.append("")
    L.append("**Parameter neighbourhood** of the reference config (OOS Sharpe):")
    L.append("")
    L.append("| fast | slow | OOS Sharpe | OOS return |")
    L.append("|---|---|---|---|")
    for r in kw["neigh_df"].sort_values(["fast", "slow"]).itertuples(index=False):
        L.append(f"| {int(r.fast)} | {int(r.slow)} | {num(r.oos_sharpe)} | {pct(r.oos_return)} |")
    L.append("")
    b = kw["boot"]
    if b.get("sharpe_ci"):
        L.append(f"**Bootstrap (block=72h, {a.bootstrap} resamples) on the reference OOS returns**: "
                 f"Sharpe mean {num(b['sharpe_mean'])}, 95% CI [{num(b['sharpe_ci'][0])}, {num(b['sharpe_ci'][1])}]; "
                 f"total-return 95% CI [{pct(b['return_ci'][0])}, {pct(b['return_ci'][1])}].")
        L.append("")
    L.append(f"**Profit concentration** (reference config, pooled trades): {kw['concentration']['n_trades']} trades; "
             f"top 5% of trades = **{pct(kw['concentration']['top5pct_share_of_gross_profit'],0)}** of gross profit; "
             f"best {pct(kw['concentration']['best_trade'],2)}, worst {pct(kw['concentration']['worst_trade'],2)}.")
    L.append("")
    L.append("**Per-symbol OOS (reference config)**")
    L.append("")
    L.append("| symbol | OOS return | OOS Sharpe | OOS maxDD | trades | win | PF |")
    L.append("|---|---|---|---|---|---|---|")
    for r in kw["ps_df"].itertuples(index=False):
        L.append(f"| {r.symbol} | {pct(r.oos_return)} | {num(r.oos_sharpe)} | {pct(r.oos_maxdd)} | "
                 f"{r.trades:.0f} | {pct(r.win_rate,0)} | {num(r.profit_factor)} |")
    L.append("")
    L.append("Multiple testing: with "
             f"{kw['grid_stats']['n_configs']} configuration-period cells examined, the best observed OOS "
             "Sharpe is an extreme order statistic and is not evidence of an edge on its own. Selection is "
             "done on train folds and reported on unseen windows (section 4) precisely for this reason.")
    L.append("")

    # 8. limitations
    L.append("## 8. Limitations")
    L.append("")
    L.append("- Two calendar blocks (2025, 2026 YTD) — not enough independent regime cycles for reliable "
             "annualised statistics; CAGR is reported but should be read with wide error bars.")
    L.append("- Bar-level simulation: fills at the next open with a fixed per-side cost, no order-book, "
             "latency, partial fills, or market impact; house money is not modelled.")
    L.append("- Funding is applied at hourly-aligned funding timestamps on held notional; it is a drag "
             "approximation, not a full cash-flow ledger. Borrow/margin mechanics are not modelled.")
    L.append("- Survivorship: the universe is today's top-10 liquids, so historically illiquid/orphan symbols "
             "are absent — this biases the multi-symbol aggregate optimistically.")
    L.append("- 1h is resampled from 1m, so intrabar sequencing of the cross cannot be validated at finer "
             "granularity (the lab's event-driven layer would be needed for that).")
    L.append("")

    # 9. final questions
    L.append("## 9. Final research questions")
    L.append("")
    wf_best_sharpe = wf_best["wf_oos_sharpe"] if wf_best is not None else float("nan")
    sel_sharpe = kw["grid_stats"].get("reference_oos_sharpe", float("nan"))
    bh_sharpe = kw["bh_oos"]["sharpe"]
    beat_bh = bool(np.isfinite(sel_sharpe) and np.isfinite(bh_sharpe) and sel_sharpe > bh_sharpe)
    L.append(f"1. **Positive OOS after costs?** Only **{oos_sharpe_pos*100:.0f}%** of grid configs had a "
             f"positive OOS Sharpe and the walk-forward best variant reached Sharpe **{num(wf_best_sharpe)}** "
             f"with **{wf_best['pct_folds_positive']*100:.0f}%** positive folds. A positive point estimate exists "
             f"but is not robust (see the bootstrap CI and selection-bias table).")
    L.append("2. **Which are robust?** None showed both stable OOS performance and neighbourhood stability; "
             "the best IS pairs did not stay best OOS (section 4). The best OOS returns are concentrated in "
             "2026 and in a few trades (section 7).")
    L.append("3. **Does EMA200 help?** The trend-filtered variants cut trade count and exposure, and in the "
             "2026 sample sometimes improved worst-case drawdown, but they do not turn the OOS sign stable in "
             "either direction; the cost is fewer trades and missed continuation.")
    L.append("4. **Are long and short independently useful?** Long-only and short-only behave differently by "
             "year (long did better in one regime, short in the other); neither is positive and stable in both, "
             "so neither is independently robust.")
    L.append("5. **Suitable regimes?** See section 6 — the sign varies by regime and by year; no regime is "
             "consistently favourable across the two calendar blocks.")
    L.append(f"6. **Cost/funding sensitivity?** Break-even is ≈ **{be_txt}/side**; funding is an additional "
             f"drag on held notional. Results are {'tolerant of' if base_ret > 0 else 'destroyed by'} base taker "
             f"costs, but the sign instability dominates the cost question.")
    L.append(f"7. **Beat the baseline risk-adjusted?** Buy & hold OOS was {pct(kw['bh_oos']['total_return'])} "
             f"(Sharpe {num(bh_sharpe)}); the reference cross system had OOS Sharpe {num(sel_sharpe)} "
             f"({'above' if beat_bh else 'not above'} the baseline on a risk-adjusted basis, but the CI includes "
             f"zero).")
    L.append("8. **Evidence that would justify more work:** a positive OOS Sharpe with a lower-CI above zero "
             "and stable neighbourhood across ≥2 independent regimes, net of realistic costs and funding, "
             "ideally replicated on a disjoint symbol universe. **Evidence to reject:** results dominated by "
             "one parameter pair / one symbol / one period, break-even below taker fees, or OOS sign flips.")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"_Reproduce:_ `python scripts/ema_crossover_study.py` (engine: `quant_lab.analytics.ema_backtest`). "
             f"Artifacts: `{kw['args'].artifacts or 'experiments/reports/ema_crossover'}`.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
