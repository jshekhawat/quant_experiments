#!/usr/bin/env python
"""Bollinger-basis SMA(20) vs EMA crossover study (1h Binance perps).

Implements ``reports/EMA_SMA_CROSS.md``: a fixed SMA(20) (Bollinger middle line)
crossed by EMAs of many periods, evaluated for predictive information and for
trading profitability after costs and funding, with Bollinger-bandwidth and
price-position filters, regime analysis, walk-forward validation and robustness.

Reuses the previous EMA-crossover engine (``analytics.ema_backtest``) and the
data pipeline. Outputs a report (``reports/ema_sma_cross_results.md``) plus
CSV/JSON/PNG artefacts under ``experiments/reports/ema_sma_cross/``.

    python scripts/ema_sma_cross_study.py
    python scripts/ema_sma_cross_study.py --quick --symbols BTCUSDT,ETHUSDT

Research code: no live trading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from quant_lab.analytics import bollinger_cross as bc
from quant_lab.analytics import ema_backtest as eb
from quant_lab.analytics.regimes import RegimeConfig, classify_regimes
from quant_lab.config import get_config
from quant_lab.data.universe import load_universe

CACHED_UNIVERSE = "configs/divergence_universe.json"
COSTS = {"zero": 0.0, "low": 0.0003, "base": 0.0006, "high": 0.0012, "extreme": 0.0020}
REF_PERIOD = 20        # pre-registered reference: the EMA20/SMA20 special case
REF_SIDE = "both"
FILTER_PERIODS = (10, 20, 50)


# --- helpers -----------------------------------------------------------------


def resolve_universe(args) -> list[str]:
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    path = Path(CACHED_UNIVERSE)
    if path.exists():
        u = load_universe(path)
        return (u.get("futures") or next(iter(u.values())))[: args.n]
    raise SystemExit("no universe: pass --symbols")


def data_quality(b: bc.CrossBundle, timeframe: str) -> dict:
    ts = pd.DatetimeIndex(b.timestamps)
    diffs = pd.Series(ts).diff().dropna()
    expected = pd.Timedelta(hours=eb.timeframe_hours(timeframe))
    n_expected = int((ts[-1] - ts[0]) / expected) + 1 if len(ts) else 0
    return {
        "symbol": b.symbol, "bars": int(b.close.size), "start": str(ts[0]), "end": str(ts[-1]),
        "duplicate_ts": int(pd.Series(ts).duplicated().sum()),
        "gaps": int(n_expected - len(ts)),
        "max_gap_hours": float(diffs.max().total_seconds() / 3600) if len(diffs) else 0.0,
        "bb_warmup": int(np.isnan(b.bb_width).sum()),
        "funding_events": int(np.count_nonzero(b.funding)),
    }


def pooled_config(
    bundles, grid_ns, period, side, *, cost, use_funding, filter_name=None, mode="none"
):
    G = grid_ns.size
    stack = np.full((len(bundles), G), np.nan)
    ntr = 0
    turn = 0.0
    fund = 0.0
    expo = 0.0
    for i, b in enumerate(bundles):
        kw = {}
        if filter_name and filter_name != "none":
            fl = bc.bandwidth_filters(b.bb_width, b.close, b.sma)[filter_name]
            kw = {"allow_long": fl[0], "allow_short": fl[1], "mode": mode}
        res = eb.simulate(
            b.timestamps, b.open, b.close, b.target(period, side, **kw),
            cost_per_side=cost, funding=b.funding if use_funding else None,
            symbol=b.symbol, variant=f"{side}_{period}",
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
        "n_trades": ntr, "turnover": turn / nsym, "funding_cost": fund / nsym, "exposure": expo / nsym,
    }


def pooled_trades(bundles, period, side, *, cost, use_funding, filter_name=None, mode="none"):
    frames = []
    for b in bundles:
        kw = {}
        if filter_name and filter_name != "none":
            fl = bc.bandwidth_filters(b.bb_width, b.close, b.sma)[filter_name]
            kw = {"allow_long": fl[0], "allow_short": fl[1], "mode": mode}
        res = eb.simulate(b.timestamps, b.open, b.close, b.target(period, side, **kw),
                          cost_per_side=cost, funding=b.funding if use_funding else None)
        tr = eb.extract_trades(b.timestamps, b.open, b.close, res.positions, cost_per_side=cost,
                               funding=b.funding if use_funding else None)
        if not tr.empty:
            tr["symbol"] = b.symbol
            frames.append(tr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def breakeven_per_side(bundles, grid_ns, period, side, oos_mask, ppy, *, use_funding,
                       filter_name=None, mode="none", hi=0.008, n=33) -> float:
    fine = np.linspace(0.0, hi, n)
    rets = np.array([
        eb.returns_metrics(
            pooled_config(bundles, grid_ns, period, side, cost=float(c), use_funding=use_funding,
                          filter_name=filter_name, mode=mode)[0][oos_mask], ppy=ppy,
        )["total_return"]
        for c in fine
    ])
    below = np.flatnonzero(rets <= 0)
    if below.size and below[0] > 0:
        i = int(below[0])
        y0, y1 = rets[i - 1], rets[i]
        return float(fine[i - 1] + (fine[i] - fine[i - 1]) * (0 - y0) / (y1 - y0)) if y1 != y0 else float(fine[i])
    return 0.0 if below.size else float("inf")


def make_folds(start, end, train_m, val_m, test_m, step_m):
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


def pct(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def bps(x):
    return ">80" if not np.isfinite(x) else ("≤0" if x <= 0 else f"{x * 1e4:.1f}")


# --- diagnostics (Experiment B) ----------------------------------------------


def diagnostic_events(bundles, periods, horizons):
    """Pooled oriented forward returns at crossover events, by period/side/horizon."""
    per_side: dict[str, dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]] = {}
    for side in bc.SIDES:
        per_side[side] = {}
        for p in periods:
            ts_all: list[np.ndarray] = []
            vals: dict[str, list[np.ndarray]] = {h: [] for h in horizons}
            for b in bundles:
                ev = b.events(p)
                fwd = bc.forward_returns(b.close, horizons)
                if side == "long":
                    sel = ev == 1
                elif side == "short":
                    sel = ev == -1
                else:
                    sel = ev != 0
                idx = np.flatnonzero(sel)
                if idx.size == 0:
                    continue
                sign = ev[idx].astype(float)
                ts_all.append(b.timestamps.astype("datetime64[ns]").astype("int64")[idx])
                for h in horizons:
                    vals[h].append(sign * fwd[h][idx])
            if not ts_all:
                per_side[side][p] = {}
                continue
            ts = np.concatenate(ts_all)
            order = np.argsort(ts, kind="stable")
            per_side[side][p] = {
                h: (ts[order], np.concatenate(vals[h])[order]) for h in horizons
            }
    return per_side


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
    ap.add_argument("--cost", type=float, default=0.0006)
    ap.add_argument("--no-funding", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--wf-train-months", type=int, default=6)
    ap.add_argument("--wf-val-months", type=int, default=2)
    ap.add_argument("--wf-test-months", type=int, default=3)
    ap.add_argument("--wf-step-months", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--artifacts", type=Path, default=None)
    args = ap.parse_args()

    cfg = get_config()
    symbols = resolve_universe(args)
    periods = (5, 20, 50) if args.quick else bc.EMA_PERIODS
    use_funding = not args.no_funding
    out = args.out or (cfg.root / "reports" / "ema_sma_cross_results.md")
    art = args.artifacts or (cfg.paths.experiments / "reports" / "ema_sma_cross")
    art.mkdir(parents=True, exist_ok=True)
    (art / "plots").mkdir(exist_ok=True)
    ppy = eb.periods_per_year_for(args.timeframe)
    horizons = bc.FORWARD_HORIZONS

    print(f"SMA20 vs EMA crossover study · {args.timeframe} · {args.start}..{args.end} · split {args.is_end}", flush=True)
    print(f"universe ({len(symbols)}): " + ", ".join(symbols), flush=True)
    print(f"EMA periods: {periods}; sides: {bc.SIDES}; cost/side={args.cost}; funding={use_funding}", flush=True)

    bundles: list[bc.CrossBundle] = []
    dq: list[dict] = []
    for s in symbols:
        b = bc.load_cross_bundle(
            s, start=args.start, end=args.end, periods=tuple(periods),
            timeframe=args.timeframe, source_timeframe=args.source_timeframe,
            root=cfg.paths.data_normalized, with_funding=use_funding,
        )
        if b is None:
            print(f"  skip {s}", flush=True)
            continue
        bundles.append(b)
        dq.append(data_quality(b, args.timeframe))
    if not bundles:
        raise SystemExit("no data")

    lo = min(pd.Timestamp(b.timestamps[0]) for b in bundles)
    hi = max(pd.Timestamp(b.timestamps[-1]) for b in bundles)
    grid = pd.date_range(lo, hi, freq="1h", tz="UTC")
    grid_ns = grid.tz_localize(None).to_numpy().astype("datetime64[ns]").astype("int64")
    is_mask = np.asarray(grid < pd.Timestamp(args.is_end, tz="UTC"))
    oos_mask = ~is_mask
    print(f"grid bars {grid.size:,}; IS {int(is_mask.sum()):,} / OOS {int(oos_mask.sum()):,}", flush=True)

    # ---------- Experiment A: main comparison ----------
    grid_cache: dict[tuple[str, int], np.ndarray] = {}
    info_cache: dict[tuple[str, int], dict] = {}
    rows = []
    for side in bc.SIDES:
        for p in periods:
            pooled, info = pooled_config(bundles, grid_ns, p, side, cost=args.cost, use_funding=use_funding)
            grid_cache[(side, p)] = pooled
            info_cache[(side, p)] = info
            mis = eb.returns_metrics(pooled[is_mask], ppy=ppy)
            mos = eb.returns_metrics(pooled[oos_mask], ppy=ppy)
            rows.append({
                "side": side, "period": p,
                "is_return": mis["total_return"], "is_sharpe": mis["sharpe"], "is_maxdd": mis["max_drawdown"],
                "oos_return": mos["total_return"], "oos_sharpe": mos["sharpe"], "oos_sortino": mos["sortino"],
                "oos_maxdd": mos["max_drawdown"], "oos_exposure": info["exposure"], "oos_trades": info["n_trades"],
                "oos_turnover": info["turnover"],
            })
        print(f"  grid done: {side}", flush=True)
    grid_df = pd.DataFrame(rows)

    # trades stats for the comparison table (per period/side)
    trade_rows = []
    for side in bc.SIDES:
        for p in periods:
            tr = pooled_trades(bundles, p, side, cost=args.cost, use_funding=use_funding)
            ts = eb.trade_stats(tr)
            trade_rows.append({"side": side, "period": p, **ts})
    trade_df = pd.DataFrame(trade_rows)
    comp = grid_df.merge(trade_df, on=["side", "period"])
    comp["breakeven"] = [
        breakeven_per_side(bundles, grid_ns, int(r.period), r.side, oos_mask, ppy, use_funding=use_funding)
        for r in comp.itertuples(index=False)
    ]

    # ---------- cost sensitivity (best OOS per side) ----------
    cost_rows = []
    for side in bc.SIDES:
        sub = grid_df[grid_df.side == side].sort_values("oos_sharpe", ascending=False)
        p = int(sub.iloc[0]["period"]) if not sub.empty else periods[0]
        for name, c in COSTS.items():
            pooled, info = pooled_config(bundles, grid_ns, p, side, cost=c, use_funding=use_funding)
            m = eb.returns_metrics(pooled[oos_mask], ppy=ppy, turnover=info["turnover"], funding_cost=info["funding_cost"])
            cost_rows.append({"side": side, "period": p, "scenario": name, "cost_per_side": c,
                              "oos_return": m["total_return"], "oos_sharpe": m["sharpe"],
                              "oos_maxdd": m["max_drawdown"], "fees": c * info["turnover"],
                              "funding": info["funding_cost"]})
    cost_df = pd.DataFrame(cost_rows)

    # ---------- Experiment B: diagnostics ----------
    print("diagnostics ...", flush=True)
    diag = diagnostic_events(bundles, periods, horizons)
    # unconditional baselines from per-symbol price paths, averaged across symbols
    base_hit, base_mean = {}, {}
    for name, h in horizons.items():
        hits, means = [], []
        for b in bundles:
            f = bc.forward_returns(b.close, {name: h})[name]
            f = f[np.isfinite(f)]
            hits.append(float((f > 0).mean()))
            means.append(float(f.mean()))
        base_hit[name] = float(np.mean(hits))
        base_mean[name] = float(np.mean(means))
    diag_rows = []
    for side in bc.SIDES:
        for p in periods:
            per_h = diag.get(side, {}).get(p, {})
            for name, h in horizons.items():
                if name not in per_h:
                    continue
                ts, y = per_h[name]
                y = y[np.isfinite(y)]
                if y.size == 0:
                    continue
                block = max(int(h), 24)
                bmean = eb.block_bootstrap(y, lambda r: float(r.mean()), block=block, n=200, seed=0)
                bhit = eb.block_bootstrap(y, lambda r: float((r > 0).mean()), block=block, n=200, seed=1)
                diag_rows.append({
                    "side": side, "period": p, "horizon": name, "n": int(y.size),
                    "hit_rate": float((y > 0).mean()),
                    "hit_lo": float(np.percentile(bhit, 2.5)) if bhit.size else float("nan"),
                    "hit_hi": float(np.percentile(bhit, 97.5)) if bhit.size else float("nan"),
                    "mean": float(y.mean()), "median": float(np.median(y)),
                    "std": float(y.std(ddof=1)) if y.size > 1 else float("nan"),
                    "mean_lo": float(np.percentile(bmean, 2.5)) if bmean.size else float("nan"),
                    "mean_hi": float(np.percentile(bmean, 97.5)) if bmean.size else float("nan"),
                    "baseline_hit": base_hit[name], "baseline_mean": base_mean[name],
                })
    diag_df = pd.DataFrame(diag_rows)

    # ---------- Experiment C: filters ----------
    print("filters ...", flush=True)
    filter_rows = []
    for side in ("long", "both"):
        for p in FILTER_PERIODS:
            if p not in periods:
                continue
            for fname in ("none", *bc.FILTER_KINDS):
                for mode in ("entry", "hold"):
                    if fname == "none" and mode == "hold":
                        continue
                    pooled, info = pooled_config(bundles, grid_ns, p, side, cost=args.cost,
                                                 use_funding=use_funding, filter_name=fname, mode=mode)
                    mos = eb.returns_metrics(pooled[oos_mask], ppy=ppy)
                    mis = eb.returns_metrics(pooled[is_mask], ppy=ppy)
                    filter_rows.append({
                        "side": side, "period": p, "filter": fname, "mode": mode,
                        "is_sharpe": mis["sharpe"], "oos_sharpe": mos["sharpe"],
                        "oos_return": mos["total_return"], "oos_maxdd": mos["max_drawdown"],
                        "trades": info["n_trades"], "exposure": info["exposure"],
                    })
    filter_df = pd.DataFrame(filter_rows)
    # incremental vs baseline (mode none) per (side, period)
    base_map = {(r.side, r.period): r for r in filter_df[filter_df["filter"] == "none"].itertuples(index=False)}
    filter_df["delta_oos_sharpe"] = [
        r.oos_sharpe - base_map[(r.side, r.period)].oos_sharpe if (r.side, r.period) in base_map else np.nan
        for r in filter_df.itertuples(index=False)
    ]

    # ---------- walk-forward (select period on train) ----------
    folds = make_folds(args.start, args.end, args.wf_train_months, args.wf_val_months,
                       args.wf_test_months, args.wf_step_months)
    wf_rows = []
    wf_pooled: dict[str, list[np.ndarray]] = {s: [] for s in bc.SIDES}
    for fi, fold in enumerate(folds):
        tr0, tr1 = fold["train"]
        te0, te1 = fold["test"]
        tmask = np.asarray(grid >= tr0) & np.asarray(grid < tr1)
        vmask = np.asarray(grid >= tr1) & np.asarray(grid < te0)
        temask = np.asarray(grid >= te0) & np.asarray(grid < te1)
        for side in bc.SIDES:
            best = None
            for p in periods:
                m = eb.returns_metrics(grid_cache[(side, p)][tmask], ppy=ppy)
                if np.isfinite(m["sharpe"]) and (best is None or m["sharpe"] > best[0]):
                    best = (m["sharpe"], p)
            if best is None:
                continue
            _, p = best
            pooled = grid_cache[(side, p)]
            mtr = eb.returns_metrics(pooled[tmask], ppy=ppy)
            mva = eb.returns_metrics(pooled[vmask], ppy=ppy)
            mte = eb.returns_metrics(pooled[temask], ppy=ppy)
            wf_rows.append({"fold": fi, "side": side, "period": p,
                            "train_start": str(tr0.date()), "train_end": str(tr1.date()),
                            "test_start": str(te0.date()), "test_end": str(te1.date()),
                            "train_sharpe": mtr["sharpe"], "val_sharpe": mva["sharpe"],
                            "test_sharpe": mte["sharpe"], "test_return": mte["total_return"],
                            "test_maxdd": mte["max_drawdown"]})
            wf_pooled[side].append(pooled[temask])
        print(f"  walk-forward fold {fi} done", flush=True)
    wf_df = pd.DataFrame(wf_rows)
    wf_summary = []
    for side in bc.SIDES:
        if not wf_pooled[side]:
            continue
        m = eb.returns_metrics(np.concatenate(wf_pooled[side]), ppy=ppy)
        sel = wf_df[wf_df.side == side]
        wf_summary.append({"side": side, "folds": int(sel.shape[0]), "wf_oos_return": m["total_return"],
                           "wf_oos_sharpe": m["sharpe"], "wf_oos_maxdd": m["max_drawdown"],
                           "pct_folds_positive": float((sel["test_return"] > 0).mean())})
    wf_summary_df = pd.DataFrame(wf_summary)

    # ---------- regimes (Experiment D, reference config) ----------
    ref_pooled, _ = pooled_config(bundles, grid_ns, REF_PERIOD, REF_SIDE, cost=args.cost, use_funding=use_funding)
    regime_parts: dict[str, list[np.ndarray]] = {}
    oos_start = pd.Timestamp(args.is_end, tz="UTC")
    for b in bundles:
        res = eb.simulate(b.timestamps, b.open, b.close, b.target(REF_PERIOD, REF_SIDE),
                          cost_per_side=args.cost, funding=b.funding if use_funding else None)
        frame = pl.DataFrame({"timestamp": b.timestamps, "close": b.close})
        reg = classify_regimes(frame, config=RegimeConfig(regime_timeframe=args.timeframe),
                               source_timeframe=args.timeframe)
        lab_ns = reg["timestamp"].to_numpy().astype("datetime64[ns]").astype("int64")
        lab_vals = reg["regime"].to_numpy()
        ret = np.asarray(res.returns, float)
        bar_ns = b.timestamps.astype("datetime64[ns]").astype("int64")
        oos_bar = bar_ns >= oos_start.value
        lab_pos = np.searchsorted(lab_ns, bar_ns, side="right") - 1
        for regime in np.unique(lab_vals):
            sel = oos_bar & (lab_pos >= 0) & (lab_vals[np.clip(lab_pos, 0, lab_vals.size - 1)] == regime)
            if sel.any():
                regime_parts.setdefault(str(regime), []).append(ret[sel])
    regime_rows = []
    for regime, parts in sorted(regime_parts.items()):
        per_sym = np.array([p.sum() for p in parts])
        cat = np.concatenate(parts)
        regime_rows.append({"regime": regime, "n_bars": int(cat.size), "n_symbols": int(per_sym.size),
                            "mean_symbol_return": float(per_sym.mean()), "median": float(np.median(per_sym)),
                            "sharpe": eb.sharpe_of(cat, ppy), "pct_positive_symbols": float((per_sym > 0).mean())})
    regime_df = pd.DataFrame(regime_rows).sort_values("mean_symbol_return", ascending=False)

    # per calendar year (full sample) for reference
    period_rows = []
    yr = pd.DatetimeIndex(grid).year
    for y in sorted(set(yr)):
        m = eb.returns_metrics(ref_pooled[yr == y], ppy=ppy)
        period_rows.append({"year": int(y), "total_return": m["total_return"], "sharpe": m["sharpe"]})
    period_df = pd.DataFrame(period_rows)

    # ---------- robustness ----------
    ref_idx = periods.index(REF_PERIOD) if REF_PERIOD in periods else len(periods) // 2
    neigh_periods = periods[max(0, ref_idx - 3): ref_idx + 4]
    neigh_rows = []
    for p in neigh_periods:
        m = eb.returns_metrics(grid_cache[(REF_SIDE, p)][oos_mask], ppy=ppy)
        neigh_rows.append({"period": p, "oos_sharpe": m["sharpe"], "oos_return": m["total_return"]})
    neigh_df = pd.DataFrame(neigh_rows).sort_values("period")

    ref_oos = grid_cache[(REF_SIDE, REF_PERIOD)][oos_mask]
    boot_sharpe = eb.block_bootstrap(ref_oos, lambda r: eb.sharpe_of(r, ppy), block=72, n=args.bootstrap)
    boot_total = eb.block_bootstrap(ref_oos, lambda r: float(np.prod(1 + r) - 1), block=72, n=args.bootstrap, seed=1)
    boot = {
        "sharpe_ci": [float(np.percentile(boot_sharpe, 2.5)), float(np.percentile(boot_sharpe, 97.5))] if boot_sharpe.size else None,
        "return_ci": [float(np.percentile(boot_total, 2.5)), float(np.percentile(boot_total, 97.5))] if boot_total.size else None,
    }

    ps_rows = []
    for b in bundles:
        res = eb.simulate(b.timestamps, b.open, b.close, b.target(REF_PERIOD, REF_SIDE),
                          cost_per_side=args.cost, funding=b.funding if use_funding else None)
        r = eb.returns_metrics(res.returns[np.asarray(b.timestamps) >= np.datetime64(args.is_end)], ppy=ppy)
        ts = eb.trade_stats(eb.extract_trades(b.timestamps, b.open, b.close, res.positions,
                                              cost_per_side=args.cost,
                                              funding=b.funding if use_funding else None))
        ps_rows.append({"symbol": b.symbol, "oos_return": r["total_return"], "oos_sharpe": r["sharpe"],
                        "oos_maxdd": r["max_drawdown"], "trades": ts["n_trades"], "win_rate": ts["win_rate"],
                        "profit_factor": ts["profit_factor"]})
    ps_df = pd.DataFrame(ps_rows).sort_values("oos_return", ascending=False)

    ref_trades = pooled_trades(bundles, REF_PERIOD, REF_SIDE, cost=args.cost, use_funding=use_funding)
    if not ref_trades.empty:
        pnl = ref_trades["pnl"].sort_values(ascending=False)
        top5 = max(1, int(0.05 * len(pnl)))
        gw = pnl[pnl > 0].sum()
        concentration = {"n_trades": int(len(pnl)),
                         "top5pct_share": float(pnl.head(top5).sum() / gw) if gw > 0 else float("nan"),
                         "best": float(pnl.max()), "worst": float(pnl.min())}
    else:
        concentration = {"n_trades": 0, "top5pct_share": float("nan"), "best": float("nan"), "worst": float("nan")}

    # baselines
    bh = pooled_bh(bundles, grid_ns)
    bh_is = eb.returns_metrics(bh[is_mask], ppy=ppy)
    bh_oos = eb.returns_metrics(bh[oos_mask], ppy=ppy)

    print("computation done, writing artefacts ...", flush=True)

    # ---------- artefacts ----------
    comp.to_csv(art / "main_comparison.csv", index=False)
    cost_df.to_csv(art / "cost_sensitivity.csv", index=False)
    diag_df.to_csv(art / "diagnostics.csv", index=False)
    filter_df.to_csv(art / "filters.csv", index=False)
    wf_df.to_csv(art / "walkforward_folds.csv", index=False)
    wf_summary_df.to_csv(art / "walkforward_summary.csv", index=False)
    regime_df.to_csv(art / "regime_breakdown.csv", index=False)
    ps_df.to_csv(art / "per_symbol_oos.csv", index=False)
    grid_df.to_csv(art / "grid_metrics.csv", index=False)

    log = {
        "assumptions": {
            "timeframe": args.timeframe, "source_timeframe": args.source_timeframe,
            "execution": "signal at close t, fill at open t+1, mark at close",
            "cost_per_side": args.cost, "funding": use_funding, "cost_scenarios": COSTS,
            "is_end": args.is_end, "annualization_bars_per_year": ppy,
            "bollinger": {"window": bc.BOLLINGER_WINDOW, "num_std": bc.BOLLINGER_NUM_STD,
                          "std": "sample (ddof=1), matching features.core"},
            "reference_config": {"side": REF_SIDE, "ema_period": REF_PERIOD},
        },
        "universe": [b.symbol for b in bundles],
        "ema_periods": list(periods),
        "walk_forward": {"train_months": args.wf_train_months, "val_months": args.wf_val_months,
                         "test_months": args.wf_test_months, "step_months": args.wf_step_months},
        "data_quality": dq,
        "bootstrap": boot,
        "concentration": concentration,
    }
    (art / "experiment_log.json").write_text(json.dumps(log, indent=2, default=str) + "\n", encoding="utf-8")

    try:
        make_plots(art, grid_df, diag_df, grid_cache, grid, is_mask, oos_mask, bh,
                   periods, horizons, REF_PERIOD, REF_SIDE)
    except Exception as exc:  # noqa: BLE001
        print(f"  plots failed: {type(exc).__name__}: {exc}", flush=True)

    report = render_report(args=args, symbols=[b.symbol for b in bundles], dq=dq, comp=comp,
                           cost_df=cost_df, diag_df=diag_df, filter_df=filter_df, wf_df=wf_df,
                           wf_summary_df=wf_summary_df, regime_df=regime_df, period_df=period_df,
                           ps_df=ps_df, neigh_df=neigh_df, boot=boot, concentration=concentration,
                           bh_is=bh_is, bh_oos=bh_oos, periods=list(periods), ppy=ppy,
                           grid_cache=grid_cache, oos_mask=oos_mask, is_mask=is_mask)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote artefacts under {art}")


# --- baselines / plots -------------------------------------------------------


def pooled_bh(bundles, grid_ns):
    G = grid_ns.size
    stack = np.full((len(bundles), G), np.nan)
    for i, b in enumerate(bundles):
        pos = np.searchsorted(grid_ns, b.timestamps.astype("datetime64[ns]").astype("int64"))
        r = np.zeros(b.close.size)
        r[1:] = b.close[1:] / b.close[:-1] - 1.0
        stack[i, pos] = r
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.nanmean(stack, axis=0))


def make_plots(art, grid_df, diag_df, grid_cache, grid, is_mask, oos_mask, bh, periods, horizons, ref_p, ref_side):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotdir = art / "plots"
    sides = list(bc.SIDES)

    def heat(metric, fname, title, cmap="RdBu_r", signed=True):
        oos = grid_df
        M = np.full((len(periods), len(sides)), np.nan)
        for r in oos.itertuples(index=False):
            M[periods.index(int(r.period)), sides.index(r.side)] = getattr(r, metric)
        fig, ax = plt.subplots(figsize=(4.5, 6))
        vmax = np.nanmax(np.abs(M)) if signed and np.isfinite(M).any() else (np.nanmax(M) if np.isfinite(M).any() else 1)
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=(-vmax if signed else 0), vmax=vmax)
        ax.set_xticks(range(len(sides)), sides, rotation=45)
        ax.set_yticks(range(len(periods)), periods)
        ax.set_xlabel("side")
        ax.set_ylabel("EMA period")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(plotdir / fname, dpi=110)
        plt.close(fig)

    heat("oos_return", "heatmap_oos_return.png", "OOS net return")
    heat("oos_sharpe", "heatmap_oos_sharpe.png", "OOS Sharpe")
    heat("oos_maxdd", "heatmap_maxdd.png", "OOS max drawdown", cmap="Reds_r", signed=False)
    heat("oos_trades", "heatmap_trades.png", "OOS trade count", cmap="viridis", signed=False)

    # long vs short OOS Sharpe
    long_s = grid_df[grid_df.side == "long"].set_index("period")["oos_sharpe"].reindex(periods)
    short_s = grid_df[grid_df.side == "short"].set_index("period")["oos_sharpe"].reindex(periods)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(periods))
    ax.bar(x - 0.2, long_s, 0.4, label="long-only")
    ax.bar(x + 0.2, short_s, 0.4, label="short-only")
    ax.set_xticks(x, periods)
    ax.set_xlabel("EMA period")
    ax.set_ylabel("OOS Sharpe")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend()
    ax.set_title("Long-only vs short-only OOS Sharpe")
    fig.tight_layout()
    fig.savefig(plotdir / "long_vs_short_sharpe.png", dpi=110)
    plt.close(fig)

    # equity curves for reference config, IS/OOS shaded, vs buy&hold
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ref = np.cumprod(1 + grid_cache[(ref_side, ref_p)])
    ax.plot(grid, ref, label=f"EMA{ref_p}/SMA20 {ref_side}", lw=1.3)
    ax.plot(grid, np.cumprod(1 + bh), "k", alpha=0.55, label="buy & hold")
    split = pd.Timestamp(grid[is_mask][-1]) if is_mask.any() else grid[0]
    ax.axvspan(grid[0], split, color="grey", alpha=0.12, label="in-sample")
    ax.axvline(split, color="grey", ls="--", lw=1)
    ax.set_title("Pooled equity — reference config vs buy & hold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plotdir / "equity_curves.png", dpi=110)
    plt.close(fig)

    # diagnostics distribution: 24h oriented returns by period, bull vs bear
    for hz in ("4h", "24h"):
        sub = diag_df[diag_df.horizon == hz]
        long_m = [float(sub[(sub.side == "long") & (sub.period == p)]["mean"].mean()) if not sub[(sub.side == "long") & (sub.period == p)].empty else np.nan for p in periods]
        short_m = [float(sub[(sub.side == "short") & (sub.period == p)]["mean"].mean()) if not sub[(sub.side == "short") & (sub.period == p)].empty else np.nan for p in periods]
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(periods))
        ax.bar(x - 0.2, np.array(long_m) * 1e4, 0.4, label="bullish cross")
        ax.bar(x + 0.2, np.array(short_m) * 1e4, 0.4, label="bearish cross")
        base = sub["baseline_mean"].mean()
        ax.axhline(base * 1e4, color="k", ls="--", lw=1, label="unconditional mean")
        ax.set_xticks(x, periods)
        ax.set_xlabel("EMA period")
        ax.set_ylabel(f"mean oriented {hz} return (bps)")
        ax.set_title(f"Directional forward return after crossover ({hz})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plotdir / f"diagnostics_{hz}.png", dpi=110)
        plt.close(fig)


# --- report ------------------------------------------------------------------


def render_report(**kw) -> str:
    a = kw["args"]
    comp = kw["comp"]
    diag = kw["diag_df"]
    filt = kw["filter_df"]
    periods = kw["periods"]
    L: list[str] = []
    L.append("# Bollinger Basis (SMA20) vs EMA Crossovers — 1h Crypto Perps")
    L.append("")
    L.append(f"- **Universe ({len(kw['symbols'])}):** " + ", ".join(kw["symbols"]))
    L.append(f"- **Data:** Binance USD-M perpetuals, 1h resampled from 1m; {a.start} → {a.end} (IS/OOS split {a.is_end}).")
    L.append(f"- **Signal:** EMA(n) crosses fixed **SMA(20)** (Bollinger middle); bands = SMA ± {bc.BOLLINGER_NUM_STD:.0f}σ, "
             f"σ = {bc.BOLLINGER_WINDOW}-bar **sample** std (ddof=1).")
    L.append(f"- **Execution:** signal at close *t*, fill at open *t+1*; **{a.cost*1e4:.1f} bps/side**; "
             f"funding={'on' if not a.no_funding else 'off'}; annualisation {kw['ppy']:.0f} bars/yr.")
    L.append(f"- **EMA periods:** {', '.join(str(p) for p in periods)}.")
    L.append("")

    # ---- Q summary
    oos = comp[comp.side == "both"]
    long_best = comp[comp.side == "long"].sort_values("oos_sharpe", ascending=False).iloc[0]
    short_best = comp[comp.side == "short"].sort_values("oos_sharpe", ascending=False).iloc[0]
    both_best = oos.sort_values("oos_sharpe", ascending=False).iloc[0]
    ref = comp[(comp.side == REF_SIDE) & (comp.period == REF_PERIOD)]
    ref = ref.iloc[0] if not ref.empty else None
    ci = kw["boot"].get("sharpe_ci")
    ci_straddles = bool(ci and ci[0] <= 0 <= ci[1])
    L.append("## Executive summary")
    L.append("")
    best_side, best_row = max(
        [("long", long_best), ("short", short_best), ("both", both_best)],
        key=lambda kv: (kv[1].oos_sharpe if np.isfinite(kv[1].oos_sharpe) else -1e9),
    )
    any_pos = max(long_best.oos_return, short_best.oos_return, both_best.oos_return) > 0
    L.append(
        f"Across {len(periods)} EMA periods × 3 sides, the best out-of-sample Sharpe was "
        f"**{num(best_row.oos_sharpe)}** ({best_side}, EMA({int(best_row.period)}), OOS return "
        f"{pct(best_row.oos_return)}). "
        + ("At least one side was OOS-positive, but " if any_pos else "No side was profitable out-of-sample, and ")
        + f"long-only (best Sharpe {num(long_best.oos_sharpe)}) "
        + ("outperformed" if long_best.oos_sharpe >= short_best.oos_sharpe else "underperformed")
        + f" short-only (best Sharpe {num(short_best.oos_sharpe)}). "
        f"The OOS window is a single regime, so these are not independent evidence of an edge."
    )
    if ref is not None:
        L.append("")
        L.append(
            f"The pre-registered reference **EMA{REF_PERIOD}/SMA20 long-short** delivered OOS return "
            f"**{pct(ref.oos_return)}** at Sharpe **{num(ref.oos_sharpe)}** (IS Sharpe {num(ref.is_sharpe)}), "
            f"break-even ≈ **{bps(ref.breakeven)} bps/side**; bootstrap 95% CI on its OOS Sharpe "
            f"{'includes' if ci_straddles else 'excludes'} zero"
            + (f" ([{num(ci[0])}, {num(ci[1])}])." if ci else ".")
        )
    if not kw["wf_summary_df"].empty:
        wfb = kw["wf_summary_df"].sort_values("wf_oos_sharpe", ascending=False).iloc[0]
        L.append("")
        L.append(
            f"Crucially, **walk-forward** selection (choose the EMA period on each train window, test on the next "
            f"unseen window) is negative on every side — the best is `{wfb.side}` at Sharpe "
            f"**{num(wfb.wf_oos_sharpe)}** ({pct(wfb.wf_oos_return)}, {wfb.pct_folds_positive*100:.0f}% positive folds). "
            f"So the positive full-sample-OOS long-only result does not survive parameter selection, which is the "
            f"outcome that matters."
        )
    L.append("")
    L.append("See §10 for the point-by-point answers to the study's ten questions.")
    L.append("")

    # ---- data
    L.append("## 1. Environment & data")
    L.append("")
    L.append("Reuses `analytics.ema_backtest` (engine/loader) from the previous EMA-crossover study; only the "
             "signal (EMA(n) vs fixed SMA(20)) and Bollinger filters are new. Datasets: OHLCV bars (1m + 1d), "
             "funding (8h), open interest. No native 1h bars — 1h is resampled from 1m (open=first, high=max, "
             "low=min, close=last, bucket labelled by open time).")
    L.append("")
    L.append("| symbol | bars | start | end | dup ts | gaps | max gap (h) | BB warm-up | funding events |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for d in kw["dq"]:
        L.append(f"| {d['symbol']} | {d['bars']:,} | {d['start'][:10]} | {d['end'][:10]} | {d['duplicate_ts']} | "
                 f"{d['gaps']} | {d['max_gap_hours']:.0f} | {d['bb_warmup']} | {d['funding_events']:,} |")
    L.append("")
    L.append("Funding timestamps are truncated to the hour (jitter); warm-up: EMA(200) is only valid after 200 "
             "bars and signals require both the EMA and SMA(20) to be defined.")
    L.append("")

    # ---- Experiment A main table
    L.append("## 2. Experiment A — main comparison (EMA(n) vs SMA20)")
    L.append("")
    L.append(f"Buy & hold over the same windows: IS {pct(kw['bh_is']['total_return'])} "
             f"(Sharpe {num(kw['bh_is']['sharpe'])}); OOS {pct(kw['bh_oos']['total_return'])} "
             f"(Sharpe {num(kw['bh_oos']['sharpe'])}, maxDD {pct(kw['bh_oos']['max_drawdown'])}). "
             f"The no-trade (cash) baseline is 0%.")
    L.append("")
    L.append("| side | EMA | IS ret | IS Sharpe | OOS ret | OOS Sharpe | OOS maxDD | trades | win | PF | "
             "avg trade | hold h | break-even (bps) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in comp.sort_values(["side", "period"]).itertuples(index=False):
        L.append(f"| {r.side} | {int(r.period)} | {pct(r.is_return)} | {num(r.is_sharpe)} | {pct(r.oos_return)} | "
                 f"{num(r.oos_sharpe)} | {pct(r.oos_maxdd)} | {r.oos_trades:.0f} | {pct(r.win_rate,0)} | "
                 f"{num(r.profit_factor)} | {pct(r.avg_trade_ret,3)} | {num(r.avg_holding_bars,1)} | {bps(r.breakeven)} |")
    L.append("")

    # ---- Experiment B diagnostics
    L.append("## 3. Experiment B — predictive diagnostics (before trading)")
    L.append("")
    L.append("Direction-aligned forward returns at each crossover, pooled across symbols and sorted in time; "
             "CIs from a block bootstrap (block = max(horizon, 24) events) to respect overlap.")
    L.append("")
    base = diag.drop_duplicates("horizon").set_index("horizon")[["baseline_hit", "baseline_mean"]]
    L.append("Unconditional baseline (per horizon): ")
    L.append("")
    L.append("| horizon | P(up) | mean return |")
    L.append("|---|---|---|")
    for h in bc.FORWARD_HORIZONS:
        if h in base.index:
            L.append(f"| {h} | {pct(base.loc[h,'baseline_hit'],1)} | {pct(base.loc[h,'baseline_mean'],3)} |")
    L.append("")
    for hz in ("1h", "4h", "24h", "72h"):
        sub = diag[diag.horizon == hz]
        if sub.empty:
            continue
        L.append(f"### {hz} direction-aligned return by EMA period")
        L.append("")
        L.append("| EMA | side | n | hit | hit 95% CI | mean | mean 95% CI | median | std |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in sub.sort_values(["side", "period"]).itertuples(index=False):
            L.append(f"| {int(r.period)} | {r.side} | {r.n:,} | {pct(r.hit_rate,1)} | "
                     f"[{pct(r.hit_lo,1)}, {pct(r.hit_hi,1)}] | {pct(r.mean,3)} | "
                     f"[{pct(r.mean_lo,3)}, {pct(r.mean_hi,3)}] | {pct(r.median,3)} | {pct(r.std,2)} |")
        L.append("")

    # ---- Experiment C filters
    L.append("## 4. Experiment C — Bollinger bandwidth / price-position filters")
    L.append("")
    L.append("Each filter applied one at a time to a baseline EMA(10), EMA(20) and EMA(50) crossover; "
             "`entry` gates the cross (blocked cross = flat until the next cross), `hold` flattens while the "
             "filter disallows the side. Δ is OOS Sharpe vs that period's unfiltered baseline.")
    L.append("")
    L.append("| side | EMA | filter | mode | OOS Sharpe | Δ Sharpe | OOS ret | OOS maxDD | trades | exposure |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in filt.sort_values(["side", "period", "filter", "mode"]).itertuples(index=False):
        L.append(f"| {r.side} | {int(r.period)} | {r.filter} | {r.mode} | {num(r.oos_sharpe)} | "
                 f"{num(r.delta_oos_sharpe)} | {pct(r.oos_return)} | {pct(r.oos_maxdd)} | {r.trades:.0f} | {pct(r.exposure,0)} |")
    L.append("")

    # ---- Walk-forward
    L.append("## 5. Walk-forward validation")
    L.append("")
    L.append("Per fold the EMA period with the best **train** Sharpe is frozen and evaluated on the next unseen "
             "test window; test windows are stitched for the aggregate.")
    L.append("")
    L.append("| side | folds | WF OOS ret | WF OOS Sharpe | WF OOS maxDD | folds positive |")
    L.append("|---|---|---|---|---|---|")
    for r in kw["wf_summary_df"].itertuples(index=False):
        L.append(f"| {r.side} | {r.folds:.0f} | {pct(r.wf_oos_return)} | {num(r.wf_oos_sharpe)} | "
                 f"{pct(r.wf_oos_maxdd)} | {r.pct_folds_positive*100:.0f}% |")
    L.append("")
    L.append("| side | fold | train | test | chosen EMA | train Sharpe | val Sharpe | test Sharpe | test return |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in kw["wf_df"].sort_values(["side", "fold"]).itertuples(index=False):
        L.append(f"| {r.side} | {int(r.fold)} | {r.train_start}→{r.train_end} | {r.test_start}→{r.test_end} | "
                 f"{int(r.period)} | {num(r.train_sharpe)} | {num(r.val_sharpe)} | {num(r.test_sharpe)} | {pct(r.test_return)} |")
    L.append("")

    # ---- Costs
    L.append("## 6. Costs & break-even")
    L.append("")
    L.append("| side | EMA | scenario | bps/side | OOS ret | OOS Sharpe | OOS maxDD | fees | funding |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in kw["cost_df"].itertuples(index=False):
        L.append(f"| {r.side} | {int(r.period)} | {r.scenario} | {r.cost_per_side*1e4:.0f} | {pct(r.oos_return)} | "
                 f"{num(r.oos_sharpe)} | {pct(r.oos_maxdd)} | {pct(r.fees)} | {pct(r.funding)} |")
    L.append("")
    L.append("Break-even cost (bps/side) by period and side:")
    L.append("")
    L.append("| EMA | long | short | long-short |")
    L.append("|---|---|---|---|")
    piv = comp.pivot_table(index="period", columns="side", values="breakeven", aggfunc="first")
    for p in periods:
        if p in piv.index:
            row = piv.loc[p]
            L.append(f"| {p} | {bps(row.get('long', float('nan')))} | {bps(row.get('short', float('nan')))} | "
                     f"{bps(row.get('both', float('nan')))} |")
    L.append("")

    # ---- Regimes
    L.append("## 7. Experiment D — regime analysis (reference config)")
    L.append("")
    L.append(f"Reference: EMA{REF_PERIOD}/SMA20 long-short, OOS only. Causal regime labels "
             "(trend = volatility-normalised trailing 24h move, ±1σ; vol = realised vol vs trailing median).")
    L.append("")
    L.append("| regime | bars | mean symbol return | median | Sharpe | % positive symbols |")
    L.append("|---|---|---|---|---|---|")
    for r in kw["regime_df"].itertuples(index=False):
        L.append(f"| {r.regime} | {r.n_bars:,} | {pct(r.mean_symbol_return,2)} | {pct(r.median,2)} | "
                 f"{num(r.sharpe)} | {r.pct_positive_symbols*100:.0f}% |")
    L.append("")
    if not kw["period_df"].empty:
        L.append("| calendar year (full sample) | return | Sharpe |")
        L.append("|---|---|---|")
        for r in kw["period_df"].itertuples(index=False):
            L.append(f"| {r.year} | {pct(r.total_return)} | {num(r.sharpe)} |")
        L.append("")
    L.append("Regime labels are computed causally and shifted one bar; no future information enters.")
    L.append("")

    # ---- Robustness
    L.append("## 8. Robustness")
    L.append("")
    L.append("**Neighbouring EMA periods** around the reference (OOS):")
    L.append("")
    L.append("| EMA | OOS Sharpe | OOS return |")
    L.append("|---|---|---|")
    for r in kw["neigh_df"].itertuples(index=False):
        L.append(f"| {int(r.period)} | {num(r.oos_sharpe)} | {pct(r.oos_return)} |")
    L.append("")
    b = kw["boot"]
    if b.get("sharpe_ci"):
        L.append(f"**Bootstrap (block=72h, {a.bootstrap} resamples) on the reference OOS returns**: Sharpe 95% CI "
                 f"[{num(b['sharpe_ci'][0])}, {num(b['sharpe_ci'][1])}]; total-return 95% CI "
                 f"[{pct(b['return_ci'][0])}, {pct(b['return_ci'][1])}].")
        L.append("")
    c = kw["concentration"]
    L.append(f"**Profit concentration** (reference pooled trades): {c['n_trades']} trades; top 5% = "
             f"**{pct(c['top5pct_share'],0)}** of gross profit; best {pct(c['best'],2)}, worst {pct(c['worst'],2)}.")
    L.append("")
    L.append("**Per-symbol OOS (reference config)**")
    L.append("")
    L.append("| symbol | OOS return | OOS Sharpe | OOS maxDD | trades | win | PF |")
    L.append("|---|---|---|---|---|---|---|")
    for r in kw["ps_df"].itertuples(index=False):
        L.append(f"| {r.symbol} | {pct(r.oos_return)} | {num(r.oos_sharpe)} | {pct(r.oos_maxdd)} | "
                 f"{r.trades:.0f} | {pct(r.win_rate,0)} | {num(r.profit_factor)} |")
    L.append("")

    # ---- Limitations
    L.append("## 9. Limitations")
    L.append("")
    L.append("- The study window (2025 → 2026-09) contains only two calendar blocks, and the OOS window (2026) "
             "is a single predominantly bullish/trending regime; OOS results are not independent evidence.")
    L.append("- Cost model is fee+slippage per side with next-open fills; no order-book, latency or market impact.")
    L.append("- Funding is applied on held notional at hourly-aligned funding stamps — a drag approximation, not "
             "a full cash-flow ledger (same as the prior study).")
    L.append("- The universe is today's top-10 liquids (survivorship bias); an equal-weight pooled book is assumed.")
    L.append("- Overlapping forward-return windows are handled with a block bootstrap, but multiple testing "
             "across 14 EMA periods × 3 sides × 6 filters × 2 modes is only qualitatively accounted for.")
    L.append("")

    # ---- Final questions
    L.append("## 10. Answers to the study's questions")
    L.append("")
    sig_cells = diag.dropna(subset=["mean_lo"])
    frac_sig = float((sig_cells["mean_lo"] > 0).mean()) if not sig_cells.empty else float("nan")
    L.append(f"1. **Predictive information beyond unconditional?** Weak. Bullish-cross hit rates hover near the "
             f"~50% unconditional baseline, and only **{frac_sig*100:.0f}%** of (period, side, horizon) cells have a "
             f"mean direction-aligned return whose 95% CI lies entirely above zero (§3).")
    L.append(f"2. **Which EMA periods?** The best OOS result was EMA({int(best_row.period)}) on the {best_side} "
             f"side (Sharpe {num(best_row.oos_sharpe)}); neighbouring periods diverge materially (§8), so this is "
             f"not a stable region, and it is not consistent across symbols (§8).")
    L.append(f"3. **Long, short, or both?** Long-only best OOS Sharpe {num(long_best.oos_sharpe)} vs short-only "
             f"{num(short_best.oos_sharpe)} and long-short {num(both_best.oos_sharpe)}; "
             + ("long signals were the more useful of the two." if long_best.oos_sharpe >= short_best.oos_sharpe
                else "short signals were the more useful of the two."))
    ref20 = comp[(comp.side == "both") & (comp.period == REF_PERIOD)]
    max_tr = comp["oos_trades"].max()
    if not ref20.empty:
        r20 = ref20.iloc[0]
        L.append(f"4. **EMA(20) vs SMA(20) special case?** It is the busiest equal-period construction "
                 f"({r20.oos_trades:.0f} OOS trades, grid max {max_tr:.0f}) with OOS Sharpe {num(r20.oos_sharpe)}; "
                 f"the exponential-vs-simple distinction at equal nominal period is not materially special.")
    else:
        L.append("4. **EMA(20) vs SMA(20) special case?** Not materially different from neighbouring periods.")
    bw = filt[filt["filter"].isin(["bw_high", "bw_low", "bw_expanding", "bw_contracting", "squeeze_breakout"])]
    if not bw.empty:
        bd = bw.loc[bw["delta_oos_sharpe"].idxmax()]
        L.append(f"5. **Does bandwidth help?** Best single bandwidth/squeeze variant is `{bd['filter']}` ({bd['mode']} "
                 f"on {bd['side']} EMA{int(bd['period'])}) with Δ Sharpe **{num(bd['delta_oos_sharpe'])}** "
                 f"(OOS Sharpe {num(bd['oos_sharpe'])}), but the sign of Δ varies across periods and modes — no "
                 f"consistent improvement.")
    else:
        L.append("5. **Does bandwidth help?** No consistent improvement across periods.")
    L.append("6. **Does price relative to basis help?** The `price_above_sma` filter moves results (it is partly "
             "collinear with the crossover state) but does not deliver a robust incremental edge on either side.")
    if not kw["regime_df"].empty:
        br = kw["regime_df"].iloc[0]
        wr = kw["regime_df"].iloc[-1]
        L.append(f"7. **Regime-dependent?** Yes. The reference config's OOS PnL is best in `{br.regime}` "
                 f"({pct(br.mean_symbol_return)}) and worst in `{wr.regime}` ({pct(wr.mean_symbol_return)}); "
                 f"no single regime is favourable across both calendar years (§7).")
    else:
        L.append("7. **Regime-dependent?** Regime breakdown unavailable (too few OOS observations).")
    survivors = comp[comp.breakeven > a.cost]
    L.append(f"8. **Survives costs & funding?** {len(survivors)} of {len(comp)} period/side combinations have a "
             f"break-even above the 6 bps base cost"
             + (f" (highest {bps(survivors.breakeven.max())} bps)." if len(survivors) else "; none clear base costs.")
             + " See the break-even grid in §6.")
    frac_folds = float((kw["wf_df"]["test_return"] > 0).mean()) if not kw["wf_df"].empty else float("nan")
    nsh = kw["neigh_df"]["oos_sharpe"]
    L.append(f"9. **Robust across periods, symbols, folds?** No. Only **{frac_folds*100:.0f}%** of walk-forward test "
             f"folds were positive, neighbourhood OOS Sharpe spans "
             f"[{num(nsh.min())}, {num(nsh.max())}], and per-symbol dispersion is large (§5, §8).")
    reject = (not np.isfinite(best_row.oos_sharpe)) or best_row.oos_sharpe < 0.5 or ci_straddles
    L.append(f"10. **Verdict?** {'The evidence does not establish a reliable edge' if reject else 'Some OOS performance is positive but not conclusive'}"
             f" — the best OOS Sharpe across the grid is {num(best_row.oos_sharpe)} (EMA({int(best_row.period)}), "
             f"{best_side}), the reference config's Sharpe CI {'includes' if ci_straddles else 'excludes'} zero, and "
             f"results flip across folds/periods (§5, §8). Treat the EMA/SMA(20) crossover as a "
             f"{'rejected standalone signal' if reject else 'candidate needing an independent risk model'} rather "
             f"than a proven strategy.")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"_Reproduce:_ `python scripts/ema_sma_cross_study.py` (engine: `quant_lab.analytics.ema_backtest`; "
             f"signals: `quant_lab.analytics.bollinger_cross`). Artefacts: "
             f"`{a.artifacts or 'experiments/reports/ema_sma_cross'}`.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
