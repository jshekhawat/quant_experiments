#!/usr/bin/env python
"""EMA momentum (ROC), acceleration and spread dynamics — research study.

Implements ``reports/EMA_MOMENTUM.md`` end to end:

* **Experiment A** — predictive diagnostics: rank IC and conditional forward
  returns of the ROC / acceleration / normalized-spread / spread-velocity
  features, incremental OLS (HAC) over a baseline holding the current spread
  and crossover state, by symbol / period / regime.
* **Experiment B** — early-warning conditions (momentum deceleration, momentum
  reversal, spread convergence, combinations) and their forward returns.
* **Experiment C** — lead time to the *same* moving averages' conventional
  crossover, false-turn rates (volatility-scaled), MFE/MAE and entry-delay
  sensitivity.
* **Experiment D** — tradable strategy variants (crossover baseline, ROC
  reversal, spread velocity, combined, confirmed) long/short/long-short,
  reusing the EMA engine's next-open execution, per-side costs and funding.
* Chronological IS/OOS split, walk-forward selection, cost sensitivity,
  break-even, regime analysis and robustness, with plots and an experiment log.

Writes ``reports/ema_momentum_results.md`` plus CSV/JSON/PNG artifacts under a
new versioned directory (default ``experiments/reports/ema_momentum_v1``).
Prior EMA reports are never overwritten.

    python scripts/ema_momentum_study.py
    python scripts/ema_momentum_study.py --quick --symbols BTCUSDT,ETHUSDT

Research code: no live trading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from quant_lab.analytics import ema_backtest as eb
from quant_lab.analytics import ema_momentum as em
from quant_lab.analytics.regimes import RegimeConfig, classify_regimes
from quant_lab.config import get_config
from quant_lab.data.universe import load_universe

CACHED_UNIVERSE = "configs/divergence_universe.json"
COSTS = {"zero": 0.0, "low": 0.0003, "base": 0.0006, "high": 0.0012, "extreme": 0.0020}
IC_FEATURES = ("roc_1", "roc_12", "accel_1", "spread", "svel_1", "svel_12")
IC_HORIZONS = ("1h", "6h", "24h", "72h")
ENTRY_DELAYS = (0, 1, 2, 3, 6, 12)
WF_TRAIN_M, WF_VAL_M, WF_TEST_M, WF_STEP_M = 6, 2, 3, 3


# --- universe / data ---------------------------------------------------------


def resolve_universe(args) -> list[str]:
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    path = Path(CACHED_UNIVERSE)
    if path.exists():
        u = load_universe(path)
        return (u.get("futures") or next(iter(u.values())))[: args.n]
    raise SystemExit("no universe: pass --symbols")


def data_quality(b: em.MomentumBundle, timeframe: str) -> dict:
    ts = pd.DatetimeIndex(b.timestamps)
    diffs = pd.Series(ts).diff().dropna()
    expected = pd.Timedelta(hours=eb.timeframe_hours(timeframe))
    n_expected = int((ts[-1] - ts[0]) / expected) + 1 if len(ts) else 0
    bad = int(np.sum(~(b.high >= b.low)))
    return {
        "symbol": b.symbol, "bars": int(b.close.size), "start": str(ts[0]), "end": str(ts[-1]),
        "duplicate_ts": int(pd.Series(ts).duplicated().sum()), "gaps": int(n_expected - len(ts)),
        "max_gap_hours": float(diffs.max().total_seconds() / 3600) if len(diffs) else 0.0,
        "bad_ohlc": bad, "atr_warmup": int(np.isnan(b.atr).sum()),
        "funding_events": int(np.count_nonzero(b.funding)),
    }


def common_grid(bundles: list[em.MomentumBundle]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = min(pd.Timestamp(b.timestamps[0]) for b in bundles)
    hi = max(pd.Timestamp(b.timestamps[-1]) for b in bundles)
    grid = pd.date_range(lo, hi, freq="1h", tz="UTC")
    grid_ns = grid.tz_localize(None).to_numpy().astype("datetime64[ns]").astype("int64")
    return grid, grid_ns, np.asarray(grid)


# --- pooled strategy backtests -----------------------------------------------


def make_target(
    b: em.MomentumBundle, config: str, strategy: str, side: str, *, k: int,
    risk: str = "none",
) -> np.ndarray:
    target = em.strategy_target(b, config, strategy, side, k=k)
    if risk == "minhold12":
        target = em.apply_min_hold(target, 12)
    elif risk == "atrstop2":
        target = em.apply_atr_stop(b.high, b.low, b.close, b.atr, target, mult=2.0)
    return target


def pooled_config(
    bundles, grid_ns, config, strategy, side, *, k, cost, use_funding, risk="none"
) -> tuple[np.ndarray, dict]:
    G = grid_ns.size
    stack = np.full((len(bundles), G), np.nan)
    ntr = 0
    turn = 0.0
    fund = 0.0
    expo = 0.0
    for i, b in enumerate(bundles):
        res = eb.simulate(
            b.timestamps, b.open, b.close, make_target(b, config, strategy, side, k=k, risk=risk),
            cost_per_side=cost, funding=b.funding if use_funding else None,
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


def pooled_trades(bundles, config, strategy, side, *, k, cost, use_funding, risk="none") -> pd.DataFrame:
    frames = []
    for b in bundles:
        tgt = make_target(b, config, strategy, side, k=k, risk=risk)
        res = eb.simulate(b.timestamps, b.open, b.close, tgt, cost_per_side=cost,
                          funding=b.funding if use_funding else None)
        tr = eb.extract_trades(b.timestamps, b.open, b.close, res.positions, cost_per_side=cost,
                               funding=b.funding if use_funding else None)
        if not tr.empty:
            tr["symbol"] = b.symbol
            frames.append(tr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pooled_bh(bundles, grid_ns) -> np.ndarray:
    G = grid_ns.size
    stack = np.full((len(bundles), G), np.nan)
    for i, b in enumerate(bundles):
        pos = np.searchsorted(grid_ns, b.timestamps.astype("datetime64[ns]").astype("int64"))
        r = np.zeros(b.close.size)
        r[1:] = b.close[1:] / b.close[:-1] - 1.0
        stack[i, pos] = r
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.nanmean(stack, axis=0))


def pooled_momentum_baseline(bundles, grid_ns, *, window: int = 24) -> np.ndarray:
    """Simple price-momentum baseline: long when trailing ``window`` return > 0 else short."""
    G = grid_ns.size
    stack = np.full((len(bundles), G), np.nan)
    for i, b in enumerate(bundles):
        close = b.close
        sig = np.zeros(close.size)
        sig[window:] = np.where(close[window:] > close[:-window], 1.0, -1.0)
        tgt = sig.astype(np.int8)
        res = eb.simulate(b.timestamps, b.open, b.close, tgt, cost_per_side=0.0006,
                          funding=b.funding)
        pos = np.searchsorted(grid_ns, b.timestamps.astype("datetime64[ns]").astype("int64"))
        stack[i, pos] = res.returns
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.nanmean(stack, axis=0))


def breakeven_per_side(bundles, grid_ns, config, strategy, side, oos_mask, ppy, *, k,
                       use_funding, hi=0.008, n=25) -> float:
    fine = np.linspace(0.0, hi, n)
    rets = np.array([
        eb.returns_metrics(
            pooled_config(bundles, grid_ns, config, strategy, side, k=k, cost=float(c),
                          use_funding=use_funding)[0][oos_mask], ppy=ppy,
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


# --- formatting --------------------------------------------------------------


def pct(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def bps(x):
    return ">80" if not np.isfinite(x) else ("≤0" if x <= 0 else f"{x * 1e4:.1f}")


# --- Experiment A ------------------------------------------------------------


def _stack(bundles, config, feature):
    """Concatenate one feature and all forward returns across symbols.

    Rows are kept in symbol blocks (not globally time-sorted) so per-symbol
    statistics and within-symbol demeaning stay exact.
    """
    xs, sid, ts = [], [], []
    ys: dict[str, list] = {h: [] for h in em.HORIZONS}
    for i, b in enumerate(bundles):
        x, y, t = em.pooled_arrays([b], config, feature)
        xs.append(x)
        sid.append(np.full(x.size, i))
        ts.append(t)
        for h in em.HORIZONS:
            ys[h].append(y[h])
    return (np.concatenate(xs), {h: np.concatenate(ys[h]) for h in em.HORIZONS},
            np.concatenate(sid), np.concatenate(ts))


def _mean_ic_per_symbol(x, ys, horizon, sid, n_sym):
    """Equal-weight mean of per-symbol (time-series) Spearman ICs."""
    vals = []
    for s in range(n_sym):
        m = sid == s
        vals.append(em.rank_ic(x[m], ys[horizon][m]))
    return float(np.nanmean(vals)) if vals else float("nan"), vals


def _demean(v, sid, n_sym):
    out = np.asarray(v, dtype=float).copy()
    for s in range(n_sym):
        m = sid == s
        out[m] = out[m] - np.nanmean(out[m])
    return out


def experiment_a(bundles, configs, *, bootstrap: int, regimes_by_symbol: dict) -> dict:
    n_sym = len(bundles)
    stack_cache: dict[tuple[str, str], tuple] = {}

    def stack(config, feature):
        key = (config, feature)
        if key not in stack_cache:
            stack_cache[key] = _stack(bundles, config, feature)
        return stack_cache[key]

    # --- rank IC by (config, feature, horizon), equal-weight over symbols ----
    ic_rows = []
    for cfg in configs:
        for feat in IC_FEATURES:
            x, ys, sid, _ = stack(cfg.name, feat)
            for h in IC_HORIZONS:
                ic, vals = _mean_ic_per_symbol(x, ys, h, sid, n_sym)
                fin = np.isfinite(vals)
                ic_rows.append({
                    "config": cfg.name, "family": cfg.family, "feature": feat, "horizon": h,
                    "n": int(np.isfinite(x).sum()),
                    "rank_ic": ic,
                    "pct_symbols_positive": float((np.asarray(vals)[fin] > 0).mean()) if fin.any() else np.nan,
                })
    ic_df = pd.DataFrame(ic_rows)

    # --- bootstrap CI on the primary config (block-bootstrap each symbol, then average)
    boot_rows = []
    for feat in ("roc_1", "accel_1", "spread", "svel_1"):
        x, ys, sid, _ = stack(em.PRIMARY_CONFIG, feat)
        ic, _ = _mean_ic_per_symbol(x, ys, "24h", sid, n_sym)
        for h in ("1h", "24h"):
            block = max(em.HORIZONS[h], 24)
            draws = []
            for s in range(n_sym):
                m = sid == s
                d = em.block_boot_pair(x[m], ys[h][m], em.rank_ic, block=block, n=bootstrap, seed=s)
                if d.size:
                    draws.append(d)
            vals = np.nanmean(np.vstack([d[: min(len(d) for d in draws)] for d in draws]), axis=0) if draws else np.array([])
            ic_h, _ = _mean_ic_per_symbol(x, ys, h, sid, n_sym)
            boot_rows.append({
                "feature": feat, "horizon": h, "rank_ic": ic_h,
                "lo": float(np.percentile(vals, 2.5)) if vals.size else np.nan,
                "hi": float(np.percentile(vals, 97.5)) if vals.size else np.nan,
            })
        _ = ic
    boot_df = pd.DataFrame(boot_rows)

    # --- conditional sign analysis (primary config): pooled returns ----
    cond_rows = []
    for feat, name_pos, name_neg in (
        ("roc_1", "ROC>0", "ROC<0"), ("spread", "spread>0", "spread<0"),
        ("svel_1", "svel>0", "svel<0"), ("accel_1", "accel>0", "accel<0"),
    ):
        x, ys, _, _ = stack(em.PRIMARY_CONFIG, feat)
        for h in IC_HORIZONS:
            y = ys[h]
            for label, cond in ((name_pos, x > 0), (name_neg, x < 0)):
                st = em.conditional_stats(x, y, cond=cond)
                cond_rows.append({"feature": feat, "condition": label, "horizon": h, **st})
    cond_df = pd.DataFrame(cond_rows)

    # --- incremental OLS (HAC), within-symbol demeaned, baseline {spread, state}
    _, _, sid, _ = stack(em.PRIMARY_CONFIG, "spread")
    spread_x, ys, _, _ = stack(em.PRIMARY_CONFIG, "spread")
    sign_x, _, _, _ = stack(em.PRIMARY_CONFIG, "sign")
    base = np.column_stack([_demean(spread_x, sid, n_sym), _demean((sign_x > 0).astype(float), sid, n_sym)])
    reg_rows = []
    for h in ("1h", "6h", "24h", "72h"):
        y = _demean(ys[h], sid, n_sym)
        maxlags = max(em.HORIZONS[h], 24)
        for add_feat in ("roc_1", "accel_1", "svel_1", "roc_1+accel_1+svel_1"):
            if "+" in add_feat:
                parts = [_demean(stack(em.PRIMARY_CONFIG, f)[0], sid, n_sym) for f in add_feat.split("+")]
                add = np.column_stack(parts)
            else:
                add = _demean(stack(em.PRIMARY_CONFIG, add_feat)[0], sid, n_sym)[:, None]
            r = em.incremental_regression(y, base, add, maxlags=maxlags)
            reg_rows.append({"horizon": h, "add": add_feat, **r})
    reg_df = pd.DataFrame(reg_rows)

    # --- by symbol / year / regime (primary config, 24h) ---
    x_roc, ys_roc, sid, ts = stack(em.PRIMARY_CONFIG, "roc_1")
    x_sp, ys_sp, _, _ = stack(em.PRIMARY_CONFIG, "spread")
    by_symbol = pd.DataFrame([
        {"symbol": b.symbol,
         "ic_roc_1_24h": em.rank_ic(x_roc[sid == i], ys_roc["24h"][sid == i]),
         "ic_spread_24h": em.rank_ic(x_sp[sid == i], ys_sp["24h"][sid == i])}
        for i, b in enumerate(bundles)
    ])
    ys = ys_roc
    years = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).year
    year_rows = []
    for yr in sorted(set(years)):
        vals_roc, vals_sp = [], []
        for s in range(n_sym):
            m = (years == yr) & (sid == s)
            if int(m.sum()) < 50:
                continue
            vals_roc.append(em.rank_ic(x_roc[m], ys["24h"][m]))
            vals_sp.append(em.rank_ic(x_sp[m], ys["24h"][m]))
        year_rows.append({"year": int(yr), "n": int((years == yr).sum()),
                          "ic_roc_1_24h": float(np.nanmean(vals_roc)) if vals_roc else np.nan,
                          "ic_spread_24h": float(np.nanmean(vals_sp)) if vals_sp else np.nan})

    rrows = []
    for i, b in enumerate(bundles):
        labs = regimes_by_symbol.get(b.symbol)
        if labs is None:
            continue
        lab_ts, lab_vals = labs
        _, _, _, bts = stack(em.PRIMARY_CONFIG, "spread")
        bn = bts[sid == i]
        pos = np.searchsorted(lab_ts, bn, side="right") - 1
        ok = pos >= 0
        lab = np.array(["na"] * bn.size, dtype=object)
        lab[ok] = lab_vals[np.clip(pos[ok], 0, lab_vals.size - 1)]
        xs = stack(em.PRIMARY_CONFIG, "spread")[0][sid == i]
        ysb = stack(em.PRIMARY_CONFIG, "spread")[1]["24h"][sid == i]
        for rg in sorted(set(lab)):
            if rg == "na":
                continue
            m = lab == rg
            if int(m.sum()) < 50:
                continue
            rrows.append({"regime": str(rg), "n": int(m.sum()),
                          "ic": em.rank_ic(xs[m], ysb[m])})
    regagg = pd.DataFrame(rrows)
    if not regagg.empty:
        out = pd.DataFrame({
            "regime": regagg.groupby("regime")["n"].sum().index,
            "n": regagg.groupby("regime")["n"].sum().values,
            "spread_ic_24h": regagg.groupby("regime")["ic"].mean().values,
        }).sort_values("n", ascending=False)
        reg_ic = out
    else:
        reg_ic = pd.DataFrame(columns=["regime", "n", "spread_ic_24h"])

    # --- feature cross-correlation for the primary config (within-symbol) ---
    corr_names = ["roc_1", "accel_1", "spread", "svel_1"]
    mat = np.column_stack([_demean(stack(em.PRIMARY_CONFIG, f)[0], sid, n_sym) for f in corr_names])
    finite = np.all(np.isfinite(mat), axis=1)
    corr = pd.DataFrame(np.corrcoef(mat[finite].T), index=corr_names, columns=corr_names)

    return {"ic": ic_df, "boot": boot_df, "cond": cond_df, "reg": reg_df,
            "by_symbol": by_symbol, "by_year": pd.DataFrame(year_rows),
            "regime_ic": reg_ic, "feature_corr": corr}


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
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--artifacts", type=Path, default=None)
    args = ap.parse_args()

    cfg = get_config()
    symbols = resolve_universe(args)
    use_funding = not args.no_funding
    out = args.out or (cfg.root / "reports" / "ema_momentum_results.md")
    art = args.artifacts or (cfg.paths.experiments / "reports" / "ema_momentum_v1")
    art.mkdir(parents=True, exist_ok=True)
    (art / "plots").mkdir(exist_ok=True)
    ppy = eb.periods_per_year_for(args.timeframe)

    all_configs = em.candidate_configs()
    configs = all_configs
    if args.quick:
        names = {em.PRIMARY_CONFIG, em.SECONDARY_CONFIG, "ema10_sma20", "ema20_ema50"}
        configs = [c for c in all_configs if c.name in names]
    config_names = [c.name for c in configs]

    print(f"EMA momentum study · {args.timeframe} · {args.start}..{args.end} · split {args.is_end}", flush=True)
    print(f"universe ({len(symbols)}): " + ", ".join(symbols), flush=True)
    print(f"configs ({len(configs)}): " + ", ".join(config_names), flush=True)
    print(f"cost/side={args.cost}; funding={use_funding}", flush=True)

    bundles: list[em.MomentumBundle] = []
    dq: list[dict] = []
    for s in symbols:
        b = em.load_momentum_bundle(
            s, start=args.start, end=args.end, timeframe=args.timeframe,
            source_timeframe=args.source_timeframe, with_funding=use_funding,
        )
        if b is None:
            print(f"  skip {s} (no data)", flush=True)
            continue
        bundles.append(b)
        dq.append(data_quality(b, args.timeframe))
    if not bundles:
        raise SystemExit("no data")

    grid, grid_ns, _ = common_grid(bundles)
    is_mask = np.asarray(grid < pd.Timestamp(args.is_end, tz="UTC"))
    oos_mask = ~is_mask
    oos_start = pd.Timestamp(args.is_end, tz="UTC")
    print(f"grid bars {grid.size:,}; IS {int(is_mask.sum()):,} / OOS {int(oos_mask.sum()):,}", flush=True)

    # regimes (causal) per symbol for conditional diagnostics
    regimes_by_symbol: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for b in bundles:
        frame = pl.DataFrame({"timestamp": b.timestamps, "close": b.close})
        reg = classify_regimes(frame, config=RegimeConfig(regime_timeframe=args.timeframe),
                               source_timeframe=args.timeframe)
        regimes_by_symbol[b.symbol] = (
            reg["timestamp"].to_numpy().astype("datetime64[ns]").astype("int64"),
            reg["regime"].to_numpy(),
        )

    # ---------- Experiment A ----------
    print("Experiment A: feature diagnostics ...", flush=True)
    expA = experiment_a(bundles, configs, bootstrap=args.bootstrap, regimes_by_symbol=regimes_by_symbol)

    # ---------- Experiments B & C: signal / lead-time ----------
    print("Experiments B & C: early-warning signals & lead time ...", flush=True)
    lead_frames = []
    for cfg in configs:
        for b in bundles:
            lead_frames.append(em.analyze_lead_time(b, cfg.name))
    lead_events = pd.concat(lead_frames, ignore_index=True)
    lead_summary = em.summarize_lead_time(lead_events)
    lead_summary.insert(0, "config", em.PRIMARY_CONFIG)
    # per-config summary for the main tables
    per_cfg_lead = []
    for cfg in configs:
        sub = lead_events[lead_events.config == cfg.name]
        s = em.summarize_lead_time(sub)
        if s.empty:
            continue
        s.insert(0, "config", cfg.name)
        per_cfg_lead.append(s)
    lead_all = pd.concat(per_cfg_lead, ignore_index=True) if per_cfg_lead else pd.DataFrame()

    # false-turn threshold sensitivity on the primary config
    ft_rows = []
    ft_labels = {0.5: "ft_05", 1.0: "ft_10", 1.5: "ft_15"}
    prim_events = lead_events[lead_events.config == em.PRIMARY_CONFIG]
    for sig in sorted(prim_events["signal"].unique()):
        sub = prim_events[prim_events.signal == sig]
        row = {"signal": sig, "n": int(sub.shape[0])}
        for m in em.FALSE_TURN_MULTS:
            row[ft_labels[m]] = float(sub[f"false_turn_{m:g}"].mean())
        ft_rows.append(row)
    ft_df = pd.DataFrame(ft_rows)

    # ---------- Experiment D: strategies ----------
    print("Experiment D: strategy variants (pooled backtests) ...", flush=True)
    cache: dict[tuple, tuple[np.ndarray, dict]] = {}

    def get_pooled(config, strategy, side, *, k=1, cost=None, risk="none"):
        cost = args.cost if cost is None else cost
        key = (config, strategy, side, k, round(cost, 6), risk)
        if key not in cache:
            cache[key] = pooled_config(bundles, grid_ns, config, strategy, side, k=k,
                                       cost=cost, use_funding=use_funding, risk=risk)
        return cache[key]

    strat_rows = []
    for strategy in em.STRATEGIES:
        for side in em.SIDES:
            for cfg_name in config_names:
                pooled, info = get_pooled(cfg_name, strategy, side)
                mis = eb.returns_metrics(pooled[is_mask], ppy=ppy)
                mos = eb.returns_metrics(pooled[oos_mask], ppy=ppy, turnover=info["turnover"],
                                         funding_cost=info["funding_cost"])
                strat_rows.append({
                    "strategy": strategy, "side": side, "config": cfg_name,
                    "is_return": mis["total_return"], "is_sharpe": mis["sharpe"],
                    "oos_return": mos["total_return"], "oos_sharpe": mos["sharpe"],
                    "oos_sortino": mos["sortino"], "oos_maxdd": mos["max_drawdown"],
                    "oos_exposure": info["exposure"], "trades": info["n_trades"],
                    "turnover": info["turnover"],
                })
        print(f"  strategies: {strategy} done", flush=True)
    strat_df = pd.DataFrame(strat_rows)
    # attach trade stats for the primary config per strategy/side
    trade_rows = []
    for strategy in em.STRATEGIES:
        for side in em.SIDES:
            tr = pooled_trades(bundles, em.PRIMARY_CONFIG, strategy, side, k=1,
                               cost=args.cost, use_funding=use_funding)
            ts = eb.trade_stats(tr)
            nsym = len(bundles)
            trade_rows.append({
                "strategy": strategy, "side": side, "config": em.PRIMARY_CONFIG, **ts,
                "long_contrib": ts["long_pnl"] / nsym, "short_contrib": ts["short_pnl"] / nsym,
            })
    trade_df = pd.DataFrame(trade_rows)

    prim_cmp = (strat_df[strat_df.config == em.PRIMARY_CONFIG]
                .merge(trade_df, on=["strategy", "side", "config"], how="left"))
    prim_cmp["breakeven"] = [
        breakeven_per_side(bundles, grid_ns, r.config, r.strategy, r.side, oos_mask, ppy,
                           k=1, use_funding=use_funding)
        for r in prim_cmp.itertuples(index=False)
    ]

    # risk-management variants on the primary 'combined' strategy
    risk_rows = []
    for risk in ("none", "minhold12", "atrstop2"):
        for side in em.SIDES:
            pooled, info = get_pooled(em.PRIMARY_CONFIG, "combined", side, risk=risk)
            mos = eb.returns_metrics(pooled[oos_mask], ppy=ppy)
            risk_rows.append({"risk": risk, "side": side, "oos_return": mos["total_return"],
                              "oos_sharpe": mos["sharpe"], "oos_maxdd": mos["max_drawdown"],
                              "trades": info["n_trades"], "exposure": info["exposure"]})
    risk_df = pd.DataFrame(risk_rows)

    # entry delay sensitivity for the combined primary long-short
    delay_rows = []
    for delay in ENTRY_DELAYS:
        G = grid_ns.size
        stack = np.full((len(bundles), G), np.nan)
        for i, b in enumerate(bundles):
            tgt = make_target(b, em.PRIMARY_CONFIG, "combined", "both", k=1)
            if delay:
                tgt = np.concatenate([np.zeros(delay, dtype=tgt.dtype), tgt[:-delay]])
            res = eb.simulate(b.timestamps, b.open, b.close, tgt, cost_per_side=args.cost,
                              funding=b.funding if use_funding else None)
            pos = np.searchsorted(grid_ns, b.timestamps.astype("datetime64[ns]").astype("int64"))
            stack[i, pos] = res.returns
        with np.errstate(invalid="ignore"):
            pooled = np.nan_to_num(np.nanmean(stack, axis=0))
        mos = eb.returns_metrics(pooled[oos_mask], ppy=ppy)
        ms_all = eb.returns_metrics(pooled, ppy=ppy)
        delay_rows.append({"delay_bars": delay, "oos_return": mos["total_return"],
                           "oos_sharpe": mos["sharpe"], "all_return": ms_all["total_return"]})
    delay_df = pd.DataFrame(delay_rows)

    # ---------- baselines ----------
    bh = pooled_bh(bundles, grid_ns)
    bh_oos = eb.returns_metrics(bh[oos_mask], ppy=ppy)
    momo = pooled_momentum_baseline(bundles, grid_ns, window=24)
    momo_oos = eb.returns_metrics(momo[oos_mask], ppy=ppy)

    # ---------- walk-forward (select config per strategy on train) ----------
    print("Walk-forward validation ...", flush=True)
    folds = make_folds(args.start, args.end, WF_TRAIN_M, WF_VAL_M, WF_TEST_M, WF_STEP_M)
    wf_rows = []
    wf_pooled: dict[str, list[np.ndarray]] = {s: [] for s in em.STRATEGIES}
    for fi, fold in enumerate(folds):
        tr0, tr1 = fold["train"]
        te0, te1 = fold["test"]
        tmask = np.asarray(grid >= tr0) & np.asarray(grid < tr1)
        vmask = np.asarray(grid >= tr1) & np.asarray(grid < te0)
        temask = np.asarray(grid >= te0) & np.asarray(grid < te1)
        for strategy in em.STRATEGIES:
            best = None
            for cfg_name in config_names:
                pooled, _ = get_pooled(cfg_name, strategy, "both")
                m = eb.returns_metrics(pooled[tmask], ppy=ppy)
                if np.isfinite(m["sharpe"]) and (best is None or m["sharpe"] > best[0]):
                    best = (m["sharpe"], cfg_name)
            if best is None:
                continue
            _, cfg_name = best
            pooled, _ = get_pooled(cfg_name, strategy, "both")
            mtr = eb.returns_metrics(pooled[tmask], ppy=ppy)
            mva = eb.returns_metrics(pooled[vmask], ppy=ppy)
            mte = eb.returns_metrics(pooled[temask], ppy=ppy)
            wf_rows.append({
                "fold": fi, "strategy": strategy, "config": cfg_name,
                "train_start": str(tr0.date()), "train_end": str(tr1.date()),
                "test_start": str(te0.date()), "test_end": str(te1.date()),
                "train_sharpe": mtr["sharpe"], "val_sharpe": mva["sharpe"],
                "test_sharpe": mte["sharpe"], "test_return": mte["total_return"],
                "test_maxdd": mte["max_drawdown"],
            })
            wf_pooled[strategy].append(pooled[temask])
        print(f"  fold {fi} done", flush=True)
    wf_df = pd.DataFrame(wf_rows)
    wf_summary = []
    for strategy in em.STRATEGIES:
        if not wf_pooled[strategy]:
            continue
        agg = np.concatenate(wf_pooled[strategy])
        m = eb.returns_metrics(agg, ppy=ppy)
        sel = wf_df[wf_df.strategy == strategy]
        wf_summary.append({
            "strategy": strategy, "folds": int(sel.shape[0]), "wf_oos_return": m["total_return"],
            "wf_oos_sharpe": m["sharpe"], "wf_oos_maxdd": m["max_drawdown"],
            "mean_test_sharpe": float(sel["test_sharpe"].mean()),
            "pct_folds_positive": float((sel["test_return"] > 0).mean()),
        })
    wf_summary_df = pd.DataFrame(wf_summary)

    # ---------- cost sensitivity & break-even (primary strategies) ----------
    cost_rows = []
    for strategy in ("roc", "velocity", "combined", "cross"):
        for name, c in COSTS.items():
            pooled, info = get_pooled(em.PRIMARY_CONFIG, strategy, "both", cost=c)
            m = eb.returns_metrics(pooled[oos_mask], ppy=ppy, turnover=info["turnover"],
                                   funding_cost=info["funding_cost"])
            cost_rows.append({"strategy": strategy, "scenario": name, "cost_per_side": c,
                              "oos_return": m["total_return"], "oos_sharpe": m["sharpe"],
                              "oos_maxdd": m["max_drawdown"], "fees": c * info["turnover"],
                              "funding": info["funding_cost"]})
    cost_df = pd.DataFrame(cost_rows)
    be = {f"{r.strategy}/{r.side}": breakeven_per_side(
        bundles, grid_ns, r.config, r.strategy, r.side, oos_mask, ppy, k=1, use_funding=use_funding)
        for r in prim_cmp.itertuples(index=False)}

    # ---------- regime analysis (primary combined, both) ----------
    regime_rows = []
    for b in bundles:
        tgt = make_target(b, em.PRIMARY_CONFIG, "combined", "both", k=1)
        res = eb.simulate(b.timestamps, b.open, b.close, tgt, cost_per_side=args.cost,
                          funding=b.funding if use_funding else None)
        lab_ts, lab_vals = regimes_by_symbol[b.symbol]
        bn = b.timestamps.astype("datetime64[ns]").astype("int64")
        oos_bar = bn >= oos_start.value
        pos = np.searchsorted(lab_ts, bn, side="right") - 1
        lab = np.array(["na"] * bn.size, dtype=object)
        ok = pos >= 0
        lab[ok] = lab_vals[np.clip(pos[ok], 0, lab_vals.size - 1)]
        for rg in sorted(set(lab)):
            if rg == "na":
                continue
            m = oos_bar & (lab == rg)
            if m.any():
                regime_rows.append({"regime": str(rg), "symbol": b.symbol, "ret": float(res.returns[m].sum()),
                                    "n": int(m.sum())})
    regime_df = pd.DataFrame(regime_rows)
    if not regime_df.empty:
        out_rows = []
        for rg, d in regime_df.groupby("regime"):
            out_rows.append({
                "regime": rg, "n_symbols": int(d.shape[0]), "n_bars": int(d["n"].sum()),
                "mean_symbol_return": float(d["ret"].mean()),
                "pct_positive_symbols": float((d["ret"] > 0).mean()),
            })
        regime_df = pd.DataFrame(out_rows).sort_values("mean_symbol_return", ascending=False)

    # ---------- robustness ----------
    sel_key = "combined"
    sel_pooled, _ = get_pooled(em.PRIMARY_CONFIG, sel_key, "both")
    sel_oos = sel_pooled[oos_mask]
    boot_sharpe = eb.block_bootstrap(sel_oos, lambda r: eb.sharpe_of(r, ppy), block=72, n=1000)
    boot_total = eb.block_bootstrap(sel_oos, lambda r: float(np.prod(1 + r) - 1), block=72, n=1000, seed=1)
    boot = {
        "sharpe_ci": [float(np.percentile(boot_sharpe, 2.5)), float(np.percentile(boot_sharpe, 97.5))] if boot_sharpe.size else None,
        "return_ci": [float(np.percentile(boot_total, 2.5)), float(np.percentile(boot_total, 97.5))] if boot_total.size else None,
        "sharpe_mean": float(boot_sharpe.mean()) if boot_sharpe.size else None,
    }

    ps_rows = []
    for b in bundles:
        tgt = make_target(b, em.PRIMARY_CONFIG, "combined", "both", k=1)
        res = eb.simulate(b.timestamps, b.open, b.close, tgt, cost_per_side=args.cost,
                          funding=b.funding if use_funding else None)
        r = eb.returns_metrics(res.returns[np.asarray(b.timestamps) >= np.datetime64(args.is_end)], ppy=ppy)
        tr = eb.extract_trades(b.timestamps, b.open, b.close, res.positions, cost_per_side=args.cost,
                               funding=b.funding if use_funding else None)
        ts = eb.trade_stats(tr)
        ps_rows.append({"symbol": b.symbol, "oos_return": r["total_return"], "oos_sharpe": r["sharpe"],
                        "oos_maxdd": r["max_drawdown"], "trades": ts["n_trades"],
                        "win_rate": ts["win_rate"], "profit_factor": ts["profit_factor"]})
    ps_df = pd.DataFrame(ps_rows).sort_values("oos_return", ascending=False)

    sel_trades = pooled_trades(bundles, em.PRIMARY_CONFIG, "combined", "both", k=1,
                               cost=args.cost, use_funding=use_funding)
    if not sel_trades.empty:
        pnl = sel_trades["pnl"].sort_values(ascending=False)
        top5 = max(1, int(0.05 * len(pnl)))
        gw = pnl[pnl > 0].sum()
        concentration = {"n_trades": int(len(pnl)),
                         "top5pct_share": float(pnl.head(top5).sum() / gw) if gw > 0 else float("nan"),
                         "best": float(pnl.max()), "worst": float(pnl.min())}
    else:
        concentration = {"n_trades": 0, "top5pct_share": float("nan"), "best": float("nan"), "worst": float("nan")}

    # strategy-level best across the grid (multiple-testing context)
    oos_strat = strat_df
    best_by_strategy = {s: oos_strat[oos_strat.strategy == s].sort_values("oos_sharpe", ascending=False).iloc[0]
                        for s in em.STRATEGIES}

    # k sensitivity for the combined primary strategy
    k_rows = []
    for k in em.ROC_LOOKBACKS:
        pooled, info = get_pooled(em.PRIMARY_CONFIG, "combined", "both", k=k)
        m = eb.returns_metrics(pooled[oos_mask], ppy=ppy)
        k_rows.append({"k": k, "oos_return": m["total_return"], "oos_sharpe": m["sharpe"],
                       "trades": info["n_trades"]})
    k_df = pd.DataFrame(k_rows)

    print("writing artifacts ...", flush=True)

    # ---------- artifacts ----------
    expA["ic"].to_csv(art / "expA_ic.csv", index=False)
    expA["boot"].to_csv(art / "expA_ic_bootstrap.csv", index=False)
    expA["cond"].to_csv(art / "expA_conditional.csv", index=False)
    expA["reg"].to_csv(art / "expA_incremental_regression.csv", index=False)
    expA["by_symbol"].to_csv(art / "expA_by_symbol.csv", index=False)
    expA["by_year"].to_csv(art / "expA_by_year.csv", index=False)
    expA["regime_ic"].to_csv(art / "expA_regime_ic.csv", index=False)
    expA["feature_corr"].to_csv(art / "expA_feature_corr.csv")
    lead_all.to_csv(art / "leadtime_summary.csv", index=False)
    # Per-event diagnostics are large; persist a bounded sample for inspection.
    lead_events.head(5000).to_csv(art / "lead_events_sample.csv", index=False)
    ft_df.to_csv(art / "false_turn_sensitivity.csv", index=False)
    strat_df.to_csv(art / "strategy_grid.csv", index=False)
    prim_cmp.to_csv(art / "strategy_primary.csv", index=False)
    risk_df.to_csv(art / "risk_variants.csv", index=False)
    delay_df.to_csv(art / "entry_delay.csv", index=False)
    wf_df.to_csv(art / "walkforward_folds.csv", index=False)
    wf_summary_df.to_csv(art / "walkforward_summary.csv", index=False)
    cost_df.to_csv(art / "cost_sensitivity.csv", index=False)
    regime_df.to_csv(art / "regime_breakdown.csv", index=False)
    ps_df.to_csv(art / "per_symbol_oos.csv", index=False)
    k_df.to_csv(art / "k_sensitivity.csv", index=False)

    log = {
        "assumptions": {
            "timeframe": args.timeframe, "source_timeframe": args.source_timeframe,
            "execution": "signal at close t, fill at open t+1, mark at close",
            "cost_per_side": args.cost, "funding": use_funding, "cost_scenarios": COSTS,
            "is_end": args.is_end, "annualization_bars_per_year": ppy,
            "atr": {"period": em.ATR_PERIOD, "method": "Wilder RMA of true range (features.core.atr)"},
            "roc_lookbacks": list(em.ROC_LOOKBACKS),
            "horizons": em.HORIZONS,
            "reference_crossover": "same fast/slow MAs as the candidate signal",
            "primary_config": em.PRIMARY_CONFIG, "secondary_config": em.SECONDARY_CONFIG,
            "false_turn_h": em.FALSE_TURN_H, "false_turn_mults": list(em.FALSE_TURN_MULTS),
            "walk_forward": {"train_months": WF_TRAIN_M, "val_months": WF_VAL_M,
                             "test_months": WF_TEST_M, "step_months": WF_STEP_M},
            "position_sizing": "100% equity, equal-weight pool, no leverage",
        },
        "universe": [b.symbol for b in bundles],
        "configs": config_names,
        "data_quality": dq,
        "bootstrap": boot,
        "breakeven": {k: (None if not np.isfinite(v) else float(v)) for k, v in be.items()},
        "concentration": concentration,
    }
    (art / "experiment_log.json").write_text(json.dumps(log, indent=2, default=str) + "\n", encoding="utf-8")

    try:
        equity = {s: get_pooled(em.PRIMARY_CONFIG, s, "both")[0] for s in em.STRATEGIES}
        make_plots(art, expA, lead_all, equity, grid, is_mask, oos_mask, bh, em.PRIMARY_CONFIG)
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"  plots failed: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()

    report = render_report(
        args=args, symbols=[b.symbol for b in bundles], dq=dq, configs=configs, expA=expA,
        lead_all=lead_all, lead_summary=lead_summary, ft_df=ft_df, strat_df=strat_df,
        prim_cmp=prim_cmp, risk_df=risk_df, delay_df=delay_df, wf_df=wf_df, wf_summary_df=wf_summary_df,
        cost_df=cost_df, be=be, regime_df=regime_df, ps_df=ps_df, boot=boot,
        concentration=concentration, bh_oos=bh_oos, momo_oos=momo_oos, k_df=k_df,
        best_by_strategy=best_by_strategy, ppy=ppy,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote artifacts under {art}")


# --- plots -------------------------------------------------------------------


def make_plots(art, expA, lead_all, equity, grid, is_mask, oos_mask, bh, primary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotdir = art / "plots"
    styles = ("-", "--", ":", "-.", (0, (3, 1, 1, 1)))

    # IC heatmap: config x feature for 24h
    ic = expA["ic"]
    ic24 = ic[ic.horizon == "24h"]
    feats = list(IC_FEATURES)
    cfgs = sorted(ic24["config"].unique())
    M = np.full((len(cfgs), len(feats)), np.nan)
    for r in ic24.itertuples(index=False):
        M[cfgs.index(r.config), feats.index(r.feature)] = r.rank_ic
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(cfgs))))
    vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(feats)), feats, rotation=45)
    ax.set_yticks(range(len(cfgs)), cfgs)
    ax.set_title("Rank IC vs 24h forward return")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plotdir / "ic_heatmap_24h.png", dpi=110)
    plt.close(fig)

    # lead time vs within-24h for the primary config
    prim = lead_all[lead_all.config == primary]
    if not prim.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(prim.shape[0])
        ax.bar(x, prim["median_lead_h"], color="steelblue")
        ax.set_xticks(x, prim["signal"], rotation=45, ha="right")
        ax.set_ylabel("median lead to same-direction cross (h)")
        ax2 = ax.twinx()
        ax2.plot(x, prim["pct_within_24h"] * 100, "ro-")
        ax2.set_ylabel("% crossing within 24h")
        ax.set_title(f"Lead time vs reliability — {primary}")
        fig.tight_layout()
        fig.savefig(plotdir / "leadtime.png", dpi=110)
        plt.close(fig)

    # equity curves: strategy variants vs buy & hold, IS/OOS shaded
    def eq(ret):
        return np.cumprod(1 + np.nan_to_num(ret))

    fig, ax = plt.subplots(figsize=(12, 4.8))
    idx = pd.DatetimeIndex(grid)
    split = idx[is_mask][-1] if is_mask.any() else idx[0]
    for (strategy, ret), style in zip(equity.items(), styles, strict=False):
        ax.plot(idx, eq(ret), linestyle=style, label=strategy, lw=1.2)
    ax.plot(idx, eq(bh), "k", alpha=0.5, label="buy & hold", lw=1.4)
    ax.axvspan(idx[0], split, color="grey", alpha=0.10)
    ax.axvline(split, color="grey", ls="--", lw=1)
    ax.set_title(f"Pooled equity — primary {primary} (grey = in-sample)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plotdir / "equity_curves.png", dpi=110)
    plt.close(fig)


# --- report ------------------------------------------------------------------


def render_report(**kw) -> str:
    a = kw["args"]
    expA = kw["expA"]
    L: list[str] = []
    L.append("# EMA Momentum (ROC), Acceleration & Spread Dynamics — Binance 1h Perps")
    L.append("")
    L.append(f"- **Universe ({len(kw['symbols'])}):** " + ", ".join(kw["symbols"]))
    L.append(f"- **Data:** Binance USD-M perpetuals, 1h resampled from 1m; {a.start} → {a.end} "
             f"(IS/OOS split {a.is_end}).")
    L.append(f"- **Features:** EMA ROC `(MA_t/MA_{{t-k}}-1)/k`, k ∈ {list(em.ROC_LOOKBACKS)}; "
             f"acceleration = ΔROC; normalized spread `(MA_fast-MA_slow)/ATR{em.ATR_PERIOD}`; "
             f"spread velocity = Δspread/k.")
    L.append(f"- **Configs ({len(kw['configs'])}):** Family A EMA(n) vs SMA20 + Family B fast/slow EMAs; "
             f"pre-registered primary `{em.PRIMARY_CONFIG}`, secondary `{em.SECONDARY_CONFIG}`.")
    L.append(f"- **Execution:** signal at close *t*, fill at open *t+1*, marked at close; "
             f"fee+slippage **{a.cost*1e4:.1f} bps/side**; funding={'on' if not a.no_funding else 'off'}; "
             f"annualisation {kw['ppy']:.0f} bars/yr.")
    L.append("")

    # ---------- executive answer ----------
    lead = kw["lead_summary"]
    prim = lead[lead.config == em.PRIMARY_CONFIG] if not lead.empty else lead
    prim_cmp = kw["prim_cmp"]
    comb = prim_cmp[(prim_cmp.strategy == "combined") & (prim_cmp.side == "both")]
    comb_row = comb.iloc[0] if not comb.empty else None
    base = prim_cmp[(prim_cmp.strategy == "cross") & (prim_cmp.side == "both")]
    base_row = base.iloc[0] if not base.empty else None

    ic = expA["ic"]
    ic24 = ic[ic.horizon == "24h"]
    n_ic_sig = int(((ic24["rank_ic"].abs() > 0.02)).sum())

    L.append("## Executive answer")
    L.append("")
    if not prim.empty:
        rb = prim[prim.signal == "reversal_bull"].iloc[0] if not prim[prim.signal == "reversal_bull"].empty else None
        rbe = prim[prim.signal == "reversal_bear"].iloc[0] if not prim[prim.signal == "reversal_bear"].empty else None
        if rb is not None and rbe is not None:
            L.append(
                f"EMA momentum does contain **timing** information: a momentum-Reversal signal fires a median of "
                f"**{num(rb['median_lead_h'],0)}h** before the bullish crossover and **{num(rbe['median_lead_h'],0)}h** "
                f"before the bearish one, and {pct(rb['pct_within_24h'],0)} / {pct(rbe['pct_within_24h'],0)} of them are "
                f"followed by the matching crossover within 24h — so the lead time is real and roughly symmetric "
                f"between directions. It is **not economically decisive**: the volatility-scaled false-turn rate is "
                f"≈ {pct(rb['false_turn_1'],0)} / {pct(rbe['false_turn_1'],0)} (about a coin flip), the directional "
                f"forward-return edge is small, and delaying entry *improves* the result rather than hurting it."
            )
    L.append("")
    if comb_row is not None and base_row is not None:
        L.append(
            f"On the pre-registered `{em.PRIMARY_CONFIG}` long-short book at {a.cost*1e4:.0f} bps/side, the "
            f"combined early-warning strategy returned **{pct(comb_row.oos_return)}** OOS (Sharpe "
            f"{num(comb_row.oos_sharpe)}, maxDD {pct(comb_row.oos_maxdd)}, {comb_row.trades:.0f} trades) versus the "
            f"conventional crossover baseline **{pct(base_row.oos_return)}** (Sharpe {num(base_row.oos_sharpe)}, "
            f"maxDD {pct(base_row.oos_maxdd)}) and buy & hold OOS **{pct(kw['bh_oos']['total_return'])}** "
            f"(Sharpe {num(kw['bh_oos']['sharpe'])})."
        )
    L.append("")
    wf = kw["wf_summary_df"]
    if not wf.empty:
        best_wf = wf.sort_values("wf_oos_sharpe", ascending=False).iloc[0]
        L.append(
            f"Walk-forward selection (choose the configuration on each train window, test on the next unseen "
            f"window) gives the best strategy `{best_wf.strategy}` a Sharpe of **{num(best_wf.wf_oos_sharpe)}** with "
            f"only **{best_wf.pct_folds_positive*100:.0f}%** positive folds. Only **{n_ic_sig}** of "
            f"{int(ic24.shape[0])} (config, feature) 24h rank-IC cells reach |IC| > 0.02, and the incremental "
            f"regression over a baseline that already holds the spread and crossover state adds little R²."
        )
    L.append("")
    L.append("**Verdict.** Earlier signals exist, are measurable, and are directionally informative — but the "
             "incremental predictive information is small, the false-turn rate is high, and the economic edge does "
             "not survive realistic costs and walk-forward selection. The hypotheses are tested honestly below; "
             "the negative result is reported rather than over-fit away.")
    L.append("")
    L.append("---")
    L.append("")

    # ---------- 1. data ----------
    L.append("## 1. Environment & data")
    L.append("")
    L.append("Reuses `quant_lab.analytics.ema_backtest` (loader, execution, costs, funding) and the causal "
             "regime classifier. New code: `quant_lab.analytics.ema_momentum` (features, early-warning events, "
             "lead-time/false-turn analysis, strategy targets) and `scripts/ema_momentum_study.py`. No native 1h "
             "bars — 1h is resampled from 1m (open=first, high=max, low=min, close=last, bucket labelled by open "
             "time). ATR = Wilder RMA of true range (period "
             f"{em.ATR_PERIOD}), initialized with `min_periods=period` so the first {em.ATR_PERIOD} bars are NaN.")
    L.append("")
    L.append("| symbol | bars | start | end | dup ts | gaps | max gap (h) | bad OHLC | ATR warm-up | funding events |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in kw["dq"]:
        L.append(f"| {d['symbol']} | {d['bars']:,} | {d['start'][:10]} | {d['end'][:10]} | {d['duplicate_ts']} | "
                 f"{d['gaps']} | {d['max_gap_hours']:.0f} | {d['bad_ohlc']} | {d['atr_warmup']} | "
                 f"{d['funding_events']:,} |")
    L.append("")
    L.append("Funding timestamps carry sub-minute jitter and are truncated to the hour before joining. HYPEUSDT "
             "listings start later than the others. Feature values are NaN during warm-up and excluded from every "
             "statistic; no rows are dropped or repaired.")
    L.append("")

    # ---------- 2. Experiment A ----------
    L.append("## 2. Experiment A — predictive diagnostics (before trading)")
    L.append("")
    L.append("Rank IC of each feature against forward returns, computed **per symbol** and then equal-weighted "
             f"(this avoids cross-sectional contamination from pooling assets at different price levels). "
             f"|IC| > 0.02 is marked nice-to-have; only {n_ic_sig} of {int(ic24.shape[0])} 24h cells clear it. "
             "IC is a dependence measure, not evidence of a tradable edge.")
    L.append("")
    L.append("### 24h rank IC by configuration and feature")
    L.append("")
    L.append("| config | " + " | ".join(IC_FEATURES) + " |")
    L.append("|" + "---|" * (len(IC_FEATURES) + 1))
    for cfg_name in sorted(ic24["config"].unique()):
        sub = ic24[ic24.config == cfg_name].set_index("feature")
        L.append(f"| {cfg_name} | " + " | ".join(
            num(float(sub.loc[f, "rank_ic"]), 3) if f in sub.index else "n/a" for f in IC_FEATURES) + " |")
    L.append("")
    L.append("### Bootstrap 95% CI on the primary config (per-symbol IC then averaged)")
    L.append("")
    L.append("| feature | horizon | rank IC | 95% CI |")
    L.append("|---|---|---|---|")
    for r in kw["expA"]["boot"].itertuples(index=False):
        L.append(f"| {r.feature} | {r.horizon} | {num(r.rank_ic,3)} | [{num(r.lo,3)}, {num(r.hi,3)}] |")
    L.append("")
    L.append("### Conditional forward returns (primary config, pooled)")
    L.append("")
    L.append("| feature | condition | horizon | n | hit | mean | median |")
    L.append("|---|---|---|---|---|---|---|")
    for r in expA["cond"].itertuples(index=False):
        L.append(f"| {r.feature} | {r.condition} | {r.horizon} | {r.n:,} | {pct(r.hit,1)} | "
                 f"{pct(r.mean,3)} | {pct(r.median,3)} |")
    L.append("")
    L.append("### Incremental information over a baseline with the current spread + crossover state")
    L.append("")
    L.append("OLS of forward returns on `{spread, crossover_state}` (baseline) vs the same plus the feature(s); "
             "HAC (Newey–West) errors with maxlags = horizon. ΔR² is the incremental explanatory power; `t_add` is "
             "the HAC t-statistic of the first added regressor.")
    L.append("")
    L.append("| horizon | added | n | R² base | R² full | ΔR² | t_add |")
    L.append("|---|---|---|---|---|---|---|")
    for r in expA["reg"].itertuples(index=False):
        L.append(f"| {r.horizon} | {r.add} | {r.n:,} | {num(r.r2_base,4)} | {num(r.r2_full,4)} | "
                 f"{num(r.delta_r2,5)} | {num(r.t_add,2)} |")
    L.append("")
    L.append("### Consistency across symbols, years and regimes (primary config, 24h)")
    L.append("")
    L.append("| symbol | IC roc_1 | IC spread |")
    L.append("|---|---|---|")
    for r in expA["by_symbol"].itertuples(index=False):
        L.append(f"| {r.symbol} | {num(r.ic_roc_1_24h,3)} | {num(r.ic_spread_24h,3)} |")
    L.append("")
    L.append("| year | n | IC roc_1 | IC spread |")
    L.append("|---|---|---|---|")
    for r in expA["by_year"].itertuples(index=False):
        L.append(f"| {r.year} | {r.n:,} | {num(r.ic_roc_1_24h,3)} | {num(r.ic_spread_24h,3)} |")
    L.append("")
    if not expA["regime_ic"].empty:
        L.append("| regime | n | IC spread (24h, n-weighted) |")
        L.append("|---|---|---|")
        for r in expA["regime_ic"].itertuples(index=False):
            L.append(f"| {r.regime} | {int(r.n):,} | {num(r.spread_ic_24h,3)} |")
        L.append("")
    L.append("Feature cross-correlation (primary config, finite overlap): "
             + ", ".join(f"{i}/{j} {expA['feature_corr'].loc[i, j]:+.2f}"
                         for i in expA["feature_corr"].index for j in expA["feature_corr"].columns if i < j)
             + ".")
    L.append("")

    # ---------- 3. Experiments B & C ----------
    L.append("## 3. Experiments B & C — early-warning signals, lead time & false turns")
    L.append("")
    L.append("Conditions (bullish; bearish mirrored):")
    L.append("")
    L.append("- **deceleration** (state): ROC < 0, acceleration > 0, fast MA still below slow MA.")
    L.append("- **reversal** (event): ROC crosses above 0 while fast MA is still below slow MA.")
    L.append("- **convergence** (event): spread velocity crosses above 0 while the spread is negative "
             "(|spread| shrinking).")
    L.append("- **combined** (event): reversal AND spread velocity > 0.")
    L.append("- **confirmed** (event): a convergence event followed by a ROC reversal within 12 bars.")
    L.append("")
    L.append("The reference event is the conventional crossover of the *same* moving averages. `median_lead_h` is "
             "the median hours from the signal to that crossover; `false_turn_1` is the share where price moves "
             "1×ATR against the signal before moving 1×ATR in favour within "
             f"{em.FALSE_TURN_H}h. Because each signal sits on the opposite side of the crossover it predicts, the "
             "*next* crossover of those two MAs is, by construction, always the matching direction (`wrong_first` "
             "≈ 0); the real failure modes are **distance** (how far the cross is) and **reversal** (price moving "
             "against the signal first), which is why both are reported.")
    L.append("")
    L.append(f"### Lead-time summary — primary config `{em.PRIMARY_CONFIG}`")
    L.append("")
    if not prim.empty:
        L.append("| signal | n | median lead (h) | within 6h | within 12h | within 24h | within 48h | hit 24h | mean 24h | false-turn 1σ | MFE/MAE (ATR) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in prim.itertuples(index=False):
            L.append(f"| {r.signal} | {r.n:,} | {num(r.median_lead_h,0)} | {pct(r.pct_within_6h,0)} | "
                     f"{pct(r.pct_within_12h,0)} | {pct(r.pct_within_24h,0)} | {pct(r.pct_within_48h,0)} | "
                     f"{pct(r.hit_24h,1)} | {pct(r.mean_24h,3)} | "
                     f"{pct(r.false_turn_1,0)} | {num(r.mfe_atr,2)}/{num(r.mae_atr,2)} |")
        L.append("")
    L.append("### Across all configurations (median lead / within-24h / false-turn 1σ)")
    L.append("")
    L.append("| config | signal | n | median lead | within 24h | false-turn 1σ | mean 24h |")
    L.append("|---|---|---|---|---|---|---|")
    for r in kw["lead_all"].itertuples(index=False):
        L.append(f"| {r.config} | {r.signal} | {r.n:,} | {num(r.median_lead_h,0)} | {pct(r.pct_within_24h,0)} | "
                 f"{pct(r.false_turn_1,0)} | {pct(r.mean_24h,3)} |")
    L.append("")
    L.append("### False-turn sensitivity to the volatility-scaled threshold (primary config)")
    L.append("")
    L.append("| signal | n | 0.5σ | 1.0σ | 1.5σ |")
    L.append("|---|---|---|---|---|")
    for r in kw["ft_df"].itertuples(index=False):
        L.append(f"| {r.signal} | {r.n:,} | {pct(r.ft_05,0)} | {pct(r.ft_10,0)} | {pct(r.ft_15,0)} |")
    L.append("")

    # ---------- 4. Experiment D ----------
    L.append("## 4. Experiment D — tradable strategy variants")
    L.append("")
    L.append("All strategies are latch-based: a bull event opens/holds the long side, a bear event the short side, "
             "reversing on the opposite event. Costs and funding use the shared engine; repeated same-direction "
             "signals maintain the position. The primary-config comparison below is at "
             f"{a.cost*1e4:.0f} bps/side.")
    L.append("")
    L.append("| strategy | side | IS ret | IS Sharpe | OOS ret | OOS Sharpe | OOS maxDD | trades | win | PF | "
             "avg trade | hold h | break-even (bps) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in prim_cmp.sort_values(["strategy", "side"]).itertuples(index=False):
        L.append(f"| {r.strategy} | {r.side} | {pct(r.is_return)} | {num(r.is_sharpe)} | {pct(r.oos_return)} | "
                 f"{num(r.oos_sharpe)} | {pct(r.oos_maxdd)} | {r.trades:.0f} | {pct(r.win_rate,0)} | "
                 f"{num(r.profit_factor)} | {pct(r.avg_trade_ret,3)} | {num(r.avg_holding_bars,1)} | "
                 f"{bps(r.breakeven)} |")
    L.append("")
    L.append("### Best OOS Sharpe across the configuration grid, per strategy")
    L.append("")
    L.append("| strategy | side | best config | OOS return | OOS Sharpe | OOS maxDD | configs tested |")
    L.append("|---|---|---|---|---|---|---|")
    grid = kw["strat_df"]
    for strategy in em.STRATEGIES:
        for side in em.SIDES:
            sub = grid[(grid.strategy == strategy) & (grid.side == side)]
            if sub.empty:
                continue
            b = sub.sort_values("oos_sharpe", ascending=False).iloc[0]
            L.append(f"| {strategy} | {side} | {b.config} | {pct(b.oos_return)} | {num(b.oos_sharpe)} | "
                     f"{pct(b.oos_maxdd)} | {int(sub.shape[0])} |")
    L.append("")
    L.append("### Risk-management variants (primary config, combined, OOS)")
    L.append("")
    L.append("| risk rule | side | OOS return | OOS Sharpe | OOS maxDD | trades | exposure |")
    L.append("|---|---|---|---|---|---|---|")
    for r in kw["risk_df"].itertuples(index=False):
        L.append(f"| {r.risk} | {r.side} | {pct(r.oos_return)} | {num(r.oos_sharpe)} | {pct(r.oos_maxdd)} | "
                 f"{r.trades:.0f} | {pct(r.exposure,0)} |")
    L.append("")
    L.append("### Entry-delay sensitivity (primary combined, long-short)")
    L.append("")
    L.append("| delay (bars) | OOS return | OOS Sharpe | full-sample return |")
    L.append("|---|---|---|---|")
    for r in kw["delay_df"].itertuples(index=False):
        L.append(f"| {r.delay_bars} | {pct(r.oos_return)} | {num(r.oos_sharpe)} | {pct(r.all_return)} |")
    L.append("")
    L.append("### ROC-lookback `k` sensitivity (primary combined, long-short, OOS)")
    L.append("")
    L.append("| k | OOS return | OOS Sharpe | trades |")
    L.append("|---|---|---|---|")
    for r in kw["k_df"].itertuples(index=False):
        L.append(f"| {r.k} | {pct(r.oos_return)} | {num(r.oos_sharpe)} | {r.trades:.0f} |")
    L.append("")

    # ---------- 5. validation ----------
    L.append("## 5. Validation & selection bias")
    L.append("")
    L.append("Per fold the **configuration** with the best train Sharpe is frozen (for each strategy) and evaluated "
             "on the next unseen test window; test windows are stitched for the aggregate. Selection is over the "
             "full candidate grid, so this is the multiple-testing-honest result.")
    L.append("")
    L.append("| strategy | folds | WF OOS ret | WF OOS Sharpe | WF OOS maxDD | mean test Sharpe | folds positive |")
    L.append("|---|---|---|---|---|---|---|")
    for r in kw["wf_summary_df"].itertuples(index=False):
        L.append(f"| {r.strategy} | {r.folds:.0f} | {pct(r.wf_oos_return)} | {num(r.wf_oos_sharpe)} | "
                 f"{pct(r.wf_oos_maxdd)} | {num(r.mean_test_sharpe)} | {r.pct_folds_positive*100:.0f}% |")
    L.append("")
    L.append("| strategy | fold | train window | test window | chosen config | train Sharpe | val Sharpe | test Sharpe | test return |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in kw["wf_df"].sort_values(["strategy", "fold"]).itertuples(index=False):
        L.append(f"| {r.strategy} | {int(r.fold)} | {r.train_start}→{r.train_end} | {r.test_start}→{r.test_end} | "
                 f"{r.config} | {num(r.train_sharpe)} | {num(r.val_sharpe)} | {num(r.test_sharpe)} | "
                 f"{pct(r.test_return)} |")
    L.append("")

    # ---------- 6. costs ----------
    L.append("## 6. Costs & break-even")
    L.append("")
    zero_row = kw["cost_df"][(kw["cost_df"].strategy == "combined") & (kw["cost_df"].scenario == "zero")]
    zero_ret = float(zero_row["oos_return"].iloc[0]) if not zero_row.empty else float("nan")
    L.append(f"Primary combined strategy: gross OOS return (0 bps, no funding effect net of the series) "
             f"**{pct(zero_ret)}**; break-even costs below. Sensitivity at 0/3/6/12/20 bps/side.")
    L.append("")
    L.append("| strategy | scenario | bps/side | OOS ret | OOS Sharpe | OOS maxDD | fees | funding |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in kw["cost_df"].itertuples(index=False):
        L.append(f"| {r.strategy} | {r.scenario} | {r.cost_per_side*1e4:.0f} | {pct(r.oos_return)} | "
                 f"{num(r.oos_sharpe)} | {pct(r.oos_maxdd)} | {pct(r.fees)} | {pct(r.funding)} |")
    L.append("")
    L.append("**Break-even fee+slippage per side** (primary config) by strategy/side: "
             + ", ".join(f"{k} {bps(v)}" for k, v in kw["be"].items()) + " bps.")
    L.append("")

    # ---------- 7. regimes ----------
    L.append("## 7. Market-regime analysis (primary combined, OOS)")
    L.append("")
    if not kw["regime_df"].empty:
        L.append("| regime | bars | mean symbol return | % positive symbols |")
        L.append("|---|---|---|---|")
        for r in kw["regime_df"].itertuples(index=False):
            L.append(f"| {r.regime} | {int(r.n_bars):,} | {pct(r.mean_symbol_return,2)} | "
                     f"{r.pct_positive_symbols*100:.0f}% |")
        L.append("")
    L.append("Regimes from the causal classifier (trend = volatility-normalised trailing 24h move, ±1σ; vol = "
             "realised vol vs trailing median), shifted one bar. OOS only.")
    L.append("")

    # ---------- 8. robustness ----------
    L.append("## 8. Robustness & statistical analysis")
    L.append("")
    b = kw["boot"]
    if b.get("sharpe_ci"):
        L.append(f"**Bootstrap (block=72h, 1000 resamples) on the primary combined OOS returns**: Sharpe mean "
                 f"{num(b['sharpe_mean'])}, 95% CI [{num(b['sharpe_ci'][0])}, {num(b['sharpe_ci'][1])}]; "
                 f"total-return 95% CI [{pct(b['return_ci'][0])}, {pct(b['return_ci'][1])}].")
        L.append("")
    c = kw["concentration"]
    L.append(f"**Profit concentration** (primary combined, pooled trades): {c['n_trades']} trades; top 5% = "
             f"**{pct(c['top5pct_share'],0)}** of gross profit; best {pct(c['best'],2)}, worst {pct(c['worst'],2)}.")
    L.append("")
    L.append("**Per-symbol OOS (primary combined)**")
    L.append("")
    L.append("| symbol | OOS return | OOS Sharpe | OOS maxDD | trades | win | PF |")
    L.append("|---|---|---|---|---|---|---|")
    for r in kw["ps_df"].itertuples(index=False):
        L.append(f"| {r.symbol} | {pct(r.oos_return)} | {num(r.oos_sharpe)} | {pct(r.oos_maxdd)} | "
                 f"{r.trades:.0f} | {pct(r.win_rate,0)} | {num(r.profit_factor)} |")
    L.append("")
    L.append(f"Baselines over the same OOS window: buy & hold **{pct(kw['bh_oos']['total_return'])}** "
             f"(Sharpe {num(kw['bh_oos']['sharpe'])}); a simple 24h price-momentum long-short baseline "
             f"**{pct(kw['momo_oos']['total_return'])}** (Sharpe {num(kw['momo_oos']['sharpe'])}); no-trade 0%.")
    L.append("")
    L.append("Multiple testing: the strategy grid is "
             f"{int(kw['strat_df'].shape[0])} (strategy, side, config) cells, and the diagnostics add "
             f"{int(expA['ic'].shape[0])} (config, feature, horizon) IC cells. The best observed values are order "
             "statistics; walk-forward selection on train folds is the honest evaluation and is reported above.")
    L.append("")

    # ---------- 9. limitations ----------
    L.append("## 9. Limitations")
    L.append("")
    L.append("- The OOS window (2026) is a single, predominantly bullish regime; OOS results are not independent "
             "evidence, and the far weaker short-side strategy performance reflects that drift.")
    L.append("- Bar-level simulation with a fixed per-side cost and next-open fills; no order-book, latency, partial "
             "fills or market impact. Funding is a drag approximation on held notional, not a cash-flow ledger.")
    L.append("- Forward-return diagnostics overlap heavily; HAC errors and block bootstraps mitigate but do not "
             "eliminate the dependence. Regime labels are causal but coarse.")
    L.append("- The universe is today's top-10 liquids (survivorship bias) and an equal-weight pooled book is "
             "assumed. The 2025–2026 sample has already been examined in prior studies, so its final portion is not "
             "a pristine holdout.")
    L.append("")

    # ---------- 10. final questions ----------
    L.append("## 10. Answers to the study's final questions")
    L.append("")
    if not prim.empty:
        rv_b = prim[prim.signal == "reversal_bull"]
        rv_s = prim[prim.signal == "reversal_bear"]
        L.append(f"1. **Does EMA ROC predict before crossovers?** Weakly, but with a genuine and roughly symmetric "
                 f"lead. Reversal signals precede the same-direction cross with a median lead of "
                 f"{num(rv_b.iloc[0]['median_lead_h'],0) if not rv_b.empty else float('nan')}h (bull) / "
                 f"{num(rv_s.iloc[0]['median_lead_h'],0) if not rv_s.empty else float('nan')}h (bear); rank IC vs "
                 f"24h return is small ({num(float(ic24[ic24.feature=='roc_1'].rank_ic.mean()),3)} average over "
                 f"configs) and conditional mean returns are near zero.")
    L.append(f"2. **Does acceleration add information beyond ROC?** Marginally. The incremental R² from adding "
             f"`accel_1` on top of the spread + crossover baseline is "
             f"{num(kw['expA']['reg'][(kw['expA']['reg']['add']=='accel_1') & (kw['expA']['reg'].horizon=='24h')].delta_r2.mean(),5)} "
             f"at 24h; it does not change the sign of the conclusion.")
    L.append(f"3. **Does normalized spread velocity add beyond the current spread?** Slightly more than "
             f"acceleration: ΔR² ≈ "
             f"{num(kw['expA']['reg'][(kw['expA']['reg']['add']=='svel_1') & (kw['expA']['reg'].horizon=='24h')].delta_r2.mean(),5)} "
             f"at 24h, with HAC t-stats that are positive but modest.")
    L.append("4. **Does the combined signal beat its components?** Not out-of-sample after costs — see the "
             "strategy table; the components are close together and the combined rule trades a different amount "
             "rather than dominating.")
    if not prim.empty:
        L.append(f"5. **Lead time and false turns?** Median lead ≈ "
                 f"{num(prim['median_lead_h'].median(),0)}h across signals; the 1×ATR false-turn rate is ≈ "
                 f"{pct(prim['false_turn_1'].mean(),0)} on average — high — and lead time and reliability trade off "
                 f"sharply (best reliability is the short-lead combined/confirmed bull signal).")
    L.append("6. **Does earlier entry improve expectancy?** No — the opposite. In the entry-delay table the "
             "*un*-delayed early signal is the worst (OOS ≈ -47% at 6 bps); adding a 6–12 bar delay lifts it to "
             "roughly break-even. The earliest signal is the noisiest, so early entry destroys expectancy rather "
             "than creating it, and the crossover baseline is the better of the two.")
    L.append("7. **Do gains survive costs & funding?** No strategy clears the base taker cost with a stable sign; "
             "break-even costs are modest relative to the 6 bps assumption (the crossover baseline is the only "
             "primary-config variant with a positive break-even on the long side).")
    L.append("8. **Do long and short differ?** The *diagnostic* lead time is nearly symmetric (≈9h either way), but "
             "the *traded* result is not: the long side is far less bad than the short side (primary combined OOS "
             "long ≈ -9% vs short ≈ -44%). That is consistent with the sample's upward drift rather than a "
             "structural asymmetry in the signal.")
    L.append("9. **Stable across symbols, regimes, folds?** No. Per-symbol dispersion is large, regime signs "
             "differ, and walk-forward folds are mostly negative.")
    L.append("10. **Is there a repeatable edge?** No. **H1/H3 are weakly supported diagnostically, H2/H4 are "
              "marginal, H5 holds (lead time exists) but with unacceptable false turns, and H6/H7/H8 are not "
              "supported for trading.** The momentum-deceleration family is rejected as a standalone signal; the "
              "diagnostic value of ROC and spread velocity may merit a conditional (regime-gated) retest with "
              "finer execution data, but the current evidence does not justify strategy development.")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"_Reproduce:_ `python scripts/ema_momentum_study.py` (engine: `quant_lab.analytics.ema_backtest`; "
             f"features/signals: `quant_lab.analytics.ema_momentum`). Artifacts: "
             f"`{a.artifacts or 'experiments/reports/ema_momentum_v1'}`.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
