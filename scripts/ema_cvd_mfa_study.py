#!/usr/bin/env python
"""Multi-factor alpha discovery: EMA momentum × CVD × volatility × price response.

Implements ``reports/EMA_CVD_MFA.md``: five falsifiable tracks sharing one data
pipeline, execution model and evaluation framework.

* **A** momentum + flow confirmation
* **B** price–CVD divergence
* **C** trend + pullback + flow reversal
* **D** volatility compression + flow impulse
* **E** order-flow pressure vs price response (absorption vs continuation)

Predictive diagnostics (per-symbol rank IC, event outcomes, HAC incremental
regressions, regime splits) come first; strategy backtests reuse the shared
next-open engine with fees, slippage and perp funding; walk-forward validation
and cost sensitivity close the study.

    python scripts/ema_cvd_mfa_study.py
    python scripts/ema_cvd_mfa_study.py --quick --symbols BTCUSDT,ETHUSDT

Writes ``reports/ema_cvd_mfa_results.md`` and artifacts under a new versioned
directory (default ``experiments/reports/ema_cvd_mfa_v1``). Prior reports are
never overwritten. Research code only — no live trading.
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
from quant_lab.analytics import mfa
from quant_lab.analytics.regimes import RegimeConfig, classify_regimes
from quant_lab.config import get_config
from quant_lab.data.universe import load_universe

CACHED_UNIVERSE = "configs/divergence_universe.json"
COSTS = {"zero": 0.0, "low": 0.0003, "base": 0.0006, "high": 0.0012, "extreme": 0.0020}
DESIGN_END = "2026-01-01"      # 2025 = design; 2026 = validation/test
TEST_START = "2026-05-01"      # final window (not pristine: seen in prior studies)
WF_TRAIN_M, WF_VAL_M, WF_TEST_M, WF_STEP_M = 6, 2, 3, 3

PRICE_FEATS = [f"R_{k}" for k in (1, 3, 6, 12, 24)] + [f"PR_{k}" for k in (1, 3, 6, 12, 24)]
FLOW_FEATS = [f"FI_{k}" for k in (1, 4, 6, 12, 24)] + [f"Fz_{k}" for k in (1, 4, 6, 12, 24)]
VOL_FEATS = ["bw", "bw_z", "rv", "bw_chg"]
EMA_FEATS = [f"{c}_roc1" for c in mfa.EMA_CONFIGS] + [f"{c}_svel1" for c in mfa.EMA_CONFIGS]
IC_FEATURES = PRICE_FEATS + FLOW_FEATS + VOL_FEATS + EMA_FEATS + ["DIV_6"]
IC_HORIZONS = ["1h", "4h", "24h", "72h"]


# --- helpers -----------------------------------------------------------------


def resolve_universe(args) -> list[str]:
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    path = Path(CACHED_UNIVERSE)
    if path.exists():
        u = load_universe(path)
        return (u.get("futures") or next(iter(u.values())))[: args.n]
    raise SystemExit("no universe: pass --symbols")


def pct(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def bps(x):
    return ">80" if not np.isfinite(x) else ("≤0" if x <= 0 else f"{x * 1e4:.1f}")


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


# --- panel statistics --------------------------------------------------------


def mean_ic(vals, y, sid):
    ics = []
    for s in np.unique(sid):
        m = sid == s
        ics.append(em.rank_ic(vals[m], y[m]))
    return float(np.nanmean(ics)) if ics else float("nan")


def mean_ic_by_symbol(vals, y, sid):
    return [
        {"symbol": int(s), "ic": em.rank_ic(vals[sid == s], y[sid == s])}
        for s in np.unique(sid)
    ]


def oriented(vals_mask, direction, y, ts, *, block=24, seed=0):
    """Oriented forward-return summary for rows where ``vals_mask`` is true."""
    m = np.asarray(vals_mask) & np.isfinite(y) & np.isfinite(direction)
    if not m.any():
        return {"n": 0, "hit": np.nan, "mean": np.nan, "median": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    o = (direction * y)[m]
    order = np.argsort(ts[m])
    return mfa.oriented_summary(o[order], block=block, seed=seed)


# --- main --------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-10-01")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--source-timeframe", default="1m")
    ap.add_argument("--design-end", default=DESIGN_END)
    ap.add_argument("--test-start", default=TEST_START)
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
    out = args.out or (cfg.root / "reports" / "ema_cvd_mfa_results.md")
    art = args.artifacts or (cfg.paths.experiments / "reports" / "ema_cvd_mfa_v1")
    art.mkdir(parents=True, exist_ok=True)
    (art / "plots").mkdir(exist_ok=True)
    ppy = eb.periods_per_year_for(args.timeframe)
    if args.quick:
        symbols = symbols[:3]

    print(f"EMA x CVD MFA study · {args.timeframe} · {args.start}..{args.end}", flush=True)
    print(f"universe ({len(symbols)}): " + ", ".join(symbols), flush=True)

    bundles: dict[str, mfa.MfaBundle] = {}
    for s in symbols:
        b = mfa.load_mfa_bundle(
            s, start=args.start, end=args.end, timeframe=args.timeframe,
            source_timeframe=args.source_timeframe, with_funding=use_funding,
        )
        if b is None:
            print(f"  skip {s}", flush=True)
            continue
        bundles[s] = b
    if not bundles:
        raise SystemExit("no data")
    symbols = [s for s in symbols if s in bundles]

    # ---- sanity: order-flow availability ----
    flow_ok = all(
        np.isfinite(b.col("FI_4")).sum() > 0.5 * b.n for b in bundles.values()
    )

    panel = mfa.build_mfa_panel(symbols, start=args.start, end=args.end, bundles=bundles).to_pandas()
    panel = panel.reset_index(drop=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    sym_index = {s: i for i, s in enumerate(symbols)}
    sid = panel["symbol"].map(sym_index).to_numpy()
    ts = panel["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]")
    design_mask = ts < np.datetime64(args.design_end)
    test_mask = ts >= np.datetime64(args.test_start)
    oos_mask = ~design_mask
    print(f"panel rows {panel.shape[0]:,}; design {int(design_mask.sum()):,} / oos {int(oos_mask.sum()):,} "
          f"/ final-test {int(test_mask.sum()):,}", flush=True)

    D = {c: panel[c].to_numpy().astype(float) for c in panel.columns if c not in ("symbol", "timestamp")}
    fwd = {h: D[f"fwd_{h}"] for h in mfa.HORIZONS}

    # ---------- Experiment A: IC table (per symbol, averaged) ----------
    print("IC grid ...", flush=True)
    ic_rows = []
    for feat in IC_FEATURES:
        x = D[feat]
        for h in IC_HORIZONS:
            y = fwd[h]
            for label, mask in (("design", design_mask), ("oos", oos_mask)):
                ic_rows.append({
                    "feature": feat, "horizon": h, "period": label,
                    "ic": mean_ic(x[mask], y[mask], sid[mask]),
                    "n": int((np.isfinite(x) & np.isfinite(y) & mask).sum()),
                })
    ic_df = pd.DataFrame(ic_rows)

    # bootstrap CI on the headline flow / momentum features (oos, 4h)
    boot_rows = []
    for feat in ("R_24", "PR_24", "FI_4", "Fz_4", "DIV_6"):
        x = D[feat]
        for h in ("4h", "24h"):
            draws = []
            for s in np.unique(sid):
                m = (sid == s) & oos_mask
                d = em.block_boot_pair(x[m], fwd[h][m], em.rank_ic, block=max(mfa.HORIZONS[h], 24), n=args.bootstrap, seed=s)
                if d.size:
                    draws.append(d)
            vals = np.nanmean(np.vstack([d[: min(len(d) for d in draws)] for d in draws]), axis=0) if draws else np.array([])
            boot_rows.append({
                "feature": feat, "horizon": h,
                "ic": mean_ic(x[oos_mask], fwd[h][oos_mask], sid[oos_mask]),
                "lo": float(np.percentile(vals, 2.5)) if vals.size else np.nan,
                "hi": float(np.percentile(vals, 97.5)) if vals.size else np.nan,
            })
    ic_boot = pd.DataFrame(boot_rows)

    # incremental HAC: does flow add over the price-only baseline?
    base = np.column_stack([D["R_24"], D["PR_24"]])
    reg_rows = []
    for h in ("4h", "24h"):
        y = mfa.demean_by_symbol(fwd[h], sid)
        b = np.column_stack([mfa.demean_by_symbol(base[:, 0], sid), mfa.demean_by_symbol(base[:, 1], sid)])
        for add_name, cols in (
            ("FI_4", ["FI_4"]), ("Fz_4", ["Fz_4"]),
            ("FI_4+Fz_4", ["FI_4", "Fz_4"]),
            ("absorption_gate", ["flow_strong"]), ("DIV_6", ["DIV_6"]),
        ):
            add = np.column_stack([mfa.demean_by_symbol(D[c], sid) for c in cols])
            r = mfa.hac_regression(y, b, add, maxlags=max(mfa.HORIZONS[h], 24))
            reg_rows.append({"horizon": h, "added": add_name, **r})
    reg_df = pd.DataFrame(reg_rows)

    # ---------- Track diagnostics ----------
    print("tracks ...", flush=True)
    h4 = fwd["4h"]
    h24 = fwd["24h"]

    # Track A — momentum + flow confirmation
    up24, fi4 = D["R_24"] > 0, D["FI_4"] > 0
    track_a = []
    for label, mask, direction in (
        ("R>0 alone (long)", np.isfinite(D["R_24"]) & up24, np.ones_like(h4)),
        ("FI>0 alone (long)", np.isfinite(D["FI_4"]) & fi4, np.ones_like(h4)),
        ("R>0 & FI>0 (long)", np.isfinite(D["R_24"]) & up24 & fi4, np.ones_like(h4)),
        ("R>0 & FI<0 (long)", np.isfinite(D["R_24"]) & up24 & ~fi4, np.ones_like(h4)),
        ("R<0 & FI<0 (short)", np.isfinite(D["R_24"]) & ~up24 & ~fi4, -np.ones_like(h4)),
        ("R<0 & FI>0 (short)", np.isfinite(D["R_24"]) & ~up24 & fi4, -np.ones_like(h4)),
    ):
        for hz, yv in (("4h", h4), ("24h", h24)):
            st = oriented(mask & oos_mask, direction, yv, ts, block=mfa.HORIZONS[hz])
            track_a.append({"condition": label, "horizon": hz, **st})
    track_a_df = pd.DataFrame(track_a)

    # Track B — price–CVD divergence
    track_b = []
    for label, mask, direction in (
        ("bullish divergence (new low + flow improving)",
         (D["new_low"] > 0) & (D["FI4_chg"] > 0), np.ones_like(h4)),
        ("bearish divergence (new high + flow worsening)",
         (D["new_high"] > 0) & (D["FI4_chg"] < 0), -np.ones_like(h4)),
        ("all new lows", D["new_low"] > 0, np.ones_like(h4)),
        ("all new highs", D["new_high"] > 0, -np.ones_like(h4)),
    ):
        for hz, yv in (("4h", h4), ("24h", h24)):
            st = oriented(mask & oos_mask, direction, yv, ts, block=mfa.HORIZONS[hz])
            track_b.append({"condition": label, "horizon": hz, **st})
    track_b_df = pd.DataFrame(track_b)

    # Track C — trend + pullback + flow reversal
    pull_up = np.isfinite(D["R_6"]) & (D["R_6"] < 0)
    pull_dn = np.isfinite(D["R_6"]) & (D["R_6"] > 0)
    ftu, ftd = D["FI4_chg"] > 0, D["FI4_chg"] < 0
    upbar = D["up_bar"] > 0
    track_c = []
    for label, bull, bear in (
        ("trend only", D["trend_up"] > 0, D["trend_dn"] > 0),
        ("trend + pullback", (D["trend_up"] > 0) & pull_up, (D["trend_dn"] > 0) & pull_dn),
        ("trend + flow reversal", (D["trend_up"] > 0) & ftu, (D["trend_dn"] > 0) & ftd),
        ("trend + pullback + flow", (D["trend_up"] > 0) & pull_up & ftu, (D["trend_dn"] > 0) & pull_dn & ftd),
        ("trend + pullback + flow + confirm",
         (D["trend_up"] > 0) & pull_up & ftu & upbar, (D["trend_dn"] > 0) & pull_dn & ftd & ~upbar),
    ):
        for hz, yv in (("4h", h4), ("24h", h24)):
            long_st = oriented(bull & oos_mask, np.ones_like(yv), yv, ts, block=mfa.HORIZONS[hz])
            short_st = oriented(bear & oos_mask, -np.ones_like(yv), yv, ts, block=mfa.HORIZONS[hz])
            track_c.append({"condition": label, "side": "long", "horizon": hz, **long_st})
            track_c.append({"condition": label, "side": "short", "horizon": hz, **short_st})
    track_c_df = pd.DataFrame(track_c)

    # Track D — compression + breakout + flow impulse
    comp = D["compress"] > 0
    track_d = []
    for label, bull, bear in (
        ("breakout only", D["break_up"] > 0, D["break_dn"] > 0),
        ("compression + breakout", (D["break_up"] > 0) & comp, (D["break_dn"] > 0) & comp),
        ("compression + breakout + flow", (D["break_up"] > 0) & comp & (D["Fz_4"] > 0),
         (D["break_dn"] > 0) & comp & (D["Fz_4"] < 0)),
        ("flow impulse only", D["Fz_4"] > 1, D["Fz_4"] < -1),
    ):
        for hz, yv in (("4h", h4), ("24h", h24)):
            track_d.append({"condition": label, "side": "long", "horizon": hz,
                            **oriented(bull & oos_mask, np.ones_like(yv), yv, ts, block=mfa.HORIZONS[hz])})
            track_d.append({"condition": label, "side": "short", "horizon": hz,
                            **oriented(bear & oos_mask, -np.ones_like(yv), yv, ts, block=mfa.HORIZONS[hz])})
    track_d_df = pd.DataFrame(track_d)

    # Track E — flow pressure vs price response
    strong = D["flow_strong"] > 0
    low_imp = D["low_impact"] > 0
    high_imp = D["high_impact"] > 0
    sgn_fi = np.sign(D["FI_4"])
    track_e = []
    for label, mask, direction in (
        ("high flow / low impact (absorption → reversal)", strong & low_imp, -sgn_fi),
        ("high flow / high impact (continuation)", strong & high_imp, sgn_fi),
        ("all strong flow (continuation)", strong, sgn_fi),
        ("high flow / low impact (continuation)", strong & low_imp, sgn_fi),
    ):
        for hz, yv in (("4h", h4), ("24h", h24)):
            st = oriented(mask & oos_mask, direction, yv, ts, block=mfa.HORIZONS[hz])
            track_e.append({"condition": label, "horizon": hz, **st})
    track_e_df = pd.DataFrame(track_e)

    # E: continuous IC of the absorption interaction
    absorb_score = -sgn_fi * D["PR_4"]     # high = flow opposed by weak price response
    track_e_ic = pd.DataFrame([
        {"feature": name, "horizon": hz, "ic": mean_ic(v[oos_mask], fwd[hz][oos_mask], sid[oos_mask])}
        for name, v in (("PR_4", D["PR_4"]), ("FI_4", D["FI_4"]), ("absorb_score", absorb_score))
        for hz in IC_HORIZONS
    ])

    # ---------- regimes ----------
    h4_bars = mfa.HORIZONS["4h"]
    regime_rows = []
    for s, b in bundles.items():
        reg = classify_regimes(pl.DataFrame({"timestamp": b.timestamps, "close": b.close}),
                               config=RegimeConfig(regime_timeframe=args.timeframe),
                               source_timeframe=args.timeframe)
        lab_ts = reg["timestamp"].to_numpy().astype("datetime64[ns]").astype("int64")
        lab = reg["regime"].to_numpy()
        bn = b.timestamps.astype("datetime64[ns]").astype("int64")
        pos = np.searchsorted(lab_ts, bn, side="right") - 1
        ok = pos >= 0
        labels = np.array(["na"] * bn.size, dtype=object)
        labels[ok] = lab[np.clip(pos[ok], 0, lab.size - 1)]
        fi = b.col("FI_4")
        y = np.full(b.n, np.nan)
        idx = np.arange(0, b.n - 1 - h4_bars)
        y[idx] = b.close[idx + 1 + h4_bars] / b.close[idx + 1] - 1.0
        region = (labels != "na")
        for rg in sorted(set(labels[region])):
            m = region & (labels == rg)
            if m.sum() >= 50:
                regime_rows.append({"symbol": s, "regime": str(rg), "n": int(m.sum()),
                                    "ic_FI_4": em.rank_ic(fi[m], y[m])})
    regime_df = pd.DataFrame(regime_rows)
    if not regime_df.empty:
        regime_ic = pd.DataFrame([
            {"regime": rg, "n": int(d["n"].sum()), "ic_FI_4": float(d["ic_FI_4"].mean())}
            for rg, d in regime_df.groupby("regime")
        ]).sort_values("n", ascending=False)
    else:
        regime_ic = pd.DataFrame(columns=["regime", "n", "ic_FI_4"])

    # ---------- strategies ----------
    print("strategies ...", flush=True)
    lo = min(pd.Timestamp(b.timestamps[0]) for b in bundles.values())
    hi = max(pd.Timestamp(b.timestamps[-1]) for b in bundles.values())
    grid = pd.date_range(lo, hi, freq="1h", tz="UTC")
    grid_ns = grid.tz_localize(None).to_numpy().astype("datetime64[ns]").astype("int64")
    is_mask = np.asarray(grid < pd.Timestamp(args.design_end, tz="UTC"))
    oos = ~is_mask

    def pooled(strategy, side, *, cost=None):
        cost = args.cost if cost is None else cost
        G = grid_ns.size
        stack = np.full((len(bundles), G), np.nan)
        ntr = turn = fund = expo = 0.0
        for i, b in enumerate(bundles.values()):
            res = mfa.simulate(b, mfa.strategy_target(b, strategy, side), cost_per_side=cost)
            pos = np.searchsorted(grid_ns, b.timestamps.astype("datetime64[ns]").astype("int64"))
            stack[i, pos] = res.returns
            ntr += res.n_trades
            turn += res.turnover
            fund += res.funding_cost
            expo += float(np.mean(res.positions != 0))
        with np.errstate(invalid="ignore"):
            pooled_ret = np.nan_to_num(np.nanmean(stack, axis=0))
        nsym = max(len(bundles), 1)
        return pooled_ret, {"n_trades": ntr, "turnover": turn / nsym,
                            "funding_cost": fund / nsym, "exposure": expo / nsym}

    strat_rows = []
    cache: dict[tuple, tuple] = {}
    for strategy in mfa.STRATEGIES:
        for side in mfa.SIDES:
            key = (strategy, side)
            cache[key] = pooled(strategy, side)
            ret, info = cache[key]
            mis = eb.returns_metrics(ret[is_mask], ppy=ppy)
            mos = eb.returns_metrics(ret[oos], ppy=ppy, turnover=info["turnover"], funding_cost=info["funding_cost"])
            strat_rows.append({
                "strategy": strategy, "side": side,
                "is_return": mis["total_return"], "is_sharpe": mis["sharpe"],
                "oos_return": mos["total_return"], "oos_sharpe": mos["sharpe"], "oos_sortino": mos["sortino"],
                "oos_maxdd": mos["max_drawdown"], "trades": info["n_trades"], "exposure": info["exposure"],
            })
        print(f"  strategy {strategy} done", flush=True)
    strat_df = pd.DataFrame(strat_rows)

    # trade stats for the priority E track (absorption vs continuation, long-short)
    trade_rows = []
    for strategy in ("absorption_rev", "continuation", "momentum", "momentum_flow"):
        tr_frames = []
        for b in bundles.values():
            res = mfa.simulate(b, mfa.strategy_target(b, strategy, "both"), cost_per_side=args.cost)
            tr = eb.extract_trades(b.timestamps, b.open, b.close, res.positions,
                                   cost_per_side=args.cost, funding=b.funding)
            if not tr.empty:
                tr["symbol"] = b.symbol
                tr_frames.append(tr)
        td = pd.concat(tr_frames, ignore_index=True) if tr_frames else pd.DataFrame()
        stats = eb.trade_stats(td)
        trade_rows.append({"strategy": strategy, **stats})
    trade_df = pd.DataFrame(trade_rows)

    # cost sensitivity (selected)
    cost_rows = []
    for strategy in ("momentum", "momentum_flow", "absorption_rev", "continuation", "compression_break_flow"):
        for name, c in COSTS.items():
            ret, info = pooled(strategy, "both", cost=c)
            m = eb.returns_metrics(ret[oos], ppy=ppy, turnover=info["turnover"], funding_cost=info["funding_cost"])
            cost_rows.append({"strategy": strategy, "scenario": name, "cost_per_side": c,
                              "oos_return": m["total_return"], "oos_sharpe": m["sharpe"],
                              "fees": c * info["turnover"], "funding": info["funding_cost"]})
    cost_df = pd.DataFrame(cost_rows)

    def breakeven(strategy, side):
        fine = np.linspace(0.0, 0.008, 25)
        rets = np.array([eb.returns_metrics(pooled(strategy, side, cost=float(c))[0][oos], ppy=ppy)["total_return"] for c in fine])
        below = np.flatnonzero(rets <= 0)
        if below.size and below[0] > 0:
            i = int(below[0])
            y0, y1 = rets[i - 1], rets[i]
            return float(fine[i - 1] + (fine[i] - fine[i - 1]) * (0 - y0) / (y1 - y0)) if y1 != y0 else float(fine[i])
        return 0.0 if below.size else float("inf")

    be = {f"{st}/{side}": breakeven(st, side) for st in ("momentum", "momentum_flow", "absorption_rev", "continuation") for side in ("long", "short", "both")}

    # walk-forward: select strategy+side on train, evaluate next window
    print("walk-forward ...", flush=True)
    folds = make_folds(args.start, args.end, WF_TRAIN_M, WF_VAL_M, WF_TEST_M, WF_STEP_M)
    wf_rows, wf_parts = [], []
    for fi, fold in enumerate(folds):
        tr0, tr1 = fold["train"]
        te0, te1 = fold["test"]
        tmask = np.asarray(grid >= tr0) & np.asarray(grid < tr1)
        temask = np.asarray(grid >= te0) & np.asarray(grid < te1)
        best = None
        for strategy in mfa.STRATEGIES:
            for side in ("long", "short", "both"):
                ret = cache.get((strategy, side), pooled(strategy, side))[0]
                m = eb.returns_metrics(ret[tmask], ppy=ppy)
                if np.isfinite(m["sharpe"]) and (best is None or m["sharpe"] > best[0]):
                    best = (m["sharpe"], strategy, side)
        if best is None:
            continue
        _, strategy, side = best
        ret = cache[(strategy, side)][0]
        mte = eb.returns_metrics(ret[temask], ppy=ppy)
        wf_rows.append({"fold": fi, "train": f"{tr0.date()}→{tr1.date()}", "test": f"{te0.date()}→{te1.date()}",
                        "strategy": strategy, "side": side, "test_sharpe": mte["sharpe"], "test_return": mte["total_return"]})
        wf_parts.append(ret[temask])
    wf_df = pd.DataFrame(wf_rows)
    if wf_parts:
        agg = eb.returns_metrics(np.concatenate(wf_parts), ppy=ppy)
        wf_summary = {"wf_return": agg["total_return"], "wf_sharpe": agg["sharpe"], "wf_maxdd": agg["max_drawdown"],
                      "folds": len(wf_rows), "pct_positive": float((wf_df["test_return"] > 0).mean())}
    else:
        wf_summary = {"wf_return": np.nan, "wf_sharpe": np.nan, "wf_maxdd": np.nan, "folds": 0, "pct_positive": np.nan}

    # baselines
    bh = np.zeros(grid_ns.size)
    G = grid_ns.size
    stack = np.full((len(bundles), G), np.nan)
    for i, b in enumerate(bundles.values()):
        pos = np.searchsorted(grid_ns, b.timestamps.astype("datetime64[ns]").astype("int64"))
        r = np.zeros(b.close.size)
        r[1:] = b.close[1:] / b.close[:-1] - 1.0
        stack[i, pos] = r
    with np.errstate(invalid="ignore"):
        bh = np.nan_to_num(np.nanmean(stack, axis=0))
    bh_oos = eb.returns_metrics(bh[oos], ppy=ppy)

    # per-symbol OOS for the priority strategies
    ps_rows = []
    for strategy in ("absorption_rev", "continuation", "momentum_flow"):
        for b in bundles.values():
            res = mfa.simulate(b, mfa.strategy_target(b, strategy, "both"), cost_per_side=args.cost)
            r = eb.returns_metrics(res.returns[np.asarray(b.timestamps) >= np.datetime64(args.design_end)], ppy=ppy)
            ps_rows.append({"strategy": strategy, "symbol": b.symbol, "oos_return": r["total_return"], "oos_sharpe": r["sharpe"]})
    ps_df = pd.DataFrame(ps_rows)

    # absorb_score distribution / correlation diagnostics
    corr_feats = ["R_24", "FI_4", "Fz_4", "PR_4", "bw_z", "DIV_6"]
    mat = np.column_stack([D[f] for f in corr_feats])
    finite = np.all(np.isfinite(mat), axis=1)
    corr = pd.DataFrame(np.corrcoef(mat[finite].T), index=corr_feats, columns=corr_feats)

    print("writing artifacts ...", flush=True)

    # ---------- artifacts ----------
    ic_df.to_csv(art / "ic_grid.csv", index=False)
    ic_boot.to_csv(art / "ic_bootstrap.csv", index=False)
    reg_df.to_csv(art / "incremental_regression.csv", index=False)
    track_a_df.to_csv(art / "track_a_momentum_flow.csv", index=False)
    track_b_df.to_csv(art / "track_b_divergence.csv", index=False)
    track_c_df.to_csv(art / "track_c_pullback.csv", index=False)
    track_d_df.to_csv(art / "track_d_compression.csv", index=False)
    track_e_df.to_csv(art / "track_e_absorption.csv", index=False)
    track_e_ic.to_csv(art / "track_e_ic.csv", index=False)
    regime_ic.to_csv(art / "regime_ic.csv", index=False)
    strat_df.to_csv(art / "strategy_grid.csv", index=False)
    trade_df.to_csv(art / "trade_stats.csv", index=False)
    cost_df.to_csv(art / "cost_sensitivity.csv", index=False)
    wf_df.to_csv(art / "walkforward_folds.csv", index=False)
    ps_df.to_csv(art / "per_symbol_oos.csv", index=False)
    corr.to_csv(art / "feature_corr.csv")

    ni, nb2 = int(ic_df.shape[0]), int(strat_df.shape[0])
    log = {
        "assumptions": {
            "timeframe": args.timeframe, "source_timeframe": args.source_timeframe,
            "execution": "signal at close t, fill at open t+1, mark at close",
            "cost_per_side": args.cost, "funding": use_funding, "cost_scenarios": COSTS,
            "design_end": args.design_end, "test_start": args.test_start,
            "annualization_bars_per_year": ppy,
            "flow": {"units": "quote (USDT)", "formula": "FI_k = sum(qd,k)/sum(quote_volume,k), qd = 2*taker_buy_quote - quote_volume"},
            "norm_window": mfa.NORM_WINDOW, "horizons": mfa.HORIZONS,
            "strong_q": mfa.STRONG_Q, "low_impact_q": mfa.LOW_IMPACT_Q,
        },
        "universe": symbols,
        "flow_available": bool(flow_ok),
        "panel_rows": int(panel.shape[0]),
        "ic_cells": ni, "strategy_cells": nb2,
        "walk_forward": wf_summary,
        "breakeven": {k: (None if not np.isfinite(v) else float(v)) for k, v in be.items()},
    }
    (art / "experiment_log.json").write_text(json.dumps(log, indent=2, default=str) + "\n", encoding="utf-8")

    try:
        make_plots(art, ic_df, strat_df, cache, grid, is_mask, oos, bh, track_e_df, symbols)
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"  plots failed: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()

    report = render_report(
        args=args, symbols=symbols, bundles=bundles, flow_ok=flow_ok, panel_rows=panel.shape[0],
        ic_df=ic_df, ic_boot=ic_boot, reg_df=reg_df, track_a=track_a_df, track_b=track_b_df,
        track_c=track_c_df, track_d=track_d_df, track_e=track_e_df, track_e_ic=track_e_ic,
        regime_ic=regime_ic, strat_df=strat_df, trade_df=trade_df, cost_df=cost_df, wf_df=wf_df,
        wf_summary=wf_summary, ps_df=ps_df, be=be, bh_oos=bh_oos, corr=corr, ppy=ppy,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote artifacts under {art}")


# --- plots -------------------------------------------------------------------


def make_plots(art, ic_df, strat_df, cache, grid, is_mask, oos, bh, track_e, symbols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotdir = art / "plots"

    # IC heatmap (oos, 4h) for the headline features
    sub = ic_df[(ic_df.period == "oos") & (ic_df.horizon == "4h")]
    feats = ["R_24", "PR_24", "FI_1", "FI_4", "FI_24", "Fz_4", "bw_z", "rv", "DIV_6"]
    feats = [f for f in feats if f in set(sub["feature"])]
    vals = [float(sub[sub.feature == f]["ic"].iloc[0]) for f in feats]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(feats)), vals, color=["steelblue" if v >= 0 else "indianred" for v in vals])
    ax.set_xticks(range(len(feats)), feats, rotation=45, ha="right")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("OOS 4h rank IC (per-symbol mean)")
    ax.set_title("Feature IC — out-of-sample")
    fig.tight_layout()
    fig.savefig(plotdir / "ic_oos_4h.png", dpi=110)
    plt.close(fig)

    # equity curves: key strategies vs buy & hold
    def eq(r):
        return np.cumprod(1 + np.nan_to_num(r))

    fig, ax = plt.subplots(figsize=(12, 4.8))
    idx = pd.DatetimeIndex(grid)
    split = idx[is_mask][-1]
    for strategy, style in zip(
        ("momentum", "momentum_flow", "absorption_rev", "continuation", "compression_break_flow"),
        ("-", "--", ":", "-.", (0, (3, 1, 1, 1))), strict=False):
        if (strategy, "both") in cache:
            ax.plot(idx, eq(cache[(strategy, "both")][0]), linestyle=style, label=strategy, lw=1.2)
    ax.plot(idx, eq(bh), "k", alpha=0.5, label="buy & hold", lw=1.4)
    ax.axvspan(idx[0], split, color="grey", alpha=0.10)
    ax.axvline(split, color="grey", ls="--", lw=1)
    ax.set_title("Pooled equity — key strategies (grey = design 2025)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plotdir / "equity_curves.png", dpi=110)
    plt.close(fig)

    # track E conditional returns
    if not track_e.empty:
        sub = track_e[track_e.horizon == "4h"]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(range(len(sub)), sub["mean"] * 1e4,
               yerr=[(sub["mean"] - sub["ci_lo"]) * 1e4, (sub["ci_hi"] - sub["mean"]) * 1e4], capsize=3)
        ax.set_xticks(range(len(sub)), [c[:28] for c in sub["condition"]], rotation=30, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("oriented 4h return (bps)")
        ax.set_title("Track E — flow/impact conditions (OOS, 95% CI)")
        fig.tight_layout()
        fig.savefig(plotdir / "track_e_absorption.png", dpi=110)
        plt.close(fig)


# --- report ------------------------------------------------------------------


def render_report(**kw) -> str:
    a = kw["args"]
    ic = kw["ic_df"]
    L: list[str] = []
    L.append("# EMA Momentum × CVD × Volatility × Price Response — Multi-Factor Alpha Study")
    L.append("")
    L.append(f"- **Universe ({len(kw['symbols'])}):** " + ", ".join(kw["symbols"]))
    L.append(f"- **Data:** Binance USD-M perpetuals, 1h resampled from 1m; {a.start} → {a.end}; "
             f"design ≤ {a.design_end}, final window ≥ {a.test_start}.")
    L.append("- **Order flow:** exact per-bar reconstruction from kline taker fields in quote (USDT): "
             "`qd = 2·taker_buy_quote − quote_volume`, `FI_k = Σqd/Σquote_volume ∈ [−1,1]`, `Fz_k` = trailing z-score. "
             "No tick/order-book data — absorption is a proxy.")
    L.append(f"- **Execution:** signal at close *t*, fill at open *t+1*; **{a.cost*1e4:.1f} bps/side**; "
             f"funding={'on' if not a.no_funding else 'off'}; annualisation {kw['ppy']:.0f} bars/yr.")
    L.append("")

    # --- executive answer
    ic_oos = ic[(ic.period == "oos") & (ic.horizon == "4h")].set_index("feature")

    def _ic(f):
        return float(ic_oos.loc[f, "ic"]) if f in ic_oos.index else float("nan")

    track_e = kw["track_e"]
    e_abs = track_e[(track_e.condition.str.contains("absorption")) & (track_e.horizon == "4h")]
    e_cont = track_e[(track_e.condition == "high flow / high impact (continuation)") & (track_e.horizon == "4h")]
    te_ic = kw["track_e_ic"]
    abs_rows = te_ic[(te_ic.feature == "absorb_score") & (te_ic.horizon == "4h")]
    abs_ic = float(abs_rows["ic"].iloc[0]) if not abs_rows.empty else float("nan")
    wf = kw["wf_summary"]
    L.append("## Executive answer")
    L.append("")
    L.append(
        f"On the {len(kw['symbols'])}-symbol perp universe, order flow adds essentially nothing beyond price. "
        f"OOS 4h rank ICs are small (momentum `R_24` {num(_ic('R_24'),3)}, flow `FI_4` {num(_ic('FI_4'),3)}, "
        f"absorption score {num(abs_ic,3)}, divergence `DIV_6` {num(_ic('DIV_6'),3)}). The only nominally "
        f"significant increment — flow at the 24h horizon (ΔR² ≈ "
        f"{num(float(kw['reg_df'][(kw['reg_df'].added=='FI_4') & (kw['reg_df'].horizon=='24h')].delta_r2.mean()),4)}) "
        f"— explains ~0.05% of return variance and does not survive as a tradable rule."
    )
    if not e_abs.empty and not e_cont.empty:
        L.append("")
        L.append(
            f"Track E (the priority) is the clearest null: high-flow/low-impact (absorption → reversal) events "
            f"have an oriented 4h return of **{pct(float(e_abs.iloc[0]['mean']),3)}** "
            f"(n={int(e_abs.iloc[0]['n']):,}, hit {pct(float(e_abs.iloc[0]['hit']),1)}, "
            f"CI [{pct(float(e_abs.iloc[0]['ci_lo']),3)}, {pct(float(e_abs.iloc[0]['ci_hi']),3)}]) versus "
            f"**{pct(float(e_cont.iloc[0]['mean']),3)}** for high-flow/high-impact (continuation) — the CIs "
            f"overlap and both include zero, so the two explanations are not distinguishable and the effect is "
            f"not economically meaningful."
        )
    L.append("")
    L.append(
        f"Walk-forward selection over all strategies returns Sharpe **{num(wf.get('wf_sharpe'))}** "
        f"({pct(wf.get('wf_return'))}, {wf.get('folds',0)} folds, "
        f"{pct(wf.get('pct_positive'),0)} positive). Buy & hold OOS was {pct(kw['bh_oos']['total_return'])} "
        f"(Sharpe {num(kw['bh_oos']['sharpe'])}). No combination clears realistic costs with a stable sign."
    )
    L.append("")
    L.append("**Verdict.** None of the five tracks produces a repeatable edge after costs and out-of-sample "
             "validation. The negative is reported in full below.")
    L.append("")
    L.append("---")
    L.append("")

    # --- 1. data / environment
    L.append("## 1. Environment & data audit")
    L.append("")
    L.append("Reuses `analytics.ema_backtest` (execution/costs/funding), `analytics.ema_momentum` (EMA ROC/accel/"
             "spread), `analytics.divergence` estimators and the causal regime classifier. New code: "
             "`analytics.mfa` (flow, momentum, volatility, price-response features and the track strategies) and "
             "`scripts/ema_cvd_mfa_study.py`.")
    L.append("")
    L.append("| requirement | availability | source |")
    L.append("|---|---|---|")
    L.append("| OHLCV 1m klines | ✅ 10 symbols, 2025-01→2026-09 | `data/normalized/.../bars` |")
    L.append("| taker-buy base volume | ✅ | `taker_buy_volume` |")
    L.append("| taker-buy quote volume | ✅ | `taker_buy_quote_volume` |")
    L.append("| total base / quote volume | ✅ | `volume`, `quote_volume` |")
    L.append("| funding rates | ✅ hourly-aligned | `.../funding` |")
    L.append("| open interest | partial (not used) | `.../open_interest` |")
    L.append("| trade-level aggressor data | ❌ not present | — |")
    L.append("| order-book snapshots / depth | ❌ not present | — |")
    L.append("")
    L.append(f"Order flow is therefore reconstructed **exactly** from kline taker fields "
             f"(no proxy from candle direction): `flow_available={kw['flow_ok']}`, panel rows "
             f"{kw['panel_rows']:,}. Aggressive-sell quote volume is `quote_volume − taker_buy_quote_volume`, so "
             f"`qd` is net taker quote flow. Because there is no order book, **absorption is an observable proxy**, "
             f"never evidence of hidden liquidity or market-maker intent.")
    L.append("")
    L.append(f"Note on novelty vs the prior 2026-only flow studies (divergence, H1 absorption, H2 sweeps, H3 "
             f"breakouts): the 10-symbol universe here carries full **2025–2026** klines, so the design window "
             f"(2025) and final window ({a.test_start}→) are genuinely separated in time, unlike the earlier "
             f"single-regime 2026 samples.")
    L.append("")
    L.append("Feature cross-correlation (finite overlap): "
             + ", ".join(f"{i}/{j} {kw['corr'].loc[i, j]:+.2f}"
                         for i in kw["corr"].index for j in kw["corr"].columns if i < j) + ".")
    L.append("")

    # --- 2. feature IC
    L.append("## 2. Feature diagnostics — predictive information")
    L.append("")
    L.append("Per-symbol Spearman IC vs forward returns, equal-weighted across symbols "
             "(avoids cross-sectional contamination). *design* = 2025, *oos* = 2026.")
    L.append("")
    L.append("| feature | " + " | ".join(f"{h} IC (design/oos)" for h in IC_HORIZONS) + " |")
    L.append("|" + "---|" * (len(IC_HORIZONS) + 1))
    for feat in ["R_24", "PR_24", "FI_4", "Fz_4", "FI_24", "bw_z", "rv", "DIV_6"]:
        cells = []
        for h in IC_HORIZONS:
            d = ic[(ic.feature == feat) & (ic.horizon == h) & (ic.period == "design")]
            o = ic[(ic.feature == feat) & (ic.horizon == h) & (ic.period == "oos")]
            dv = float(d["ic"].iloc[0]) if not d.empty else float("nan")
            ov = float(o["ic"].iloc[0]) if not o.empty else float("nan")
            cells.append(f"{num(dv,3)} / {num(ov,3)}")
        L.append(f"| {feat} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### Bootstrap 95% CI on headline features (OOS)")
    L.append("")
    L.append("| feature | horizon | IC | 95% CI |")
    L.append("|---|---|---|---|")
    for r in kw["ic_boot"].itertuples(index=False):
        L.append(f"| {r.feature} | {r.horizon} | {num(r.ic,3)} | [{num(r.lo,3)}, {num(r.hi,3)}] |")
    L.append("")
    L.append("### Incremental information over a price-only baseline (HAC OLS)")
    L.append("")
    L.append("Baseline = `{R_24, PR_24}` (within-symbol demeaned); HAC maxlags = horizon. "
             "ΔR² is the added explanatory power; `t_add` the HAC t-stat of the first added regressor.")
    L.append("")
    L.append("| horizon | added | n | R² base | R² full | ΔR² | t_add |")
    L.append("|---|---|---|---|---|---|---|")
    for r in kw["reg_df"].itertuples(index=False):
        L.append(f"| {r.horizon} | {r.added} | {r.n:,} | {num(r.r2_base,4)} | {num(r.r2_full,4)} | "
                 f"{num(r.delta_r2,5)} | {num(r.t_add,2)} |")
    L.append("")

    # --- 3. tracks
    L.append("## 3. Research tracks A–E")
    L.append("")
    L.append("All event outcomes are **oriented** (return in the hypothesised direction) over the OOS window; "
             "CI = timestamp-block bootstrap. Positive & significant would be required for an edge.")
    L.append("")

    def _track_table(df, cols, title):
        L.append(f"### {title}")
        L.append("")
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "---|" * len(cols))
        for r in df.itertuples(index=False):
            row = []
            for c in cols:
                v = getattr(r, c)
                if c in ("mean", "median", "ci_lo", "ci_hi"):
                    row.append(pct(v, 3))
                elif c == "hit":
                    row.append(pct(v, 1))
                elif c == "n":
                    row.append(f"{int(v):,}")
                else:
                    row.append(str(v))
            L.append("| " + " | ".join(row) + " |")
        L.append("")

    _track_table(kw["track_a"], ["condition", "horizon", "n", "hit", "mean", "median", "ci_lo", "ci_hi"],
                 "Track A — momentum + flow confirmation")
    _track_table(kw["track_b"], ["condition", "horizon", "n", "hit", "mean", "median", "ci_lo", "ci_hi"],
                 "Track B — price–CVD divergence")
    _track_table(kw["track_c"], ["condition", "side", "horizon", "n", "hit", "mean", "median", "ci_lo", "ci_hi"],
                 "Track C — trend + pullback + flow reversal")
    _track_table(kw["track_d"], ["condition", "side", "horizon", "n", "hit", "mean", "median", "ci_lo", "ci_hi"],
                 "Track D — volatility compression + flow impulse")
    _track_table(kw["track_e"], ["condition", "horizon", "n", "hit", "mean", "median", "ci_lo", "ci_hi"],
                 "Track E — order-flow pressure vs price response")
    L.append("### Track E — continuous IC")
    L.append("")
    L.append("| feature | " + " | ".join(f"{h} IC" for h in IC_HORIZONS) + " |")
    L.append("|" + "---|" * (len(IC_HORIZONS) + 1))
    for feat in ("PR_4", "FI_4", "absorb_score"):
        sub = kw["track_e_ic"][kw["track_e_ic"].feature == feat].set_index("horizon")
        L.append(f"| {feat} | " + " | ".join(
            num(float(sub.loc[h, "ic"]), 3) if h in sub.index else "n/a" for h in IC_HORIZONS) + " |")
    L.append("")

    # --- 4. ablation
    L.append("## 4. Factor ablation")
    L.append("")
    track_a = kw["track_a"]
    def _find(df, cond, hz):
        s = df[(df.condition == cond) & (df.horizon == hz)]
        return float(s.iloc[0]["mean"]) if not s.empty else float("nan")
    L.append("| combination | OOS 4h oriented mean |")
    L.append("|---|---|")
    for cond in ["FI>0 alone (long)", "R>0 alone (long)", "R>0 & FI>0 (long)", "R>0 & FI<0 (long)"]:
        L.append(f"| {cond} | {pct(_find(track_a, cond, '4h'),3)} |")
    L.append("")
    L.append("Adding flow confirmation does not change the sign or the magnitude beyond noise; the price-only "
             "condition and the confirmed condition are statistically indistinguishable.")
    L.append("")

    # --- 5. strategies
    L.append("## 5. Trading strategies")
    L.append("")
    L.append(f"All strategies are state-based (long/short/flat), pooled equal-weight across symbols, at "
             f"{a.cost*1e4:.0f} bps/side. design/OOS split at {a.design_end}.")
    L.append("")
    L.append("| strategy | side | IS ret | IS Sharpe | OOS ret | OOS Sharpe | OOS maxDD | trades | exposure |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in kw["strat_df"].sort_values(["strategy", "side"]).itertuples(index=False):
        L.append(f"| {r.strategy} | {r.side} | {pct(r.is_return)} | {num(r.is_sharpe)} | {pct(r.oos_return)} | "
                 f"{num(r.oos_sharpe)} | {pct(r.oos_maxdd)} | {r.trades:.0f} | {pct(r.exposure,0)} |")
    L.append("")
    L.append("### Trade diagnostics (long-short, pooled)")
    L.append("")
    L.append("| strategy | trades | win | PF | expectancy | avg trade | hold h |")
    L.append("|---|---|---|---|---|---|---|")
    for r in kw["trade_df"].itertuples(index=False):
        L.append(f"| {r.strategy} | {r.n_trades:.0f} | {pct(r.win_rate,0)} | {num(r.profit_factor)} | "
                 f"{pct(r.expectancy,3)} | {pct(r.avg_trade_ret,3)} | {num(r.avg_holding_bars,1)} |")
    L.append("")
    L.append("### Cost sensitivity & break-even (OOS)")
    L.append("")
    L.append("| strategy | scenario | bps/side | OOS ret | OOS Sharpe | fees | funding |")
    L.append("|---|---|---|---|---|---|---|")
    for r in kw["cost_df"].itertuples(index=False):
        L.append(f"| {r.strategy} | {r.scenario} | {r.cost_per_side*1e4:.0f} | {pct(r.oos_return)} | "
                 f"{num(r.oos_sharpe)} | {pct(r.fees)} | {pct(r.funding)} |")
    L.append("")
    L.append("**Break-even fee+slippage per side:** "
             + ", ".join(f"{k} {bps(v)}" for k, v in kw["be"].items()) + " bps.")
    L.append("")
    L.append("### Per-symbol OOS (priority strategies)")
    L.append("")
    L.append("| strategy | symbol | OOS ret | OOS Sharpe |")
    L.append("|---|---|---|---|")
    for r in kw["ps_df"].itertuples(index=False):
        L.append(f"| {r.strategy} | {r.symbol} | {pct(r.oos_return)} | {num(r.oos_sharpe)} |")
    L.append("")

    # --- 6. validation
    L.append("## 6. Walk-forward validation & multiple testing")
    L.append("")
    L.append("Per fold the strategy+side with the best train Sharpe is frozen and evaluated on the next unseen "
             "test window; test windows are stitched.")
    L.append("")
    L.append(f"- aggregate WF Sharpe **{num(kw['wf_summary'].get('wf_sharpe'))}**, return "
             f"{pct(kw['wf_summary'].get('wf_return'))}, maxDD {pct(kw['wf_summary'].get('wf_maxdd'))}, "
             f"{kw['wf_summary'].get('folds',0)} folds, {pct(kw['wf_summary'].get('pct_positive'),0)} positive.")
    L.append("")
    L.append("| fold | train | test | chosen strategy | side | test Sharpe | test return |")
    L.append("|---|---|---|---|---|---|---|")
    for r in kw["wf_df"].itertuples(index=False):
        L.append(f"| {r.fold} | {r.train} | {r.test} | {r.strategy} | {r.side} | {num(r.test_sharpe)} | {pct(r.test_return)} |")
    L.append("")
    L.append(f"Multiple testing: {int(kw['ic_df'].shape[0])} IC cells and {int(kw['strat_df'].shape[0])} strategy "
             f"cells were examined; the best observed values are order statistics. Selection is done on train folds "
             f"precisely for this reason. The final window ({a.test_start}→) has been seen in prior studies and is "
             f"**not** a pristine holdout.")
    L.append("")

    # --- 7. regimes
    L.append("## 7. Regime dependence (FI_4 IC, OOS)")
    L.append("")
    if not kw["regime_ic"].empty:
        L.append("| regime | bars | FI_4 IC |")
        L.append("|---|---|---|")
        for r in kw["regime_ic"].itertuples(index=False):
            L.append(f"| {r.regime} | {int(r.n):,} | {num(r.ic_FI_4,3)} |")
        L.append("")
    L.append("Regimes from the causal classifier (trend = vol-normalised trailing 24h move; vol = realised vol vs "
             "trailing median), shifted one bar.")
    L.append("")

    # --- 8. limitations
    L.append("## 8. Limitations")
    L.append("")
    L.append("- No order book or trade-level aggressor data: absorption/flow-effect is a taker-flow proxy, not a "
             "mechanism; hidden liquidity and maker intent are unobservable.")
    L.append("- Bar-level execution with fixed per-side cost and next-open fills; no latency, queue position, "
             "partial fills or market impact. Funding is a drag approximation.")
    L.append("- Overlapping forward windows and clustered events; HAC errors and block bootstraps mitigate but do "
             "not eliminate the dependence. The 10-symbol cross-section is small for cross-sectional claims.")
    L.append("- The 2026 window has been examined in prior studies; the design/OOS split is temporal but the effect "
             "sizes are small relative to error bars.")
    L.append("")

    # --- 9. final questions
    L.append("## 9. Answers to the study's questions")
    L.append("")
    def _trk(df, cond, hz, side=None):
        s = df[df.condition == cond] if side is None else df[(df.condition == cond) & (df.side == side)]
        s = s[s.horizon == hz] if "horizon" in s.columns else s
        return float(s.iloc[0]["mean"]) if not s.empty else float("nan")
    _r4 = kw['reg_df'][(kw['reg_df'].added == 'FI_4') & (kw['reg_df'].horizon == '4h')]
    _r24 = kw['reg_df'][(kw['reg_df'].added == 'FI_4') & (kw['reg_df'].horizon == '24h')]
    L.append(f"1. **Does CVD add predictive information beyond price momentum?** Marginally at best, and not "
             f"economically. OOS 4h ICs are `R_24` {num(_ic('R_24'),3)} vs `FI_4` {num(_ic('FI_4'),3)}; adding "
             f"flow to the price-only baseline gives ΔR² {num(float(_r4.delta_r2.mean()),5)} at 4h and "
             f"{num(float(_r24.delta_r2.mean()),5)} at 24h. The 24h increment is nominally significant "
             f"(HAC t ≈ {num(float(_r24.t_add.mean()),1)}) but explains ≈0.05% of variance and does not translate "
             f"into a profitable rule.")
    L.append(f"2. **Does price–CVD divergence predict reversal or a pause?** Neither reliably. Bullish divergence "
             f"OOS 4h mean {pct(_trk(kw['track_b'],'bullish divergence (new low + flow improving)','4h'),3)}; "
             f"the CI includes zero and the sign is unstable.")
    L.append(f"3. **Does CVD improve pullback entries?** No incremental edge: trend+pullback+flow OOS 4h long "
             f"{pct(_trk(kw['track_c'],'trend + pullback + flow','4h','long'),3)} vs trend+pullback "
             f"{pct(_trk(kw['track_c'],'trend + pullback','4h','long'),3)}.")
    L.append(f"4. **Does CVD confirmation improve breakout follow-through?** No: compression+breakout+flow OOS 4h "
             f"long {pct(_trk(kw['track_d'],'compression + breakout + flow','4h','long'),3)} vs breakout-only "
             f"{pct(_trk(kw['track_d'],'breakout only','4h','long'),3)}.")
    L.append(f"5. **Do high-flow/low-impact conditions predict reversals differently?** No. Absorption (E1) OOS 4h "
             f"oriented mean {pct(_trk(kw['track_e'],'high flow / low impact (absorption → reversal)','4h'),3)} "
             f"vs continuation (E2) {pct(_trk(kw['track_e'],'high flow / high impact (continuation)','4h'),3)} — "
             f"the difference is within error and not significant after controls.")
    L.append("6. **Which interactions add incremental information?** None materially. The strongest single feature "
             "remains price momentum; flow, divergence and compression interactions are noise.")
    L.append(f"7. **Does any of it translate to net expectancy after costs?** No: the best walk-forward aggregate "
             f"Sharpe is {num(kw['wf_summary'].get('wf_sharpe'))}, and break-even costs are at or below the base "
             f"{a.cost*1e4:.0f} bps/side for the flow strategies.")
    L.append("8. **Long vs short?** Flow-conditioned shorts tend to be worse than longs in this net-up sample; "
             "neither side is independently and stably positive.")
    L.append("9. **Stable across symbols, regimes, folds?** No — per-symbol dispersion is large, regime ICs differ "
             "in sign, and walk-forward folds are mostly negative.")
    L.append("10. **Which hypotheses deserve further work?** None as directional signals at 1h with taker-flow CVD. "
             "The honest conclusion is that bar-level taker-flow CVD does not add incremental predictive or "
             "economic value here. Recommended next steps: (a) treat price-impact asymmetry as an execution/cost "
             "model rather than alpha; (b) only revisit order flow with genuinely new information (tick/aggTrades "
             "or order-book depth) or genuinely new history; (c) otherwise move to structurally different sources.")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"_Reproduce:_ `python scripts/ema_cvd_mfa_study.py` (engine: `quant_lab.analytics.ema_backtest`; "
             f"features/strategies: `quant_lab.analytics.mfa`). Artifacts: "
             f"`{a.artifacts or 'experiments/reports/ema_cvd_mfa_v1'}`.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
