#!/usr/bin/env python
"""Correlation study: BTCUSDT vs ZECUSDT daily bars (with ETHUSDT as control).

Loads the normalized daily bar datasets, aligns them on the UTC day, and runs a
full correlation battery: levels vs returns, Pearson/Spearman/Kendall, Fisher
and bootstrap confidence intervals, OLS with HAC errors, conditional and tail
dependence, lead-lag cross-correlation, rolling correlations and betas, monthly
stability, and leave-one-out outlier influence.

Outputs ``summary.json`` plus ``series.csv`` under the ``--out`` directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from quant_lab.config import get_config
from quant_lab.data.schemas import DatasetType
from quant_lab.data.storage import DatasetKey, read_dataset

PAIR = ("BTCUSDT", "ZECUSDT")
CONTROL = ("BTCUSDT", "ETHUSDT")


def _load(cfg, symbol: str, start: str, end: str) -> pd.Series:
    key = DatasetKey("binance", "futures", DatasetType.BARS, symbol, "1d")
    df = read_dataset(cfg.paths.data_normalized, key, start=start, end=end)
    pdf = df.select(["timestamp", "close"]).to_pandas().set_index("timestamp")
    s = pdf["close"].astype(float)
    s.index = pd.to_datetime(s.index, utc=True).normalize()
    return s.sort_index()


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 3 or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, stat, n_boot: int = 20000, seed: int = 7):
    rng = np.random.default_rng(seed)
    n = len(x)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            out[b] = np.nan
        else:
            out[b] = stat(xb, yb)
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def corr_bundle(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    pear = stats.pearsonr(x, y)
    spear = stats.spearmanr(x, y)
    kend = stats.kendalltau(x, y)
    lo, hi = fisher_ci(pear.statistic, n)
    blo, bhi = bootstrap_ci(x, y, lambda a, b: stats.pearsonr(a, b).statistic)
    same_dir = float(np.mean(np.sign(x) == np.sign(y)))
    return {
        "n": int(n),
        "pearson": float(pear.statistic),
        "pearson_p": float(pear.pvalue),
        "fisher_lo": lo,
        "fisher_hi": hi,
        "boot_lo": blo,
        "boot_hi": bhi,
        "spearman": float(spear.statistic),
        "spearman_p": float(spear.pvalue),
        "kendall": float(kend.statistic),
        "kendall_p": float(kend.pvalue),
        "r_squared": float(pear.statistic**2),
        "same_direction_share": same_dir,
    }


def ols_hac(y: np.ndarray, x: np.ndarray, maxlags: int = 5) -> dict:
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {
        "alpha": float(model.params[0]),
        "alpha_se": float(model.bse[0]),
        "alpha_p": float(model.pvalues[0]),
        "beta": float(model.params[1]),
        "beta_se": float(model.bse[1]),
        "beta_t": float(model.tvalues[1]),
        "beta_p": float(model.pvalues[1]),
        "beta_lo": float(model.conf_int()[1][0]),
        "beta_hi": float(model.conf_int()[1][1]),
        "r_squared": float(model.rsquared),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-09-23")
    ap.add_argument("--sept-start", default="2026-09-01")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--out", default="experiments/reports/btc_zec_correlation")
    args = ap.parse_args()

    cfg = get_config()
    syms = ["BTCUSDT", "ZECUSDT", "ETHUSDT"]
    series = {s: _load(cfg, s, args.start, args.end) for s in syms}
    prices = pd.DataFrame(series).dropna()
    rets = np.log(prices).diff().dropna()

    sept = prices[(prices.index >= pd.Timestamp(args.sept_start, tz="UTC"))]
    sept_rets = np.log(sept).diff().dropna()

    z = rets["ZECUSDT"].to_numpy()
    b = rets["BTCUSDT"].to_numpy()
    e = rets["ETHUSDT"].to_numpy()

    zs = sept_rets["ZECUSDT"].to_numpy()
    bs = sept_rets["BTCUSDT"].to_numpy()
    es = sept_rets["ETHUSDT"].to_numpy()

    summary: dict = {"meta": {"start": str(prices.index[0].date()), "end": str(prices.index[-1].date()),
                              "sept_start": args.sept_start, "n_full": int(len(rets)), "n_sept": int(len(sept_rets)),
                              "window": args.window,
                              "symbols": syms, "market": "binance futures USD-M perpetual"}}

    # --- coverage -------------------------------------------------------
    summary["coverage"] = {
        "full": {"n": int(len(prices)), "start": str(prices.index[0].date()), "end": str(prices.index[-1].date())},
        "september": {"n": int(len(sept)), "start": str(sept.index[0].date()), "end": str(sept.index[-1].date()),
                      "calendar_days": int((sept.index[-1] - sept.index[0]).days) + 1,
                      "missing_days": int((sept.index[-1] - sept.index[0]).days) + 1 - len(sept)},
        "first_close": {s: float(prices[s].iloc[0]) for s in syms},
        "last_close": {s: float(prices[s].iloc[-1]) for s in syms},
        "sept_first_close": {s: float(sept[s].iloc[0]) for s in syms},
        "sept_last_close": {s: float(sept[s].iloc[-1]) for s in syms},
    }

    # --- stationarity of levels ----------------------------------------
    summary["adf_levels"] = {s: {"stat": float(adfuller(prices[s].to_numpy())[0]),
                                 "p": float(adfuller(prices[s].to_numpy())[1])} for s in syms}
    summary["adf_returns"] = {s: {"stat": float(adfuller(rets[s].to_numpy())[0]),
                                  "p": float(adfuller(rets[s].to_numpy())[1])} for s in syms}

    # --- level correlation (spurious) ----------------------------------
    summary["levels_september"] = corr_bundle(np.log(sept["BTCUSDT"].to_numpy()), np.log(sept["ZECUSDT"].to_numpy()))
    summary["levels_full"] = corr_bundle(np.log(prices["BTCUSDT"].to_numpy()), np.log(prices["ZECUSDT"].to_numpy()))

    # --- return correlation --------------------------------------------
    summary["returns_september"] = corr_bundle(bs, zs)
    summary["returns_full"] = corr_bundle(b, z)
    summary["control_september"] = corr_bundle(bs, es)
    summary["control_full"] = corr_bundle(b, e)

    # --- regressions ----------------------------------------------------
    summary["regression_sept_zec_on_btc"] = ols_hac(zs, bs, maxlags=3)
    summary["regression_full_zec_on_btc"] = ols_hac(z, b, maxlags=5)
    summary["regression_sept_eth_on_btc"] = ols_hac(es, bs, maxlags=3)
    summary["regression_full_eth_on_btc"] = ols_hac(e, b, maxlags=5)

    # --- conditional / tail dependence ---------------------------------
    up = b >= 0
    dn = ~up
    summary["conditional_full"] = {
        "zec_on_btc_up": corr_bundle(b[up], z[up]),
        "zec_on_btc_down": corr_bundle(b[dn], z[dn]),
        "vol_corr_abs": corr_bundle(np.abs(b), np.abs(z)),
    }
    big = np.abs(b) >= np.quantile(np.abs(b), 0.9)
    summary["conditional_full"]["big_move_days"] = corr_bundle(b[big], z[big])
    # downside / upside semi-correlations (only joint co-movement tails)
    lo_q = np.quantile(b, 0.2)
    hi_q = np.quantile(b, 0.8)
    summary["conditional_full"]["btc_lower_quintile"] = corr_bundle(b[b <= lo_q], z[b <= lo_q])
    summary["conditional_full"]["btc_upper_quintile"] = corr_bundle(b[b >= hi_q], z[b >= hi_q])

    up_s = bs >= 0
    dn_s = ~up_s
    summary["conditional_sept"] = {
        "zec_on_btc_up": corr_bundle(bs[up_s], zs[up_s]),
        "zec_on_btc_down": corr_bundle(bs[dn_s], zs[dn_s]),
        "vol_corr_abs": corr_bundle(np.abs(bs), np.abs(zs)),
        "n_up": int(up_s.sum()), "n_down": int(dn_s.sum()),
    }

    # --- lead-lag: corr( ZEC_t , BTC_{t+k} ) ----------------------------
    btc_ret = rets["BTCUSDT"]
    zec_ret = rets["ZECUSDT"]

    def leadlag(zs_, bs_, lags_):
        out = []
        for k in lags_:
            pair = pd.concat([zs_, bs_.shift(-k)], axis=1).dropna()
            n = len(pair)
            r = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic if n > 2 else float("nan")
            out.append({"lag": k, "corr": float(r), "n": int(n)})
        return out

    lags = list(range(-5, 6))
    ll = leadlag(zec_ret, btc_ret, lags)
    summary["leadlag_full"] = ll
    ll_s = leadlag(sept_rets["ZECUSDT"], sept_rets["BTCUSDT"], lags)
    summary["leadlag_sept"] = ll_s

    # --- rolling --------------------------------------------------------
    roll = rets["BTCUSDT"].rolling(args.window).corr(rets["ZECUSDT"]).dropna()
    roll_eth = rets["BTCUSDT"].rolling(args.window).corr(rets["ETHUSDT"]).dropna()
    roll_beta = (rets["ZECUSDT"].rolling(args.window).cov(rets["BTCUSDT"])
                 / rets["BTCUSDT"].rolling(args.window).var()).dropna()
    summary["rolling"] = {
        "window": args.window,
        "dates": [str(d.date()) for d in roll.index],
        "zec": [float(v) for v in roll.to_numpy()],
        "eth": [float(v) for v in roll_eth.reindex(roll.index).to_numpy()],
        "beta_zec": [float(v) for v in roll_beta.reindex(roll.index).to_numpy()],
        "mean": float(roll.mean()), "std": float(roll.std()),
        "min": float(roll.min()), "min_date": str(roll.idxmin().date()),
        "max": float(roll.max()), "max_date": str(roll.idxmax().date()),
        "share_negative": float((roll < 0).mean()),
        "sept_mean": float(roll[roll.index >= pd.Timestamp(args.sept_start, tz="UTC")].mean()),
        "sept_n": int((roll.index >= pd.Timestamp(args.sept_start, tz="UTC")).sum()),
    }

    # --- monthly stability ---------------------------------------------
    months = rets.copy()
    months["month"] = months.index.to_period("M").astype(str)
    rows = []
    for m, g in months.groupby("month"):
        if len(g) < 5:
            continue
        rows.append({
            "month": m,
            "n": int(len(g)),
            "zec": float(g["BTCUSDT"].corr(g["ZECUSDT"])),
            "eth": float(g["BTCUSDT"].corr(g["ETHUSDT"])),
            "btc_ret": float(np.expm1(g["BTCUSDT"].sum())),
            "zec_ret": float(np.expm1(g["ZECUSDT"].sum())),
            "eth_ret": float(np.expm1(g["ETHUSDT"].sum())),
        })
    summary["monthly"] = rows
    zec_m = np.array([r["zec"] for r in rows])
    summary["monthly_summary"] = {
        "n_months": int(len(rows)),
        "zec_mean": float(np.mean(zec_m)),
        "zec_median": float(np.median(zec_m)),
        "zec_negative_months": int((zec_m < 0).sum()),
        "zec_min": float(np.min(zec_m)), "zec_min_month": rows[int(np.argmin(zec_m))]["month"],
        "zec_max": float(np.max(zec_m)), "zec_max_month": rows[int(np.argmax(zec_m))]["month"],
        "sept_value": float(rows[-1]["zec"]) if rows[-1]["month"].startswith("2026-09") else None,
    }

    # --- leave-one-out influence ---------------------------------------
    loo = []
    for i in range(len(b)):
        xb = np.delete(b, i)
        yb = np.delete(z, i)
        loo.append(float(stats.pearsonr(xb, yb).statistic))
    loo = np.array(loo)
    summary["leave_one_out"] = {
        "min": float(loo.min()), "min_date": str(rets.index[int(loo.argmin())].date()),
        "max": float(loo.max()), "max_date": str(rets.index[int(loo.argmax())].date()),
        "base": float(stats.pearsonr(b, z).statistic),
    }
    # september LOO
    loos = []
    for i in range(len(bs)):
        loos.append(float(stats.pearsonr(np.delete(bs, i), np.delete(zs, i)).statistic))
    loos = np.array(loos)
    summary["leave_one_out_sept"] = {
        "min": float(loos.min()), "min_date": str(sept_rets.index[int(loos.argmin())].date()),
        "max": float(loos.max()), "max_date": str(sept_rets.index[int(loos.argmax())].date()),
        "base": float(stats.pearsonr(bs, zs).statistic),
    }

    # --- series for plotting -------------------------------------------
    idx = prices / prices.iloc[0] * 100.0
    summary["series"] = {
        "dates": [str(d.date()) for d in prices.index],
        "btc": [float(v) for v in idx["BTCUSDT"]],
        "zec": [float(v) for v in idx["ZECUSDT"]],
        "eth": [float(v) for v in idx["ETHUSDT"]],
        "btc_ret": [float(v) for v in rets["BTCUSDT"].to_numpy()],
        "zec_ret": [float(v) for v in rets["ZECUSDT"].to_numpy()],
        "eth_ret": [float(v) for v in rets["ETHUSDT"].to_numpy()],
    }
    sidx = sept / sept.iloc[0] * 100.0
    summary["sept_series"] = {
        "dates": [str(d.date()) for d in sept.index],
        "btc": [float(v) for v in sidx["BTCUSDT"]],
        "zec": [float(v) for v in sidx["ZECUSDT"]],
        "eth": [float(v) for v in sidx["ETHUSDT"]],
        "btc_close": [float(v) for v in sept["BTCUSDT"]],
        "zec_close": [float(v) for v in sept["ZECUSDT"]],
        "eth_close": [float(v) for v in sept["ETHUSDT"]],
        "btc_ret": [float(v) for v in sept_rets["BTCUSDT"]],
        "zec_ret": [float(v) for v in sept_rets["ZECUSDT"]],
        "eth_ret": [float(v) for v in sept_rets["ETHUSDT"]],
    }
    # --- ZEC top idiosyncratic moves -----------------------------------
    top = rets.assign(_abs=np.abs(rets["ZECUSDT"])).sort_values("_abs", ascending=False).head(6)
    summary["zec_top_moves"] = [{"date": str(d.date()), "zec_ret": float(r["ZECUSDT"]),
                                 "btc_ret": float(r["BTCUSDT"]), "eth_ret": float(r["ETHUSDT"])}
                                for d, r in top.iterrows()]

    # --- risk / volatility ---------------------------------------------
    def vol(a: np.ndarray, ann: int) -> float:
        return float(np.std(a, ddof=1) * np.sqrt(ann))

    summary["risk"] = {
        "sept": {s: {"daily_pct": float(np.std(sept_rets[s].to_numpy(), ddof=1) * 100),
                     "annual_pct": vol(sept_rets[s].to_numpy(), 365) * 100,
                     "cum_pct": float((sept[s].iloc[-1] / sept[s].iloc[0] - 1) * 100)} for s in syms},
        "full": {s: {"daily_pct": float(np.std(rets[s].to_numpy(), ddof=1) * 100),
                     "annual_pct": vol(rets[s].to_numpy(), 365) * 100,
                     "cum_pct": float((prices[s].iloc[-1] / prices[s].iloc[0] - 1) * 100)} for s in syms},
    }
    summary["sept_direction"] = {
        "same": int(np.sum(np.sign(bs) == np.sign(zs))),
        "opposite": int(np.sum(np.sign(bs) != np.sign(zs))),
        "n": int(len(bs)),
        "opposite_dates": [str(sept_rets.index[i].date()) for i in range(len(bs)) if np.sign(bs[i]) != np.sign(zs[i])],
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    merged = prices.join(rets, rsuffix="_ret")
    merged.to_csv(out / "series.csv")

    # --- console report -------------------------------------------------
    def line1(t: str) -> None:
        print("\n" + t)
        print("-" * len(t))

    print(f"sample: {summary['meta']['start']} -> {summary['meta']['end']}  (n_full={len(rets)}, n_sept={len(sept_rets)})")
    line1("September returns: ZEC vs BTC")
    for k in ("pearson", "pearson_p", "spearman", "kendall", "r_squared", "same_direction_share", "fisher_lo", "fisher_hi", "boot_lo", "boot_hi"):
        print(f"  {k:22s} {summary['returns_september'][k]: .4f}")
    line1("September returns: ETH vs BTC (control)")
    for k in ("pearson", "spearman", "r_squared", "same_direction_share"):
        print(f"  {k:22s} {summary['control_september'][k]: .4f}")
    line1("Full-sample returns: ZEC vs BTC")
    for k in ("pearson", "pearson_p", "spearman", "r_squared", "same_direction_share", "boot_lo", "boot_hi"):
        print(f"  {k:22s} {summary['returns_full'][k]: .4f}")
    line1("September levels (spurious): ZEC vs BTC")
    print(f"  pearson {summary['levels_september']['pearson']:.4f}  r2 {summary['levels_september']['r_squared']:.4f}")
    line1("Regression ZEC ~ BTC (Sep, HAC)")
    print("  " + json.dumps(summary["regression_sept_zec_on_btc"], indent=2).replace("\n", "\n  "))
    line1("Lead-lag corr(ZEC_t, BTC_t+k), full sample")
    for r in ll:
        print(f"  k={r['lag']:+d}  r={r['corr']: .4f}  n={r['n']}")
    line1("Lead-lag (Sep)")
    for r in ll_s:
        print(f"  k={r['lag']:+d}  r={r['corr']: .4f}  n={r['n']}")
    line1("Monthly ZEC corr")
    for r in rows:
        print(f"  {r['month']}  n={r['n']:2d}  zec={r['zec']: .3f}  eth={r['eth']: .3f}  btc={r['btc_ret']*100:6.1f}%  zec_px={r['zec_ret']*100:7.1f}%")
    line1("Rolling corr summary")
    print("  " + json.dumps({k: summary["rolling"][k] for k in ("mean", "std", "min", "min_date", "max", "max_date", "share_negative", "sept_mean", "sept_n")}, indent=2))
    line1("Leave-one-out (full / Sep)")
    print("  full:", json.dumps(summary["leave_one_out"]))
    print("  sept:", json.dumps(summary["leave_one_out_sept"]))
    line1("ZEC top |daily moves|")
    print("  " + json.dumps(summary["zec_top_moves"]))
    line1("September direction agreement")
    print("  " + json.dumps({k: summary["sept_direction"][k] for k in ("same", "opposite", "n", "opposite_dates")}))
    print(f"\nwrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
