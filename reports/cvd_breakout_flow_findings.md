# H3 — Flow-Confirmed Breakout Continuation: Findings & Recommendation

**Status:** complete — negative result.
**Design:** `reports/cvd_breakout_flow_study.md` (frozen before results).
**Machine outputs:** `experiments/reports/breakouts/` and `breakouts_all/`
(`summary.json`, `by_symbol.json`, `leave_one_out.json`, `quintiles.json`,
`robustness.json`, `events.parquet`).
**Reproduce:**
```bash
python scripts/breakout_study.py                    # 62 crypto perps (primary)
python scripts/breakout_study.py --include-noncrypto # 100 symbols incl. TradFi
```

---

## 1. Verdict

**NO EDGE (H0)** in both universes. All preregistered criteria R1–R7 fail.
Persistent same-direction aggressive flow does **not** improve breakout
continuation after accounting for the breakout itself.

## 2. What was tested

Close-beyond-prior-range breakouts (trailing `L=48` extreme, excludes current
bar), 1h bars, aligned activity-normalised flow over `w=4` bars (`flow_align_z`),
24-bar cooldown, 4h primary horizon; outcomes = oriented forward return,
continuation-failure rate, MFE/MAE; baselines = price-only breakout, momentum-only,
and the prior study's `cvd_20>0` filter.

## 3. Results

| universe | design IC | test IC | test 95% CI | matched spread | incremental coef |
|---|---|---|---|---|---|
| **62 crypto perps** (6,806 events) | −0.0296 | **−0.0146** | [−0.0504, +0.0259] | +21.8 bps [−27.6, +62.7] | −0.00093 [−0.00235, +0.00047] |
| 100 symbols incl. TradFi (9,469 events) | −0.0107 | **−0.0057** | [−0.0362, +0.0243] | +6.9 bps [−17.4, +35.9] | −0.00015 [−0.00094, +0.00060] |

- **Horizons (crypto-62 test IC):** 1h −0.0179 · 4h −0.0146 · 24h −0.0115 ·
  3d −0.0086 — negative everywhere.
- **Matched controls** and the **incremental regression** both include zero; the
  regression point estimate is **negative** in both universes.
- **Robustness (27 cells):** at 15m/1h design and test IC are consistently
  negative; at 4h the design IC is strongly negative (−0.15…−0.21) while test
  flips small-positive — instability, not a stable region.
- **Failure rate** shows a mild decline with flow alignment (0.588 → 0.441 across
  quintiles) but forward returns are non-monotone and inverted — no tradable
  structure.

## 4. Relationship to the prior breakout study

`reports/breakout_regime_study.md` found the `cvd_20>0` breakout filter
insignificant (validate lift 1.07). H3's result is **directionally consistent**:
flow confirmation does not rescue breakouts, and where anything is detected it is
slightly negative intraday. Four flow-conditioning ideas now fail on the same
2026 sample (divergence, H1 absorption, H2 sweeps, H3 breakouts).

## 5. Limitations

- **Single regime, in-sample 2026** — no cross-regime out-of-sample.
- 1h resolution, taker-flow proxy, event-level samples of a few thousand to ~7k.
- Overlapping events and clusters handled by cooldown + timestamp-block bootstrap,
  but serial dependence remains a caveat.

## 6. Recommendation

1. **Do not re-tune H3.** No stable cell; both universes, all horizons, matched
   controls and the incremental regression agree on "no edge".
2. **The recurring blocker is out-of-sample history.** Four hypotheses have now
   been judged on a single 2026 regime. **Building 2024–2025 perp 1m data
   (~2 h, ~3 GB) is the single highest-value action** before any further flow
   hypothesis; without it, negative results are as unconfirmable as positives.
3. **Consider stepping back from bar-level taker-flow CVD.** Four independent
   flow features (spot-vs-perp divergence, impact residual, sweep-aligned flow,
   breakout flow persistence) show no incremental value at 1m→1h in this sample.
   Before more variants, either (a) raise the information content (tick/aggTrades
   for true sub-bar impact — but this is costly and lowers breadth), or (b) move
   to a structurally different information source (funding/basis carry,
   cross-sectional relative strength, liquidation-adjacent proxies where data
   exists).
4. **If continuing the H-series:** H6 (spot lead–lag) is the last genuinely
   distinct flow test but risks re-treading the failed divergence study; H5
   (impact asymmetry) is better kept as an **execution/cost model** than a signal.
5. **Do not buy tick data yet**; the binding constraints are regime breadth and
   effect existence, not resolution.

**Bottom line: abandon H3; prioritise out-of-sample history, then question whether
1m-bar taker-flow CVD carries incremental information at all.**
