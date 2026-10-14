# H2 — Failed Liquidity Sweeps: Findings & Recommendation

**Status:** complete — negative result.
**Design:** `reports/cvd_failed_sweeps_study.md` (frozen before results).
**Machine outputs:** `experiments/reports/sweeps/{summary,by_symbol,leave_one_out,robustness}.json`, `events.parquet`.
**Reproduce:** `python scripts/sweep_study.py`.

---

## 1. Verdict

**NO EDGE (H0).** All preregistered criteria Q1–Q7 fail. Aggressive flow aligned
with a failed sweep does **not** predict the strength of the subsequent reversal.

## 2. What was tested

Failed sweeps of a trailing prior-range extreme (wick beyond, close back inside),
1h bars, crypto top-10, `L = 48`, 24-bar cooldown, 4h primary horizon; signal =
activity-normalised flow aligned to the sweep direction (`af_z`); outcome = oriented
forward return; baselines = all sweeps / low-flow sweeps / matched controls.
1,262 events (508 design, 754 test).

## 3. Results

| quantity | design | test |
|---|---|---|
| event pooled rank IC(`af_z`, `y_4h`) | +0.0495 (n=508) | **−0.0022** (n=754) |
| test 95% block-bootstrap CI | — | [−0.0807, +0.0654] |

- **Horizons (test IC):** 1h −0.0067 · 4h −0.0022 · 24h +0.0068 · 3d +0.0063.
- **Matched control** (high-`af` − low-`af` sweeps): **−32.0 bps**, CI
  [−125.4, +46.0] → **includes zero**, wrong sign.
- **Control regression:** `af_z` coefficient ≈ **0.00002**, CI
  [−0.00070, +0.00080] → indistinguishable from zero.
- **Per-symbol test IC:** wildly mixed (BTCUSDT +0.152, ZECUSDT −0.172, DOGEUSDT
  +0.163, BNBUSDT −0.135; n≈80 each) — the spread of signs at n≈80 is what noise
  looks like.
- **Quintiles:** non-monotone.

## 4. Robustness sweep (18 cells)

Timeframe ∈ {15m,1h,4h} × lookback ∈ {24,48,96} × flow-window ∈ {1,4}:

- Test IC ranges −0.093 … +0.026 and **design and test signs disagree** across
  cells; no cell is stable, and the larger |IC| values sit on the smallest
  samples (4h, n≈200–400).
- The single "promising" design value (0.0495 at the primary spec) reverses in
  test — a textbook selection artefact.

## 5. Interpretation

The failed-sweep story (trapped aggressors → sharper reversal) is as plausible as
H1's, and the event frequency is ample. The data still reject it: the design-period
signal does not survive, the matched-control spread is negative and
insignificant, the regression coefficient is zero, and the parameter sweep shows
no stable region. This is the **third** independent order-flow negative on the
same 2026 sample (spot–perp divergence, H1 absorption, H2 sweeps).

## 6. Limitations

- **Single regime, in-sample 2026** — no true OOS across regimes; BTCUSDT is the
  only multi-year symbol.
- **Event-level sample is small** (1.3k events on top-10; 754 in test): low power
  for a modest effect. The wider 62-symbol crypto universe yields ~22k events and
  would materially raise power.
- **Taker-flow proxy**; 1h resolution; no tick data.

## 7. Recommendation

1. **Do not re-tune H2.** The sweep shows no stable cell; the design positive is
   noise. Rescuing it would be curve-fitting.
2. **Highest-leverage step remains: build 2024–2025 perp 1m history** (~2 h,
   ~3 GB). Three hypotheses have now been judged on a single regime; none can be
   confirmed without out-of-sample data.
3. **If continuing bar-level flow work, widen the universe first.** Event-based
   hypotheses (H2) are power-limited on top-10; the 62-symbol crypto set gives
   ~17× the events at no extra download. Re-run any survivor on the wide set.
4. **Next candidate: H3 (flow-confirmed breakout continuation), on the wide
   universe.** It is the most strategy-like, has a clean price-only baseline, and
   does not need new primitives. H6 (spot lead–lag) is the other distinct test,
   but note it risks re-treading the failed divergence study.
5. **H5 as execution infrastructure.** Directional price-impact asymmetry is worth
   keeping as a slippage/cost model even without alpha.
6. **Do not buy tick data yet.** The binding constraints are regime breadth and
   statistical power, not resolution.

**Bottom line: abandon H2; the priority order is (a) out-of-sample history,
(b) wider universe, (c) H3 — before any further event-level flow hypotheses.**
