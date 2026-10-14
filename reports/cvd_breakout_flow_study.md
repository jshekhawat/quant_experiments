# H3 — Flow-Confirmed Breakout Continuation: Design (frozen)

**Status:** frozen design — awaiting go-ahead before implementation.
**Scope:** H3 only. H1 (`cvd_absorption_findings.md`) and H2
(`cvd_failed_sweeps_findings.md`) are completed negatives; H4–H6 not covered.
**Related:** `reports/breakout_regime_study.md` (**prior negative — see A.2**).

---

# Part A — Assessment and the overlap to beat

## A.1 Reusable machinery

- Levels: `analytics/sweeps.prior_range_levels` (trailing prior-range high/low,
  excludes the current bar) — exactly the "breakout of a historical range" construct.
- Estimators: `analytics/divergence` (`pooled_ic`, `decile_table`,
  `block_bootstrap_ic`) and `analytics/absorption` (`block_bootstrap`,
  `matched_spread`, `absorption_regression`, `_oriented_extremes`).
- Regimes: `analytics/regimes`. Splits/resampling as before.
- **No new data needed** — perp 1m bars on disk (2026).

## A.2 This is not a fresh question — the prior study already came close

`reports/breakout_regime_study.md` tested breakout triggers with a confluence
filter `cvd_20 > 0` (20-bar CVD positive) and reported **validate lift 1.07,
not significant** (design 0.99 / validate 1.07), part of an overall **negative**
result. H3 must therefore *add something* over that filter or it is a re-run:

H3's additions, all beyond the prior test:
1. **Flow persistence** across multiple bars (not a single `cvd_20>0` sign).
2. **Breakout displacement** and displacement-conditioned flow.
3. **Failure-rate** and **adverse-excursion (MAE)** outcomes, not just return.
4. **Explicit incremental-vs-price-only** test (does flow beat a breakout alone?).
5. A **wider universe** (all 62 crypto perps) for power, and **stated prior** that
   a negative is likely.

The prior study is cited prominently so a null H3 is correctly read as
"consistent with the earlier breakout negative", not a surprise.

## A.3 Event frequency is ample (measured, 1h, 2026)

Breakout = close beyond the trailing `L`-bar extreme (excluding current bar):

| universe | L=24 | L=48 | L=96 | base continuation (fwd 4h > 12 bps) |
|---|---|---|---|---|
| crypto top-10 | 5,053 | 3,266 | 2,074 | ~42% |
| all 62 crypto perps | 30,680 | 19,902 | 13,042 | ~43% |

Full power even on top-10; the 62-symbol set gives ~6× more. No tick data required.
Limitation unchanged: **single regime (2026)** → exploratory verdict.

---

# Part B — Frozen H3 specification

## B.1 Hypothesis and rationale

**Question.** Do breakouts supported by **persistent same-direction aggressive
flow** continue more than breakouts with weak flow or immediate flow reversal,
**after accounting for the breakout itself** (price-only baseline)?

**H3.** Breakouts with high aligned/persistent taker flow exhibit **better
continuation** (larger oriented forward return, lower failure rate, smaller
adverse excursion) than breakouts with weak or opposing flow.

**Direction.** Breakout direction `d = +1` (close above prior high) or `d = −1`
(close below prior low). Oriented forward return `y_h = d · fwd_ret_h`; H3 predicts
`y_h` increases with flow alignment/persistence.

**Proxy caveat.** Taker flow is a proxy for genuine aggressive intent; the prior
negative makes a null outcome plausible.

## B.2 Definitions (causal)

Per symbol, on **1h** bars (from 1m):

```
prior_high_t = max(high[t-L .. t-1])          # excludes t (sweeps.prior_range_levels)
prior_low_t  = min(low[t-L .. t-1])

bull_t = close_t > prior_high_t
bear_t = close_t < prior_low_t
d_t    = +1 if bull_t else (-1 if bear_t else 0)
break_disp_t = d_t * (close_t - level_t)/level_t      # how far beyond, signed
range_width_t = (prior_high_t - prior_low_t)/close_t  # volatility scale

delta_t   = 2*taker_buy_quote_volume - quote_volume    # signed taker flow, USDT
act_t     = mean(|delta|, trailing A)                  # activity scale
F_t       = sum(delta over [t-w+1 .. t])               # w-bar flow
flow_align_t = d_t * F_t / (w * act_t)                 # aligned, normalized
persist_t    = d_t * mean(sign(delta) over [t-w+1 .. t])  # ∈ [-1,1] signed persistence
flow_align_z_t = flow_align_t / std_W(flow_align)      # z-scored
```

Signals evaluated at the **close of `t`**; outcomes start at `t+1`. A per-symbol
**cooldown** (primary 24 bars) de-duplicates clustered breakouts.

## B.3 Required input data

Perp 1m klines (2026, on disk): OHLCV + taker-buy base/quote volume. Controls use
regimes; funding optional. **No spot, no tick data, no downloads.**

## B.4 Baselines and matched controls

1. **Price-only breakout baseline** — all breakouts, no flow conditioning
   (unconditional oriented forward return / failure rate).
2. **Momentum-only baseline** — condition on past return / trend only.
3. **Prior-study filter** — the single-bar `cvd_20 > 0` condition, for direct
   comparison with `breakout_regime_study.md`.
4. **Matched controls** — high-persistence vs low-persistence breakouts matched on
   `break_disp`, `range_width`, trailing realized vol, `past_ret`, and `regime`.

## B.5 Primary outcome and horizon

- **Primary:** oriented forward return over **h = 4h** (entry `t+1`, exit `t+1+h`).
- **Secondary:** continuation **failure rate** (price closes back through the
  breakout level within `h`), oriented **MFE/MAE**, directional **hit rate**.
- **Robustness horizons:** 1h, 24h, 3d.

## B.6 Confounders and leakage

- **Causal levels** — exclude bar `t`; explicit causality test.
- **Event clustering** — cooldown; report whether overlaps change results.
- **Volatility/range confound** — control and match on `range_width`, realized vol.
- **Trend confound** — breakouts cluster in trends; control on `past_ret`, stratify
  by regime, and report the momentum-only baseline.
- **Mechanical** — flow is contemporaneous with the breakout bar (known at close);
  forward window starts `t+1`.
- **Multiple testing** — one primary spec, sweep reported whole with BH-FDR.

## B.7 Statistical evaluation & cost assumptions

- **Event pooled rank IC(`flow_align_z`, `y_4h`)** (headline) + block-bootstrap CI;
  the same for `persist_z`.
- **Tercile/quintile event study** + monotonicity; **failure rate** by quintile.
- **Matched-control spread** (high − low alignment), paired within strata, CI.
- **Regression:** `y ~ flow_align_z + break_disp + range_width + realized_vol +
  past_ret + regime FE`, clustered by symbol×day; the `flow_align_z` coefficient
  and CI is the **incremental-value** test vs the price-only baseline.
- **Costs:** ~12 bps round-trip; a tradable claim must clear it.
- **Stability:** per-symbol, per-regime, chronological design/test.

## B.8 Pass/fail criteria (preregistered)

**Declare an edge iff ALL hold on the test period:**

| # | criterion | threshold |
|---|---|---|
| R1 | event pooled rank IC(`flow_align_z`, `y_4h`) | `> 0.02`, 95% block-bootstrap CI excludes 0 |
| R2 | sign-stable | design IC and test IC both `> 0` |
| R3 | matched-control spread (high−low alignment) | `> 12 bps`, CI excludes 0, correct sign |
| R4 | incremental vs price-only | `flow_align_z` coefficient CI excludes 0, `> 0` |
| R5 | monotone quintiles | Spearman(quintile idx, mean `y`) `> 0.7` |
| R6 | not single-name / mega-cap | holds dropping any symbol; holds ex-BTC/ETH |
| R7 | horizon-consistent | `> 0` at 4h **and** ≥1 of {1h, 24h} |

Anything short of all seven → **H0 (no edge)**.

## B.9 Feasibility with existing data

**Yes.** Event counts verified (A.3); all primitives either exist or are trivial
to compose (`prior_range_levels` reused). No tick data. Binding limits: single
regime (2026) → exploratory; overlap with the prior breakout negative (A.2).

---

# Part C — Frozen primary specification

| element | primary | robustness |
|---|---|---|
| universe | **crypto-only, all 62 perps** (power) | top-10; ex-BTC/ETH |
| timeframe | 1h (from 1m) | 15m, 4h |
| prior range | `L = 48` | `L ∈ {24, 96}` |
| breakout | close beyond prior extreme | require min `break_disp`; wick-then-close |
| flow window | `w = 4` bars | `w ∈ {1, 12}` |
| signal | `flow_align_z` (aligned, activity-normalised, z-scored) | `persist_z`; raw align |
| horizon | 4h | 1h, 24h, 3d |
| cooldown | 24 bars/symbol | 12, 48; none |
| design/test | 2026-01…04 / 2026-05…09 | walk-forward months |
| cost | 12 bps round-trip | 24 bps |

## C.1 Implementation plan (after approval)

- New module `src/quant_lab/analytics/breakouts.py`:
  `detect_breakouts` (reuse `sweeps.prior_range_levels`), `flow_alignment`
  (align + persistence), `failure_outcomes`, `run_breakout_study`, `robustness_sweep`.
- Reuse `analytics/divergence` and `analytics/absorption` estimators; reuse
  `regimes`.
- New tests `tests/test_breakouts.py`: level causality, breakout direction on
  hand-built OHLC, flow-alignment identity, failure-outcome correctness,
  forward-return alignment/no-leakage, synthetic edge/no-edge fixtures.
- New script `scripts/breakout_study.py`; writes
  `reports/cvd_breakout_flow_results.md` + machine outputs under
  `experiments/reports/breakouts/`.
- **No existing reports overwritten; no existing research code behaviour changed.**

---

# Part D — Explicit non-goals

- Not implementing H4–H6.
- Not re-running the existing `breakout_regime_study` triggers
  (`bollinger_breakout` etc.); H3 is a flow-conditioning event study, not a
  strategy backtest.
- Not tuning thresholds to find a positive result; the sweep is reported whole.
- Not using tick/aggTrades.
