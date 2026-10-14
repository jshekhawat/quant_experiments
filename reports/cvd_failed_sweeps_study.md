# H2 — Failed Liquidity Sweeps: Design (frozen)

**Status:** frozen design — awaiting go-ahead before implementation.
**Scope:** H2 only (failed sweeps). H1 is a completed negative
(`reports/cvd_absorption_findings.md`); H3–H6 are not implemented here.
**Related:** `reports/cvd_absorption_study.md`, `reports/spot_cvd_divergence_study.md`.

---

# Part A — What H2 adds, and what must be built

## A.1 The one missing capability

The repo has **no swing / pivot detection** (verified: grep for
`swing|pivot|sweep|breakout` finds only the four breakout *strategies*, none of
which detect structural levels causally). H2 therefore must build a **causal
level detector** as its first component. Everything else is reusable:
`resample_bars`, `read_dataset`, `analytics/regimes`, `analytics/divergence`
estimators (`pooled_ic`, `decile_table`, `block_bootstrap_ic`), and the
`absorption.py` helpers (`matched_spread`, `block_bootstrap`).

## A.2 Level definition: trailing extreme (primary), pivot (robustness)

Two candidate "previously established level" definitions:

1. **Prior range extreme (primary).** As of bar `t`, the level is the highest
   high (for a swing *high*) over the trailing `L` bars **strictly before `t`**:
   `prior_high_t = max(high[t-L .. t-1])`. This needs no pivot detection, has a
   single parameter `L`, is unambiguously causal, and is the standard
   "N-bar-high failed breakout" construct. Symmetric for `prior_low_t`.
2. **Confirmed fractal pivot (robustness).** A pivot high at `p` is confirmed once
   `k` bars have closed without exceeding it; at time `t` only pivots with
   `p + k ≤ t` are usable. More literal to "swing high", but adds a knob `k` and
   a sparser event set.

**Primary = (1)** because it is look-ahead-free by construction and knob-light;
(2) is reported as a robustness variant.

## A.3 Event frequency is sufficient (measured)

1h bars, 2026-01…09, prior-range definition, count of wick-beyond-then-close-inside:

| universe | L=24 | L=48 | L=96 |
|---|---|---|---|
| crypto top-10 (65,520 bars) | 2,884 up / 2,788 down (8.7%) | 1,850 / 1,771 (5.5%) | 1,178 / 1,083 (3.5%) |
| all 62 crypto perps (391,584 bars) | 17,142 / 17,516 (8.9%) | 11,104 / 11,530 (5.8%) | 7,293 / 7,215 (3.7%) |

So even the top-10 yields ~3.6k events at `L=48` — ample, and the wider crypto
set gives ~22k. **No tick data required.** The only limitation remains regime
breadth (2026-only; BTC is the sole multi-year symbol).

---

# Part B — Frozen H2 specification

## B.1 Hypothesis and economic rationale

**Question.** When price sweeps a previously established swing high/low and
closes back inside the prior range (a **failed sweep** / stop-run), does the
**magnitude and direction of aggressive flow during the sweep** predict the
strength of the subsequent reversal?

**H2.** A failed sweep with **large aggressive flow in the sweep direction**
(chasers buying an upside sweep, or selling a downside sweep) reverses
**harder** than a similar failed sweep without such flow. Rationale: trapped
aggressors and absorbed liquidity produce an outsized move once the sweep fails.
**Proxy, not mechanism** — we cannot see stopping/absorption identities.

**Directional definition (fixed).**
- Upside sweep (price wicks above the prior high, closes back below) → predicted
  reversal `d = −1` (down).
- Downside sweep (wicks below prior low, closes back above) → `d = +1` (up).
- **Aligned flow** `af_t = −d_t · F_t` (flow in the sweep direction is positive):
  upside → net buying positive; downside → net selling positive.
- **Oriented forward return** `y_t = d_t · fwd_ret_t`. H2 predicts `y_t` increases
  with `af_t`.

## B.2 Event and signal definitions (causal)

Per symbol, on **1h** bars (resampled from 1m):

```
prior_high_t = max(high[t-L .. t-1])        # trailing, excludes t
prior_low_t  = min(low[t-L .. t-1])

upside_sweep_t  = high_t > prior_high_t and close_t < prior_high_t
downside_sweep_t= low_t  < prior_low_t  and close_t > prior_low_t

delta_t   = 2*taker_buy_quote_volume - quote_volume          # signed taker flow, USDT
F_t       = sum(delta over the sweep bar)   (primary: the sweep bar only)
act_t     = mean(|delta|, trailing A)                        # activity scale
af_t      = (-d_t) * F_t / act_t                             # aligned, normalized
af_z_t    = af_t / std_W(af)                                 # z-scored
sweep_depth_t = (high_t/prior_high_t - 1)  or (1 - low_t/prior_low_t)
range_width_t = (prior_high_t - prior_low_t) / close_t
```

Signals are evaluated at the **close of `t`**; outcomes start at `t+1`. The
level excludes bar `t` (no look-ahead in level formation).

## B.3 Required input data

Perp 1m klines (2026, already on disk): `timestamp, open, high, low, close,
volume, quote_volume, taker_buy_volume, taker_buy_quote_volume`. Controls use
regimes (computed from bars) and, where available, funding (100 symbols). OI is
**not** required (partial coverage would only limit an optional robustness slice).

## B.4 Baselines and matched controls

1. **All failed sweeps (no flow conditioning)** — unconditional oriented return.
2. **Low-flow failed sweeps** — the natural control group (bottom tercile `af`).
3. **Successful breakouts** — close beyond the level (not a failed sweep); tests
   whether the *failure* is what matters, not the wick per se.
4. **Matched controls** — high-`af` vs low-`af` failed sweeps matched on
   `sweep_depth`, `range_width` (ATR-normalised), trailing realized vol, trailing
   return and `regime`.

## B.5 Primary outcome and horizon

- **Primary:** oriented forward return over **h = 4h** (entry close `t+1`, exit
  close `t+1+h`) on failed-sweep events, pooled across symbols.
- **Secondary:** oriented MFE/MAE, directional hit rate, time-to-reversal.
- **Robustness horizons:** 1h, 24h, 3d (reported in full).

## B.6 Confounders and leakage

- **Look-ahead in levels** — `prior_high/low` exclude `t`; pivots used only after
  confirmation. Explicit causality test.
- **Event clustering** — de-duplicate with a **cooldown** of `C` bars per symbol
  (primary `C = 24`; one event per day), so overlapping sweeps are not
  double-counted; also dedupe consecutive bars.
- **Volatility / range confound** — control and match on `range_width`, realized
  vol, ATR.
- **Trend confound** — a failed low-sweep in an uptrend is not symmetric to one in
  a downtrend; control for trailing return and stratify by regime.
- **Mechanical** — the flow is contemporaneous with the sweep bar (known at
  close); forward window starts `t+1`.
- **Multiple testing** — one primary spec; sweep reported whole with BH-FDR.

## B.7 Statistical evaluation & cost assumptions

- **Pooled rank IC(`af_z`, `y_4h`)** on events (headline), block bootstrap CI
  (blocks ≈ horizon; also cluster-aware via timestamp blocks).
- **Tercile/quintile event study** of `y_4h` by `af_z`, monotonicity check.
- **Matched-control spread** (high-`af` − low-`af`), paired within strata, CI.
- **Regression:** `y ~ af_z + sweep_depth + range_width + realized_vol + past_ret
  + regime FE`, clustered by symbol×day; report the `af_z` coefficient + CI.
- **Hit rate** of predicted reversal direction; **event counts** and frequency.
- **Costs:** ~12 bps round-trip; a tradable claim must clear it.
- **Stability:** per-symbol and per-regime; chronological design/test.

## B.8 Pass/fail criteria (preregistered)

**Declare an edge iff ALL hold on the test period:**

| # | criterion | threshold |
|---|---|---|
| Q1 | pooled rank IC(`af_z`, `y_4h`) | `> 0.02`, 95% block-bootstrap CI excludes 0 |
| Q2 | sign-stable | design IC and test IC both `> 0` |
| Q3 | matched-control spread (high−low `af`) | `> 12 bps`, CI excludes 0, correct sign |
| Q4 | survives controls | `af_z` coefficient CI excludes 0, `> 0` |
| Q5 | monotone quintiles | Spearman(quintile idx, mean `y`) `> 0.7` |
| Q6 | not single-name / mega-cap | holds dropping any symbol; holds ex-BTC/ETH |
| Q7 | horizon-consistent | `> 0` at 4h **and** ≥1 of {1h, 24h} |

Anything short of all seven → **H0 (no edge)**, reported as a negative result.

## B.9 Feasibility with existing data

**Yes.** 1m perp bars suffice (event counts verified, §A.3). No tick data, no new
downloads. The binding limitations are: (i) the level detector must be built and
causality-tested, and (ii) **single regime (2026) → exploratory verdict**, with
BTCUSDT as the only multi-year cross-check.

---

# Part C — Frozen primary specification

| element | primary | robustness |
|---|---|---|
| universe | crypto-only **top-10** perps (`configs/divergence_universe.json`) | top-30, all 62 crypto |
| timeframe | 1h (from 1m) | 15m, 4h |
| level | prior-range extreme, `L = 48` | `L ∈ {24, 96}`; fractal pivot `k ∈ {3, 5}` |
| sweep | wick beyond, close inside | require min sweep depth; close-direction filter |
| flow window | sweep bar only | trailing `{1, 4}` bars |
| signal | `af_z` (aligned, activity-normalised, z-scored) | rank-transformed; raw `af` |
| horizon | 4h | 1h, 24h, 3d |
| cooldown | 24 bars/symbol | 12, 48; none |
| design/test | 2026-01…04 / 2026-05…09 | walk-forward months |
| cost | 12 bps round-trip | 24 bps |

## C.1 Implementation plan (after approval)

- New module `src/quant_lab/analytics/sweeps.py`:
  `prior_range_levels`, `detect_sweeps`, `aligned_flow`, `sweep_signal`,
  `matched_spread` (reuse from `absorption`), `run_sweep_study`.
- Reuse `analytics/divergence.py` estimators and `analytics/absorption.py`
  (`block_bootstrap`, `matched_spread`); reuse `regimes`.
- New tests `tests/test_sweeps.py`: level causality (level excludes bar `t`),
  sweep detection correctness on hand-built OHLC, cooldown de-duplication,
  forward-return alignment/no-leakage, synthetic edge and no-edge fixtures.
- New script `scripts/sweep_study.py`; writes `reports/cvd_failed_sweeps_results.md`
  + machine outputs under `experiments/reports/sweeps/`.
- **No existing reports overwritten; no existing research code changed.**

---

# Part D — Explicit non-goals

- Not implementing H3–H6 in this pass.
- Not tuning thresholds to find a positive result; the sweep is reported whole.
- Not using tick/aggTrades.
- Not claiming cross-regime generality from a 2026-only sample.
