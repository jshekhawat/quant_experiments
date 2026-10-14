# H1 — Price Absorption: Design & Repository Assessment

**Status:** frozen design — awaiting go-ahead before implementation.
**Question owner:** (you)
**Scope:** H1 only (price absorption). H2–H6 are *not* implemented here.
**Related:** `reports/spot_cvd_divergence_study.md` (completed negative result),
`reports/spot_cvd_divergence_results.md`.

---

# Part A — Repository assessment

## A.1 What already exists and is reusable

**Data layer** (`src/quant_lab/data/`)
- `storage.py` — `DatasetKey`, `read_dataset` (lazy Parquet scan, date-filtered,
  predicate push-down), `write_partition`, monthly partition layout.
- `resample.py::resample_bars` — canonical OHLCV aggregation; `volume`,
  `quote_volume`, `taker_buy_*` are **summed**, `timestamp` is the bucket *open*.
  This is exactly what H1 needs to go 1m→1h.
- `sources/binance.py`, `build.py` — resumable bulk downloader/normalizer.
- `universe.py` — top-N by live volume, `is_non_crypto`, `cross_market_universe`.
- `cvd.py` — `volume_delta`/`cvd` construction + monthly partitions.
- `validate.py` — data-quality report.

**Feature layer** (`src/quant_lab/features/`)
- `core.py` — `sma, ema, atr, rsi, vwap, returns, rolling_volume,
  realized_volatility, zscore, roc, stdev, rolling_max/min, momentum,
  true_range, price_range` + `safe_div`.
- `market.py` — `volume_delta, cvd, cvd_change, open_interest_change,
  funding_rate, basis, relative_volume, range_expansion, trade_imbalance`.
- `compute_features(frame, [FeatureSpec(...)])` registry.

**Analytics layer** (`src/quant_lab/analytics/`)
- `splits.py` — `chronological_split`, `split_by_dates`, `walk_forward_splits`.
- `excursions.py::analyze_excursions` — post-signal **time-to-reversion,
  max-favourable-move, excursion-to-reversion**; close/touch basis. H1 reuses the
  path-outcome machinery (MFE/MAE-style).
- `robustness.py` — `bootstrap_returns` (**block bootstrap**), `monte_carlo_trades`,
  `parameter_sensitivity`, `metrics_from_returns`.
- `regimes.py` — **causal** trend×vol classifier (`classify_regimes`,
  `attach_regimes`, `evaluate_by_regime`), shifted one bar.
- `confluence.py` — signal labelling, design/validate evaluation, `enrich_frame`
  (as-of joins of spot/funding/OI), `default_filters`.
- `divergence.py` (added this session) — `build_divergence_panel` plus **reusable
  estimators**: `pooled_ic`, `cross_sectional_ic`, `decile_table`, `event_study`,
  `block_bootstrap_ic` (vectorised). H1 reuses these directly.

**Reporting / execution**
- `reporting/report.py` (summary.md/json + plots + tearsheet);
  `analytics/*` expose `to_markdown()`.
- `backtest/` (VectorBT engine, metrics, explicit assumptions). Execution defaults
  in `configs/default.toml`: **fees 4 bps, slippage 2 bps per side → ~12 bps
  round-trip**.
- CLI `quantlab analyze …`, thin scripts in `scripts/`.
- Tests: `tests/conftest.py` fixtures (`bars`, `small_bars`, `tmp_root`); strong
  existing emphasis on **causality / lookahead** tests.

## A.2 What does *not* exist (must be built)

- No **swing/pivot detection** (relevant to H2, not H1).
- No **price-impact / absorption / flow-effectiveness** model — H1 is new.
- No rolling price-impact regression or Kyle-lambda / Amihud estimator.
- No regime-conditional regression helper.
- `AlphaFlow` (Rust) is a separate project and is **not present** in this repo.

## A.3 Data coverage (the binding constraint)

Measured from `data/normalized/binance` (this repo):

| dataset | coverage |
|---|---|
| perp 1m bars | **100 symbols, ~2026-01…2026-09.** Only **BTCUSDT** spans 2024-01…2026-09; ETHUSDT partial |
| spot 1m bars | 100 symbols, 2026; BTCUSDT 2025-01…2026-09 |
| funding | 100 symbols (2026) |
| open interest | 20 symbols so far (backfill job running) |
| 1d bars | BTC/ETH/ZEC 2025-01…2026-09 |

**Implication.** Any H1 test on the current universe is **single-regime, in-sample
2026** and must be labelled *exploratory* — the same limitation called out in the
divergence study. The only genuine multi-regime cross-check is **BTCUSDT**
(2024–2026), and to a lesser extent ETHUSDT. This is stated in every pass/fail
claim; we will not assert generality across regimes we did not observe.

## A.4 Data caveats carried into H1

- Base vs quote units: H1 works in **quote (USDT)** units throughout.
- `delta = 2*taker_buy_volume − volume`; per-bar flow is *net taker* flow, not the
  full book — absorption is a **proxy**, not a proven mechanism.
- Contemporaneous vs predictive: the same-bar price impact is an *observable
  conditioning feature* (known at bar close); it must never enter the forward
  window (start at `t+1`).
- Missing bars / abnormal volume: drop bars with nulls; normalize flow by trailing
  activity so outliers are relative, not absolute.
- Tick data is **not** required and not authorized; 1m bars suffice.

---

# Part B — Frozen H1 specification

## B.1 Hypothesis and economic rationale

**Question.** Does the *effectiveness of aggressive flow in moving price* carry
information about subsequent price behaviour?

**H1.** When aggressive **selling** produces unusually **little downward** price
movement (absorption), price subsequently **outperforms** comparable strong-selling
events. Symmetric: absorbed aggressive **buying** (little upward movement) precedes
**underperformance**.

**Rationale (a proxy, not a mechanism).** If large aggressive flow moves price much
less than a trailing model predicts, resting/passive liquidity likely opposed it.
*If* that passive side is persistent, price should later move against the
aggressor. Alternative explanation: low impact simply means low-information flow,
which may mean-revert. H1 tests whether the residual is informative, **net of**
volatility, trend and flow magnitude.

**Directional definition (fixed).**
- Predicted direction `d_t = −sign(f_t)` (absorbed selling → up; absorbed buying → down).
- **Oriented forward return** `y_t = d_t · fwd_ret_t`. H1 predicts `y_t > 0` when
  absorption is high.

## B.2 Event definitions and formulas (all causal, trailing only)

Per symbol, on **1h** bars (resampled from 1m):

```
qd_t      = 2 * taker_buy_quote_volume - quote_volume        # signed taker flow, USDT
r_t       = log(close_t / close_{t-1})                        # bar log return
adv_t     = mean(quote_volume, 20)  (trailing, min_periods=20)  # activity scale
f_t       = qd_t / adv_t             (guard: null if adv_t == 0)  # normalized flow
```

**Expected impact (regression-based, not a naive ratio).** A rolling OLS of the
bar return on normalized flow over a trailing window `W` (primary `W = 720` ≈ 30d),
intercept included, using only bars `≤ t`:

```
r_i ≈ alpha_t + beta_t * f_i          for i in (t-W, t]
rhat_t = alpha_t + beta_t * f_t       # expected same-bar impact
s_t    = r_t - rhat_t                 # impact residual (signed surprise)
```

This replaces the unstable `r/qd` ratio with a conditional estimate, per the
brief. Bars where the denominator would explode are excluded, not divided.

**Absorption score (the signal):**

```
A_t = -sign(f_t) * s_t                     # >0 = aggressor absorbed
A_t = A_t / std_W(A)   (trailing std)      # scale-free z-score, per symbol
```

**Strong-flow event:** `|f_t| ≥ q90(|f|)` where `q90` is the trailing 90th
percentile of `|f|` over `W` (self-normalising; no absolute threshold). Primary
tests are **within strong-flow events**; `A_t` is also tested continuously.

**Execution delay.** Signals are evaluated at the **close of `t`**; all outcomes
begin at `t+1`.

## B.3 Required input data

Perp 1m klines for the universe (already on disk for 2026). Columns used:
`timestamp, open, high, low, close, volume, quote_volume, taker_buy_volume,
taker_buy_quote_volume`. No spot, funding or OI is required for H1's primary test;
funding/OI/regimes are **controls**, so partial OI coverage (20 symbols) limits
only the robustness slice, not the primary test.

## B.4 Baselines and matched controls

1. **Price-action-only baseline** — strong-flow events with *no* absorption
   conditioning: unconditional oriented forward return.
2. **Flow-only baseline** — condition only on `|f_t|` (and `sign(f_t)`), ignoring
   the impact residual. If absorption adds nothing over “big flow”, H1 fails.
3. **Matched controls** — each absorbed event is compared with non-absorbed
   strong-flow events **matched** on trailing buckets of: realized vol, trailing
   return (trend), `|f_t|`, and `regime` label. Pairwise within-cell difference is
   reported alongside the pooled view.

## B.5 Primary outcome and horizon

- **Primary:** oriented forward return over **h = 4h** (entry close `t+1`, exit
  close `t+1+h`), pooled across the universe.
- **Secondary:** MFE/MAE of the oriented return; directional hit rate.
- **Robustness horizons:** 15m, 1h, 24h, 3d. Reported in full.

## B.6 Confounders and leakage

- **Mechanical vol dependence** — residual is smaller when vol is low; normalize
  flow and residual by trailing activity/vol; match on realized vol.
- **Mean reversion / trend** — absorbed events may follow large moves; control and
  match on trailing return; regime stratification.
- **Liquidity** — normalize flow by trailing ADV; match on `|f|`.
- **Contemporaneous→predictive** — `s_t` uses only bar `t`; forward window starts
  `t+1`; explicit alignment test.
- **Look-ahead in the regression/quantiles** — all rolling windows trailing,
  `W`/`q90` computed from past bars only; causality test on shifted inputs.
- **Multiple testing** — one primary spec; sweeps reported with Benjamini–Hochberg
  FDR.

## B.7 Statistical evaluation & cost assumptions

- **Pooled rank IC** of `A_t` vs `y_t` (headline), plus within-strong-flow IC.
- **Block bootstrap** (blocks ≈ horizon) for CIs — respects overlapping windows.
- **Decile / quantile monotonicity** of `y_t` across `A_t`.
- **Matched-control paired difference** (absorbed − control), with CI.
- **Regression:** `y_t ~ A_t + past_ret + realized_vol + |f_t| + regime FE`,
  clustered by symbol×day; report the `A_t` coefficient and bootstrap CI.
- **Costs:** ~12 bps round-trip; a tradable claim must clear it.
- **Stability:** per-symbol and per-regime tables; chronological design/test.

## B.8 Pass/fail criteria (preregistered)

**Declare an edge iff ALL hold on the test period:**

| # | criterion | threshold |
|---|---|---|
| P1 | Pooled rank IC(`A_t`, `y_t`) | `> 0.02`, 95% block-bootstrap CI excludes 0 |
| P2 | Sign-stable | design IC and test IC both `> 0` |
| P3 | Matched-control spread (absorbed − control, 4h) | `> 12 bps`, correct sign, CI excludes 0 |
| P4 | Survives controls | `A_t` coefficient CI excludes 0, `> 0` |
| P5 | Monotone quintiles | Spearman(quintile idx, mean `y_t`) `> 0.7` |
| P6 | Not single-name / mega-cap | holds dropping any one symbol; holds ex-BTC/ETH |
| P7 | Horizon-consistent | `> 0` at 4h **and** at ≥1 of {1h, 24h} |

Anything short of all seven → **H0 (no edge)**. Thresholds frozen before results.
No parameter is chosen on the test period.

## B.9 Feasibility with existing data

**Yes.** 1m perp bars (2026) → 1h via `resample_bars`; rolling OLS and trailing
quantiles in Polars/NumPy; estimators reuse `analytics/divergence.py`. **No new
data download and no tick data required.** The binding limitation is regime
breadth (2026 only), which forces the *exploratory* label.

---

# Part C — Frozen primary specification

| element | primary | robustness |
|---|---|---|
| universe | crypto-only **top-10** perps (`configs/divergence_universe.json`) | top-30, ex-BTC/ETH |
| timeframe | 1h (from 1m) | 15m, 4h |
| flow scaling | `f = qd / ADV(quote, 20)` | ADV 60; base-unit flow |
| impact model | rolling OLS `r ~ f`, `W = 720` | `W ∈ {240, 720, 1440}`; add vol term |
| event | `|f| ≥ q90` trailing | `q95`; continuous (no event gate) |
| signal | `A = −sign(f)·s`, z-scored | rank-transformed |
| horizon | 4h | 15m, 1h, 24h, 3d |
| design/test | design 2026-01…04 / test 2026-05…09 | walk-forward months |
| cost | 12 bps round-trip | 24 bps |

## C.1 Implementation plan (after approval)

- New module `src/quant_lab/analytics/absorption.py`:
  `normalized_flow`, `rolling_impact_model` (returns `alpha,beta`), `absorption_signal`,
  `strong_flow_events`, `match_controls`, `event_outcomes`, `run_absorption_study`.
- Reuse `analytics/divergence.py` estimators (`pooled_ic`, `decile_table`,
  `event_study`, `block_bootstrap_ic`) and `analytics/regimes.attach_regimes`.
- New tests `tests/test_absorption.py`: formula correctness, **causality**
  (appending future bars must not change past `A_t`), rolling-OLS equals a
  hand-computed window, forward-return alignment/no-leakage, synthetic edge and
  no-edge fixtures.
- New script `scripts/absorption_study.py` (reproducible CLI); writes
  `reports/cvd_absorption_results.md` + machine-readable outputs under
  `experiments/reports/absorption/`.
- **No existing reports overwritten; no existing research code changed.** If a
  shared helper is needed, it is imported, not duplicated.

---

# Part D — Explicit non-goals

- Not implementing H2–H6 in this pass.
- Not tuning H1 parameters to find a positive result; the sweep is reported whole.
- Not using tick/aggTrade data.
- Not claiming cross-regime generality from a 2026-only sample.
