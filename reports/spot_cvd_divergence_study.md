# Spot–Perp CVD Divergence Study — Methodology (frozen)

**Status:** frozen — all decisions recorded (§12). No results inspected yet.
**Question owner:** (you)
**Instrument universe:** top-10 crypto, cross-market (spot∩perp) — 10 symbols,
`configs/divergence_universe.json` (see §3.2)
**Period:** 2026-01-01 → 2026-09-30 — **single-regime, in-sample / exploratory**
**Primary horizon:** 4h
**Author:** quant-lab research
**Related:** `reports/breakout_regime_study.md` (the negative-result template this
study deliberately imitates), `src/quant_lab/data/cvd.py`

---

## 0. How to review this document

This is a **pre-registration**: definitions, horizons and pass/fail rules are
fixed *before* looking at results, so a positive finding is credible and a
negative finding is conclusive. Review in this order: §1 (question), §2
(definitions), §8 (pass/fail bars), §12 (decision log).

The breakout study is the cautionary tale: a strong 2026 regime effect vanished
in 2025. This study cannot run that replication test (§3.3) — so its verdict is
**exploratory, not confirmatory**, and is labelled as such.

---

## 1. Research question and hypotheses

**Question.** Does *divergence between spot order-flow and perp order-flow*
(measured as cumulative volume delta, CVD) carry information about future price
moves, and is that information tradeable after costs?

**Why it might exist (economic rationale).**
- Spot taker flow is closer to "real" demand (custody, spot accumulation,
  non-leveraged buyers). Perp taker flow is dominated by leveraged, reflexive
  positioning.
- A move **led by spot** (spot net buying ≥ perp net buying) may be more durable.
- A move **led by perps** (perp net buying > spot) may be crowded leverage and
  prone to reversal, especially when funding is already positive.
- Conversely, perp selling *against* spot buying (shorts building into spot
  accumulation) is the classic setup for a short squeeze.

**Hypotheses (direction pre-specified).**

- **H1 (primary).** `D = spot_flow − perp_flow` (standardised, §2) has *positive*
  rank information for the forward return: `IC(D_t, r_{t→t+h}) > 0`.
- **H2.** The effect ranks relative future returns across the universe
  (market-neutral), not merely market direction (BTC beta).
- **H3 (variant).** Price–flow divergence (price makes a new high, CVD does not)
  predicts **negative** forward returns, and vice versa.
- **H0 (null).** No stable, cost-surviving predictive relationship.

**We commit to reporting H0 if H1–H3 fail.**

---

## 2. Construct definitions (fixed)

All flow is **taker flow** from 1-minute klines (which carry the taker-buy
split). For any bar:

```
delta       = 2 * taker_buy_volume - volume               # net taker flow, base units
quote_delta = 2 * taker_buy_quote_volume - quote_volume   # net taker flow, USDT
cvd         = cumsum(delta)                               # anchored at range start
```

`cvd` is non-stationary and is **never used raw**; only windowed/normalised
transforms are.

### 2.1 Rolling flows over window `w`

```
Fs_t = sum(delta_spot[t-w+1 .. t])     # spot net taker flow
Fp_t = sum(delta_perp[t-w+1 .. t])     # perp net taker flow
```

Primary `w` = 60 bars (1h bars → 60h). Robustness `w ∈ {15, 60, 240}`.

### 2.2 Standardisation

Each rolling flow is z-scored against its own trailing distribution over `W`
(primary `W` = 720 bars ≈ 30 days of 1h). All windows trailing, up to `t`.

```
zs_t = (Fs_t - mean_W(Fs)) / std_W(Fs)
zp_t = (Fp_t - mean_W(Fp)) / std_W(Fp)
```

### 2.3 The divergence signal (primary)

```
D_t = zs_t - zp_t          # continuous, symmetric about 0
sign(D_t) = spot-vs-perp leadership
```

Two complementary forms, both pre-registered:

- **Continuous:** `D_t` is the feature; test rank-IC against forward returns (§5).
- **Event:** an event fires when `|D_t| ≥ q`, `q` = trailing 95th percentile of
  `|D|` (self-normalising extreme, no absolute threshold).

Primary test uses the **continuous** form. The event form is the robustness /
"impact on prices" view.

### 2.4 Alternative normalisations (robustness only)

- **Sign agreement:** `sign(Fs) * sign(Fp)`.
- **Ratio:** `(Fs_t - Fp_t) / (|Fs_t| + |Fp_t|) ∈ [-1, 1]`.
- **Quote-flow** version using `quote_delta`.

A result that survives only one normalisation is noise.

### 2.5 Price–flow divergence (H3, secondary)

Over lookback `L` = 60 bars: `rp = close_t/close_{t-L} - 1`, `F` = rolling sum
of `delta` (spot), standardised as §2.2:

```
bear_div_t = (rp_t >= +k) and (F_t <= -m)     # price up, flow down
bull_div_t = (rp_t <= -k) and (F_t >= +m)     # price down, flow up
```

`k = m = 1.0` primary (z-units); swept in {0.5, 1.0, 1.5}. Predicted sign:
bearish divergence → negative forward return.

---

## 3. Data, universe and alignment

### 3.1 Sources

| series | source | how |
|---|---|---|
| spot bars 1m | `data/normalized/binance/spot/bars/{sym}/1m` | existing downloader |
| perp bars 1m | `data/normalized/binance/futures/bars/{sym}/1m` | existing downloader |
| spot CVD | `data/derived/cvd/spot/{sym}/1m` | `scripts/recreate_cvd.py` (done) |
| perp CVD | `data/derived/cvd/futures/{sym}/1m` | `scripts/recreate_cvd.py` (done) |
| funding | `data/normalized/binance/futures/funding/{sym}/na` | `recreate_funding_oi.py` |
| open interest | `data/normalized/binance/futures/open_interest/{sym}/na` | `recreate_funding_oi.py` |

CVD is exact per-bar from klines; no tick data is required. Spot and perp are
always kept as **separate series** and never summed.

### 3.2 Universe (frozen)

**Top-10 crypto, cross-market.** Rank each market's crypto-only top-200 by live
24h quote volume, take the **intersection** (both legs required), then keep the
**10 by combined spot+perp 24h quote volume**:

| # | symbol | combined 24h (USDT) |
|---|---|---|
| 1 | BTCUSDT | 17.96 B |
| 2 | ETHUSDT | 14.62 B |
| 3 | SOLUSDT | 3.76 B |
| 4 | ZECUSDT | 2.88 B |
| 5 | NEARUSDT | 1.52 B |
| 6 | XRPUSDT | 1.52 B |
| 7 | OGNUSDT | 1.12 B |
| 8 | HYPEUSDT | 0.80 B |
| 9 | BNBUSDT | 0.71 B |
| 10 | DOGEUSDT | 0.70 B |

Saved to `configs/divergence_universe.json`. Non-crypto instruments
(`XAUUSDT`, `PAXGUSDT`, `NVDAUSDT`, `AAPLUSDT`, tokenised equities/ETFs, …) are
excluded up front via `universe.is_non_crypto`; `METUSDT` (Meteora) is crypto but
does not make the top 10.

**Why 10, not 72.** The cross-section is thin, so the study leans on the pooled
time series (§5). A top-10 universe also concentrates on the most liquid, least
microstructure-noisy names.

**Survivorship caveat:** ranked as of today; symbols that died during 2026 are
absent (§10).

### 3.3 Period, split and regime caveat

**Window: 2026-01-01 → 2026-09-30.** This is a **single-regime, in-sample**
study — *exploratory*, not confirmatory. The breakout study failed because a
strong 2026 effect did not replicate in 2025; that replication is **not
available here**.

Available guard rails (within 2026):

- **Chronological design/test:** design = 2026-01 → 2026-04, test = 2026-05 →
  2026-09. Nothing is chosen on test.
- **Walk-forward** month to month (stability strip).
- **Leave-one-symbol-out** (n=10): no single name may drive the result.

If the exploratory result is promising, the natural follow-up is to build
2024–2025 CVD and re-run as a genuine out-of-sample test before risking capital.

### 3.4 Timeframe and alignment

- Analysis on **1h** bars (resampled from 1m, canonical resampler), primary; 15m
  and 1d secondary/robustness.
- Join spot and perp on the bar's **open** timestamp (shared UTC minute grid).
- Signal at the **close of bar `t`**; forward return from `t+1 … t+h`
  (execution delay 1 bar) — nothing at `t` predicts `t`.
- Funding/OI joined **as-of backward + forward-filled** (same convention as
  `analytics/confluence.py::enrich_frame`).

---

## 4. Outcome variables ("impact on prices")

Price = perp close. For signal bar `t`, horizon `h`:

| outcome | definition |
|---|---|
| `fwd_ret_h` | `close[t+h]/close[t+1] - 1` (gross) |
| `fwd_ret_h_cs` | `fwd_ret_h` minus cross-sectional mean at `t+h` (market-neutral) |
| `MFE_h` / `MAE_h` | max favourable / adverse excursion over `(t, t+h]`, % |
| `hit_h` | `sign(fwd_ret_h) == sign(D_t)` |
| `spread_h` | mean `fwd_ret_h_cs` in top-`D` decile − bottom-`D` decile |

Primary horizon `h = 4h` (4 bars). Robustness `h ∈ {1h, 24h, 3d}`.

**Cost anchor.** Default execution config: `fees 4 bps + slippage 2 bps` per
side → **~12 bps round-trip**. The long–short spread must exceed this.

---

## 5. Estimators and statistics

For each timeframe-horizon cell:

1. **Pooled time-series IC (headline).** Spearman of `D_t` vs `fwd_ret_h` pooled
   across all (symbol, time) observations, with a **block bootstrap over time**
   (block ≈ horizon, respecting overlapping windows) for the CI. Per-symbol ICs
   reported too (median, sign count).
2. **Cross-sectional rank IC (H2).** At each timestamp, Spearman of `D` vs
   `fwd_ret_h_cs` across the 10 symbols; average over time. **With n=10 this is
   low-powered** — reported for completeness, not the headline.
3. **Event study.** Forward return by `|D|` quantile / event bucket, mean and
   median, with block-bootstrap 95% CI.
4. **Monotonicity.** Decile table of `D` → `fwd_ret_h_cs`; decile-spread
   Spearman must be high before believing a spread.
5. **Significance.** t-stat from the block bootstrap (Newey–West style), not
   naive i.i.d. p-values. Event counts and base rates reported.

All estimators applied identically to design and test.

---

## 6. Controls and confounders (must survive all)

The biggest threat is **mechanical reverse causality**: taker flow and the
same-bar return are the same event. Require the effect to survive:

1. **Lagged past return** `r_{t-L→t}` (is `D` just momentum?).
2. **Realized volatility** (is `D` just a vol proxy?).
3. **Funding rate** and **OI 1h change** (is `D` just carry?).
4. **Cross-sectional de-meaning** at the horizon (removes BTC beta).
5. **Regime stratification** (`analytics/regimes.py`, trend × vol).

Regression (per horizon, pooled, cluster by symbol×day):

```
fwd_ret_cs ~ D + past_ret + realized_vol + funding + oi_change + regime FE
```

Report the `D` coefficient and its bootstrap CI. Edge claim = the `D` coefficient
is significant and sign-stable across design/test.

---

## 7. Validation protocol

- **Chronological, never random.** Design = 2026-01→04, test = 2026-05→09.
- **Walk-forward** month to month (`analytics.splits.walk_forward_splits`) —
  a stability strip, since this is a fixed-signal study (nothing is "fit").
- **Parameter discipline.** One primary spec (`w=60, W=720, h=4h`, continuous
  `D`, top-10). Sweeps reported **in full**, never used to pick a winner, and
  assessed with **Benjamini–Hochberg FDR** across the grid.
- **Leave-one-symbol-out** (n=10). **Sign consistency** design vs test; a flip is
  a kill. **Mega-cap exclusion** (drop BTC and ETH) as a final sanity view.

---

## 8. Pre-registered decision criteria

**Declare an edge if and only if ALL hold on the *test* period:**

| # | criterion | threshold |
|---|---|---|
| C1 | Pooled time-series rank IC (primary spec) | `> 0.02` with 95% block-bootstrap CI excluding 0 |
| C2 | Sign-stable | design IC and test IC both `> 0` |
| C3 | Monotone deciles | decile-spread Spearman `> 0.7` |
| C4 | Survives controls | `D` coefficient CI excludes 0, `> 0` |
| C5 | Cost-surviving | top–bottom decile `spread_h` `> 12 bps` |
| C6 | Leave-one-symbol-out | dropping any single symbol keeps IC `> 0` |
| C7 | Not mega-cap-driven | IC `> 0` with BTCUSDT and ETHUSDT removed |

**Anything short of all seven = H0 (no edge), reported as a negative result.**
Criteria will not be relaxed after seeing results.

---

## 9. Implementation plan (repo mapping)

**New module** `src/quant_lab/analytics/divergence.py`:

| function | responsibility |
|---|---|
| `build_divergence_panel(...)` | load spot+perp bars, resample, compute `Fs,Fp,zs,zp,D`, forward returns, controls → tidy panel |
| `flow_divergence(...)` | §2.1–2.4 transforms |
| `price_flow_divergence(...)` | §2.5 |
| `event_study(panel, signal, horizons)` | §5.3 |
| `rank_ic(panel, signal, fwd, ...)` | §5.1–5.2 |
| `run_divergence_study(...)` | orchestration + design/test tables |

**CLI** `quantlab analyze divergence` (thin), **script**
`scripts/divergence_study.py` writing `reports/spot_cvd_divergence_results.md`
and artifacts under `experiments/reports/divergence/`.

**Reuse:** `analytics/splits.py`, `analytics/robustness.py`,
`analytics/regimes.py`, `analytics/excursions.py`, `data/resample.py`,
`data/cvd.py`.

**Tests** (`tests/test_divergence.py`): causality (appending future bars must not
change past `D`), normalisation against a hand-computed window, forward-return
alignment (no same-bar leakage), deterministic synthetic edge / no-edge fixture.

**Steps:** (bars+CVD already built) → implement `divergence.py` + tests → run
design (Jan–Apr), freeze table → run test (May–Sep) → apply §8 mechanically →
write results doc.

---

## 10. Limitations and threats to validity

- **In-sample, single regime (2026).** The dominant limitation; verdict is
  exploratory.
- **n = 10 cross-section.** Cross-sectional IC is low-powered; headline is the
  pooled time series.
- **CVD is taker flow only** — no maker/limit intent or cancellations.
- **Mechanical contemporaneity** — all claims are strictly `t → t+h`, 1-bar delay.
- **Spot/perp microstructure differ**; a "divergence" can be a venue artifact.
- **1-minute bars** cap resolution.
- **Survivorship** from selecting today's top-10.
- **Multiple testing** across horizons/normalisations; FDR + pre-registration are
  mitigations, not a fix.
- **Costs** are a flat per-side fee+slippage; no market impact.

---

## 11. Deliverables

1. This methodology (frozen).
2. `divergence.py` + tests.
3. `reports/spot_cvd_divergence_results.md` — design/test tables, deciles, event
   study, controls, full robustness grid, verdict.
4. Artifacts under `experiments/reports/divergence/`.
5. A one-line verdict: **edge / no edge**, with the exact criteria passed/failed,
   and an explicit "exploratory (in-sample)" label.

---

## 12. Decision log (frozen)

| # | decision | value |
|---|---|---|
| 1 | Primary construct | **spot-CVD vs perp-CVD divergence (H1)**; price–flow (H3) secondary |
| 2 | Universe | **top-10 crypto cross-market** — `configs/divergence_universe.json` |
| 3 | Primary horizon | **4h** (secondary 24h; robustness 1h/1d/3d) |
| 4 | Period | **2026-01→09 only** — in-sample/exploratory; internal design/test split |
| 5 | Cost hurdle (C5) | **~12 bps** round-trip |
| 6 | Extra definitions | none beyond §2.4 / §2.5 |

All decisions frozen. Implementation proceeds without inspecting results first.
