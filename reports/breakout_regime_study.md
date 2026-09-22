# Breakout, Confluence and Regime Study — BTCUSDT

**Status:** complete
**Instrument:** BTCUSDT (Binance USD-M perpetual + spot)
**Period covered:** 2025-01-01 → 2026-08-31 (plus 2024-01..03 futures bars, partial derivatives)
**Author:** quant-lab research
**Code:** `src/quant_lab` (see "Reproducibility" at the end)

---

## 0. Executive summary

**Question.** Does a volatility/band breakout on BTCUSDT have an edge, and can
that edge be recovered with (a) confluence filters or (b) a market-regime gate?

**Answer.** No. The base breakout trigger has no durable edge, and neither
confluence filters nor a regime gate produced a robust, statistically
significant improvement. The one apparently strong result — breakouts in an
"up-trend, high-volatility" regime looked much better in 2026 — **failed to
replicate on 2025**, where the best regime was in fact a *down* regime. Regime
gating reduced losses substantially, but that was almost entirely a reduction
in trade count and fees, not a discovered edge. Even gated, the strategy lost
money in both 2025 and 2026.

This is a **negative result**. It is reported in full because the negative
result is the deliverable: it prevents further effort on a dead trigger family.

**Headline numbers**

| finding | value |
|---|---|
| 1m Bollinger breakout, "good" outcome rate | ~5% |
| 15m, same definition | ~18–20% |
| 1h, same definition | ~32–36% |
| best confluence filter (15m), out-of-sample lift | 1.23× (not significant, z≈0.6) |
| `up_highvol` regime, 2026 | 35.0% vs 17.8% baseline |
| `up_highvol` regime, 2025 | 19.8% vs 18.8% baseline (**no edge**) |
| gated vs ungated backtest, 2025 | −25.4% vs −78.5% (still losing) |
| gated vs ungated backtest, 2026 | −7.0% vs −58.0% (still losing) |

---

## 1. Research question and thesis

The lab's reference strategy is a volatility/band breakout: price crosses above
a volatility band on expanding volume. The working hypothesis was:

> H1: raw breakouts are noise, but a *confluence* of conditions (volume,
> order-flow, trend, volatility regime, derivatives) selects the subset that
> trends.

A second hypothesis followed from the data:

> H2: the edge is *regime-dependent* — breakouts work in trending/expanding
> regimes and fail in range/contraction regimes. A regime classifier should
> therefore separate signal from noise.

Both were tested on a design period and a separate validation period, with a
pre-registered outcome definition, and H2 was additionally stress-tested by
replicating 2026 on 2025.

---

## 2. Data

### 2.1 Sources and coverage

All data from Binance Public Data bulk archives (`data.binance.vision`), via a
resumable, checksum-verified downloader (`quant_lab.data.sources.binance`).

| dataset | market | 2025 rows | 2026 (Jan–Aug) rows | notes |
|---|---|---|---|---|
| bars 1m | futures (USD-M) | 525,600 | 349,920 | OHLCV + taker-buy volume |
| bars 1m | spot | 525,600 | 349,920 | for spot–perp basis |
| funding | futures | 1,095 | 729 | 8-hourly |
| open interest | futures | 105,117 | 69,984 | 5-minute, derived from daily `metrics` |

Ranges: 2025-01-01 → 2025-12-31 and 2026-01-01 → 2026-08-31. Also present:
BTCUSDT/ETHUSDT futures 2024-01..03, and 2024-01/02 funding + 5 days OI.

### 2.2 Storage and validation

Canonical schemas (`quant_lab.data.schemas`), Parquet partitioned by
`exchange/market_type/dataset/symbol/timeframe/year/month.parquet`, queried via
DuckDB. Validation (`quant_lab.data.validate`) checks missing/duplicate/
non-monotonic timestamps, invalid prices/quantities, gaps, outliers and
timezone consistency. **All datasets used here passed with 0 errors and 0
warnings.** No data was fabricated or silently repaired.

### 2.3 Known data caveats

- Monthly archives cover whole calendar months; query-time filtering bounds use.
- Funding timestamps carry ~1–15 ms jitter; truncated to the minute for joins.
- `open_interest` is derived from the daily `metrics` archive (5-minute grid).
- Liquidations are unavailable (no reliable bulk historical source) and were
  not used.
- Trade-level data (`trades`/`aggTrades`) was **not** downloaded: 1m bars
  already carry volume, quote volume and taker-buy volume, which is sufficient
  for the flow features used. Tick data would be needed only for sub-minute
  order-flow or execution-realism work.

---

## 3. Methodology

### 3.1 Triggers tested

| strategy | definition | params (defaults) |
|---|---|---|
| `bollinger_breakout` | close crosses **above** upper Bollinger band; exit close < mid | window 20, 2σ |
| `volatility_breakout` | close > EMA and \|Δclose\| > k·ATR and volume > m·mean(volume) | EMA 100, ATR 14, k 1.5, m 1.5 |
| `squeeze_release` | Bollinger cross **only after** a compressed-bandwidth squeeze, with bandwidth expanding | window 20, squeeze z ≤ −0.5, lookback 96 |
| `range_breakout_hold` | close above prior N-bar high and **holds** for M consecutive closes | lookback 60, hold 2, exit lookback 30 |

All strategies are long-only by default, use trailing (causal) windows only, and
are validated for causality by tests that assert signal/feature values do not
change when future bars are appended.

### 3.2 Execution assumptions (all backtests)

Recorded with every run; defaults are non-zero:

| assumption | value |
|---|---|
| initial capital | 100,000 |
| fees (per side) | 0.0004 |
| slippage (per side) | 0.0002 |
| position size | 100% of equity |
| leverage | 1× |
| execution delay | 1 bar |
| timing | signal at bar *t* close → fill at bar *t+1* open |

Higher timeframes are produced by canonical OHLCV resampling of 1m bars
(`quant_lab.data.resample`): open=first, high=max, low=min, close=last,
volumes summed, bucket labelled by its **open** time; partial trailing buckets
are dropped. Funding and open interest are as-of joined (backward) and
forward-filled; spot is resampled identically.

### 3.3 Outcome measures

For every signal, the forward path is measured from the **close of the signal
bar** over a 7-day horizon:

- **time to reversion** — first bar whose *close* returns to the signal price
  (a stricter "touch" variant using the bar low was also computed).
- **excursion to reversion** — the maximum favourable price move (highest high
  for longs) between the signal and the reversion bar.
- **good outcome (label)** — `excursion_to_reversion ≥ X` **and**
  `time_to_reversion > 1h`. Used at X = 0.25% (1m study) and X = 0.5% (main
  study). A "never reverted" signal counts as slow.

Reversion and excursion are also reported as a **cumulative reversion curve**
and a **bucketed excursion table** (disjoint reversion-speed buckets).

### 3.4 Confluence filters

Each filter is a single explicit boolean condition evaluated at the signal bar
(no model): `volume ≥ 1.5×mean`, `taker_buy/volume ≥ 0.55`, 20-bar CVD > 0,
`close > EMA200`, strong close (close in top 30% of bar range), bandwidth
z ≤ −0.5, spot–perp basis > 0, funding ≤ 0, OI 1h change > 0. Combinations are
greedy ANDs of the best standalone filters on the design period.

### 3.5 Regime classifier

A **causal** classifier (`quant_lab.analytics.regimes`), computed on a 1h
regime timeframe from trailing statistics only:

- **trend**: `up`/`down`/`range` from a volatility-normalised trailing move —
  the 24h log return divided by `realized_vol × √24`, threshold ±1.0.
- **volatility**: `highvol`/`lowvol` from realized vol vs its trailing median.
- combined label `{trend}_{vol}` (e.g. `up_highvol`), **shifted one regime bar**
  so a signal only ever sees regimes that had already closed.

### 3.6 Evaluation protocol

- **Design / validation split.** Filters and the regime hypothesis were chosen
  on a design window and reported on a disjoint validation window. The final
  test used design = **2025** and validation = **2026 Jan–Aug**.
- **Baseline and lift.** For each filter, `lift = P(good | filter) / P(good |
  baseline)`.
- **Significance.** Two-proportion z-test of the filtered rate against the
  baseline rate (reported as a 2-sided p-value). This is essential because the
  base rate is low and the samples are small.
- **Multiple testing.** Many filters × periods were tested; results are
  interpreted with that in mind (no correction is claimed, so only large,
  replicated effects would be credible).

---

## 4. Results

### 4.1 Phase 1 — Reversion and excursion at 1m (Jul–Aug 2026)

2,492 Bollinger breakout signals, 1m, close-basis reversion, 7-day horizon.

**Time to reversion (cumulative):**

| within | 1m | 5m | 15m | 1h | 4h | 1d | 7d |
|---|---|---|---|---|---|---|---|
| fraction reverted | 55.8% | 77.1% | 85.8% | 92.9% | 96.5% | 98.2% | 99.2% |

Median 1 bar (1 min), mean 51.2 min, p75 4 min; 21 signals (0.8%) never
reverted within 7 days.

**Excursion before reversion, by reversion-speed bucket** (value in USDT):

| bucket | n | mean | median | min | max | mean % |
|---|---|---|---|---|---|---|
| ≤1m | 1390 | 10.4 | 2.8 | 0.0 | 384.9 | 0.015% |
| 1m–5m | 532 | 43.9 | 34.1 | 0.1 | 460.9 | 0.066% |
| 5m–15m | 216 | 98.6 | 70.3 | 2.9 | 1409.5 | 0.143% |
| 15m–1h | 178 | 201.6 | 161.6 | 9.2 | 979.3 | 0.300% |
| 1h–4h | 88 | 384.7 | 311.9 | 52.6 | 2171.4 | 0.564% |
| 4h–1d | 42 | 702.1 | 567.5 | 129.3 | 2404.7 | 1.048% |
| 1d–7d | 25 | 1997.6 | 1831.5 | 607.7 | 4299.7 | 2.955% |
| not reverted | 21 | 9748.8 | 9343.2 | 1952.0 | 16947.6 | 14.871% |

**Interpretation.** Most 1m breakouts revert within a minute and move almost
nothing first (median excursion 2.8 USDT; the ≤1m bucket is 56% of all
signals). Excursion and reversion time are strongly, mechanically related:
only signals that keep running stay out of reversion. The raw 1m trigger is
essentially microstructure noise.

### 4.2 Phase 2 — Confluence at 1m (Jul design / Aug validate, 0.25% label)

Baseline P(good): 6.28% design, 6.16% validate.

| filter | design lift | validate lift | validate p (2-sided) |
|---|---|---|---|
| volume ≥ 1.5× | 0.90 | 1.17 | 0.37 |
| taker_ratio ≥ 0.55 | 0.98 | 1.06 | 0.72 |
| cvd_20 > 0 | 0.99 | 1.07 | 0.70 |
| close > EMA200 | 1.03 | 1.10 | 0.57 |
| strong_close ≥ 0.7 | 0.94 | 1.02 | 0.89 |
| bandwidth_z ≤ −0.5 | 0.97 | 1.08 | 0.70 |
| oi_change_1h > 0 | 1.05 | 0.94 | 0.75 |
| basis > 0 | n/a (0 signals) | 0.87 | 0.81 |
| funding ≤ 0 | 0.00 (n=12) | n/a (0 signals) | — |

No filter was significant. `volume` flipped sign between periods. The greedy
combination (design) selected `oi_change` first, which then reversed in
validation — a clean demonstration of selection overfitting.

### 4.3 Phase 3 — Higher timeframes (Jul/Aug split, 0.5% label)

| trigger | tf | P(good) design | P(good) validate | signals/mo |
|---|---|---|---|---|
| bollinger | 1m | — | 4.9% | ~1250 |
| bollinger | 15m | 18.0% | 19.5% | ~85 |
| bollinger | 1h | 30.4% | 32.1% | ~25 |
| squeeze_release | 15m | 14.3% | 21.9% | ~67 |
| squeeze_release | 1h | 16.7% | 29.4% | ~15 |
| range_breakout_hold | 15m | 16.7% | 30.0% | ~11 |
| range_breakout_hold | 1h | 0.0% | 0.0% | ~3 |

The base rate rises ~4× (15m) to ~6× (1h) versus 1m, but signal counts collapse
and `range_breakout_hold` is too restrictive to evaluate. No filter was
significant at any timeframe.

### 4.4 Phase 4 — Full 2026, design Jan–Jul / validate Aug (0.5% label)

| trigger | tf | baseline design | baseline validate | best filter (design/validate lift) |
|---|---|---|---|---|
| bollinger | 1m | 5.0% (n=8304) | 5.0% (n=1234) | none > 1.11, ns |
| bollinger | 15m | 17.7% (n=566) | 19.5% (n=82) | volume+EMA200: 1.19 / **1.23** (z=0.59) |
| bollinger | 1h | 35.5% (n=138) | 32.1% (n=28) | none (all ≈1.0 or below) |
| squeeze_release | 15m | 18.1% (n=507) | 21.9% (n=64) | volume+EMA200: 1.17 validate |
| squeeze_release | 1h | 36.2% (n=116) | 29.4% (n=17) | none; OI negative |

At 15m, two filters were positive in **both** windows: `volume ≥ 1.5×`
(1.13/1.14) and `close > EMA200` (1.11/1.15); their combination gave 1.19/1.23.
This was the first directionally consistent signal, though not significant.
Adding `oi_change_1h` produced a large validation lift (2.05, n=20, z=1.94) but
its design lift was only ~1.06 — treated as regime-specific, not real. At 1h no
filter helped; breakouts there are already good ~⅓ of the time.

### 4.5 Phase 5 — Regime classifier, 2026

Baseline: 17.7% design (Jan–Jul), 19.5% validate (Aug).

| regime | design n | design P(good) | validate n | validate P(good) |
|---|---|---|---|---|
| **up_highvol** | 46 | **23.9%** | 16 | **62.5%** |
| up_lowvol | 55 | 16.4% | 3 | 0.0% |
| range_highvol | 168 | 18.5% | 22 | 13.6% |
| range_lowvol | 212 | 17.5% | 14 | 7.1% |
| down_lowvol | 36 | 11.1% | — | — |
| down_highvol | 24 | 12.5% | 4 | 0.0% |

`up_highvol` was the best regime in both windows, and down-regimes were worst.
At this point H2 looked supported — pending the 2025 replication.

### 4.6 Phase 6 — 2025 replication (design = 2025, validation = 2026)

Baseline: 18.8% (2025), 17.8% (2026 Jan–Aug).

| regime | 2025 n | 2025 P(good) | 2026 n | 2026 P(good) |
|---|---|---|---|---|
| **up_highvol** | 81 | **19.8%** | 60 | **35.0%** |
| up_lowvol | 80 | 15.0% | 62 | 14.5% |
| range_highvol | 282 | 21.6% | 186 | 17.7% |
| range_lowvol | 400 | 17.2% | 250 | 16.4% |
| **down_highvol** | 57 | **24.6%** | 27 | 11.1% |
| down_lowvol | 41 | 12.2% | 39 | 10.3% |

**H2 is refuted.** In 2025, `up_highvol` (19.8%) is indistinguishable from the
18.8% baseline and is *not* the best regime — `down_highvol` (24.6%) is. The
strong 2026 `up_highvol` effect did not replicate. No regime is robustly good
across both years.

**Regime-gated backtest** (gate = `up_highvol` + `up_lowvol`):

| period | variant | trades | total return | max DD | Sharpe | win | PF | fees |
|---|---|---|---|---|---|---|---|---|
| 2025 | ungated | 619 | −78.5% | −79.3% | −4.29 | 27.8% | 0.68 | 34,189 |
| 2025 | gated | 107 | −25.4% | −26.4% | −3.11 | 23.4% | 0.46 | 7,363 |
| 2026 | ungated | 417 | −58.0% | −62.6% | −5.28 | 25.2% | 0.56 | 23,008 |
| 2026 | gated | 79 | −7.0% | −19.2% | −0.87 | 29.1% | 0.86 | 6,116 |

Gating cut losses sharply in both years, but trade count fell 5–6× and fees
fell correspondingly — the improvement is **fee/turnover avoidance**, not a
recovered edge. Even gated, both years lose money and profit factor stays below
1.

---

## 5. Interpretation — why it fails

1. **The trigger has no edge.** A band/volatility cross fires on a large
   fraction of bars and, at 1m, is dominated by microstructure; at 15m/1h it is
   better but still a coin-flip biased toward fast reversion.
2. **The "good" outcome and PnL diverge.** "Excursion ≥ 0.5% before reversion"
   is not a profit measure: the strategy exits at the mid-band, pays fees on
   both sides, and gives back the excursion. A signal can be labelled good and
   still lose money.
3. **Regime effects are not stable.** The apparently strong 2026 `up_highvol`
   result is a single-sample artifact; 2025 contradicts it. With ~60–80 signals
   per regime-year, confidence intervals are wide enough to manufacture such
   effects by chance.
4. **Low base rates destroy statistical power.** At 1m the good rate is ~5%
   with thousands of signals; at 1h it is ~⅓ with only ~25/month. There is no
   window where both the effect size and the sample are large enough to prove a
   modest (1.2–1.3×) lift.
5. **Gating's apparent benefit is a turnover effect.** Fewer trades → fewer
   fees → smaller losses. That is risk reduction, not alpha.

---

## 6. Limitations

- **Single instrument** (BTCUSDT). No cross-sectional validation.
- **Two calendar blocks** (2025, 2026 Jan–Aug) with different regime mixes;
  neither is a large sample at higher timeframes.
- **Multiple comparisons.** Many filters × timeframes × periods were tested;
  no formal correction applied. Any single small lift should be discounted.
- **Outcome definition is a proxy.** The `good` label is path-based, not
  PnL-based; it ignores the exit rule, funding and slippage on exits.
- **Backtest simplifications.** Bar-based fills, fixed per-side slippage,
  intrabar stop/target ambiguity handled by an explicit policy, no order-book
  or latency modelling, funding applied as a periodic drag.
- **Regime thresholds are hand-set** (24h trend z, ±1.0; vol vs trailing
  median) and were not themselves optimised or cross-validated.
- **Derivatives history is short** relative to bars; OI/funding filters were
  regime-dependent and sometimes empty (e.g. no non-positive funding in Aug
  2026; no positive basis in Jul 2026).

---

## 7. Conclusion

The hypothesis that a breakout trigger plus confluence or a regime gate yields a
durable BTCUSDT edge is **not supported** by 2025–2026 data.

- Timeframe is the only robust lever: it raises the "good" base rate from ~5%
  (1m) to ~18–20% (15m) to ~32–36% (1h), consistently.
- No confluence filter reached significance at any timeframe; the best
  candidate (`volume + EMA200` at 15m) was directionally consistent but weak
  (1.23× out-of-sample, z≈0.6).
- The regime hypothesis failed out-of-sample: `up_highvol` was special only in
  2026; in 2025 a down regime was best.
- Regime gating reduced losses mainly by cutting turnover; it did not make the
  strategy profitable.

**Recommendation.** Stop refining this trigger family. If research continues,
change the *source of edge* rather than the filters — e.g. funding/basis carry,
cross-sectional relative strength across a universe of perps, or a
mean-reversion trigger with a proper risk model — and design the study around a
PnL-based objective with enough sample to detect the effect size of interest.

---

## 8. Reproducibility

### Data

```bash
quantlab data build binance --market futures --symbol BTCUSDT --interval 1m \
    --start 2025-01-01 --end 2026-01-01
quantlab data build binance --market futures --symbol BTCUSDT --dataset funding \
    --start 2025-01-01 --end 2026-01-01
quantlab data build binance --market futures --symbol BTCUSDT --dataset open_interest \
    --start 2025-01-01 --end 2026-01-01
quantlab data build binance --market spot --symbol BTCUSDT --interval 1m \
    --start 2025-01-01 --end 2026-01-01
# (repeat for 2026-01-01 → 2026-09-01)
quantlab data validate --symbol BTCUSDT --dataset bars --interval 1m
```

### Reversion / excursion study

```bash
quantlab analyze excursions --strategy bollinger_breakout --symbol BTCUSDT \
    --timeframe 1m --start 2026-07-01 --end 2026-09-01 \
    --reversion-only --horizon-days 7 --reversion-basis close \
    --out experiments/reports/bb_reversion_only
```

### Confluence and regime evaluation

```bash
quantlab analyze confluence --strategy bollinger_breakout --symbol BTCUSDT \
    --timeframe 15m --source-timeframe 1m \
    --start 2025-01-01 --end 2026-01-01 \
    --validate-start 2026-01-01 --validate-end 2026-09-01 \
    --excursion-pct 0.005 --min-reversion-hours 1 --horizon-days 7 \
    --by-regime --regime-timeframe 1h \
    --out experiments/reports/regime_2025_2026
```

### Regime-gated backtest

```bash
quantlab analyze regime-backtest --strategy bollinger_breakout --symbol BTCUSDT \
    --timeframe 15m --start 2025-01-01 --end 2026-01-01 \
    --allow-regimes up_highvol,up_lowvol \
    --out experiments/reports/regime_bt_2025
```

### Key artifacts

| path | contents |
|---|---|
| `experiments/reports/bb_reversion_only/` | reversion CDF, excursion-by-bucket, per-signal events |
| `experiments/reports/confluence_bb/` | 1m confluence tables (design/validate) |
| `experiments/reports/cf2_*/` | full-2026 confluence by trigger/timeframe |
| `experiments/reports/regime_bb_15m/` | 2026 regime tables |
| `experiments/reports/regime_2025_2026/` | 2025 vs 2026 regime tables |
| `experiments/reports/regime_bt_2025/`, `regime_bt_2026/` | gated vs ungated metrics |

### Code map

| module | responsibility |
|---|---|
| `data/sources/binance.py`, `data/build.py` | download + normalize |
| `data/resample.py` | canonical OHLCV resampling |
| `data/validate.py` | data-quality reporting |
| `features/core.py`, `features/market.py` | indicators and flow features |
| `strategies/*` | breakout triggers |
| `backtest/engine.py`, `backtest/metrics.py` | VectorBT engine, metrics |
| `analytics/excursions.py` | reversion / excursion analysis |
| `analytics/confluence.py` | filter evaluation, design/validate |
| `analytics/regimes.py` | regime classifier + gate |
| `analytics/gating.py` | regime-gated backtest |

**Tests:** `uv run pytest` — 97 tests, including deterministic end-to-end,
lookahead/causality, resampling, regime and gating tests.
