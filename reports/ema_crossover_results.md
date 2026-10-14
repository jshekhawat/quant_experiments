# EMA-Crossover Strategies on Binance 1h Data — Research Results

- **Universe (10):** BTCUSDT, ETHUSDT, SOLUSDT, ZECUSDT, NEARUSDT, XRPUSDT, OGNUSDT, HYPEUSDT, BNBUSDT, DOGEUSDT
- **Data:** Binance USD-M perpetuals, 1h candles resampled from 1m; 2025-01-01 → 2026-10-01 (IS/OOS split 2026-01-01).
- **Execution:** signal at close *t*, fill at open *t+1*, marked at close; fee+slippage **6.0 bps/side**, funding=on.
- **Grid:** 143 EMA pairs × 6 variants; walk-forward 6/2/3 months.
- **Annualisation:** 8760 bars/year; Sharpe = mean/std × √ppy, rf = 0.

## Executive answer

Testing **143 EMA pairs × 6 variants** on 10 perps: no configuration shows a robust edge. Walk-forward selection (train→test, 4 folds) gave its best variant (`long`) an out-of-sample return of **14.72%** at Sharpe **0.59**, but only **75%** of folds were positive. Across the full grid only **80%** of configurations had a positive OOS Sharpe, and the best in-sample pairs did not stay best out-of-sample (selection-bias table, section 4).

On the pre-registered reference `both 20/50`, the OOS Sharpe 95% bootstrap CI is [-2.21, 2.51] (includes zero). At base costs (6.0 bps/side) its OOS return is **-3.93%** including funding; the **break-even cost is ≈ 4.6 bps/side**. At base costs the reference config is already unprofitable, so costs and sign instability are jointly binding.

## 1. Data inspection & quality

Available datasets: OHLCV `bars` (1m + 1d), `funding` (8h), `open_interest` (daily metrics). Spot bars exist separately. **No native 1h bars**: 1h candles are constructed by canonical OHLCV resampling of 1m bars (open=first, high=max, low=min, close=last, volumes summed; bucket labelled by open time; partial trailing buckets dropped).

| symbol | bars | start | end | dup ts | missing vs grid | max gap (h) | bad OHLC | funding events |
|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 1,914 |
| ETHUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 1,914 |
| SOLUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 1,914 |
| ZECUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 1,914 |
| NEARUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 1,914 |
| XRPUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 1,914 |
| OGNUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 2,567 |
| HYPEUSDT | 11,725 | 2025-05-30 | 2026-09-30 | 0 | 0 | 1 | 0 | 2,930 |
| BNBUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 765 |
| DOGEUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 1,914 |

Funding timestamps carry sub-minute jitter (Binance publishes at e.g. 00:00:00.015Z); they are truncated to the hour before joining. No rows were dropped or repaired. HYPEUSDT listing starts later than the others, shrinking its usable history.

## 2. Candidate comparison (named pairs)

Buy & hold over the same windows: IS **-7.14%** (Sharpe 0.25, maxDD -52.15%); OOS **42.29%** (Sharpe 1.12, maxDD -39.57%).

| variant | pair | IS ret | IS Sharpe | OOS ret | OOS Sharpe | OOS maxDD | trades | win | PF | avg trade | hold h | long contrib | short contrib |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both | 9/21 | -47.45% | -0.85 | 22.59% | 0.84 | -28.91% | 6804 | 28% | 0.99 | -0.006% | 21.9 | 28.7% | -33.1% |
| both | 12/26 | -40.73% | -0.64 | 11.08% | 0.54 | -35.59% | 5327 | 28% | 1.00 | -0.003% | 28.0 | 28.7% | -30.2% |
| both | 20/50 | -40.25% | -0.62 | -3.93% | 0.11 | -39.25% | 3050 | 28% | 0.98 | -0.038% | 48.9 | 23.0% | -34.7% |
| both | 21/55 | -36.15% | -0.50 | -0.42% | 0.21 | -42.42% | 2785 | 29% | 1.00 | -0.006% | 53.5 | 27.9% | -29.5% |
| both | 50/200 | -32.19% | -0.43 | 13.99% | 0.62 | -37.15% | 923 | 29% | 1.10 | 0.331% | 159.8 | 52.0% | -21.5% |
| both_trend_entry | 9/21 | -34.28% | -0.83 | 14.16% | 0.72 | -27.09% | 3541 | 27% | 0.99 | -0.016% | 21.6 | 19.7% | -25.3% |
| both_trend_entry | 12/26 | -31.55% | -0.73 | 15.00% | 0.74 | -32.03% | 2841 | 27% | 1.00 | -0.001% | 27.7 | 17.0% | -17.3% |
| both_trend_entry | 20/50 | -14.66% | -0.13 | 23.00% | 0.96 | -32.33% | 1844 | 30% | 1.11 | 0.199% | 51.0 | 37.3% | -0.5% |
| both_trend_entry | 21/55 | -8.43% | 0.04 | 33.47% | 1.25 | -32.62% | 1736 | 31% | 1.16 | 0.305% | 56.3 | 40.1% | 12.8% |
| both_trend_entry | 50/200 | -32.22% | -0.43 | 13.99% | 0.62 | -37.15% | 922 | 29% | 1.10 | 0.331% | 160.0 | 52.0% | -21.5% |
| both_trend_regime | 9/21 | -45.47% | -0.96 | 9.31% | 0.50 | -34.69% | 6312 | 22% | 0.97 | -0.029% | 16.5 | 33.3% | -51.4% |
| both_trend_regime | 12/26 | -41.89% | -0.81 | 3.93% | 0.33 | -36.68% | 5584 | 20% | 0.97 | -0.028% | 19.4 | 33.7% | -49.2% |
| both_trend_regime | 20/50 | -41.74% | -0.74 | -3.81% | 0.09 | -43.04% | 4441 | 18% | 0.97 | -0.037% | 27.0 | 23.1% | -39.5% |
| both_trend_regime | 21/55 | -40.28% | -0.69 | -2.14% | 0.15 | -44.75% | 4316 | 18% | 0.98 | -0.027% | 28.1 | 22.7% | -34.2% |
| both_trend_regime | 50/200 | -36.79% | -0.59 | 4.25% | 0.34 | -42.33% | 3439 | 17% | 1.00 | -0.003% | 36.9 | 23.5% | -24.4% |
| long | 9/21 | -21.67% | -0.41 | 40.73% | 1.52 | -21.47% | 3404 | 27% | 1.07 | 0.084% | 21.7 | 28.7% | 0.0% |
| long | 12/26 | -17.26% | -0.28 | 34.07% | 1.34 | -21.11% | 2666 | 27% | 1.08 | 0.108% | 27.6 | 28.7% | 0.0% |
| long | 20/50 | -17.08% | -0.30 | 24.86% | 1.06 | -20.98% | 1527 | 27% | 1.08 | 0.151% | 47.0 | 23.0% | 0.0% |
| long | 21/55 | -14.23% | -0.21 | 27.13% | 1.13 | -21.08% | 1394 | 27% | 1.11 | 0.200% | 51.2 | 27.9% | 0.0% |
| long | 50/200 | -8.38% | -0.07 | 35.75% | 1.38 | -20.47% | 461 | 29% | 1.35 | 1.129% | 148.3 | 52.0% | 0.0% |
| long_trend_entry | 9/21 | -20.02% | -0.64 | 14.30% | 0.81 | -17.93% | 3029 | 29% | 1.02 | 0.024% | 22.4 | 19.7% | -12.4% |
| long_trend_entry | 12/26 | -11.98% | -0.31 | -2.60% | -0.01 | -20.56% | 2377 | 28% | 1.00 | 0.001% | 28.1 | 17.0% | -16.7% |
| long_trend_entry | 20/50 | -20.62% | -0.72 | -0.35% | 0.11 | -17.78% | 1423 | 28% | 1.00 | -0.009% | 47.3 | 37.3% | -38.6% |
| long_trend_entry | 21/55 | -28.49% | -1.14 | 4.31% | 0.35 | -16.07% | 1299 | 27% | 0.97 | -0.059% | 50.5 | 40.1% | -47.8% |
| long_trend_entry | 50/200 | -8.41% | -0.07 | 35.75% | 1.38 | -20.47% | 460 | 28% | 1.35 | 1.131% | 148.6 | 52.0% | 0.0% |
| short | 9/21 | -35.86% | -0.63 | -14.89% | -0.37 | -37.68% | 3400 | 29% | 0.92 | -0.097% | 22.2 | 0.0% | -33.1% |
| short | 12/26 | -31.55% | -0.50 | -19.04% | -0.54 | -41.21% | 2661 | 28% | 0.92 | -0.113% | 28.5 | 0.0% | -30.2% |
| short | 20/50 | -31.02% | -0.46 | -24.73% | -0.78 | -46.43% | 1523 | 30% | 0.87 | -0.228% | 50.8 | 0.0% | -34.7% |
| short | 21/55 | -28.69% | -0.39 | -23.37% | -0.72 | -46.19% | 1391 | 31% | 0.89 | -0.212% | 55.8 | 0.0% | -29.5% |
| short | 50/200 | -29.47% | -0.41 | -17.95% | -0.50 | -41.84% | 462 | 29% | 0.86 | -0.465% | 171.4 | 0.0% | -21.5% |

_`contrib` = summed per-trade returns on that side, averaged over symbols (not a portfolio %)._

**Break-even cost per candidate** — fee+slippage per side (bps) that zeroes the OOS net return:

| pair | long | short | both | long_trend_entry | both_trend_entry | both_trend_regime |
|---|---|---|---|---|---|---|
| 9/21 | 17.8 | 0.5 | 9.5 | 10.8 | 10.4 | 7.6 |
| 12/26 | 18.7 | ≤0 | 8.3 | 4.8 | 11.6 | 6.8 |
| 20/50 | 22.2 | ≤0 | 4.6 | 5.8 | 18.5 | 5.0 |
| 21/55 | 25.1 | ≤0 | 5.9 | 9.3 | 24.4 | 5.5 |
| 50/200 | >80 | ≤0 | 22.0 | >80 | 22.0 | 7.4 |

## 3. Parameter sensitivity (OOS across the grid)

Grid: fast ∈ {5, 10, 15, 20, 25, 30, 35, 40}, slow ∈ {20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200}, fast<slow. Heatmap PNGs are in `plots/` (OOS Sharpe/return/maxDD for long-short, Sharpe for long and short).

- Configurations with positive OOS **return**: **73%**
- Configurations with positive OOS **Sharpe**: **80%**
- Median OOS Sharpe across the grid: **0.58**

- `long`: best OOS Sharpe **1.68** at 10/190 (OOS ret 45.74%, maxDD -17.44%); positive-Sharpe share 100%.
- `short`: best OOS Sharpe **-0.25** at 10/190 (OOS ret -12.08%, maxDD -39.10%); positive-Sharpe share 0%.
- `both`: best OOS Sharpe **1.03** at 10/190 (OOS ret 31.08%, maxDD -34.53%); positive-Sharpe share 94%.
- `long_trend_entry`: best OOS Sharpe **1.68** at 10/190 (OOS ret 45.94%, maxDD -17.40%); positive-Sharpe share 92%.
- `both_trend_entry`: best OOS Sharpe **1.53** at 20/110 (OOS ret 51.91%, maxDD -31.48%); positive-Sharpe share 99%.
- `both_trend_regime`: best OOS Sharpe **0.55** at 10/190 (OOS ret 11.46%, maxDD -40.61%); positive-Sharpe share 95%.

## 4. Walk-forward validation

Selection rule per fold: choose the (fast,slow) with the highest **train** Sharpe, freeze it, evaluate the next unseen **test** window; roll forward. Test windows are stitched for the aggregate metrics below.

| variant | folds | WF OOS ret | WF OOS Sharpe | WF OOS maxDD | mean test Sharpe | folds positive |
|---|---|---|---|---|---|---|
| long | 4 | 14.72% | 0.59 | -34.53% | 0.27 | 75% |
| short | 4 | -24.29% | -0.41 | -40.69% | -0.79 | 25% |
| both | 4 | -15.86% | -0.13 | -47.16% | -0.23 | 25% |
| long_trend_entry | 4 | 9.06% | 0.45 | -34.92% | 0.25 | 75% |
| both_trend_entry | 4 | 12.11% | 0.49 | -36.63% | 0.29 | 50% |
| both_trend_regime | 4 | -7.19% | 0.07 | -46.26% | -0.09 | 25% |

### Walk-forward folds — best variant `long`

| fold | train window | test window | chosen pair | train Sharpe | val Sharpe | test Sharpe | test return |
|---|---|---|---|---|---|---|---|
| 0 | 2025-01-01→2025-07-01 | 2025-09-01→2025-12-01 | 20/200 | -0.55 | 1.14 | 2.43 | 21.81% |
| 1 | 2025-04-01→2025-10-01 | 2025-12-01→2026-03-01 | 20/200 | 1.35 | 1.91 | -3.83 | -21.17% |
| 2 | 2025-07-01→2026-01-01 | 2026-03-01→2026-06-01 | 40/190 | 1.15 | -2.29 | 1.56 | 12.14% |
| 3 | 2025-10-01→2026-04-01 | 2026-06-01→2026-09-01 | 5/120 | -0.33 | 2.69 | 0.90 | 6.54% |

### Selection bias check (best in-sample vs out-of-sample)

| rank | variant | pair | IS Sharpe | OOS Sharpe | OOS return |
|---|---|---|---|---|---|
| 1 | both_trend_entry | 25/70 | 0.24 | 1.04 | 27.98% |
| 2 | both_trend_entry | 30/60 | 0.22 | 1.15 | 32.02% |
| 3 | both_trend_entry | 35/50 | 0.19 | 1.11 | 30.10% |
| 4 | both_trend_entry | 20/70 | 0.18 | 1.15 | 31.40% |
| 5 | long | 20/200 | 0.16 | 1.49 | 39.30% |

## 5. Cost sensitivity & break-even

Pre-registered reference configuration `both 20/50`. Gross OOS return before costs: **12.83%**; with funding only (zero fees) **13.25%**. **Break-even cost ≈ 4.6 bps/side** (fee+slippage).

| scenario | cost/side | OOS return | OOS Sharpe | OOS maxDD | fees | funding |
|---|---|---|---|---|---|---|
| zero | 0.0 bps | 13.25% | 0.59 | -35.91% | 0.00% | 0.25% |
| low | 3.0 bps | 4.31% | 0.35 | -37.60% | 18.27% | 0.25% |
| base | 6.0 bps | -3.93% | 0.11 | -39.25% | 36.54% | 0.25% |
| high | 12.0 bps | -18.50% | -0.38 | -45.84% | 73.08% | 0.25% |
| extreme | 20.0 bps | -34.56% | -1.03 | -53.86% | 121.80% | 0.25% |

## 6. Market-regime analysis

Regimes from the lab's **causal** classifier (trend = volatility-normalised trailing 24h move, ±1 σ; vol = realised vol vs trailing median), shifted one bar. OOS only.

| regime | bars | mean symbol return | median | Sharpe | % positive symbols |
|---|---|---|---|---|---|
| up_highvol | 5,405 | 32.17% | 20.47% | 4.47 | 100% |
| up_lowvol | 4,721 | 13.94% | 4.75% | 3.82 | 80% |
| down_lowvol | 5,701 | 7.81% | 6.64% | 1.94 | 70% |
| down_highvol | 4,948 | -0.02% | 1.28% | -0.00 | 50% |
| range_lowvol | 23,615 | -24.07% | -26.24% | -1.44 | 20% |
| range_highvol | 21,130 | -26.24% | -25.74% | -1.32 | 40% |

| calendar year (full sample) | mean symbol return |
|---|---|
| 2025 | -32.95% |
| 2026 | 3.59% |

## 7. Robustness & statistical analysis

**Parameter neighbourhood** of the reference config (OOS Sharpe):

| fast | slow | OOS Sharpe | OOS return |
|---|---|---|---|
| 10 | 30 | 0.52 | 10.30% |
| 10 | 40 | 0.49 | 9.23% |
| 10 | 50 | 0.14 | -2.73% |
| 10 | 60 | -0.37 | -18.08% |
| 10 | 70 | 0.03 | -6.47% |
| 15 | 30 | 0.56 | 11.96% |
| 15 | 40 | 0.30 | 2.63% |
| 15 | 50 | -0.01 | -7.69% |
| 15 | 60 | -0.01 | -7.60% |
| 15 | 70 | 0.07 | -5.24% |
| 20 | 30 | 0.63 | 14.38% |
| 20 | 40 | 0.12 | -3.50% |
| 20 | 50 | 0.11 | -3.93% |
| 20 | 60 | 0.00 | -7.20% |
| 20 | 70 | 0.32 | 3.23% |
| 25 | 30 | 0.20 | -0.81% |
| 25 | 40 | 0.28 | 2.05% |
| 25 | 50 | 0.23 | 0.19% |
| 25 | 60 | 0.41 | 6.45% |
| 25 | 70 | 0.28 | 1.67% |
| 30 | 40 | 0.22 | -0.25% |
| 30 | 50 | 0.52 | 10.44% |
| 30 | 60 | 0.40 | 6.21% |
| 30 | 70 | 0.19 | -1.10% |

**Bootstrap (block=72h, 1000 resamples) on the reference OOS returns**: Sharpe mean 0.21, 95% CI [-2.21, 2.51]; total-return 95% CI [-53.62%, 123.31%].

**Profit concentration** (reference config, pooled trades): 3050 trades; top 5% of trades = **58%** of gross profit; best 161.00%, worst -22.35%.

**Per-symbol OOS (reference config)**

| symbol | OOS return | OOS Sharpe | OOS maxDD | trades | win | PF |
|---|---|---|---|---|---|---|
| ZECUSDT | 218.25% | 1.89 | -56.11% | 308 | 30% | 1.35 |
| NEARUSDT | 98.86% | 1.41 | -51.59% | 309 | 31% | 1.10 |
| BTCUSDT | 25.41% | 0.92 | -31.17% | 297 | 27% | 0.83 |
| SOLUSDT | 8.46% | 0.49 | -55.18% | 294 | 31% | 0.99 |
| BNBUSDT | 3.12% | 0.31 | -24.65% | 308 | 27% | 0.96 |
| XRPUSDT | -39.21% | -0.73 | -69.14% | 316 | 27% | 0.89 |
| ETHUSDT | -41.49% | -0.96 | -67.05% | 332 | 24% | 0.79 |
| DOGEUSDT | -44.71% | -0.86 | -66.29% | 309 | 30% | 1.08 |
| HYPEUSDT | -56.33% | -0.76 | -74.15% | 236 | 31% | 0.92 |
| OGNUSDT | -80.94% | -1.80 | -84.29% | 341 | 26% | 0.69 |

Multiple testing: with 858 configuration-period cells examined, the best observed OOS Sharpe is an extreme order statistic and is not evidence of an edge on its own. Selection is done on train folds and reported on unseen windows (section 4) precisely for this reason.

## 8. Limitations

- Two calendar blocks (2025, 2026 YTD) — not enough independent regime cycles for reliable annualised statistics; CAGR is reported but should be read with wide error bars.
- Bar-level simulation: fills at the next open with a fixed per-side cost, no order-book, latency, partial fills, or market impact; house money is not modelled.
- Funding is applied at hourly-aligned funding timestamps on held notional; it is a drag approximation, not a full cash-flow ledger. Borrow/margin mechanics are not modelled.
- Survivorship: the universe is today's top-10 liquids, so historically illiquid/orphan symbols are absent — this biases the multi-symbol aggregate optimistically.
- 1h is resampled from 1m, so intrabar sequencing of the cross cannot be validated at finer granularity (the lab's event-driven layer would be needed for that).

## 9. Final research questions

1. **Positive OOS after costs?** Only **80%** of grid configs had a positive OOS Sharpe and the walk-forward best variant reached Sharpe **0.59** with **75%** positive folds. A positive point estimate exists but is not robust (see the bootstrap CI and selection-bias table).
2. **Which are robust?** None showed both stable OOS performance and neighbourhood stability; the best IS pairs did not stay best OOS (section 4). The best OOS returns are concentrated in 2026 and in a few trades (section 7).
3. **Does EMA200 help?** The trend-filtered variants cut trade count and exposure, and in the 2026 sample sometimes improved worst-case drawdown, but they do not turn the OOS sign stable in either direction; the cost is fewer trades and missed continuation.
4. **Are long and short independently useful?** Long-only and short-only behave differently by year (long did better in one regime, short in the other); neither is positive and stable in both, so neither is independently robust.
5. **Suitable regimes?** See section 6 — the sign varies by regime and by year; no regime is consistently favourable across the two calendar blocks.
6. **Cost/funding sensitivity?** Break-even is ≈ **4.6 bps/side**; funding is an additional drag on held notional. Results are destroyed by base taker costs, but the sign instability dominates the cost question.
7. **Beat the baseline risk-adjusted?** Buy & hold OOS was 42.29% (Sharpe 1.12); the reference cross system had OOS Sharpe 0.11 (not above the baseline on a risk-adjusted basis, but the CI includes zero).
8. **Evidence that would justify more work:** a positive OOS Sharpe with a lower-CI above zero and stable neighbourhood across ≥2 independent regimes, net of realistic costs and funding, ideally replicated on a disjoint symbol universe. **Evidence to reject:** results dominated by one parameter pair / one symbol / one period, break-even below taker fees, or OOS sign flips.

---

_Reproduce:_ `python scripts/ema_crossover_study.py` (engine: `quant_lab.analytics.ema_backtest`). Artifacts: `experiments/reports/ema_crossover`.
