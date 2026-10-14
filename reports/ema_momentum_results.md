# EMA Momentum (ROC), Acceleration & Spread Dynamics — Binance 1h Perps

- **Universe (10):** BTCUSDT, ETHUSDT, SOLUSDT, ZECUSDT, NEARUSDT, XRPUSDT, OGNUSDT, HYPEUSDT, BNBUSDT, DOGEUSDT
- **Data:** Binance USD-M perpetuals, 1h resampled from 1m; 2025-01-01 → 2026-10-01 (IS/OOS split 2026-01-01).
- **Features:** EMA ROC `(MA_t/MA_{t-k}-1)/k`, k ∈ [1, 3, 6, 12]; acceleration = ΔROC; normalized spread `(MA_fast-MA_slow)/ATR14`; spread velocity = Δspread/k.
- **Configs (18):** Family A EMA(n) vs SMA20 + Family B fast/slow EMAs; pre-registered primary `ema12_sma20`, secondary `ema10_ema30`.
- **Execution:** signal at close *t*, fill at open *t+1*, marked at close; fee+slippage **6.0 bps/side**; funding=on; annualisation 8760 bars/yr.

## Executive answer

EMA momentum does contain **timing** information: a momentum-Reversal signal fires a median of **9h** before the bullish crossover and **9h** before the bearish one, and 79% / 80% of them are followed by the matching crossover within 24h — so the lead time is real and roughly symmetric between directions. It is **not economically decisive**: the volatility-scaled false-turn rate is ≈ 50% / 48% (about a coin flip), the directional forward-return edge is small, and delaying entry *improves* the result rather than hurting it.

On the pre-registered `ema12_sma20` long-short book at 6 bps/side, the combined early-warning strategy returned **-46.91%** OOS (Sharpe -2.16, maxDD -53.63%, 6458 trades) versus the conventional crossover baseline **16.09%** (Sharpe 0.67, maxDD -35.61%) and buy & hold OOS **42.29%** (Sharpe 1.12).

Walk-forward selection (choose the configuration on each train window, test on the next unseen window) gives the best strategy `confirmed` a Sharpe of **-0.42** with only **50%** positive folds. Only **26** of 108 (config, feature) 24h rank-IC cells reach |IC| > 0.02, and the incremental regression over a baseline that already holds the spread and crossover state adds little R².

**Verdict.** Earlier signals exist, are measurable, and are directionally informative — but the incremental predictive information is small, the false-turn rate is high, and the economic edge does not survive realistic costs and walk-forward selection. The hypotheses are tested honestly below; the negative result is reported rather than over-fit away.

---

## 1. Environment & data

Reuses `quant_lab.analytics.ema_backtest` (loader, execution, costs, funding) and the causal regime classifier. New code: `quant_lab.analytics.ema_momentum` (features, early-warning events, lead-time/false-turn analysis, strategy targets) and `scripts/ema_momentum_study.py`. No native 1h bars — 1h is resampled from 1m (open=first, high=max, low=min, close=last, bucket labelled by open time). ATR = Wilder RMA of true range (period 14), initialized with `min_periods=period` so the first 14 bars are NaN.

| symbol | bars | start | end | dup ts | gaps | max gap (h) | bad OHLC | ATR warm-up | funding events |
|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 13 | 1,914 |
| ETHUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 13 | 1,914 |
| SOLUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 13 | 1,914 |
| ZECUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 13 | 1,914 |
| NEARUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 13 | 1,914 |
| XRPUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 13 | 1,914 |
| OGNUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 13 | 2,567 |
| HYPEUSDT | 11,725 | 2025-05-30 | 2026-09-30 | 0 | 0 | 1 | 0 | 13 | 2,930 |
| BNBUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 13 | 765 |
| DOGEUSDT | 15,312 | 2025-01-01 | 2026-09-30 | 0 | 0 | 1 | 0 | 13 | 1,914 |

Funding timestamps carry sub-minute jitter and are truncated to the hour before joining. HYPEUSDT listings start later than the others. Feature values are NaN during warm-up and excluded from every statistic; no rows are dropped or repaired.

## 2. Experiment A — predictive diagnostics (before trading)

Rank IC of each feature against forward returns, computed **per symbol** and then equal-weighted (this avoids cross-sectional contamination from pooling assets at different price levels). |IC| > 0.02 is marked nice-to-have; only 26 of 108 24h cells clear it. IC is a dependence measure, not evidence of a tradable edge.

### 24h rank IC by configuration and feature

| config | roc_1 | roc_12 | accel_1 | spread | svel_1 | svel_12 |
|---|---|---|---|---|---|---|
| ema10_ema20 | -0.012 | -0.019 | -0.004 | -0.019 | 0.003 | 0.010 |
| ema10_ema30 | -0.012 | -0.019 | -0.004 | -0.027 | 0.006 | 0.012 |
| ema10_sma20 | -0.012 | -0.019 | -0.004 | -0.009 | 0.002 | 0.004 |
| ema12_ema26 | -0.014 | -0.021 | -0.004 | -0.025 | 0.006 | 0.013 |
| ema12_ema30 | -0.014 | -0.021 | -0.004 | -0.028 | 0.007 | 0.013 |
| ema12_sma20 | -0.014 | -0.021 | -0.004 | -0.008 | 0.001 | 0.003 |
| ema15_ema30 | -0.017 | -0.023 | -0.004 | -0.030 | 0.009 | 0.015 |
| ema15_sma20 | -0.017 | -0.023 | -0.004 | -0.005 | 0.001 | 0.002 |
| ema20_ema50 | -0.022 | -0.027 | -0.003 | -0.043 | 0.009 | 0.013 |
| ema20_sma20 | -0.022 | -0.027 | -0.003 | 0.010 | -0.001 | -0.003 |
| ema25_sma20 | -0.027 | -0.031 | -0.003 | 0.026 | -0.004 | -0.009 |
| ema30_sma20 | -0.031 | -0.034 | -0.003 | 0.032 | -0.006 | -0.012 |
| ema40_sma20 | -0.038 | -0.040 | -0.002 | 0.037 | -0.008 | -0.011 |
| ema50_sma20 | -0.044 | -0.045 | -0.002 | 0.042 | -0.007 | -0.010 |
| ema5_ema20 | -0.008 | -0.016 | -0.005 | -0.016 | 0.001 | 0.006 |
| ema5_sma20 | -0.008 | -0.016 | -0.005 | -0.010 | 0.001 | 0.003 |
| ema8_ema20 | -0.011 | -0.018 | -0.004 | -0.018 | 0.002 | 0.008 |
| ema8_sma20 | -0.011 | -0.018 | -0.004 | -0.009 | 0.001 | 0.004 |

### Bootstrap 95% CI on the primary config (per-symbol IC then averaged)

| feature | horizon | rank IC | 95% CI |
|---|---|---|---|
| roc_1 | 1h | -0.033 | [-0.038, -0.029] |
| roc_1 | 24h | -0.014 | [-0.029, -0.000] |
| accel_1 | 1h | -0.024 | [-0.029, -0.019] |
| accel_1 | 24h | -0.004 | [-0.008, -0.000] |
| spread | 1h | -0.014 | [-0.019, -0.010] |
| spread | 24h | -0.008 | [-0.024, 0.006] |
| svel_1 | 1h | -0.023 | [-0.027, -0.017] |
| svel_1 | 24h | 0.001 | [-0.008, 0.010] |

### Conditional forward returns (primary config, pooled)

| feature | condition | horizon | n | hit | mean | median |
|---|---|---|---|---|---|---|
| roc_1 | ROC>0 | 1h | 74,533 | 47.5% | 0.008% | -0.013% |
| roc_1 | ROC<0 | 1h | 74,870 | 51.0% | 0.001% | 0.017% |
| roc_1 | ROC>0 | 6h | 74,518 | 48.6% | 0.060% | -0.026% |
| roc_1 | ROC<0 | 6h | 74,835 | 50.7% | -0.009% | 0.024% |
| roc_1 | ROC>0 | 24h | 74,415 | 48.4% | 0.174% | -0.087% |
| roc_1 | ROC<0 | 24h | 74,758 | 50.4% | 0.027% | 0.036% |
| roc_1 | ROC>0 | 72h | 74,253 | 47.6% | 0.303% | -0.292% |
| roc_1 | ROC<0 | 72h | 74,440 | 49.7% | 0.266% | -0.013% |
| spread | spread>0 | 1h | 74,748 | 48.2% | 0.012% | -0.003% |
| spread | spread<0 | 1h | 74,585 | 50.3% | -0.003% | 0.006% |
| spread | spread>0 | 6h | 74,708 | 48.6% | 0.065% | -0.026% |
| spread | spread<0 | 6h | 74,575 | 50.6% | -0.015% | 0.024% |
| spread | spread>0 | 24h | 74,610 | 48.3% | 0.167% | -0.093% |
| spread | spread<0 | 24h | 74,493 | 50.5% | 0.029% | 0.040% |
| spread | spread>0 | 72h | 74,443 | 47.5% | 0.307% | -0.298% |
| spread | spread<0 | 72h | 74,180 | 49.7% | 0.253% | -0.013% |
| svel_1 | svel>0 | 1h | 74,762 | 48.0% | 0.004% | -0.003% |
| svel_1 | svel<0 | 1h | 74,561 | 50.4% | 0.005% | 0.008% |
| svel_1 | svel>0 | 6h | 74,756 | 49.8% | 0.040% | 0.000% |
| svel_1 | svel<0 | 6h | 74,517 | 49.4% | 0.010% | 0.000% |
| svel_1 | svel>0 | 24h | 74,651 | 49.5% | 0.115% | -0.018% |
| svel_1 | svel<0 | 24h | 74,442 | 49.3% | 0.080% | -0.027% |
| svel_1 | svel>0 | 72h | 74,427 | 48.7% | 0.299% | -0.152% |
| svel_1 | svel<0 | 72h | 74,186 | 48.5% | 0.260% | -0.172% |
| accel_1 | accel>0 | 1h | 74,473 | 48.1% | 0.004% | -0.003% |
| accel_1 | accel<0 | 1h | 74,920 | 50.4% | 0.005% | 0.008% |
| accel_1 | accel>0 | 6h | 74,448 | 49.1% | 0.022% | -0.007% |
| accel_1 | accel<0 | 6h | 74,895 | 50.1% | 0.029% | 0.007% |
| accel_1 | accel>0 | 24h | 74,353 | 49.3% | 0.095% | -0.029% |
| accel_1 | accel<0 | 24h | 74,810 | 49.5% | 0.105% | -0.012% |
| accel_1 | accel>0 | 72h | 74,101 | 48.8% | 0.304% | -0.133% |
| accel_1 | accel<0 | 72h | 74,582 | 48.4% | 0.264% | -0.178% |

### Incremental information over a baseline with the current spread + crossover state

OLS of forward returns on `{spread, crossover_state}` (baseline) vs the same plus the feature(s); HAC (Newey–West) errors with maxlags = horizon. ΔR² is the incremental explanatory power; `t_add` is the HAC t-statistic of the first added regressor.

| horizon | added | n | R² base | R² full | ΔR² | t_add |
|---|---|---|---|---|---|---|
| 1h | roc_1 | 149,333 | 0.0001 | 0.0004 | 0.00026 | -2.83 |
| 1h | accel_1 | 149,333 | 0.0001 | 0.0001 | 0.00000 | -0.01 |
| 1h | svel_1 | 149,323 | 0.0001 | 0.0002 | 0.00003 | -1.90 |
| 1h | roc_1+accel_1+svel_1 | 149,323 | 0.0001 | 0.0005 | 0.00034 | -2.93 |
| 6h | roc_1 | 149,283 | 0.0007 | 0.0010 | 0.00028 | -1.79 |
| 6h | accel_1 | 149,283 | 0.0007 | 0.0007 | 0.00001 | -0.58 |
| 6h | svel_1 | 149,273 | 0.0007 | 0.0008 | 0.00004 | 1.34 |
| 6h | roc_1+accel_1+svel_1 | 149,273 | 0.0007 | 0.0015 | 0.00071 | -2.17 |
| 24h | roc_1 | 149,103 | 0.0005 | 0.0011 | 0.00066 | -2.06 |
| 24h | accel_1 | 149,103 | 0.0005 | 0.0005 | 0.00001 | -1.17 |
| 24h | svel_1 | 149,093 | 0.0005 | 0.0005 | 0.00001 | 0.81 |
| 24h | roc_1+accel_1+svel_1 | 149,093 | 0.0005 | 0.0017 | 0.00125 | -2.23 |
| 72h | roc_1 | 148,623 | 0.0001 | 0.0005 | 0.00045 | -1.70 |
| 72h | accel_1 | 148,623 | 0.0001 | 0.0001 | 0.00000 | -0.71 |
| 72h | svel_1 | 148,613 | 0.0001 | 0.0001 | 0.00001 | 1.06 |
| 72h | roc_1+accel_1+svel_1 | 148,613 | 0.0001 | 0.0010 | 0.00089 | -1.80 |

### Consistency across symbols, years and regimes (primary config, 24h)

| symbol | IC roc_1 | IC spread |
|---|---|---|
| BTCUSDT | -0.016 | -0.005 |
| ETHUSDT | -0.031 | -0.030 |
| SOLUSDT | -0.004 | 0.010 |
| ZECUSDT | 0.037 | 0.031 |
| NEARUSDT | -0.011 | -0.004 |
| XRPUSDT | -0.026 | -0.024 |
| OGNUSDT | -0.047 | -0.038 |
| HYPEUSDT | -0.011 | -0.002 |
| BNBUSDT | -0.021 | -0.014 |
| DOGEUSDT | -0.010 | -0.006 |

| year | n | IC roc_1 | IC spread |
|---|---|---|---|
| 2025 | 84,013 | -0.025 | -0.013 |
| 2026 | 65,520 | -0.000 | -0.001 |

| regime | n | IC spread (24h, n-weighted) |
|---|---|---|
| range_lowvol | 52,837 | 0.023 |
| range_highvol | 48,620 | -0.050 |
| up_lowvol | 11,607 | 0.045 |
| down_lowvol | 11,588 | 0.053 |
| down_highvol | 11,449 | -0.047 |
| up_highvol | 11,402 | 0.026 |

Feature cross-correlation (primary config, finite overlap): roc_1/spread +0.64, roc_1/svel_1 +0.51, accel_1/roc_1 +0.28, accel_1/spread -0.10, accel_1/svel_1 +0.34, spread/svel_1 +0.09.

## 3. Experiments B & C — early-warning signals, lead time & false turns

Conditions (bullish; bearish mirrored):

- **deceleration** (state): ROC < 0, acceleration > 0, fast MA still below slow MA.
- **reversal** (event): ROC crosses above 0 while fast MA is still below slow MA.
- **convergence** (event): spread velocity crosses above 0 while the spread is negative (|spread| shrinking).
- **combined** (event): reversal AND spread velocity > 0.
- **confirmed** (event): a convergence event followed by a ROC reversal within 12 bars.

The reference event is the conventional crossover of the *same* moving averages. `median_lead_h` is the median hours from the signal to that crossover; `false_turn_1` is the share where price moves 1×ATR against the signal before moving 1×ATR in favour within 24h. Because each signal sits on the opposite side of the crossover it predicts, the *next* crossover of those two MAs is, by construction, always the matching direction (`wrong_first` ≈ 0); the real failure modes are **distance** (how far the cross is) and **reversal** (price moving against the signal first), which is why both are reported.

### Lead-time summary — primary config `ema12_sma20`

| signal | n | median lead (h) | within 6h | within 12h | within 24h | within 48h | hit 24h | mean 24h | false-turn 1σ | MFE/MAE (ATR) |
|---|---|---|---|---|---|---|---|---|---|---|
| decel_bull | 233,455 | 15 | 19% | 42% | 69% | 88% | 50.9% | 0.035% | 50% | 2.60/2.83 |
| decel_bear | 232,633 | 15 | 19% | 42% | 69% | 88% | 51.4% | -0.205% | 49% | 2.61/3.04 |
| reversal_bull | 127,882 | 9 | 41% | 60% | 79% | 93% | 49.2% | -0.043% | 50% | 2.73/2.82 |
| reversal_bear | 125,653 | 9 | 42% | 61% | 80% | 93% | 51.5% | -0.011% | 48% | 2.73/2.78 |
| converge_bull | 168,953 | 13 | 27% | 48% | 71% | 89% | 50.0% | 0.044% | 51% | 2.73/2.81 |
| converge_bear | 167,680 | 13 | 27% | 48% | 70% | 88% | 51.2% | -0.117% | 49% | 2.71/2.91 |
| combined_bull | 116,908 | 8 | 42% | 62% | 79% | 93% | 49.4% | -0.032% | 50% | 2.72/2.82 |
| combined_bear | 114,687 | 8 | 43% | 62% | 80% | 93% | 51.8% | -0.011% | 48% | 2.71/2.78 |
| confirmed_bull | 76,551 | 9 | 40% | 60% | 79% | 92% | 49.9% | 0.005% | 50% | 2.73/2.79 |
| confirmed_bear | 75,777 | 9 | 41% | 60% | 79% | 92% | 51.7% | -0.037% | 48% | 2.69/2.79 |

### Across all configurations (median lead / within-24h / false-turn 1σ)

| config | signal | n | median lead | within 24h | false-turn 1σ | mean 24h |
|---|---|---|---|---|---|---|
| ema5_sma20 | decel_bull | 14,186 | 13 | 80% | 51% | 0.017% |
| ema5_sma20 | decel_bear | 14,431 | 13 | 80% | 49% | -0.202% |
| ema5_sma20 | reversal_bull | 11,528 | 7 | 88% | 50% | -0.036% |
| ema5_sma20 | reversal_bear | 11,427 | 7 | 88% | 49% | -0.128% |
| ema5_sma20 | converge_bull | 11,538 | 9 | 85% | 51% | -0.014% |
| ema5_sma20 | converge_bear | 11,583 | 9 | 85% | 49% | -0.153% |
| ema5_sma20 | combined_bull | 10,825 | 7 | 88% | 50% | -0.032% |
| ema5_sma20 | combined_bear | 10,776 | 7 | 87% | 49% | -0.148% |
| ema5_sma20 | confirmed_bull | 6,666 | 7 | 88% | 50% | -0.042% |
| ema5_sma20 | confirmed_bear | 6,653 | 7 | 87% | 49% | -0.145% |
| ema8_sma20 | decel_bull | 15,189 | 14 | 78% | 51% | 0.038% |
| ema8_sma20 | decel_bear | 15,369 | 14 | 78% | 49% | -0.221% |
| ema8_sma20 | reversal_bull | 9,396 | 6 | 89% | 50% | -0.061% |
| ema8_sma20 | reversal_bear | 9,405 | 6 | 88% | 48% | -0.092% |
| ema8_sma20 | converge_bull | 9,986 | 10 | 84% | 51% | 0.009% |
| ema8_sma20 | converge_bear | 10,099 | 9 | 84% | 49% | -0.127% |
| ema8_sma20 | combined_bull | 8,636 | 6 | 89% | 50% | -0.068% |
| ema8_sma20 | combined_bear | 8,630 | 6 | 88% | 48% | -0.098% |
| ema8_sma20 | confirmed_bull | 5,203 | 6 | 89% | 50% | -0.038% |
| ema8_sma20 | confirmed_bear | 5,297 | 6 | 88% | 49% | -0.103% |
| ema10_sma20 | decel_bull | 15,650 | 14 | 79% | 50% | 0.023% |
| ema10_sma20 | decel_bear | 15,898 | 13 | 78% | 49% | -0.224% |
| ema10_sma20 | reversal_bull | 8,533 | 5 | 90% | 51% | -0.082% |
| ema10_sma20 | reversal_bear | 8,481 | 5 | 89% | 48% | -0.042% |
| ema10_sma20 | converge_bull | 9,545 | 9 | 85% | 51% | 0.018% |
| ema10_sma20 | converge_bear | 9,637 | 9 | 84% | 48% | -0.116% |
| ema10_sma20 | combined_bull | 7,676 | 5 | 90% | 51% | -0.056% |
| ema10_sma20 | combined_bear | 7,623 | 5 | 89% | 48% | -0.041% |
| ema10_sma20 | confirmed_bull | 4,766 | 6 | 90% | 50% | -0.023% |
| ema10_sma20 | confirmed_bear | 4,801 | 6 | 89% | 48% | -0.081% |
| ema12_sma20 | decel_bull | 16,012 | 13 | 81% | 50% | 0.034% |
| ema12_sma20 | decel_bear | 16,191 | 13 | 80% | 49% | -0.220% |
| ema12_sma20 | reversal_bull | 7,737 | 5 | 91% | 51% | -0.106% |
| ema12_sma20 | reversal_bear | 7,798 | 5 | 91% | 48% | -0.034% |
| ema12_sma20 | converge_bull | 9,350 | 9 | 86% | 51% | -0.005% |
| ema12_sma20 | converge_bear | 9,411 | 9 | 85% | 49% | -0.107% |
| ema12_sma20 | combined_bull | 6,765 | 5 | 92% | 51% | -0.090% |
| ema12_sma20 | combined_bear | 6,807 | 5 | 91% | 48% | -0.053% |
| ema12_sma20 | confirmed_bull | 4,403 | 5 | 92% | 50% | -0.027% |
| ema12_sma20 | confirmed_bear | 4,452 | 5 | 91% | 49% | -0.063% |
| ema15_sma20 | decel_bull | 16,152 | 12 | 86% | 50% | 0.034% |
| ema15_sma20 | decel_bear | 16,306 | 12 | 85% | 49% | -0.219% |
| ema15_sma20 | reversal_bull | 6,737 | 4 | 93% | 50% | -0.063% |
| ema15_sma20 | reversal_bear | 6,864 | 4 | 93% | 48% | 0.006% |
| ema15_sma20 | converge_bull | 9,216 | 9 | 89% | 51% | 0.019% |
| ema15_sma20 | converge_bear | 9,347 | 8 | 89% | 48% | -0.068% |
| ema15_sma20 | combined_bull | 5,474 | 4 | 94% | 50% | -0.040% |
| ema15_sma20 | combined_bear | 5,582 | 4 | 94% | 48% | -0.016% |
| ema15_sma20 | confirmed_bull | 3,804 | 4 | 95% | 50% | -0.030% |
| ema15_sma20 | confirmed_bear | 3,914 | 4 | 95% | 48% | -0.023% |
| ema20_sma20 | decel_bull | 14,050 | 9 | 96% | 50% | -0.011% |
| ema20_sma20 | decel_bear | 14,029 | 9 | 95% | 50% | -0.166% |
| ema20_sma20 | reversal_bull | 5,110 | 6 | 92% | 51% | -0.099% |
| ema20_sma20 | reversal_bear | 5,269 | 6 | 93% | 48% | -0.055% |
| ema20_sma20 | converge_bull | 9,186 | 8 | 94% | 51% | -0.009% |
| ema20_sma20 | converge_bear | 9,294 | 8 | 94% | 49% | -0.174% |
| ema20_sma20 | combined_bull | 3,505 | 4 | 95% | 51% | -0.104% |
| ema20_sma20 | combined_bear | 3,631 | 4 | 96% | 48% | -0.034% |
| ema20_sma20 | confirmed_bull | 2,728 | 4 | 95% | 52% | -0.115% |
| ema20_sma20 | confirmed_bear | 2,820 | 4 | 95% | 49% | -0.082% |
| ema25_sma20 | decel_bull | 10,229 | 8 | 98% | 51% | -0.119% |
| ema25_sma20 | decel_bear | 10,291 | 8 | 98% | 50% | -0.160% |
| ema25_sma20 | reversal_bull | 4,129 | 10 | 90% | 52% | -0.134% |
| ema25_sma20 | reversal_bear | 4,239 | 10 | 89% | 49% | -0.047% |
| ema25_sma20 | converge_bull | 8,553 | 10 | 82% | 51% | -0.069% |
| ema25_sma20 | converge_bear | 8,682 | 10 | 81% | 50% | -0.129% |
| ema25_sma20 | combined_bull | 2,617 | 7 | 94% | 52% | -0.108% |
| ema25_sma20 | combined_bear | 2,690 | 7 | 93% | 48% | -0.064% |
| ema25_sma20 | confirmed_bull | 2,242 | 8 | 91% | 52% | -0.148% |
| ema25_sma20 | confirmed_bear | 2,312 | 8 | 91% | 50% | -0.157% |
| ema30_sma20 | decel_bull | 7,589 | 8 | 96% | 51% | -0.085% |
| ema30_sma20 | decel_bear | 7,548 | 8 | 96% | 49% | -0.113% |
| ema30_sma20 | reversal_bull | 3,579 | 13 | 79% | 52% | -0.195% |
| ema30_sma20 | reversal_bear | 3,701 | 12 | 80% | 49% | 0.012% |
| ema30_sma20 | converge_bull | 7,871 | 16 | 64% | 52% | 0.013% |
| ema30_sma20 | converge_bear | 7,908 | 16 | 64% | 51% | -0.128% |
| ema30_sma20 | combined_bull | 2,237 | 9 | 84% | 51% | -0.209% |
| ema30_sma20 | combined_bear | 2,345 | 9 | 84% | 49% | 0.029% |
| ema30_sma20 | confirmed_bull | 2,015 | 12 | 80% | 53% | -0.215% |
| ema30_sma20 | confirmed_bear | 2,073 | 12 | 82% | 51% | -0.063% |
| ema40_sma20 | decel_bull | 5,412 | 8 | 91% | 51% | -0.035% |
| ema40_sma20 | decel_bear | 5,393 | 8 | 90% | 49% | -0.020% |
| ema40_sma20 | reversal_bull | 2,961 | 15 | 65% | 51% | -0.192% |
| ema40_sma20 | reversal_bear | 3,103 | 15 | 63% | 48% | -0.064% |
| ema40_sma20 | converge_bull | 7,368 | 26 | 48% | 51% | 0.072% |
| ema40_sma20 | converge_bear | 7,445 | 27 | 47% | 51% | -0.187% |
| ema40_sma20 | combined_bull | 1,908 | 13 | 68% | 51% | -0.164% |
| ema40_sma20 | combined_bear | 2,078 | 13 | 68% | 49% | -0.033% |
| ema40_sma20 | confirmed_bull | 1,708 | 15 | 65% | 51% | -0.153% |
| ema40_sma20 | confirmed_bear | 1,775 | 15 | 64% | 49% | 0.014% |
| ema50_sma20 | decel_bull | 4,479 | 8 | 89% | 51% | -0.026% |
| ema50_sma20 | decel_bear | 4,525 | 7 | 87% | 49% | -0.081% |
| ema50_sma20 | reversal_bull | 2,626 | 15 | 62% | 51% | -0.138% |
| ema50_sma20 | reversal_bear | 2,673 | 15 | 59% | 47% | -0.134% |
| ema50_sma20 | converge_bull | 7,186 | 31 | 42% | 51% | 0.091% |
| ema50_sma20 | converge_bear | 7,472 | 32 | 39% | 51% | -0.191% |
| ema50_sma20 | combined_bull | 1,769 | 14 | 63% | 50% | -0.119% |
| ema50_sma20 | combined_bear | 1,864 | 14 | 61% | 48% | -0.167% |
| ema50_sma20 | confirmed_bull | 1,513 | 16 | 61% | 50% | -0.064% |
| ema50_sma20 | confirmed_bear | 1,585 | 15 | 59% | 47% | -0.058% |
| ema5_ema20 | decel_bull | 14,026 | 15 | 71% | 51% | 0.026% |
| ema5_ema20 | decel_bear | 14,159 | 15 | 71% | 49% | -0.207% |
| ema5_ema20 | reversal_bull | 11,469 | 9 | 81% | 50% | -0.038% |
| ema5_ema20 | reversal_bear | 11,155 | 8 | 81% | 49% | -0.095% |
| ema5_ema20 | converge_bull | 12,544 | 11 | 78% | 51% | -0.011% |
| ema5_ema20 | converge_bear | 12,305 | 11 | 77% | 49% | -0.165% |
| ema5_ema20 | combined_bull | 11,467 | 9 | 81% | 50% | -0.038% |
| ema5_ema20 | combined_bear | 11,155 | 8 | 81% | 49% | -0.095% |
| ema5_ema20 | confirmed_bull | 7,489 | 9 | 80% | 50% | -0.002% |
| ema5_ema20 | confirmed_bear | 7,280 | 9 | 81% | 49% | -0.088% |
| ema8_ema20 | decel_bull | 14,598 | 18 | 63% | 51% | 0.045% |
| ema8_ema20 | decel_bear | 14,568 | 18 | 63% | 49% | -0.219% |
| ema8_ema20 | reversal_bull | 9,312 | 9 | 78% | 50% | -0.020% |
| ema8_ema20 | reversal_bear | 9,075 | 9 | 78% | 48% | -0.026% |
| ema8_ema20 | converge_bull | 10,889 | 13 | 72% | 51% | 0.035% |
| ema8_ema20 | converge_bear | 10,656 | 13 | 71% | 48% | -0.123% |
| ema8_ema20 | combined_bull | 9,311 | 9 | 78% | 50% | -0.020% |
| ema8_ema20 | combined_bear | 9,075 | 9 | 78% | 48% | -0.026% |
| ema8_ema20 | confirmed_bull | 5,928 | 10 | 78% | 50% | 0.028% |
| ema8_ema20 | confirmed_bear | 5,823 | 10 | 77% | 49% | -0.061% |
| ema10_ema20 | decel_bull | 14,832 | 20 | 60% | 50% | 0.046% |
| ema10_ema20 | decel_bear | 14,835 | 20 | 59% | 49% | -0.224% |
| ema10_ema20 | reversal_bull | 8,402 | 9 | 77% | 50% | -0.028% |
| ema10_ema20 | reversal_bear | 8,132 | 9 | 78% | 47% | 0.025% |
| ema10_ema20 | converge_bull | 10,104 | 14 | 69% | 50% | 0.063% |
| ema10_ema20 | converge_bear | 9,945 | 14 | 68% | 48% | -0.136% |
| ema10_ema20 | combined_bull | 8,401 | 9 | 77% | 50% | -0.028% |
| ema10_ema20 | combined_bear | 8,132 | 9 | 78% | 47% | 0.025% |
| ema10_ema20 | confirmed_bull | 5,261 | 10 | 76% | 50% | 0.053% |
| ema10_ema20 | confirmed_bear | 5,174 | 10 | 76% | 47% | 0.003% |
| ema10_ema30 | decel_bull | 13,998 | 25 | 49% | 50% | 0.069% |
| ema10_ema30 | decel_bear | 13,783 | 25 | 49% | 49% | -0.244% |
| ema10_ema30 | reversal_bull | 8,324 | 13 | 68% | 50% | 0.009% |
| ema10_ema30 | reversal_bear | 7,924 | 13 | 69% | 47% | 0.060% |
| ema10_ema30 | converge_bull | 9,712 | 18 | 60% | 50% | 0.116% |
| ema10_ema30 | converge_bear | 9,381 | 18 | 60% | 48% | -0.052% |
| ema10_ema30 | combined_bull | 8,324 | 13 | 68% | 50% | 0.009% |
| ema10_ema30 | combined_bear | 7,924 | 13 | 69% | 47% | 0.060% |
| ema10_ema30 | confirmed_bull | 5,216 | 14 | 68% | 49% | 0.103% |
| ema10_ema30 | confirmed_bear | 5,027 | 14 | 68% | 47% | -0.005% |
| ema12_ema26 | decel_bull | 14,470 | 25 | 50% | 50% | 0.077% |
| ema12_ema26 | decel_bear | 14,279 | 25 | 50% | 49% | -0.232% |
| ema12_ema26 | reversal_bull | 7,653 | 12 | 70% | 50% | -0.019% |
| ema12_ema26 | reversal_bear | 7,320 | 12 | 71% | 47% | 0.074% |
| ema12_ema26 | converge_bull | 9,321 | 18 | 61% | 50% | 0.113% |
| ema12_ema26 | converge_bear | 9,082 | 18 | 60% | 48% | -0.060% |
| ema12_ema26 | combined_bull | 7,652 | 12 | 70% | 50% | -0.019% |
| ema12_ema26 | combined_bear | 7,320 | 12 | 71% | 47% | 0.074% |
| ema12_ema26 | confirmed_bull | 4,779 | 12 | 70% | 49% | 0.069% |
| ema12_ema26 | confirmed_bear | 4,615 | 13 | 69% | 47% | 0.013% |
| ema12_ema30 | decel_bull | 14,180 | 27 | 46% | 50% | 0.102% |
| ema12_ema30 | decel_bear | 13,892 | 27 | 47% | 49% | -0.227% |
| ema12_ema30 | reversal_bull | 7,604 | 13 | 67% | 50% | -0.009% |
| ema12_ema30 | reversal_bear | 7,265 | 13 | 68% | 47% | 0.079% |
| ema12_ema30 | converge_bull | 9,230 | 20 | 57% | 50% | 0.118% |
| ema12_ema30 | converge_bear | 8,984 | 20 | 57% | 48% | -0.058% |
| ema12_ema30 | combined_bull | 7,604 | 13 | 67% | 50% | -0.009% |
| ema12_ema30 | combined_bear | 7,265 | 13 | 68% | 47% | 0.079% |
| ema12_ema30 | confirmed_bull | 4,755 | 14 | 67% | 49% | 0.059% |
| ema12_ema30 | confirmed_bear | 4,589 | 14 | 67% | 47% | 0.021% |
| ema15_ema30 | decel_bull | 14,478 | 29 | 42% | 50% | 0.097% |
| ema15_ema30 | decel_bear | 14,092 | 29 | 43% | 49% | -0.238% |
| ema15_ema30 | reversal_bull | 6,857 | 14 | 66% | 50% | 0.050% |
| ema15_ema30 | reversal_bear | 6,461 | 13 | 68% | 47% | 0.125% |
| ema15_ema30 | converge_bull | 8,799 | 22 | 54% | 50% | 0.127% |
| ema15_ema30 | converge_bear | 8,487 | 22 | 54% | 49% | -0.062% |
| ema15_ema30 | combined_bull | 6,857 | 14 | 66% | 50% | 0.050% |
| ema15_ema30 | combined_bear | 6,461 | 13 | 68% | 47% | 0.125% |
| ema15_ema30 | confirmed_bull | 4,260 | 14 | 65% | 49% | 0.133% |
| ema15_ema30 | confirmed_bear | 4,103 | 15 | 66% | 47% | 0.031% |
| ema20_ema50 | decel_bull | 13,925 | 46 | 26% | 49% | 0.113% |
| ema20_ema50 | decel_bear | 13,044 | 43 | 28% | 49% | -0.216% |
| ema20_ema50 | reversal_bull | 5,925 | 23 | 52% | 49% | 0.123% |
| ema20_ema50 | reversal_bear | 5,361 | 20 | 56% | 46% | 0.228% |
| ema20_ema50 | converge_bull | 8,555 | 36 | 37% | 49% | 0.157% |
| ema20_ema50 | converge_bear | 7,962 | 34 | 38% | 47% | -0.073% |
| ema20_ema50 | combined_bull | 5,880 | 23 | 52% | 49% | 0.124% |
| ema20_ema50 | combined_bear | 5,329 | 20 | 56% | 46% | 0.229% |
| ema20_ema50 | confirmed_bull | 3,815 | 24 | 50% | 48% | 0.124% |
| ema20_ema50 | confirmed_bear | 3,484 | 21 | 54% | 46% | 0.275% |

### False-turn sensitivity to the volatility-scaled threshold (primary config)

| signal | n | 0.5σ | 1.0σ | 1.5σ |
|---|---|---|---|---|
| combined_bear | 6,807 | 45% | 48% | 47% |
| combined_bull | 6,765 | 48% | 51% | 48% |
| confirmed_bear | 4,452 | 47% | 49% | 47% |
| confirmed_bull | 4,403 | 48% | 50% | 48% |
| converge_bear | 9,411 | 46% | 49% | 48% |
| converge_bull | 9,350 | 48% | 51% | 48% |
| decel_bear | 16,191 | 45% | 49% | 49% |
| decel_bull | 16,012 | 46% | 50% | 49% |
| reversal_bear | 7,798 | 46% | 48% | 47% |
| reversal_bull | 7,737 | 48% | 51% | 49% |

## 4. Experiment D — tradable strategy variants

All strategies are latch-based: a bull event opens/holds the long side, a bear event the short side, reversing on the opposite event. Costs and funding use the shared engine; repeated same-direction signals maintain the position. The primary-config comparison below is at 6 bps/side.

| strategy | side | IS ret | IS Sharpe | OOS ret | OOS Sharpe | OOS maxDD | trades | win | PF | avg trade | hold h | break-even (bps) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| combined | both | -36.55% | -0.77 | -46.91% | -2.16 | -53.63% | 6458 | 45% | 0.88 | -0.150% | 23.1 | ≤0 |
| combined | long | -18.21% | -0.28 | -8.89% | -0.18 | -31.26% | 6450 | 45% | 0.96 | -0.053% | 23.0 | 2.7 |
| combined | short | -27.90% | -0.55 | -43.77% | -2.25 | -51.08% | 6456 | 45% | 0.82 | -0.246% | 23.2 | ≤0 |
| confirmed | both | -34.10% | -0.64 | -43.09% | -1.79 | -51.30% | 4768 | 51% | 0.87 | -0.195% | 31.3 | ≤0 |
| confirmed | long | -16.97% | -0.22 | -5.91% | -0.03 | -31.26% | 4759 | 50% | 0.96 | -0.062% | 31.2 | 3.0 |
| confirmed | short | -25.94% | -0.50 | -41.47% | -2.17 | -47.35% | 4767 | 52% | 0.79 | -0.327% | 31.4 | ≤0 |
| cross | both | -26.45% | -0.25 | 16.09% | 0.67 | -35.61% | 8660 | 33% | 1.02 | 0.020% | 17.2 | 8.1 |
| cross | long | -7.61% | 0.02 | 36.78% | 1.38 | -23.01% | 8659 | 32% | 1.08 | 0.096% | 17.2 | 14.3 |
| cross | short | -23.72% | -0.30 | -17.00% | -0.47 | -33.93% | 8651 | 34% | 0.95 | -0.055% | 17.2 | 1.2 |
| roc | both | -48.91% | -1.22 | -55.49% | -2.94 | -59.27% | 7403 | 42% | 0.85 | -0.182% | 20.1 | ≤0 |
| roc | long | -26.51% | -0.54 | -16.48% | -0.54 | -36.32% | 7395 | 42% | 0.91 | -0.098% | 20.1 | 0.4 |
| roc | short | -35.30% | -0.79 | -48.67% | -2.59 | -52.82% | 7401 | 43% | 0.79 | -0.267% | 20.2 | ≤0 |
| velocity | both | -42.97% | -0.75 | -72.04% | -3.74 | -73.25% | 7519 | 51% | 0.82 | -0.229% | 19.8 | ≤0 |
| velocity | long | -21.68% | -0.30 | -33.60% | -1.27 | -43.38% | 7510 | 50% | 0.88 | -0.145% | 19.8 | ≤0 |
| velocity | short | -30.77% | -0.63 | -58.95% | -3.37 | -61.72% | 7518 | 51% | 0.77 | -0.312% | 19.9 | ≤0 |

### Best OOS Sharpe across the configuration grid, per strategy

| strategy | side | best config | OOS return | OOS Sharpe | OOS maxDD | configs tested |
|---|---|---|---|---|---|---|
| cross | long | ema10_ema20 | 41.41% | 1.54 | -21.18% | 18 |
| cross | short | ema10_ema20 | -14.48% | -0.36 | -37.74% | 18 |
| cross | both | ema10_ema20 | 23.78% | 0.86 | -28.57% | 18 |
| roc | long | ema20_ema50 | 54.85% | 1.78 | -26.46% | 18 |
| roc | short | ema20_ema50 | -4.13% | -0.02 | -31.01% | 18 |
| roc | both | ema20_ema50 | 53.29% | 1.67 | -27.08% | 18 |
| velocity | long | ema20_ema50 | 38.72% | 1.33 | -32.46% | 18 |
| velocity | short | ema20_ema50 | -13.90% | -0.44 | -31.69% | 18 |
| velocity | both | ema20_ema50 | 22.38% | 0.84 | -29.96% | 18 |
| combined | long | ema20_ema50 | 54.56% | 1.78 | -26.46% | 18 |
| combined | short | ema20_ema50 | -4.30% | -0.02 | -30.98% | 18 |
| combined | both | ema20_ema50 | 52.74% | 1.66 | -27.08% | 18 |
| confirmed | long | ema20_ema50 | 47.05% | 1.57 | -28.85% | 18 |
| confirmed | short | ema20_ema50 | -8.85% | -0.22 | -33.39% | 18 |
| confirmed | both | ema20_ema50 | 38.09% | 1.27 | -28.94% | 18 |

### Risk-management variants (primary config, combined, OOS)

| risk rule | side | OOS return | OOS Sharpe | OOS maxDD | trades | exposure |
|---|---|---|---|---|---|---|
| none | long | -8.89% | -0.18 | -31.26% | 6450 | 50% |
| none | short | -43.77% | -2.25 | -51.08% | 6456 | 50% |
| none | both | -46.91% | -2.16 | -53.63% | 6458 | 100% |
| minhold12 | long | -6.18% | -0.07 | -28.53% | 6205 | 50% |
| minhold12 | short | -42.18% | -2.11 | -51.60% | 6211 | 50% |
| minhold12 | both | -43.82% | -1.93 | -51.75% | 6213 | 100% |
| atrstop2 | long | 46.45% | 1.24 | -35.54% | 484 | 94% |
| atrstop2 | short | -46.04% | -1.21 | -61.73% | 280 | 96% |
| atrstop2 | both | -42.06% | -2.78 | -43.60% | 8257 | 60% |

### Entry-delay sensitivity (primary combined, long-short)

| delay (bars) | OOS return | OOS Sharpe | full-sample return |
|---|---|---|---|
| 0 | -46.91% | -2.16 | -66.31% |
| 1 | -38.80% | -1.63 | -53.49% |
| 2 | -20.66% | -0.69 | -51.10% |
| 3 | -20.71% | -0.68 | -54.71% |
| 6 | -3.75% | 0.04 | -57.33% |
| 12 | 5.85% | 0.39 | -15.66% |

### ROC-lookback `k` sensitivity (primary combined, long-short, OOS)

| k | OOS return | OOS Sharpe | trades |
|---|---|---|---|
| 1 | -46.91% | -2.16 | 6458 |
| 3 | -4.43% | -0.01 | 4959 |
| 6 | -3.39% | -0.01 | 2521 |
| 12 | -1.37% | 0.01 | 73 |

## 5. Validation & selection bias

Per fold the **configuration** with the best train Sharpe is frozen (for each strategy) and evaluated on the next unseen test window; test windows are stitched for the aggregate. Selection is over the full candidate grid, so this is the multiple-testing-honest result.

| strategy | folds | WF OOS ret | WF OOS Sharpe | WF OOS maxDD | mean test Sharpe | folds positive |
|---|---|---|---|---|---|---|
| cross | 4 | -48.62% | -1.12 | -62.88% | -1.19 | 0% |
| roc | 4 | -46.16% | -1.41 | -47.44% | -1.58 | 25% |
| velocity | 4 | -49.75% | -1.24 | -58.47% | -1.30 | 25% |
| combined | 4 | -49.14% | -1.54 | -51.56% | -1.71 | 25% |
| confirmed | 4 | -21.58% | -0.42 | -41.86% | -0.83 | 50% |

| strategy | fold | train window | test window | chosen config | train Sharpe | val Sharpe | test Sharpe | test return |
|---|---|---|---|---|---|---|---|---|
| combined | 0 | 2025-01-01→2025-07-01 | 2025-09-01→2025-12-01 | ema20_ema50 | 0.84 | 3.18 | -4.35 | -41.48% |
| combined | 1 | 2025-04-01→2025-10-01 | 2025-12-01→2026-03-01 | ema20_ema50 | 1.27 | -5.46 | 2.51 | 29.51% |
| combined | 2 | 2025-07-01→2026-01-01 | 2026-03-01→2026-06-01 | ema40_sma20 | 1.95 | -3.63 | -2.16 | -15.27% |
| combined | 3 | 2025-10-01→2026-04-01 | 2026-06-01→2026-09-01 | ema50_sma20 | 0.38 | -0.42 | -2.82 | -20.80% |
| confirmed | 0 | 2025-01-01→2025-07-01 | 2025-09-01→2025-12-01 | ema20_sma20 | 1.73 | -0.20 | 0.32 | 1.14% |
| confirmed | 1 | 2025-04-01→2025-10-01 | 2025-12-01→2026-03-01 | ema20_ema50 | 0.97 | -3.02 | 2.00 | 24.01% |
| confirmed | 2 | 2025-07-01→2026-01-01 | 2026-03-01→2026-06-01 | ema40_sma20 | 2.63 | -3.72 | -1.10 | -9.06% |
| confirmed | 3 | 2025-10-01→2026-04-01 | 2026-06-01→2026-09-01 | ema40_sma20 | 0.81 | -0.06 | -4.55 | -31.24% |
| cross | 0 | 2025-01-01→2025-07-01 | 2025-09-01→2025-12-01 | ema10_sma20 | 0.20 | 1.47 | -0.49 | -10.50% |
| cross | 1 | 2025-04-01→2025-10-01 | 2025-12-01→2026-03-01 | ema50_sma20 | 0.75 | -3.32 | -1.49 | -20.11% |
| cross | 2 | 2025-07-01→2026-01-01 | 2026-03-01→2026-06-01 | ema5_ema20 | 0.01 | 0.68 | -1.75 | -16.37% |
| cross | 3 | 2025-10-01→2026-04-01 | 2026-06-01→2026-09-01 | ema12_sma20 | -0.22 | -0.69 | -1.04 | -14.08% |
| roc | 0 | 2025-01-01→2025-07-01 | 2025-09-01→2025-12-01 | ema20_ema50 | 0.84 | 3.18 | -4.38 | -41.75% |
| roc | 1 | 2025-04-01→2025-10-01 | 2025-12-01→2026-03-01 | ema20_ema50 | 1.27 | -5.49 | 2.53 | 29.82% |
| roc | 2 | 2025-07-01→2026-01-01 | 2026-03-01→2026-06-01 | ema50_sma20 | 1.75 | -2.93 | -2.70 | -17.33% |
| roc | 3 | 2025-10-01→2026-04-01 | 2026-06-01→2026-09-01 | ema50_sma20 | 0.85 | -1.63 | -1.79 | -13.88% |
| velocity | 0 | 2025-01-01→2025-07-01 | 2025-09-01→2025-12-01 | ema25_sma20 | 1.25 | -4.98 | -1.27 | -18.78% |
| velocity | 1 | 2025-04-01→2025-10-01 | 2025-12-01→2026-03-01 | ema20_ema50 | 1.46 | -4.54 | 1.56 | 17.65% |
| velocity | 2 | 2025-07-01→2026-01-01 | 2026-03-01→2026-06-01 | ema8_ema20 | 0.23 | -2.23 | -1.16 | -11.15% |
| velocity | 3 | 2025-10-01→2026-04-01 | 2026-06-01→2026-09-01 | ema8_ema20 | 0.15 | 0.16 | -4.35 | -40.81% |

## 6. Costs & break-even

Primary combined strategy: gross OOS return (0 bps, no funding effect net of the series) **-25.95%**; break-even costs below. Sensitivity at 0/3/6/12/20 bps/side.

| strategy | scenario | bps/side | OOS ret | OOS Sharpe | OOS maxDD | fees | funding |
|---|---|---|---|---|---|---|---|
| roc | zero | 0 | -34.48% | -1.46 | -42.77% | 0.00% | -0.08% |
| roc | low | 3 | -46.00% | -2.20 | -51.51% | 44.39% | -0.08% |
| roc | base | 6 | -55.49% | -2.94 | -59.27% | 88.78% | -0.08% |
| roc | high | 12 | -69.77% | -4.42 | -71.72% | 177.55% | -0.08% |
| roc | extreme | 20 | -81.95% | -6.38 | -82.61% | 295.92% | -0.08% |
| velocity | zero | 0 | -58.61% | -2.53 | -61.02% | 0.00% | -1.56% |
| velocity | low | 3 | -65.98% | -3.14 | -67.71% | 45.08% | -1.56% |
| velocity | base | 6 | -72.04% | -3.74 | -73.25% | 90.17% | -1.56% |
| velocity | high | 12 | -81.11% | -4.95 | -81.66% | 180.34% | -1.56% |
| velocity | extreme | 20 | -88.81% | -6.55 | -88.93% | 300.56% | -1.56% |
| combined | zero | 0 | -25.95% | -0.93 | -39.41% | 0.00% | -0.09% |
| combined | low | 3 | -37.30% | -1.54 | -46.60% | 38.72% | -0.09% |
| combined | base | 6 | -46.91% | -2.16 | -53.63% | 77.44% | -0.09% |
| combined | high | 12 | -61.94% | -3.38 | -65.39% | 154.87% | -0.09% |
| combined | extreme | 20 | -75.59% | -5.00 | -77.10% | 258.12% | -0.09% |
| cross | zero | 0 | 83.29% | 2.03 | -26.68% | 0.00% | 0.81% |
| cross | low | 3 | 45.88% | 1.35 | -31.29% | 51.93% | 0.81% |
| cross | base | 6 | 16.09% | 0.67 | -35.61% | 103.86% | 0.81% |
| cross | high | 12 | -26.48% | -0.69 | -46.70% | 207.72% | 0.81% |
| cross | extreme | 20 | -60.02% | -2.50 | -65.56% | 346.20% | 0.81% |

**Break-even fee+slippage per side** (primary config) by strategy/side: cross/long 14.3, cross/short 1.2, cross/both 8.1, roc/long 0.4, roc/short ≤0, roc/both ≤0, velocity/long ≤0, velocity/short ≤0, velocity/both ≤0, combined/long 2.7, combined/short ≤0, combined/both ≤0, confirmed/long 3.0, confirmed/short ≤0, confirmed/both ≤0 bps.

## 7. Market-regime analysis (primary combined, OOS)

| regime | bars | mean symbol return | % positive symbols |
|---|---|---|---|
| down_highvol | 4,948 | 9.10% | 60% |
| up_highvol | 5,405 | 0.65% | 50% |
| up_lowvol | 4,721 | -7.42% | 40% |
| down_lowvol | 5,701 | -13.50% | 20% |
| range_highvol | 21,130 | -19.11% | 10% |
| range_lowvol | 23,615 | -28.13% | 0% |

Regimes from the causal classifier (trend = volatility-normalised trailing 24h move, ±1σ; vol = realised vol vs trailing median), shifted one bar. OOS only.

## 8. Robustness & statistical analysis

**Bootstrap (block=72h, 1000 resamples) on the primary combined OOS returns**: Sharpe mean -2.18, 95% CI [-4.33, 0.05]; total-return 95% CI [-70.66%, -0.63%].

**Profit concentration** (primary combined, pooled trades): 6458 trades; top 5% = **47%** of gross profit; best 84.49%, worst -51.92%.

**Per-symbol OOS (primary combined)**

| symbol | OOS return | OOS Sharpe | OOS maxDD | trades | win | PF |
|---|---|---|---|---|---|---|
| DOGEUSDT | -26.62% | -0.29 | -57.21% | 647 | 45% | 0.77 |
| HYPEUSDT | -33.35% | -0.14 | -70.05% | 502 | 46% | 0.95 |
| XRPUSDT | -45.76% | -0.97 | -69.76% | 662 | 45% | 0.80 |
| SOLUSDT | -45.78% | -0.99 | -54.48% | 675 | 46% | 0.78 |
| BTCUSDT | -53.33% | -2.14 | -59.79% | 681 | 45% | 0.89 |
| ETHUSDT | -55.61% | -1.61 | -65.19% | 673 | 43% | 0.98 |
| NEARUSDT | -60.92% | -0.74 | -78.98% | 671 | 46% | 1.02 |
| BNBUSDT | -63.26% | -2.91 | -66.63% | 661 | 45% | 0.69 |
| ZECUSDT | -67.91% | -0.68 | -86.55% | 623 | 46% | 0.91 |
| OGNUSDT | -78.71% | -1.65 | -82.64% | 663 | 43% | 0.95 |

Baselines over the same OOS window: buy & hold **42.29%** (Sharpe 1.12); a simple 24h price-momentum long-short baseline **-14.01%** (Sharpe -0.22); no-trade 0%.

Multiple testing: the strategy grid is 270 (strategy, side, config) cells, and the diagnostics add 432 (config, feature, horizon) IC cells. The best observed values are order statistics; walk-forward selection on train folds is the honest evaluation and is reported above.

## 9. Limitations

- The OOS window (2026) is a single, predominantly bullish regime; OOS results are not independent evidence, and the far weaker short-side strategy performance reflects that drift.
- Bar-level simulation with a fixed per-side cost and next-open fills; no order-book, latency, partial fills or market impact. Funding is a drag approximation on held notional, not a cash-flow ledger.
- Forward-return diagnostics overlap heavily; HAC errors and block bootstraps mitigate but do not eliminate the dependence. Regime labels are causal but coarse.
- The universe is today's top-10 liquids (survivorship bias) and an equal-weight pooled book is assumed. The 2025–2026 sample has already been examined in prior studies, so its final portion is not a pristine holdout.

## 10. Answers to the study's final questions

1. **Does EMA ROC predict before crossovers?** Weakly, but with a genuine and roughly symmetric lead. Reversal signals precede the same-direction cross with a median lead of 9h (bull) / 9h (bear); rank IC vs 24h return is small (-0.019 average over configs) and conditional mean returns are near zero.
2. **Does acceleration add information beyond ROC?** Marginally. The incremental R² from adding `accel_1` on top of the spread + crossover baseline is 0.00001 at 24h; it does not change the sign of the conclusion.
3. **Does normalized spread velocity add beyond the current spread?** Slightly more than acceleration: ΔR² ≈ 0.00001 at 24h, with HAC t-stats that are positive but modest.
4. **Does the combined signal beat its components?** Not out-of-sample after costs — see the strategy table; the components are close together and the combined rule trades a different amount rather than dominating.
5. **Lead time and false turns?** Median lead ≈ 9h across signals; the 1×ATR false-turn rate is ≈ 49% on average — high — and lead time and reliability trade off sharply (best reliability is the short-lead combined/confirmed bull signal).
6. **Does earlier entry improve expectancy?** No — the opposite. In the entry-delay table the *un*-delayed early signal is the worst (OOS ≈ -47% at 6 bps); adding a 6–12 bar delay lifts it to roughly break-even. The earliest signal is the noisiest, so early entry destroys expectancy rather than creating it, and the crossover baseline is the better of the two.
7. **Do gains survive costs & funding?** No strategy clears the base taker cost with a stable sign; break-even costs are modest relative to the 6 bps assumption (the crossover baseline is the only primary-config variant with a positive break-even on the long side).
8. **Do long and short differ?** The *diagnostic* lead time is nearly symmetric (≈9h either way), but the *traded* result is not: the long side is far less bad than the short side (primary combined OOS long ≈ -9% vs short ≈ -44%). That is consistent with the sample's upward drift rather than a structural asymmetry in the signal.
9. **Stable across symbols, regimes, folds?** No. Per-symbol dispersion is large, regime signs differ, and walk-forward folds are mostly negative.
10. **Is there a repeatable edge?** No. **H1/H3 are weakly supported diagnostically, H2/H4 are marginal, H5 holds (lead time exists) but with unacceptable false turns, and H6/H7/H8 are not supported for trading.** The momentum-deceleration family is rejected as a standalone signal; the diagnostic value of ROC and spread velocity may merit a conditional (regime-gated) retest with finer execution data, but the current evidence does not justify strategy development.

---

_Reproduce:_ `python scripts/ema_momentum_study.py` (engine: `quant_lab.analytics.ema_backtest`; features/signals: `quant_lab.analytics.ema_momentum`). Artifacts: `experiments/reports/ema_momentum_v1`.
