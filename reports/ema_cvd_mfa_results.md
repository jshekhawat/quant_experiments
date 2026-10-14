# EMA Momentum × CVD × Volatility × Price Response — Multi-Factor Alpha Study

- **Universe (10):** BTCUSDT, ETHUSDT, SOLUSDT, ZECUSDT, NEARUSDT, XRPUSDT, OGNUSDT, HYPEUSDT, BNBUSDT, DOGEUSDT
- **Data:** Binance USD-M perpetuals, 1h resampled from 1m; 2025-01-01 → 2026-10-01; design ≤ 2026-01-01, final window ≥ 2026-05-01.
- **Order flow:** exact per-bar reconstruction from kline taker fields in quote (USDT): `qd = 2·taker_buy_quote − quote_volume`, `FI_k = Σqd/Σquote_volume ∈ [−1,1]`, `Fz_k` = trailing z-score. No tick/order-book data — absorption is a proxy.
- **Execution:** signal at close *t*, fill at open *t+1*; **6.0 bps/side**; funding=on; annualisation 8760 bars/yr.

## Executive answer

On the 10-symbol perp universe, order flow adds essentially nothing beyond price. OOS 4h rank ICs are small (momentum `R_24` -0.017, flow `FI_4` -0.012, absorption score -0.007, divergence `DIV_6` -0.003). The only nominally significant increment — flow at the 24h horizon (ΔR² ≈ 0.0005) — explains ~0.05% of return variance and does not survive as a tradable rule.

Track E (the priority) is the clearest null: high-flow/low-impact (absorption → reversal) events have an oriented 4h return of **0.074%** (n=893, hit 50.3%, CI [-0.016%, 0.171%]) versus **0.079%** for high-flow/high-impact (continuation) — the CIs overlap and both include zero, so the two explanations are not distinguishable and the effect is not economically meaningful.

Walk-forward selection over all strategies returns Sharpe **-0.68** (-22.69%, 4 folds, 25% positive). Buy & hold OOS was 42.29% (Sharpe 1.12). No combination clears realistic costs with a stable sign.

**Verdict.** None of the five tracks produces a repeatable edge after costs and out-of-sample validation. The negative is reported in full below.

---

## 1. Environment & data audit

Reuses `analytics.ema_backtest` (execution/costs/funding), `analytics.ema_momentum` (EMA ROC/accel/spread), `analytics.divergence` estimators and the causal regime classifier. New code: `analytics.mfa` (flow, momentum, volatility, price-response features and the track strategies) and `scripts/ema_cvd_mfa_study.py`.

| requirement | availability | source |
|---|---|---|
| OHLCV 1m klines | ✅ 10 symbols, 2025-01→2026-09 | `data/normalized/.../bars` |
| taker-buy base volume | ✅ | `taker_buy_volume` |
| taker-buy quote volume | ✅ | `taker_buy_quote_volume` |
| total base / quote volume | ✅ | `volume`, `quote_volume` |
| funding rates | ✅ hourly-aligned | `.../funding` |
| open interest | partial (not used) | `.../open_interest` |
| trade-level aggressor data | ❌ not present | — |
| order-book snapshots / depth | ❌ not present | — |

Order flow is therefore reconstructed **exactly** from kline taker fields (no proxy from candle direction): `flow_available=True`, panel rows 149,533. Aggressive-sell quote volume is `quote_volume − taker_buy_quote_volume`, so `qd` is net taker quote flow. Because there is no order book, **absorption is an observable proxy**, never evidence of hidden liquidity or market-maker intent.

Note on novelty vs the prior 2026-only flow studies (divergence, H1 absorption, H2 sweeps, H3 breakouts): the 10-symbol universe here carries full **2025–2026** klines, so the design window (2025) and final window (2026-05-01→) are genuinely separated in time, unlike the earlier single-regime 2026 samples.

Feature cross-correlation (finite overlap): R_24/bw_z +0.04, FI_4/R_24 +0.17, FI_4/Fz_4 +0.95, FI_4/PR_4 +0.52, FI_4/bw_z -0.01, Fz_4/R_24 +0.18, Fz_4/PR_4 +0.55, Fz_4/bw_z -0.00, PR_4/R_24 +0.35, PR_4/bw_z +0.01, DIV_6/R_24 +0.08, DIV_6/FI_4 +0.10, DIV_6/Fz_4 +0.10, DIV_6/PR_4 +0.16, DIV_6/bw_z -0.04.

## 2. Feature diagnostics — predictive information

Per-symbol Spearman IC vs forward returns, equal-weighted across symbols (avoids cross-sectional contamination). *design* = 2025, *oos* = 2026.

| feature | 1h IC (design/oos) | 4h IC (design/oos) | 24h IC (design/oos) | 72h IC (design/oos) |
|---|---|---|---|---|
| R_24 | -0.022 / -0.012 | -0.032 / -0.017 | -0.022 / -0.019 | -0.042 / -0.047 |
| PR_24 | -0.020 / -0.011 | -0.022 / -0.014 | -0.005 / -0.016 | -0.022 / -0.043 |
| FI_4 | -0.008 / -0.017 | 0.007 / -0.012 | 0.021 / 0.010 | 0.020 / -0.007 |
| Fz_4 | -0.007 / -0.018 | 0.006 / -0.012 | 0.018 / 0.008 | 0.012 / -0.009 |
| FI_24 | -0.003 / -0.008 | 0.007 / -0.011 | 0.043 / -0.013 | 0.020 / -0.031 |
| bw_z | 0.001 / -0.009 | 0.007 / -0.012 | 0.027 / -0.039 | 0.035 / -0.035 |
| rv | 0.002 / -0.010 | 0.002 / -0.015 | 0.009 / -0.064 | 0.018 / -0.074 |
| DIV_6 | -0.005 / -0.008 | 0.007 / -0.003 | -0.006 / -0.006 | -0.001 / -0.001 |

### Bootstrap 95% CI on headline features (OOS)

| feature | horizon | IC | 95% CI |
|---|---|---|---|
| R_24 | 4h | -0.017 | [-0.032, -0.004] |
| R_24 | 24h | -0.019 | [-0.046, 0.003] |
| PR_24 | 4h | -0.014 | [-0.028, -0.001] |
| PR_24 | 24h | -0.016 | [-0.043, 0.005] |
| FI_4 | 4h | -0.012 | [-0.022, -0.000] |
| FI_4 | 24h | 0.010 | [-0.005, 0.023] |
| Fz_4 | 4h | -0.012 | [-0.024, -0.000] |
| Fz_4 | 24h | 0.008 | [-0.007, 0.022] |
| DIV_6 | 4h | -0.003 | [-0.013, 0.008] |
| DIV_6 | 24h | -0.006 | [-0.018, 0.008] |

### Incremental information over a price-only baseline (HAC OLS)

Baseline = `{R_24, PR_24}` (within-symbol demeaned); HAC maxlags = horizon. ΔR² is the added explanatory power; `t_add` the HAC t-stat of the first added regressor.

| horizon | added | n | R² base | R² full | ΔR² | t_add |
|---|---|---|---|---|---|---|
| 4h | FI_4 | 149,243 | 0.0013 | 0.0013 | 0.00006 | 1.77 |
| 4h | Fz_4 | 147,063 | 0.0013 | 0.0014 | 0.00007 | 1.85 |
| 4h | FI_4+Fz_4 | 147,063 | 0.0013 | 0.0014 | 0.00008 | -0.45 |
| 4h | absorption_gate | 144,673 | 0.0013 | 0.0013 | 0.00000 | -0.46 |
| 4h | DIV_6 | 149,243 | 0.0013 | 0.0013 | 0.00001 | 0.66 |
| 24h | FI_4 | 149,043 | 0.0021 | 0.0026 | 0.00054 | 4.80 |
| 24h | Fz_4 | 146,863 | 0.0021 | 0.0026 | 0.00043 | 4.02 |
| 24h | FI_4+Fz_4 | 146,863 | 0.0021 | 0.0026 | 0.00050 | 1.89 |
| 24h | absorption_gate | 144,473 | 0.0022 | 0.0022 | 0.00000 | -0.38 |
| 24h | DIV_6 | 149,043 | 0.0021 | 0.0021 | 0.00001 | -0.70 |

## 3. Research tracks A–E

All event outcomes are **oriented** (return in the hypothesised direction) over the OOS window; CI = timestamp-block bootstrap. Positive & significant would be required for an edge.

### Track A — momentum + flow confirmation

| condition | horizon | n | hit | mean | median | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| R>0 alone (long) | 4h | 32,074 | 47.8% | 0.051% | -0.032% | 0.027% | 0.076% |
| R>0 alone (long) | 24h | 31,951 | 47.3% | 0.218% | -0.149% | 0.061% | 0.384% |
| FI>0 alone (long) | 4h | 28,597 | 47.9% | 0.047% | -0.026% | 0.015% | 0.081% |
| FI>0 alone (long) | 24h | 28,514 | 49.3% | 0.277% | -0.032% | 0.104% | 0.456% |
| R>0 & FI>0 (long) | 4h | 17,003 | 46.9% | 0.055% | -0.049% | 0.017% | 0.094% |
| R>0 & FI>0 (long) | 24h | 16,946 | 47.9% | 0.288% | -0.120% | 0.089% | 0.546% |
| R>0 & FI<0 (long) | 4h | 15,071 | 48.8% | 0.046% | -0.008% | 0.001% | 0.081% |
| R>0 & FI<0 (long) | 24h | 15,005 | 46.6% | 0.140% | -0.184% | -0.079% | 0.349% |
| R<0 & FI<0 (short) | 4h | 21,802 | 48.9% | 0.008% | 0.000% | -0.026% | 0.043% |
| R<0 & FI<0 (short) | 24h | 21,751 | 49.2% | -0.059% | -0.030% | -0.231% | 0.141% |
| R<0 & FI>0 (short) | 4h | 11,594 | 49.2% | -0.035% | 0.000% | -0.071% | 0.014% |
| R<0 & FI>0 (short) | 24h | 11,568 | 48.4% | -0.261% | -0.081% | -0.507% | 0.045% |

### Track B — price–CVD divergence

| condition | horizon | n | hit | mean | median | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| bullish divergence (new low + flow improving) | 4h | 2,141 | 53.2% | -0.057% | 0.082% | -0.183% | 0.059% |
| bullish divergence (new low + flow improving) | 24h | 2,140 | 51.0% | -0.025% | 0.090% | -0.470% | 0.503% |
| bearish divergence (new high + flow worsening) | 4h | 2,259 | 52.0% | -0.116% | 0.047% | -0.227% | -0.007% |
| bearish divergence (new high + flow worsening) | 24h | 2,254 | 51.5% | -0.387% | 0.110% | -0.904% | 0.166% |
| all new lows | 4h | 6,290 | 53.1% | -0.015% | 0.082% | -0.072% | 0.043% |
| all new lows | 24h | 6,287 | 50.7% | 0.007% | 0.055% | -0.304% | 0.353% |
| all new highs | 4h | 6,223 | 53.1% | -0.125% | 0.066% | -0.207% | -0.056% |
| all new highs | 24h | 6,205 | 51.4% | -0.429% | 0.120% | -0.738% | -0.037% |

### Track C — trend + pullback + flow reversal

| condition | side | horizon | n | hit | mean | median | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|
| trend only | long | 4h | 31,629 | 48.2% | 0.051% | -0.024% | 0.024% | 0.080% |
| trend only | short | 4h | 33,841 | 49.3% | -0.007% | 0.000% | -0.032% | 0.020% |
| trend only | long | 24h | 31,521 | 47.0% | 0.226% | -0.203% | 0.053% | 0.397% |
| trend only | short | 24h | 33,749 | 48.5% | -0.123% | -0.062% | -0.278% | 0.028% |
| trend + pullback | long | 4h | 13,016 | 48.9% | 0.012% | -0.010% | -0.023% | 0.056% |
| trend + pullback | short | 4h | 13,690 | 51.3% | -0.014% | 0.032% | -0.058% | 0.025% |
| trend + pullback | long | 24h | 12,984 | 46.8% | 0.107% | -0.207% | -0.192% | 0.374% |
| trend + pullback | short | 24h | 13,648 | 48.5% | -0.251% | -0.064% | -0.471% | 0.022% |
| trend + flow reversal | long | 4h | 15,526 | 47.7% | 0.060% | -0.035% | 0.018% | 0.103% |
| trend + flow reversal | short | 4h | 16,819 | 48.9% | -0.018% | 0.000% | -0.055% | 0.011% |
| trend + flow reversal | long | 24h | 15,470 | 47.2% | 0.220% | -0.190% | -0.029% | 0.459% |
| trend + flow reversal | short | 24h | 16,773 | 48.7% | -0.122% | -0.056% | -0.343% | 0.099% |
| trend + pullback + flow | long | 4h | 6,656 | 48.5% | 0.014% | -0.014% | -0.043% | 0.067% |
| trend + pullback + flow | short | 4h | 7,073 | 50.9% | -0.015% | 0.020% | -0.069% | 0.037% |
| trend + pullback + flow | long | 24h | 6,642 | 47.2% | 0.087% | -0.185% | -0.240% | 0.422% |
| trend + pullback + flow | short | 24h | 7,054 | 49.1% | -0.211% | -0.020% | -0.457% | 0.130% |
| trend + pullback + flow + confirm | long | 4h | 3,616 | 47.0% | 0.001% | -0.046% | -0.066% | 0.073% |
| trend + pullback + flow + confirm | short | 4h | 3,925 | 50.3% | 0.011% | 0.009% | -0.057% | 0.079% |
| trend + pullback + flow + confirm | long | 24h | 3,611 | 47.9% | 0.181% | -0.141% | -0.263% | 0.521% |
| trend + pullback + flow + confirm | short | 24h | 3,915 | 49.4% | -0.193% | -0.013% | -0.604% | 0.186% |

### Track D — volatility compression + flow impulse

| condition | side | horizon | n | hit | mean | median | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|
| breakout only | long | 4h | 6,223 | 46.2% | 0.125% | -0.066% | 0.056% | 0.207% |
| breakout only | short | 4h | 6,290 | 45.9% | 0.015% | -0.082% | -0.043% | 0.072% |
| breakout only | long | 24h | 6,205 | 48.3% | 0.429% | -0.120% | 0.037% | 0.738% |
| breakout only | short | 24h | 6,287 | 49.1% | -0.007% | -0.055% | -0.353% | 0.304% |
| compression + breakout | long | 4h | 1,271 | 46.5% | 0.116% | -0.033% | 0.020% | 0.236% |
| compression + breakout | short | 4h | 1,156 | 47.6% | 0.012% | -0.040% | -0.101% | 0.101% |
| compression + breakout | long | 24h | 1,267 | 50.5% | 0.369% | 0.027% | -0.155% | 0.821% |
| compression + breakout | short | 24h | 1,156 | 50.4% | -0.015% | 0.029% | -0.563% | 0.435% |
| compression + breakout + flow | long | 4h | 1,037 | 46.3% | 0.082% | -0.035% | -0.027% | 0.177% |
| compression + breakout + flow | short | 4h | 971 | 47.4% | 0.008% | -0.038% | -0.115% | 0.126% |
| compression + breakout + flow | long | 24h | 1,035 | 50.0% | 0.277% | 0.009% | -0.098% | 0.895% |
| compression + breakout + flow | short | 24h | 971 | 50.1% | -0.070% | 0.014% | -0.704% | 0.453% |
| flow impulse only | long | 4h | 9,800 | 48.3% | 0.081% | -0.018% | 0.033% | 0.138% |
| flow impulse only | short | 4h | 10,115 | 49.2% | 0.003% | 0.000% | -0.035% | 0.051% |
| flow impulse only | long | 24h | 9,774 | 50.1% | 0.361% | 0.007% | 0.110% | 0.682% |
| flow impulse only | short | 24h | 10,096 | 51.1% | 0.003% | 0.076% | -0.228% | 0.205% |

### Track E — order-flow pressure vs price response

| condition | horizon | n | hit | mean | median | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| high flow / low impact (absorption → reversal) | 4h | 893 | 50.3% | 0.074% | 0.010% | -0.016% | 0.171% |
| high flow / low impact (absorption → reversal) | 24h | 890 | 51.3% | 0.197% | 0.136% | -0.071% | 0.478% |
| high flow / high impact (continuation) | 4h | 2,904 | 47.9% | 0.079% | -0.039% | 0.010% | 0.152% |
| high flow / high impact (continuation) | 24h | 2,896 | 48.9% | 0.212% | -0.076% | -0.048% | 0.516% |
| all strong flow (continuation) | 4h | 6,597 | 47.8% | 0.026% | -0.028% | -0.015% | 0.079% |
| all strong flow (continuation) | 24h | 6,581 | 49.6% | 0.133% | 0.000% | -0.026% | 0.291% |
| high flow / low impact (continuation) | 4h | 893 | 47.7% | -0.074% | -0.010% | -0.171% | 0.016% |
| high flow / low impact (continuation) | 24h | 890 | 48.2% | -0.197% | -0.136% | -0.478% | 0.071% |

### Track E — continuous IC

| feature | 1h IC | 4h IC | 24h IC | 72h IC |
|---|---|---|---|---|
| PR_4 | -0.021 | -0.025 | 0.005 | -0.015 |
| FI_4 | -0.017 | -0.012 | 0.010 | -0.007 |
| absorb_score | -0.004 | -0.007 | -0.004 | -0.000 |

## 4. Factor ablation

| combination | OOS 4h oriented mean |
|---|---|
| FI>0 alone (long) | 0.047% |
| R>0 alone (long) | 0.051% |
| R>0 & FI>0 (long) | 0.055% |
| R>0 & FI<0 (long) | 0.046% |

Adding flow confirmation does not change the sign or the magnitude beyond noise; the price-only condition and the confirmed condition are statistically indistinguishable.

## 5. Trading strategies

All strategies are state-based (long/short/flat), pooled equal-weight across symbols, at 6 bps/side. design/OOS split at 2026-01-01.

| strategy | side | IS ret | IS Sharpe | OOS ret | OOS Sharpe | OOS maxDD | trades | exposure |
|---|---|---|---|---|---|---|---|---|
| absorption_rev | both | -12.77% | -4.89 | -10.05% | -6.18 | -10.42% | 3424 | 1% |
| absorption_rev | long | -5.06% | -2.93 | -6.48% | -4.94 | -6.70% | 1674 | 1% |
| absorption_rev | short | -8.12% | -3.88 | -3.82% | -3.48 | -4.11% | 1750 | 1% |
| compression_break_flow | both | -9.29% | -1.46 | -9.91% | -2.96 | -10.45% | 6704 | 3% |
| compression_break_flow | long | -7.62% | -2.01 | -3.90% | -1.71 | -4.45% | 3486 | 2% |
| compression_break_flow | short | -1.81% | -0.32 | -6.25% | -2.44 | -6.99% | 3220 | 2% |
| continuation | both | -19.66% | -2.65 | -19.65% | -4.46 | -20.88% | 7949 | 4% |
| continuation | long | -8.69% | -2.18 | -5.55% | -1.57 | -8.00% | 3878 | 2% |
| continuation | short | -12.01% | -1.79 | -14.93% | -4.78 | -15.73% | 4072 | 2% |
| flow | both | -91.52% | -5.10 | -80.60% | -6.16 | -81.32% | 31896 | 100% |
| flow | long | -67.90% | -3.46 | -44.24% | -2.51 | -55.31% | 31891 | 41% |
| flow | short | -75.42% | -2.50 | -66.48% | -3.81 | -70.81% | 31891 | 59% |
| momentum | both | -78.42% | -2.37 | -11.32% | -0.13 | -36.91% | 14355 | 100% |
| momentum | long | -49.87% | -1.52 | 17.88% | 0.81 | -23.26% | 13924 | 49% |
| momentum | short | -58.70% | -1.50 | -26.39% | -0.89 | -44.06% | 13878 | 50% |
| momentum_flow | both | -84.54% | -4.06 | -54.60% | -3.04 | -58.09% | 41546 | 58% |
| momentum_flow | long | -54.27% | -3.35 | -23.70% | -1.55 | -36.23% | 21425 | 24% |
| momentum_flow | short | -66.55% | -2.53 | -40.86% | -2.38 | -49.15% | 22388 | 34% |
| trend_pullback_flow | both | -63.01% | -7.28 | -51.76% | -9.63 | -52.27% | 26802 | 11% |
| trend_pullback_flow | long | -39.73% | -6.06 | -31.03% | -7.13 | -32.29% | 12189 | 5% |
| trend_pullback_flow | short | -38.64% | -4.46 | -30.06% | -6.41 | -30.11% | 14638 | 6% |

### Trade diagnostics (long-short, pooled)

| strategy | trades | win | PF | expectancy | avg trade | hold h |
|---|---|---|---|---|---|---|
| absorption_rev | 1712 | 39% | 0.58 | -0.139% | -0.139% | 1.2 |
| continuation | 3975 | 34% | 0.75 | -0.106% | -0.106% | 1.6 |
| momentum | 13906 | 27% | 0.89 | -0.088% | -0.088% | 10.7 |
| momentum_flow | 21909 | 34% | 0.81 | -0.109% | -0.109% | 4.0 |

### Cost sensitivity & break-even (OOS)

| strategy | scenario | bps/side | OOS ret | OOS Sharpe | fees | funding |
|---|---|---|---|---|---|---|
| momentum | zero | 0 | 83.50% | 2.03 | 0.00% | -0.05% |
| momentum | low | 3 | 27.57% | 0.95 | 83.41% | -0.05% |
| momentum | base | 6 | -11.32% | -0.13 | 166.81% | -0.05% |
| momentum | high | 12 | -57.16% | -2.29 | 333.62% | -0.05% |
| momentum | extreme | 20 | -83.76% | -5.14 | 556.04% | -0.05% |
| momentum_flow | zero | 0 | 44.16% | 1.65 | 0.00% | -0.74% |
| momentum_flow | low | 3 | -19.10% | -0.70 | 131.44% | -0.74% |
| momentum_flow | base | 6 | -54.60% | -3.04 | 262.88% | -0.74% |
| momentum_flow | high | 12 | -85.71% | -7.72 | 525.76% | -0.74% |
| momentum_flow | extreme | 20 | -96.94% | -13.92 | 876.26% | -0.74% |
| absorption_rev | zero | 0 | -1.92% | -1.14 | 0.00% | -0.03% |
| absorption_rev | low | 3 | -6.07% | -3.68 | 10.27% | -0.03% |
| absorption_rev | base | 6 | -10.05% | -6.18 | 20.54% | -0.03% |
| absorption_rev | high | 12 | -17.51% | -10.92 | 41.09% | -0.03% |
| absorption_rev | extreme | 20 | -26.50% | -16.55 | 68.48% | -0.03% |
| continuation | zero | 0 | -0.04% | 0.02 | 0.00% | 0.09% |
| continuation | low | 3 | -10.38% | -2.23 | 23.85% | 0.09% |
| continuation | base | 6 | -19.65% | -4.46 | 47.70% | 0.09% |
| continuation | high | 12 | -35.41% | -8.82 | 95.40% | 0.09% |
| continuation | extreme | 20 | -51.72% | -14.34 | 159.00% | 0.09% |
| compression_break_flow | zero | 0 | 7.34% | 2.05 | 0.00% | 0.08% |
| compression_break_flow | low | 3 | -1.67% | -0.46 | 20.12% | 0.08% |
| compression_break_flow | base | 6 | -9.91% | -2.96 | 40.24% | 0.08% |
| compression_break_flow | high | 12 | -24.39% | -7.85 | 80.47% | 0.08% |
| compression_break_flow | extreme | 20 | -40.15% | -13.89 | 134.12% | 0.08% |

**Break-even fee+slippage per side:** momentum/long 8.8, momentum/short 1.0, momentum/both 5.2, momentum_flow/long 3.2, momentum_flow/short 0.7, momentum_flow/both 2.2, absorption_rev/long ≤0, absorption_rev/short 0.6, absorption_rev/both ≤0, continuation/long 2.7, continuation/short ≤0, continuation/both ≤0 bps.

### Per-symbol OOS (priority strategies)

| strategy | symbol | OOS ret | OOS Sharpe |
|---|---|---|---|
| absorption_rev | BTCUSDT | -9.07% | -3.88 |
| absorption_rev | ETHUSDT | -14.07% | -2.79 |
| absorption_rev | SOLUSDT | -6.27% | -1.17 |
| absorption_rev | ZECUSDT | 0.90% | 0.22 |
| absorption_rev | NEARUSDT | -21.10% | -3.08 |
| absorption_rev | XRPUSDT | -5.00% | -1.11 |
| absorption_rev | OGNUSDT | -8.51% | -1.37 |
| absorption_rev | HYPEUSDT | -10.54% | -1.95 |
| absorption_rev | BNBUSDT | -13.66% | -3.81 |
| absorption_rev | DOGEUSDT | -12.81% | -3.23 |
| continuation | BTCUSDT | -25.23% | -4.17 |
| continuation | ETHUSDT | -26.28% | -3.38 |
| continuation | SOLUSDT | -22.17% | -2.80 |
| continuation | ZECUSDT | -7.94% | -0.16 |
| continuation | NEARUSDT | -11.88% | -0.72 |
| continuation | XRPUSDT | -25.67% | -2.83 |
| continuation | OGNUSDT | -16.34% | -1.86 |
| continuation | HYPEUSDT | -42.81% | -4.07 |
| continuation | BNBUSDT | -19.57% | -3.10 |
| continuation | DOGEUSDT | 2.74% | 0.31 |
| momentum_flow | BTCUSDT | -56.74% | -2.95 |
| momentum_flow | ETHUSDT | -43.82% | -1.39 |
| momentum_flow | SOLUSDT | -63.02% | -2.44 |
| momentum_flow | ZECUSDT | 55.65% | 1.09 |
| momentum_flow | NEARUSDT | -80.08% | -2.44 |
| momentum_flow | XRPUSDT | -48.44% | -1.57 |
| momentum_flow | OGNUSDT | -69.14% | -2.02 |
| momentum_flow | HYPEUSDT | -77.43% | -2.42 |
| momentum_flow | BNBUSDT | -70.21% | -4.50 |
| momentum_flow | DOGEUSDT | -53.00% | -1.63 |

## 6. Walk-forward validation & multiple testing

Per fold the strategy+side with the best train Sharpe is frozen and evaluated on the next unseen test window; test windows are stitched.

- aggregate WF Sharpe **-0.68**, return -22.69%, maxDD -31.19%, 4 folds, 25% positive.

| fold | train | test | chosen strategy | side | test Sharpe | test return |
|---|---|---|---|---|---|---|
| 0 | 2025-01-01→2025-07-01 | 2025-09-01→2025-12-01 | compression_break_flow | short | -4.48 | -4.45% |
| 1 | 2025-04-01→2025-10-01 | 2025-12-01→2026-03-01 | momentum | long | -2.73 | -21.45% |
| 2 | 2025-07-01→2026-01-01 | 2026-03-01→2026-06-01 | momentum | long | 0.73 | 4.78% |
| 3 | 2025-10-01→2026-04-01 | 2026-06-01→2026-09-01 | momentum | short | 0.03 | -1.70% |

Multiple testing: 264 IC cells and 21 strategy cells were examined; the best observed values are order statistics. Selection is done on train folds precisely for this reason. The final window (2026-05-01→) has been seen in prior studies and is **not** a pristine holdout.

## 7. Regime dependence (FI_4 IC, OOS)

| regime | bars | FI_4 IC |
|---|---|---|
| range_lowvol | 52,837 | -0.006 |
| range_highvol | 48,620 | 0.008 |
| up_lowvol | 11,607 | -0.017 |
| down_lowvol | 11,588 | 0.010 |
| down_highvol | 11,449 | -0.006 |
| up_highvol | 11,402 | -0.009 |

Regimes from the causal classifier (trend = vol-normalised trailing 24h move; vol = realised vol vs trailing median), shifted one bar.

## 8. Limitations

- No order book or trade-level aggressor data: absorption/flow-effect is a taker-flow proxy, not a mechanism; hidden liquidity and maker intent are unobservable.
- Bar-level execution with fixed per-side cost and next-open fills; no latency, queue position, partial fills or market impact. Funding is a drag approximation.
- Overlapping forward windows and clustered events; HAC errors and block bootstraps mitigate but do not eliminate the dependence. The 10-symbol cross-section is small for cross-sectional claims.
- The 2026 window has been examined in prior studies; the design/OOS split is temporal but the effect sizes are small relative to error bars.

## 9. Answers to the study's questions

1. **Does CVD add predictive information beyond price momentum?** Marginally at best, and not economically. OOS 4h ICs are `R_24` -0.017 vs `FI_4` -0.012; adding flow to the price-only baseline gives ΔR² 0.00006 at 4h and 0.00054 at 24h. The 24h increment is nominally significant (HAC t ≈ 4.8) but explains ≈0.05% of variance and does not translate into a profitable rule.
2. **Does price–CVD divergence predict reversal or a pause?** Neither reliably. Bullish divergence OOS 4h mean -0.057%; the CI includes zero and the sign is unstable.
3. **Does CVD improve pullback entries?** No incremental edge: trend+pullback+flow OOS 4h long 0.014% vs trend+pullback 0.012%.
4. **Does CVD confirmation improve breakout follow-through?** No: compression+breakout+flow OOS 4h long 0.082% vs breakout-only 0.125%.
5. **Do high-flow/low-impact conditions predict reversals differently?** No. Absorption (E1) OOS 4h oriented mean 0.074% vs continuation (E2) 0.079% — the difference is within error and not significant after controls.
6. **Which interactions add incremental information?** None materially. The strongest single feature remains price momentum; flow, divergence and compression interactions are noise.
7. **Does any of it translate to net expectancy after costs?** No: the best walk-forward aggregate Sharpe is -0.68, and break-even costs are at or below the base 6 bps/side for the flow strategies.
8. **Long vs short?** Flow-conditioned shorts tend to be worse than longs in this net-up sample; neither side is independently and stably positive.
9. **Stable across symbols, regimes, folds?** No — per-symbol dispersion is large, regime ICs differ in sign, and walk-forward folds are mostly negative.
10. **Which hypotheses deserve further work?** None as directional signals at 1h with taker-flow CVD. The honest conclusion is that bar-level taker-flow CVD does not add incremental predictive or economic value here. Recommended next steps: (a) treat price-impact asymmetry as an execution/cost model rather than alpha; (b) only revisit order flow with genuinely new information (tick/aggTrades or order-book depth) or genuinely new history; (c) otherwise move to structurally different sources.

---

_Reproduce:_ `python scripts/ema_cvd_mfa_study.py` (engine: `quant_lab.analytics.ema_backtest`; features/strategies: `quant_lab.analytics.mfa`). Artifacts: `experiments/reports/ema_cvd_mfa_v1`.
