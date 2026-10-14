# H1 — Price Absorption — Results (exploratory, 2026 in-sample)

- universe (10): BTCUSDT, ETHUSDT, SOLUSDT, ZECUSDT, NEARUSDT, XRPUSDT, OGNUSDT, HYPEUSDT, BNBUSDT, DOGEUSDT
- timeframe 1h · horizon 4h · impact_window 720 · event q0.90

## Notes
- Exploratory: single regime (2026), in-sample; n=10 perp symbols.
- Design IC positive but flipped sign out-of-sample — selection artifact, not an edge.

## Primary: pooled rank IC(A, oriented forward return)

| period | IC | n |
|---|---|---|
| design | 0.0049 | 23830 |
| test | -0.0187 | 36670 |

test IC 95% bootstrap CI: [-0.0379, 0.0007]
strong-flow-only test IC: -0.0047

## Horizon consistency (test IC)

| horizon | IC |
|---|---|
| 1h | -0.0171 |
| 4h | -0.0187 |
| 24h | 0.0014 |
| 3d | -0.0035 |

## Test quintiles (A → oriented return, strong-flow)

| quintile | n | mean | median |
|---|---|---|---|
| 0 | 732 | -0.00018 | 0.00113 |
| 1 | 731 | -0.00079 | 0.00037 |
| 2 | 731 | -0.00057 | 0.00010 |
| 3 | 731 | 0.00008 | 0.00076 |
| 4 | 732 | 0.00049 | 0.00035 |

## Matched control (absorbed − control, strong-flow)

- spread: 17.19 bps (CI [-7.58, 36.96])
- n events: 3657

## Control regression (A coefficient)

- coef: 0.00001 (CI [-0.00039, 0.00042])

## Per-symbol test IC

| symbol | IC | n |
|---|---|---|
| BTCUSDT | -0.0279 | 3667 |
| ETHUSDT | -0.0379 | 3667 |
| SOLUSDT | -0.0207 | 3667 |
| ZECUSDT | -0.0275 | 3667 |
| NEARUSDT | -0.0197 | 3667 |
| XRPUSDT | -0.0149 | 3667 |
| OGNUSDT | -0.0479 | 3667 |
| HYPEUSDT | 0.0203 | 3667 |
| BNBUSDT | 0.0060 | 3667 |
| DOGEUSDT | -0.0170 | 3667 |

## Robustness sweep (test IC; sign/magnitude stability)

| timeframe | window_days | event_q | design_ic | test_ic | strong_ic | n_test |
|---|---|---|---|---|---|---|
| 15m | 15 | 0.90 | 0.0061 | -0.0099 | -0.0169 | 146710 |
| 15m | 15 | 0.95 | 0.0061 | -0.0099 | -0.0205 | 146710 |
| 15m | 30 | 0.90 | 0.0062 | -0.0112 | -0.0227 | 146710 |
| 15m | 30 | 0.95 | 0.0062 | -0.0112 | -0.0243 | 146710 |
| 15m | 60 | 0.90 | 0.0096 | -0.0129 | -0.0251 | 146710 |
| 15m | 60 | 0.95 | 0.0096 | -0.0129 | -0.0320 | 146710 |
| 1h | 15 | 0.90 | 0.0085 | -0.0189 | -0.0077 | 36670 |
| 1h | 15 | 0.95 | 0.0085 | -0.0189 | 0.0118 | 36670 |
| 1h | 30 | 0.90 | 0.0049 | -0.0187 | -0.0047 | 36670 |
| 1h | 30 | 0.95 | 0.0049 | -0.0187 | 0.0130 | 36670 |
| 1h | 60 | 0.90 | 0.0082 | -0.0200 | -0.0147 | 36670 |
| 1h | 60 | 0.95 | 0.0082 | -0.0200 | 0.0067 | 36670 |
| 4h | 15 | 0.90 | -0.0100 | -0.0226 | 0.0067 | 9160 |
| 4h | 15 | 0.95 | -0.0100 | -0.0226 | 0.0626 | 9160 |
| 4h | 30 | 0.90 | -0.0007 | -0.0233 | 0.0077 | 9160 |
| 4h | 30 | 0.95 | -0.0007 | -0.0233 | 0.0323 | 9160 |
| 4h | 60 | 0.90 | -0.0184 | -0.0191 | -0.0158 | 9160 |
| 4h | 60 | 0.95 | -0.0184 | -0.0191 | 0.0262 | 9160 |

## Verdict: **NO EDGE (H0)**

| # | criterion | pass |
|---|---|---|
| P1 | | ❌ |
| P2 | | ❌ |
| P3 | | ❌ |
| P4 | | ❌ |
| P5 | | ❌ |
| P6 | | ❌ |
| P7 | | ❌ |
