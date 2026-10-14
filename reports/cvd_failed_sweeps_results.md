# H2 — Failed Liquidity Sweeps — Results (exploratory, 2026 in-sample)

- universe (10): BTCUSDT, ETHUSDT, SOLUSDT, ZECUSDT, NEARUSDT, XRPUSDT, OGNUSDT, HYPEUSDT, BNBUSDT, DOGEUSDT
- timeframe 1h · horizon 4h · lookback 48 · cooldown 24
- sweep events (design+test): 1262

## Notes
- Exploratory: single regime (2026), in-sample; n=10 perp symbols.
- Design IC positive but flipped sign out-of-sample — selection artifact, not an edge.

## Primary: event pooled rank IC(af_z, oriented return)

| period | IC | n |
|---|---|---|
| design | 0.0495 | 508 |
| test | -0.0022 | 754 |

test IC 95% bootstrap CI: [-0.0807, 0.0654]

## Horizon consistency (test event IC)

| horizon | IC |
|---|---|
| 1h | -0.0067 |
| 4h | -0.0022 |
| 24h | 0.0068 |
| 3d | 0.0063 |

## Test quintiles (af_z → oriented return)

| quintile | n | mean | median |
|---|---|---|---|
| 0 | 151 | -0.00027 | -0.00000 |
| 1 | 151 | 0.00019 | 0.00148 |
| 2 | 150 | 0.00034 | 0.00086 |
| 3 | 151 | -0.00069 | 0.00083 |
| 4 | 151 | 0.00002 | 0.00127 |

## Matched control (high-af − low-af sweeps)

- spread: -31.96 bps (CI [-125.36, 46.00])
- n: 754

## Control regression (af_z coefficient)

- coef: 0.00002 (CI [-0.00070, 0.00080])

## Per-symbol test event IC

| symbol | IC | n |
|---|---|---|
| BTCUSDT | 0.1518 | 80 |
| ETHUSDT | -0.1259 | 75 |
| SOLUSDT | 0.0698 | 75 |
| ZECUSDT | -0.1721 | 63 |
| NEARUSDT | -0.1032 | 83 |
| XRPUSDT | -0.0459 | 83 |
| OGNUSDT | 0.0646 | 65 |
| HYPEUSDT | 0.0861 | 85 |
| BNBUSDT | -0.1347 | 74 |
| DOGEUSDT | 0.1634 | 71 |

## Robustness sweep (test event IC)

| timeframe | lookback | flow_window | design_ic | test_ic | n_events |
|---|---|---|---|---|---|
| 15m | 24 | 1 | -0.0445 | 0.0175 | 6485 |
| 15m | 24 | 4 | -0.0235 | -0.0194 | 6485 |
| 15m | 48 | 1 | 0.0072 | 0.0095 | 4993 |
| 15m | 48 | 4 | -0.0157 | 0.0221 | 4993 |
| 15m | 96 | 1 | -0.0035 | -0.0053 | 3605 |
| 15m | 96 | 4 | -0.0145 | 0.0131 | 3605 |
| 1h | 24 | 1 | -0.0047 | 0.0148 | 1683 |
| 1h | 24 | 4 | -0.0790 | 0.0198 | 1683 |
| 1h | 48 | 1 | 0.0495 | -0.0022 | 1262 |
| 1h | 48 | 4 | 0.0098 | -0.0227 | 1262 |
| 1h | 96 | 1 | 0.0525 | -0.0488 | 857 |
| 1h | 96 | 4 | -0.0112 | -0.0566 | 857 |
| 4h | 24 | 1 | -0.0347 | -0.0927 | 399 |
| 4h | 24 | 4 | -0.0051 | -0.0523 | 399 |
| 4h | 48 | 1 | 0.1024 | 0.0256 | 306 |
| 4h | 48 | 4 | 0.1596 | -0.0495 | 306 |
| 4h | 96 | 1 | 0.2299 | -0.0163 | 221 |
| 4h | 96 | 4 | 0.1530 | -0.0860 | 221 |

## Verdict: **NO EDGE (H0)**

| # | criterion | pass |
|---|---|---|
| Q1 | | ❌ |
| Q2 | | ❌ |
| Q3 | | ❌ |
| Q4 | | ❌ |
| Q5 | | ❌ |
| Q6 | | ❌ |
| Q7 | | ❌ |
