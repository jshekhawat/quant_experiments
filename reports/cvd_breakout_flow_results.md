# H3 — Flow-Confirmed Breakout Continuation — Results (exploratory, 2026)

- universe (62): 1000PEPEUSDT, AAVEUSDT, ADAUSDT, AKEUSDT, ALGOUSDT, APTUSDT, ARBUSDT, ASTERUSDT, AVAXUSDT, BCHUSDT, BNBUSDT, BTCUSDT ...
- timeframe 1h · horizon 4h · lookback 48 · flow_window 4
- breakout events (design+test): 6806

## Notes
- Exploratory: single regime (2026), in-sample; n=62 perp symbols.

## Primary: event pooled rank IC(flow_align_z, oriented return)

| period | IC | n |
|---|---|---|
| design | -0.0296 | 2666 |
| test | -0.0146 | 4140 |

test IC 95% bootstrap CI: [-0.0504, 0.0259]

## Horizon consistency (test event IC)

| horizon | IC |
|---|---|
| 1h | -0.0179 |
| 4h | -0.0146 |
| 24h | -0.0115 |
| 3d | -0.0086 |

## Test quintiles (flow_align_z → return / failure rate)

| qubit | n | mean | median | fail_rate |
|---|---|---|---|---|
| 0 | 828 | 0.00078 | -0.00107 | 0.588 |
| 1 | 828 | 0.00037 | -0.00156 | 0.599 |
| 2 | 828 | 0.00072 | -0.00073 | 0.540 |
| 3 | 828 | 0.00148 | -0.00154 | 0.504 |
| 4 | 828 | 0.00077 | -0.00154 | 0.441 |

## Matched control (high − low alignment)

- spread: 21.83 bps (CI [-27.57, 62.67])
- n: 4140

## Incremental regression (flow_align_z coefficient)

- coef: -0.00093 (CI [-0.00235, 0.00047])

## Per-symbol test event IC

| symbol | IC | n |
|---|---|---|
| 1000PEPEUSDT | -0.0050 | 64 |
| AAVEUSDT | 0.1447 | 68 |
| ADAUSDT | -0.1407 | 72 |
| AKEUSDT | 0.3050 | 53 |
| ALGOUSDT | -0.2408 | 70 |
| APTUSDT | -0.0143 | 75 |
| ARBUSDT | -0.1076 | 65 |
| ASTERUSDT | -0.1422 | 61 |
| AVAXUSDT | 0.0053 | 58 |
| BCHUSDT | 0.2586 | 63 |
| BNBUSDT | -0.0775 | 67 |
| BTCUSDT | -0.0660 | 71 |
| BTWUSDT | 0.0383 | 41 |
| CRVUSDT | -0.0417 | 76 |
| CTSIUSDT | 0.1391 | 62 |
| DOGEUSDT | -0.0807 | 64 |
| DOTUSDT | -0.0537 | 72 |
| DRAMUSDT | 0.1315 | 58 |
| ENAUSDT | 0.0334 | 77 |
| ETHFIUSDT | 0.1782 | 73 |
| ETHUSDT | -0.2388 | 63 |
| FETUSDT | -0.0882 | 75 |
| FILUSDT | -0.1731 | 66 |
| GTCUSDT | -0.2620 | 64 |
| HBARUSDT | -0.2424 | 70 |
| HYPEUSDT | -0.0963 | 75 |
| INJUSDT | 0.0131 | 76 |
| JCTUSDT | -0.0576 | 64 |
| JUPUSDT | -0.1387 | 67 |
| LINKUSDT | -0.0428 | 67 |
| LITUSDT | 0.0506 | 71 |
| LTCUSDT | -0.0608 | 69 |
| METUSDT | -0.0687 | 67 |
| MINAUSDT | -0.1380 | 68 |
| NEARUSDT | 0.0160 | 74 |
| OGNUSDT | -0.0533 | 62 |
| ONDOUSDT | 0.1061 | 72 |
| ONUSDT | 0.0503 | 62 |
| ORCAUSDT | -0.1301 | 67 |
| PENGUUSDT | 0.1000 | 69 |
| PONSUSDT | -0.1000 | 5 |
| PUMPUSDT | 0.0632 | 74 |
| PYTHUSDT | 0.0674 | 72 |
| QNTUSDT | -0.0236 | 70 |
| RAYSOLUSDT | -0.0044 | 73 |
| RLCUSDT | 0.0702 | 72 |
| SANDUSDT | 0.0485 | 64 |
| SOLUSDT | 0.1486 | 74 |
| STRKUSDT | -0.0425 | 67 |
| SUIUSDT | -0.0872 | 66 |
| TAOUSDT | -0.2236 | 67 |
| TIAUSDT | 0.1463 | 69 |
| TRUMPUSDT | 0.1202 | 60 |
| UNIUSDT | 0.0694 | 59 |
| VVVUSDT | -0.0365 | 69 |
| WLDUSDT | -0.0087 | 76 |
| WUSDT | -0.1122 | 70 |
| XLMUSDT | 0.1138 | 72 |
| XRPUSDT | -0.1736 | 68 |
| ZECUSDT | 0.0438 | 73 |
| ZROUSDT | -0.0504 | 75 |
| 龙虾USDT | -0.0960 | 67 |

## Robustness sweep (test event IC)

| timeframe | lookback | flow_window | design_ic | test_ic | n_events |
|---|---|---|---|---|---|
| 15m | 24 | 1 | -0.0076 | -0.0008 | 37610 |
| 15m | 24 | 4 | -0.0185 | -0.0008 | 37610 |
| 15m | 24 | 12 | -0.0110 | -0.0013 | 37610 |
| 15m | 48 | 1 | -0.0177 | -0.0025 | 27866 |
| 15m | 48 | 4 | -0.0257 | -0.0091 | 27866 |
| 15m | 48 | 12 | -0.0166 | -0.0024 | 27866 |
| 15m | 96 | 1 | -0.0060 | -0.0070 | 19251 |
| 15m | 96 | 4 | -0.0127 | -0.0028 | 19251 |
| 15m | 96 | 12 | -0.0125 | -0.0077 | 19251 |
| 1h | 24 | 1 | -0.0163 | -0.0246 | 9340 |
| 1h | 24 | 4 | -0.0437 | -0.0123 | 9340 |
| 1h | 24 | 12 | -0.0560 | -0.0173 | 9340 |
| 1h | 48 | 1 | -0.0073 | -0.0153 | 6806 |
| 1h | 48 | 4 | -0.0296 | -0.0146 | 6806 |
| 1h | 48 | 12 | -0.0787 | 0.0043 | 6806 |
| 1h | 96 | 1 | -0.0218 | -0.0590 | 4651 |
| 1h | 96 | 4 | -0.0500 | -0.0353 | 4651 |
| 1h | 96 | 12 | -0.0773 | -0.0106 | 4651 |
| 4h | 24 | 1 | -0.1763 | 0.0472 | 2343 |
| 4h | 24 | 4 | -0.1740 | 0.0332 | 2343 |
| 4h | 24 | 12 | -0.0831 | 0.0161 | 2343 |
| 4h | 48 | 1 | -0.1549 | 0.0482 | 1720 |
| 4h | 48 | 4 | -0.1561 | 0.0262 | 1720 |
| 4h | 48 | 12 | -0.0759 | 0.0280 | 1720 |
| 4h | 96 | 1 | -0.1985 | 0.0299 | 1178 |
| 4h | 96 | 4 | -0.2076 | 0.0165 | 1178 |
| 4h | 96 | 12 | -0.1273 | 0.0201 | 1178 |

## Verdict: **NO EDGE (H0)**

| # | criterion | pass |
|---|---|---|
| R1 | | ❌ |
| R2 | | ❌ |
| R3 | | ❌ |
| R4 | | ❌ |
| R5 | | ❌ |
| R6 | | ❌ |
| R7 | | ❌ |
