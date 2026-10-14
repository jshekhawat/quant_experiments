# H3 — Flow-Confirmed Breakout Continuation — Results (exploratory, 2026)

- universe (100): 1000PEPEUSDT, AAOIUSDT, AAPLUSDT, AAVEUSDT, ADAUSDT, AKEUSDT, ALGOUSDT, AMDUSDT, APTUSDT, ARBUSDT, ASTERUSDT, AVAXUSDT ...
- timeframe 1h · horizon 4h · lookback 48 · flow_window 4
- breakout events (design+test): 9469

## Notes
- Exploratory: single regime (2026), in-sample; n=100 perp symbols.

## Primary: event pooled rank IC(flow_align_z, oriented return)

| period | IC | n |
|---|---|---|
| design | -0.0107 | 3084 |
| test | -0.0057 | 6385 |

test IC 95% bootstrap CI: [-0.0362, 0.0243]

## Horizon consistency (test event IC)

| horizon | IC |
|---|---|
| 1h | 0.0004 |
| 4h | -0.0057 |
| 24h | -0.0075 |
| 3d | -0.0029 |

## Test quintiles (flow_align_z → return / failure rate)

| qubit | n | mean | median | fail_rate |
|---|---|---|---|---|
| 0 | 1277 | 0.00047 | -0.00049 | 0.561 |
| 1 | 1277 | -0.00090 | -0.00087 | 0.575 |
| 2 | 1277 | 0.00030 | -0.00076 | 0.548 |
| 3 | 1277 | 0.00141 | -0.00065 | 0.504 |
| 4 | 1277 | 0.00059 | -0.00089 | 0.441 |

## Matched control (high − low alignment)

- spread: 6.86 bps (CI [-17.36, 35.86])
- n: 6385

## Incremental regression (flow_align_z coefficient)

- coef: -0.00015 (CI [-0.00094, 0.00060])

## Per-symbol test event IC

| symbol | IC | n |
|---|---|---|
| 1000PEPEUSDT | -0.0050 | 64 |
| AAOIUSDT | -0.1997 | 48 |
| AAPLUSDT | -0.0972 | 66 |
| AAVEUSDT | 0.1447 | 68 |
| ADAUSDT | -0.1407 | 72 |
| AKEUSDT | 0.3050 | 53 |
| ALGOUSDT | -0.2408 | 70 |
| AMDUSDT | -0.0168 | 67 |
| APTUSDT | -0.0143 | 75 |
| ARBUSDT | -0.1076 | 65 |
| ASTERUSDT | -0.1422 | 61 |
| AVAXUSDT | 0.0053 | 58 |
| AVGOUSDT | 0.0340 | 64 |
| BCHUSDT | 0.2586 | 63 |
| BNBUSDT | -0.0775 | 67 |
| BTCUSDT | -0.0660 | 71 |
| BTWUSDT | 0.0383 | 41 |
| BZUSDT | -0.0065 | 71 |
| CBRSUSDT | 0.0884 | 60 |
| CLUSDT | 0.1571 | 71 |
| CRCLUSDT | -0.1246 | 66 |
| CRVUSDT | -0.0417 | 76 |
| CTSIUSDT | 0.1391 | 62 |
| DOGEUSDT | -0.0807 | 64 |
| DOTUSDT | -0.0537 | 72 |
| DRAMUSDT | 0.1315 | 58 |
| ENAUSDT | 0.0334 | 77 |
| ETHFIUSDT | 0.1782 | 73 |
| ETHUSDT | -0.2388 | 63 |
| EWYUSDT | 0.1406 | 64 |
| FETUSDT | -0.0882 | 75 |
| FILUSDT | -0.1731 | 66 |
| GOOGLUSDT | 0.1112 | 67 |
| GTCUSDT | -0.2620 | 64 |
| HBARUSDT | -0.2424 | 70 |
| HYPEUSDT | -0.0963 | 75 |
| INJUSDT | 0.0131 | 76 |
| INTCUSDT | -0.0943 | 69 |
| JCTUSDT | -0.0576 | 64 |
| JUPUSDT | -0.1387 | 67 |
| KORUUSDT | -0.1496 | 37 |
| LINKUSDT | -0.0428 | 67 |
| LITEUSDT | 0.1038 | 63 |
| LITUSDT | 0.0506 | 71 |
| LTCUSDT | -0.0608 | 69 |
| METAUSDT | 0.0511 | 67 |
| METUSDT | -0.0687 | 67 |
| MINAUSDT | -0.1380 | 68 |
| MRVLUSDT | 0.0396 | 58 |
| MSTRUSDT | 0.0384 | 68 |
| MUUSDT | -0.1469 | 71 |
| MUUUSDT | -0.1079 | 29 |
| NATGASUSDT | 0.1294 | 63 |
| NBISUSDT | -0.0327 | 54 |
| NEARUSDT | 0.0160 | 74 |
| NVDAUSDT | -0.0285 | 69 |
| OGNUSDT | -0.0533 | 62 |
| ONDOUSDT | 0.1061 | 72 |
| ONUSDT | 0.0503 | 62 |
| ORCAUSDT | -0.1301 | 67 |
| PAXGUSDT | -0.0465 | 69 |
| PENGUUSDT | 0.1000 | 69 |
| PLTRUSDT | -0.2772 | 63 |
| PONSUSDT | -0.1000 | 5 |
| PUMPUSDT | 0.0632 | 74 |
| PYTHUSDT | 0.0674 | 72 |
| QNTUSDT | -0.0236 | 70 |
| QQQUSDT | -0.1132 | 74 |
| RAYSOLUSDT | -0.0044 | 73 |
| RLCUSDT | 0.0702 | 72 |
| SAMSUNGUSDT | 0.0012 | 49 |
| SANDUSDT | 0.0485 | 64 |
| SKHYNIXUSDT | 0.2589 | 53 |
| SKHYUSDT | 0.5077 | 32 |
| SNDKUSDT | 0.2465 | 68 |
| SNXXUSDT | 0.4794 | 30 |
| SOLUSDT | 0.1486 | 74 |
| SOXLUSDT | -0.1727 | 57 |
| SOXSUSDT | 0.0629 | 27 |
| SPCXUSDT | -0.0283 | 51 |
| SPYUSDT | 0.0452 | 69 |
| STRKUSDT | -0.0425 | 67 |
| SUIUSDT | -0.0872 | 66 |
| TAOUSDT | -0.2236 | 67 |
| TIAUSDT | 0.1463 | 69 |
| TQQQUSDT | 0.1645 | 36 |
| TRUMPUSDT | 0.1202 | 60 |
| TSLAUSDT | 0.0043 | 72 |
| UNIUSDT | 0.0694 | 59 |
| VVVUSDT | -0.0365 | 69 |
| WLDUSDT | -0.0087 | 76 |
| WUSDT | -0.1122 | 70 |
| XAGUSDT | 0.4126 | 65 |
| XAUTUSDT | 0.0617 | 69 |
| XAUUSDT | -0.0425 | 69 |
| XLMUSDT | 0.1138 | 72 |
| XRPUSDT | -0.1736 | 68 |
| ZECUSDT | 0.0438 | 73 |
| ZROUSDT | -0.0504 | 75 |
| 龙虾USDT | -0.0960 | 67 |

## Robustness sweep (test event IC)

| timeframe | lookback | flow_window | design_ic | test_ic | n_events |
|---|---|---|---|---|---|
| 15m | 24 | 1 | -0.0050 | -0.0090 | 51980 |
| 15m | 24 | 4 | -0.0157 | -0.0088 | 51980 |
| 15m | 24 | 12 | -0.0088 | -0.0045 | 51980 |
| 15m | 48 | 1 | -0.0135 | -0.0149 | 38549 |
| 15m | 48 | 4 | -0.0219 | -0.0152 | 38549 |
| 15m | 48 | 12 | -0.0141 | -0.0114 | 38549 |
| 15m | 96 | 1 | -0.0070 | -0.0222 | 26983 |
| 15m | 96 | 4 | -0.0099 | -0.0103 | 26983 |
| 15m | 96 | 12 | -0.0132 | -0.0124 | 26983 |
| 1h | 24 | 1 | -0.0109 | -0.0329 | 12932 |
| 1h | 24 | 4 | -0.0334 | -0.0212 | 12932 |
| 1h | 24 | 12 | -0.0428 | -0.0185 | 12932 |
| 1h | 48 | 1 | 0.0066 | -0.0076 | 9469 |
| 1h | 48 | 4 | -0.0107 | -0.0057 | 9469 |
| 1h | 48 | 12 | -0.0544 | 0.0013 | 9469 |
| 1h | 96 | 1 | -0.0110 | -0.0238 | 6556 |
| 1h | 96 | 4 | -0.0372 | -0.0212 | 6556 |
| 1h | 96 | 12 | -0.0624 | -0.0111 | 6556 |
| 4h | 24 | 1 | -0.1443 | 0.0341 | 3215 |
| 4h | 24 | 4 | -0.1472 | 0.0302 | 3215 |
| 4h | 24 | 12 | -0.0619 | 0.0102 | 3215 |
| 4h | 48 | 1 | -0.1214 | 0.0133 | 2389 |
| 4h | 48 | 4 | -0.1311 | 0.0063 | 2389 |
| 4h | 48 | 12 | -0.0686 | 0.0075 | 2389 |
| 4h | 96 | 1 | -0.1915 | 0.0064 | 1646 |
| 4h | 96 | 4 | -0.2134 | 0.0019 | 1646 |
| 4h | 96 | 12 | -0.1284 | 0.0064 | 1646 |

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
