# Out-of-Sample Order-Flow Hypotheses — design 2025 / test 2026

- design: 2025-01-01 → 2026-01-01 · test: 2026-01-01 → 2026-10-01
- universe: 58 crypto perps (51 cross-market for divergence)
- These are the frozen H-series methods re-run on a genuine temporal split; verdicts are provisional (2025 vs 2026 is still only two regimes).

# Spot–Perp CVD Divergence — Results (exploratory, in-sample 2026)

- universe (51): AAVEUSDT, ADAUSDT, ALGOUSDT, APTUSDT, ARBUSDT, ASTERUSDT, AVAXUSDT, BCHUSDT, BNBUSDT, BTCUSDT, CRVUSDT, CTSIUSDT, DOGEUSDT, DOTUSDT, ENAUSDT, ETHFIUSDT, ETHUSDT, FETUSDT, FILUSDT, GTCUSDT, HBARUSDT, INJUSDT, JUPUSDT, LINKUSDT, LITUSDT, LTCUSDT, METUSDT, MINAUSDT, NEARUSDT, OGNUSDT, ONDOUSDT, ORCAUSDT, PENGUUSDT, PUMPUSDT, PYTHUSDT, QNTUSDT, RLCUSDT, SANDUSDT, SOLUSDT, STRKUSDT, SUIUSDT, TAOUSDT, TIAUSDT, TRUMPUSDT, UNIUSDT, WLDUSDT, WUSDT, XLMUSDT, XRPUSDT, ZECUSDT, ZROUSDT
- timeframe: 1h · primary horizon: 4h
- window=60 · norm_window=720

## Notes

- Exploratory only: single regime (2026), in-sample, n=51 cross-section.

## Primary: pooled rank IC

| period | pooled IC | cross-sectional IC | n |
|---|---|---|---|
| design | 0.0026 | 0.0001 | 415741 |
| test | 0.0019 | -0.0014 | 327600 |

test IC 95% bootstrap CI: [-0.0091, 0.0120]

## Verdict: **NO EDGE (H0)**

| # | criterion | value | pass |
|---|---|---|---|
| C1 | | test IC 0.0019, CI [-0.0091, 0.0120] | ❌ |
| C2 | | design 0.0026 / test 0.0019 | ✅ |
| C3 | | decile monotonicity -0.212 | ❌ |
| C5 | | top-bottom decile spread -0.44 bps | ❌ |
| C6 | | min leave-one-out IC 0.0009 | ✅ |
| C7 | | IC ex-BTC/ETH 0.0010 | ✅ |

## Test deciles (D → forward return, market-neutral)

| decile | n | mean | median |
|---|---|---|---|
| 0 | 32735 | 0.00008 | -0.00074 |
| 1 | 32735 | 0.00024 | -0.00058 |
| 2 | 32735 | -0.00011 | -0.00065 |
| 3 | 32735 | -0.00009 | -0.00076 |
| 4 | 32735 | -0.00003 | -0.00064 |
| 5 | 32735 | 0.00010 | -0.00048 |
| 6 | 32735 | 0.00001 | -0.00047 |
| 7 | 32735 | -0.00023 | -0.00067 |
| 8 | 32735 | -0.00001 | -0.00068 |
| 9 | 32735 | 0.00004 | -0.00083 |

## Test event study (|D| quintiles)

| bucket | n | mean | median |
|---|---|---|---|
| 0 | 65470 | 0.00016 | -0.00067 |
| 1 | 65470 | -0.00010 | -0.00071 |
| 2 | 65470 | 0.00004 | -0.00056 |
| 3 | 65470 | -0.00011 | -0.00057 |
| 4 | 65470 | 0.00001 | -0.00075 |

## Test leave-one-symbol-out pooled IC

| dropped | IC |
|---|---|
| AAVEUSDT | 0.0013 |
| ADAUSDT | 0.0019 |
| ALGOUSDT | 0.0018 |
| APTUSDT | 0.0028 |
| ARBUSDT | 0.0013 |
| ASTERUSDT | 0.0022 |
| AVAXUSDT | 0.0020 |
| BCHUSDT | 0.0021 |
| BNBUSDT | 0.0016 |
| BTCUSDT | 0.0012 |
| CRVUSDT | 0.0013 |
| CTSIUSDT | 0.0027 |
| DOGEUSDT | 0.0017 |
| DOTUSDT | 0.0018 |
| ENAUSDT | 0.0018 |
| ETHFIUSDT | 0.0028 |
| ETHUSDT | 0.0017 |
| FETUSDT | 0.0027 |
| FILUSDT | 0.0017 |
| GTCUSDT | 0.0026 |
| HBARUSDT | 0.0021 |
| INJUSDT | 0.0019 |
| JUPUSDT | 0.0021 |
| LINKUSDT | 0.0019 |
| LITUSDT | 0.0019 |
| LTCUSDT | 0.0021 |
| METUSDT | 0.0013 |
| MINAUSDT | 0.0023 |
| NEARUSDT | 0.0020 |
| OGNUSDT | 0.0022 |
| ONDOUSDT | 0.0021 |
| ORCAUSDT | 0.0028 |
| PENGUUSDT | 0.0012 |
| PUMPUSDT | 0.0025 |
| PYTHUSDT | 0.0021 |
| QNTUSDT | 0.0013 |
| RLCUSDT | 0.0024 |
| SANDUSDT | 0.0014 |
| SOLUSDT | 0.0016 |
| STRKUSDT | 0.0017 |
| SUIUSDT | 0.0019 |
| TAOUSDT | 0.0009 |
| TIAUSDT | 0.0017 |
| TRUMPUSDT | 0.0019 |
| UNIUSDT | 0.0017 |
| WLDUSDT | 0.0030 |
| WUSDT | 0.0009 |
| XLMUSDT | 0.0024 |
| XRPUSDT | 0.0020 |
| ZECUSDT | 0.0024 |
| ZROUSDT | 0.0017 |


---

# H1 — Price Absorption — Results (exploratory, 2026 in-sample)

- universe (58): 1000PEPEUSDT, AAVEUSDT, ADAUSDT, AKEUSDT, ALGOUSDT, APTUSDT, ARBUSDT, ASTERUSDT, AVAXUSDT, BCHUSDT, BNBUSDT, BTCUSDT, CRVUSDT, CTSIUSDT, DOGEUSDT, DOTUSDT, ENAUSDT, ETHFIUSDT, ETHUSDT, FETUSDT, FILUSDT, GTCUSDT, HBARUSDT, HYPEUSDT, INJUSDT, JCTUSDT, JUPUSDT, LINKUSDT, LITUSDT, LTCUSDT, METUSDT, MINAUSDT, NEARUSDT, OGNUSDT, ONDOUSDT, ONUSDT, ORCAUSDT, PENGUUSDT, PUMPUSDT, PYTHUSDT, QNTUSDT, RAYSOLUSDT, RLCUSDT, SANDUSDT, SOLUSDT, STRKUSDT, SUIUSDT, TAOUSDT, TIAUSDT, TRUMPUSDT, UNIUSDT, VVVUSDT, WLDUSDT, WUSDT, XLMUSDT, XRPUSDT, ZECUSDT, ZROUSDT
- timeframe 1h · horizon 4h · impact_window 720 · event q0.90

## Notes
- Exploratory: single regime (2026), in-sample; n=58 perp symbols.

## Primary: pooled rank IC(A, oriented forward return)

| period | IC | n |
|---|---|---|
| design | -0.0097 | 428326 |
| test | -0.0103 | 379248 |

test IC 95% bootstrap CI: [-0.0206, 0.0014]
strong-flow-only test IC: -0.0161

## Horizon consistency (test IC)

| horizon | IC |
|---|---|
| 1h | -0.0134 |
| 4h | -0.0103 |
| 24h | -0.0039 |
| 3d | -0.0006 |

## Test quintiles (A → oriented return, strong-flow)

| quintile | n | mean | median |
|---|---|---|---|
| 0 | 7651 | 0.00008 | 0.00089 |
| 1 | 7651 | -0.00011 | 0.00044 |
| 2 | 7650 | 0.00018 | 0.00011 |
| 3 | 7651 | 0.00001 | -0.00000 |
| 4 | 7651 | 0.00050 | 0.00029 |

## Matched control (absorbed − control, strong-flow)

- spread: 5.78 bps (CI [-5.37, 17.21])
- n events: 38254

## Control regression (A coefficient)

- coef: 0.00002 (CI [-0.00024, 0.00033])

## Per-symbol test IC

| symbol | IC | n |
|---|---|---|
| 1000PEPEUSDT | -0.0003 | 6547 |
| AAVEUSDT | -0.0046 | 6547 |
| ADAUSDT | -0.0089 | 6547 |
| AKEUSDT | 0.0030 | 6547 |
| ALGOUSDT | 0.0070 | 6547 |
| APTUSDT | -0.0284 | 6547 |
| ARBUSDT | -0.0142 | 6547 |
| ASTERUSDT | -0.0234 | 6547 |
| AVAXUSDT | -0.0203 | 6547 |
| BCHUSDT | -0.0397 | 6547 |
| BNBUSDT | 0.0139 | 6547 |
| BTCUSDT | -0.0161 | 6547 |
| CRVUSDT | -0.0161 | 6547 |
| CTSIUSDT | -0.0345 | 6547 |
| DOGEUSDT | 0.0087 | 6547 |
| DOTUSDT | -0.0062 | 6547 |
| ENAUSDT | -0.0100 | 6547 |
| ETHFIUSDT | -0.0280 | 6547 |
| ETHUSDT | -0.0237 | 6547 |
| FETUSDT | -0.0227 | 6547 |
| FILUSDT | 0.0086 | 6547 |
| GTCUSDT | -0.0023 | 6547 |
| HBARUSDT | -0.0250 | 6547 |
| HYPEUSDT | 0.0054 | 6547 |
| INJUSDT | -0.0059 | 6547 |
| JCTUSDT | -0.0187 | 6547 |
| JUPUSDT | -0.0079 | 6547 |
| LINKUSDT | -0.0111 | 6547 |
| LITUSDT | -0.0033 | 6069 |
| LTCUSDT | -0.0375 | 6547 |
| METUSDT | 0.0238 | 6547 |
| MINAUSDT | -0.0088 | 6547 |
| NEARUSDT | -0.0174 | 6547 |
| OGNUSDT | -0.0355 | 6547 |
| ONDOUSDT | -0.0079 | 6547 |
| ONUSDT | 0.0132 | 6547 |
| ORCAUSDT | -0.0153 | 6547 |
| PENGUUSDT | -0.0265 | 6547 |
| PUMPUSDT | -0.0119 | 6547 |
| PYTHUSDT | -0.0080 | 6547 |
| QNTUSDT | -0.0226 | 6547 |
| RAYSOLUSDT | -0.0333 | 6547 |
| RLCUSDT | -0.0179 | 6547 |
| SANDUSDT | -0.0010 | 6547 |
| SOLUSDT | -0.0121 | 6547 |
| STRKUSDT | -0.0142 | 6547 |
| SUIUSDT | -0.0103 | 6547 |
| TAOUSDT | -0.0057 | 6547 |
| TIAUSDT | -0.0098 | 6547 |
| TRUMPUSDT | -0.0016 | 6547 |
| UNIUSDT | -0.0118 | 6547 |
| VVVUSDT | 0.0002 | 6547 |
| WLDUSDT | 0.0088 | 6547 |
| WUSDT | -0.0078 | 6547 |
| XLMUSDT | -0.0182 | 6547 |
| XRPUSDT | 0.0024 | 6547 |
| ZECUSDT | -0.0163 | 6547 |
| ZROUSDT | -0.0011 | 6547 |

## Robustness sweep (test IC; sign/magnitude stability)

| timeframe | window_days | event_q | design_ic | test_ic | strong_ic | n_test |
|---|---|---|---|---|---|---|
| 15m | 15 | 0.90 | -0.0032 | -0.0069 | -0.0185 | 1518120 |
| 15m | 15 | 0.95 | -0.0032 | -0.0069 | -0.0271 | 1518120 |
| 15m | 30 | 0.90 | -0.0039 | -0.0075 | -0.0215 | 1517160 |
| 15m | 30 | 0.95 | -0.0039 | -0.0075 | -0.0287 | 1517160 |
| 15m | 60 | 0.90 | -0.0038 | -0.0084 | -0.0241 | 1515240 |
| 15m | 60 | 0.95 | -0.0038 | -0.0084 | -0.0346 | 1515240 |
| 1h | 15 | 0.90 | -0.0076 | -0.0105 | -0.0169 | 379488 |
| 1h | 15 | 0.95 | -0.0076 | -0.0105 | -0.0129 | 379488 |
| 1h | 30 | 0.90 | -0.0097 | -0.0103 | -0.0161 | 379248 |
| 1h | 30 | 0.95 | -0.0097 | -0.0103 | -0.0139 | 379248 |
| 1h | 60 | 0.90 | -0.0099 | -0.0115 | -0.0159 | 378768 |
| 1h | 60 | 0.95 | -0.0099 | -0.0115 | -0.0262 | 378768 |
| 4h | 15 | 0.90 | -0.0103 | -0.0163 | -0.0130 | 94830 |
| 4h | 15 | 0.95 | -0.0103 | -0.0163 | -0.0018 | 94830 |
| 4h | 30 | 0.90 | -0.0130 | -0.0176 | -0.0214 | 94770 |
| 4h | 30 | 0.95 | -0.0130 | -0.0176 | -0.0038 | 94770 |
| 4h | 60 | 0.90 | -0.0156 | -0.0173 | -0.0288 | 94650 |
| 4h | 60 | 0.95 | -0.0156 | -0.0173 | -0.0116 | 94650 |

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


---

# H2 — Failed Liquidity Sweeps — Results (exploratory, 2026 in-sample)

- universe (58): 1000PEPEUSDT, AAVEUSDT, ADAUSDT, AKEUSDT, ALGOUSDT, APTUSDT, ARBUSDT, ASTERUSDT, AVAXUSDT, BCHUSDT, BNBUSDT, BTCUSDT, CRVUSDT, CTSIUSDT, DOGEUSDT, DOTUSDT, ENAUSDT, ETHFIUSDT, ETHUSDT, FETUSDT, FILUSDT, GTCUSDT, HBARUSDT, HYPEUSDT, INJUSDT, JCTUSDT, JUPUSDT, LINKUSDT, LITUSDT, LTCUSDT, METUSDT, MINAUSDT, NEARUSDT, OGNUSDT, ONDOUSDT, ONUSDT, ORCAUSDT, PENGUUSDT, PUMPUSDT, PYTHUSDT, QNTUSDT, RAYSOLUSDT, RLCUSDT, SANDUSDT, SOLUSDT, STRKUSDT, SUIUSDT, TAOUSDT, TIAUSDT, TRUMPUSDT, UNIUSDT, VVVUSDT, WLDUSDT, WUSDT, XLMUSDT, XRPUSDT, ZECUSDT, ZROUSDT
- timeframe 1h · horizon 4h · lookback 48 · cooldown 24
- sweep events (design+test): 17109

## Notes
- Exploratory: single regime (2026), in-sample; n=58 perp symbols.

## Primary: event pooled rank IC(af_z, oriented return)

| period | IC | n |
|---|---|---|
| design | -0.0024 | 9353 |
| test | 0.0062 | 7756 |

test IC 95% bootstrap CI: [-0.0201, 0.0354]

## Horizon consistency (test event IC)

| horizon | IC |
|---|---|
| 1h | 0.0018 |
| 4h | 0.0062 |
| 24h | -0.0103 |
| 3d | 0.0090 |

## Test quintiles (af_z → oriented return)

| quintile | n | mean | median |
|---|---|---|---|
| 0 | 1552 | -0.00160 | -0.00000 |
| 1 | 1551 | -0.00051 | 0.00071 |
| 2 | 1551 | -0.00033 | 0.00093 |
| 3 | 1551 | -0.00123 | -0.00021 |
| 4 | 1551 | -0.00063 | 0.00130 |

## Matched control (high-af − low-af sweeps)

- spread: -0.81 bps (CI [-23.10, 23.49])
- n: 7756

## Control regression (af_z coefficient)

- coef: 0.00020 (CI [-0.00031, 0.00081])

## Per-symbol test event IC

| symbol | IC | n |
|---|---|---|
| 1000PEPEUSDT | 0.1654 | 140 |
| AAVEUSDT | 0.0361 | 136 |
| ADAUSDT | -0.0549 | 133 |
| AKEUSDT | -0.0999 | 125 |
| ALGOUSDT | 0.0018 | 147 |
| APTUSDT | -0.0736 | 137 |
| ARBUSDT | -0.0693 | 142 |
| ASTERUSDT | -0.1626 | 135 |
| AVAXUSDT | 0.0357 | 131 |
| BCHUSDT | 0.0707 | 127 |
| BNBUSDT | 0.0513 | 129 |
| BTCUSDT | 0.0441 | 142 |
| CRVUSDT | 0.0128 | 127 |
| CTSIUSDT | 0.0832 | 125 |
| DOGEUSDT | 0.1228 | 131 |
| DOTUSDT | -0.0661 | 137 |
| ENAUSDT | 0.1402 | 138 |
| ETHFIUSDT | -0.0686 | 135 |
| ETHUSDT | 0.0264 | 130 |
| FETUSDT | -0.0255 | 138 |
| FILUSDT | -0.0856 | 117 |
| GTCUSDT | -0.0426 | 105 |
| HBARUSDT | 0.0951 | 143 |
| HYPEUSDT | 0.0262 | 144 |
| INJUSDT | 0.0088 | 140 |
| JCTUSDT | -0.0176 | 127 |
| JUPUSDT | 0.0427 | 130 |
| LINKUSDT | -0.0335 | 134 |
| LITUSDT | 0.0306 | 147 |
| LTCUSDT | 0.0045 | 144 |
| METUSDT | 0.1245 | 142 |
| MINAUSDT | 0.0159 | 131 |
| NEARUSDT | -0.1019 | 148 |
| OGNUSDT | -0.0155 | 112 |
| ONDOUSDT | -0.0701 | 136 |
| ONUSDT | 0.0247 | 132 |
| ORCAUSDT | -0.0407 | 131 |
| PENGUUSDT | 0.0595 | 144 |
| PUMPUSDT | -0.0357 | 136 |
| PYTHUSDT | 0.0363 | 133 |
| QNTUSDT | 0.0684 | 134 |
| RAYSOLUSDT | -0.0331 | 131 |
| RLCUSDT | -0.0695 | 130 |
| SANDUSDT | 0.0460 | 136 |
| SOLUSDT | 0.0902 | 129 |
| STRKUSDT | 0.0305 | 131 |
| SUIUSDT | -0.0056 | 134 |
| TAOUSDT | -0.0150 | 131 |
| TIAUSDT | 0.0184 | 147 |
| TRUMPUSDT | -0.0872 | 121 |
| UNIUSDT | -0.1204 | 131 |
| VVVUSDT | 0.0799 | 138 |
| WLDUSDT | -0.0303 | 130 |
| WUSDT | 0.0872 | 134 |
| XLMUSDT | -0.0364 | 137 |
| XRPUSDT | 0.0042 | 134 |
| ZECUSDT | 0.0114 | 122 |
| ZROUSDT | 0.0841 | 145 |

## Robustness sweep (test event IC)

| timeframe | lookback | flow_window | design_ic | test_ic | n_events |
|---|---|---|---|---|---|
| 15m | 24 | 1 | 0.0029 | 0.0012 | 87425 |
| 15m | 24 | 4 | 0.0007 | 0.0082 | 87425 |
| 15m | 48 | 1 | -0.0104 | 0.0031 | 68195 |
| 15m | 48 | 4 | -0.0021 | 0.0062 | 68195 |
| 15m | 96 | 1 | -0.0028 | 0.0043 | 48808 |
| 15m | 96 | 4 | -0.0092 | 0.0129 | 48808 |
| 1h | 24 | 1 | -0.0086 | 0.0019 | 22394 |
| 1h | 24 | 4 | -0.0292 | 0.0012 | 22394 |
| 1h | 48 | 1 | -0.0024 | 0.0062 | 17109 |
| 1h | 48 | 4 | -0.0183 | 0.0046 | 17109 |
| 1h | 96 | 1 | -0.0005 | 0.0249 | 11899 |
| 1h | 96 | 4 | -0.0154 | 0.0140 | 11899 |
| 4h | 24 | 1 | 0.0072 | -0.0069 | 5473 |
| 4h | 24 | 4 | -0.0355 | -0.0137 | 5473 |
| 4h | 48 | 1 | -0.0219 | -0.0079 | 4192 |
| 4h | 48 | 4 | -0.0269 | -0.0068 | 4192 |
| 4h | 96 | 1 | -0.0168 | 0.0490 | 2931 |
| 4h | 96 | 4 | 0.0362 | 0.0252 | 2931 |

## Verdict: **NO EDGE (H0)**

| # | criterion | pass |
|---|---|---|
| Q1 | | ❌ |
| Q2 | | ❌ |
| Q3 | | ❌ |
| Q4 | | ❌ |
| Q5 | | ❌ |
| Q6 | | ✅ |
| Q7 | | ✅ |


---

# H3 — Flow-Confirmed Breakout Continuation — Results (exploratory, 2026)

- universe (58): 1000PEPEUSDT, AAVEUSDT, ADAUSDT, AKEUSDT, ALGOUSDT, APTUSDT, ARBUSDT, ASTERUSDT, AVAXUSDT, BCHUSDT, BNBUSDT, BTCUSDT ...
- timeframe 1h · horizon 4h · lookback 48 · flow_window 4
- breakout events (design+test): 15467

## Notes
- Exploratory: single regime (2026), in-sample; n=58 perp symbols.
- Design IC positive but flipped sign out-of-sample — selection artifact, not an edge.

## Primary: event pooled rank IC(flow_align_z, oriented return)

| period | IC | n |
|---|---|---|
| design | 0.0203 | 8518 |
| test | -0.0206 | 6949 |

test IC 95% bootstrap CI: [-0.0551, 0.0105]

## Horizon consistency (test event IC)

| horizon | IC |
|---|---|
| 1h | -0.0224 |
| 4h | -0.0206 |
| 24h | -0.0254 |
| 3d | -0.0134 |

## Test quintiles (flow_align_z → return / failure rate)

| qubit | n | mean | median | fail_rate |
|---|---|---|---|---|
| 0 | 1390 | 0.00110 | -0.00056 | 0.555 |
| 1 | 1390 | 0.00148 | -0.00014 | 0.558 |
| 2 | 1389 | 0.00041 | -0.00053 | 0.528 |
| 3 | 1390 | 0.00155 | -0.00055 | 0.484 |
| 4 | 1390 | 0.00003 | -0.00109 | 0.411 |

## Matched control (high − low alignment)

- spread: 11.16 bps (CI [-17.08, 38.67])
- n: 6949

## Incremental regression (flow_align_z coefficient)

- coef: -0.00124 (CI [-0.00250, -0.00006])

## Per-symbol test event IC

| symbol | IC | n |
|---|---|---|
| 1000PEPEUSDT | -0.1357 | 117 |
| AAVEUSDT | 0.0321 | 120 |
| ADAUSDT | -0.0879 | 120 |
| AKEUSDT | 0.2990 | 107 |
| ALGOUSDT | -0.2281 | 125 |
| APTUSDT | 0.0652 | 129 |
| ARBUSDT | -0.0794 | 118 |
| ASTERUSDT | -0.2342 | 110 |
| AVAXUSDT | 0.0249 | 104 |
| BCHUSDT | 0.1377 | 114 |
| BNBUSDT | -0.1420 | 122 |
| BTCUSDT | -0.1234 | 126 |
| CRVUSDT | 0.0196 | 124 |
| CTSIUSDT | -0.0479 | 105 |
| DOGEUSDT | -0.0767 | 112 |
| DOTUSDT | -0.1272 | 122 |
| ENAUSDT | 0.0962 | 131 |
| ETHFIUSDT | 0.1790 | 125 |
| ETHUSDT | -0.0993 | 114 |
| FETUSDT | -0.0539 | 124 |
| FILUSDT | -0.1706 | 112 |
| GTCUSDT | -0.1138 | 102 |
| HBARUSDT | -0.1634 | 121 |
| HYPEUSDT | 0.0162 | 135 |
| INJUSDT | -0.0255 | 122 |
| JCTUSDT | -0.0388 | 114 |
| JUPUSDT | -0.0816 | 121 |
| LINKUSDT | -0.0178 | 121 |
| LITUSDT | 0.0538 | 122 |
| LTCUSDT | -0.0854 | 121 |
| METUSDT | -0.1601 | 115 |
| MINAUSDT | -0.0152 | 114 |
| NEARUSDT | 0.0288 | 120 |
| OGNUSDT | -0.0803 | 102 |
| ONDOUSDT | -0.0326 | 123 |
| ONUSDT | -0.0148 | 113 |
| ORCAUSDT | -0.0969 | 117 |
| PENGUUSDT | 0.2045 | 119 |
| PUMPUSDT | 0.1491 | 130 |
| PYTHUSDT | -0.0080 | 123 |
| QNTUSDT | -0.0357 | 123 |
| RAYSOLUSDT | -0.0043 | 130 |
| RLCUSDT | -0.0184 | 130 |
| SANDUSDT | 0.0867 | 125 |
| SOLUSDT | -0.0351 | 127 |
| STRKUSDT | 0.0062 | 126 |
| SUIUSDT | 0.0095 | 116 |
| TAOUSDT | -0.1479 | 117 |
| TIAUSDT | 0.0552 | 123 |
| TRUMPUSDT | 0.0743 | 111 |
| UNIUSDT | -0.0169 | 107 |
| VVVUSDT | -0.0057 | 129 |
| WLDUSDT | -0.0576 | 123 |
| WUSDT | -0.0253 | 124 |
| XLMUSDT | 0.0624 | 122 |
| XRPUSDT | -0.0348 | 118 |
| ZECUSDT | -0.0187 | 127 |
| ZROUSDT | -0.0893 | 135 |

## Robustness sweep (test event IC)

| timeframe | lookback | flow_window | design_ic | test_ic | n_events |
|---|---|---|---|---|---|
| 15m | 24 | 1 | -0.0037 | -0.0059 | 83737 |
| 15m | 24 | 4 | -0.0024 | -0.0094 | 83737 |
| 15m | 24 | 12 | -0.0017 | -0.0064 | 83737 |
| 15m | 48 | 1 | 0.0080 | -0.0108 | 63154 |
| 15m | 48 | 4 | 0.0076 | -0.0171 | 63154 |
| 15m | 48 | 12 | 0.0066 | -0.0120 | 63154 |
| 15m | 96 | 1 | 0.0196 | -0.0086 | 43886 |
| 15m | 96 | 4 | 0.0283 | -0.0060 | 43886 |
| 15m | 96 | 12 | 0.0198 | -0.0101 | 43886 |
| 1h | 24 | 1 | 0.0345 | -0.0198 | 20927 |
| 1h | 24 | 4 | 0.0167 | -0.0269 | 20927 |
| 1h | 24 | 12 | 0.0291 | -0.0321 | 20927 |
| 1h | 48 | 1 | 0.0236 | -0.0144 | 15467 |
| 1h | 48 | 4 | 0.0203 | -0.0206 | 15467 |
| 1h | 48 | 12 | 0.0207 | -0.0268 | 15467 |
| 1h | 96 | 1 | 0.0400 | -0.0412 | 10624 |
| 1h | 96 | 4 | 0.0406 | -0.0374 | 10624 |
| 1h | 96 | 12 | 0.0296 | -0.0303 | 10624 |
| 4h | 24 | 1 | 0.0130 | -0.0552 | 5140 |
| 4h | 24 | 4 | 0.0249 | -0.0501 | 5140 |
| 4h | 24 | 12 | -0.0323 | -0.0201 | 5140 |
| 4h | 48 | 1 | 0.0301 | -0.0377 | 3820 |
| 4h | 48 | 4 | 0.0338 | -0.0449 | 3820 |
| 4h | 48 | 12 | -0.0055 | -0.0193 | 3820 |
| 4h | 96 | 1 | 0.0372 | -0.0615 | 2602 |
| 4h | 96 | 4 | 0.0242 | -0.0658 | 2602 |
| 4h | 96 | 12 | -0.0489 | -0.0348 | 2602 |

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


---
