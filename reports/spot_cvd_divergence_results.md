# Spot–Perp CVD Divergence — Results (exploratory, in-sample 2026)

- universe (10): BTCUSDT, ETHUSDT, SOLUSDT, ZECUSDT, NEARUSDT, XRPUSDT, OGNUSDT, HYPEUSDT, BNBUSDT, DOGEUSDT
- timeframe: 1h · primary horizon: 4h
- window=60 · norm_window=720

## Notes

- Exploratory only: single regime (2026), in-sample, n=10 cross-section.
- Design IC was positive and above threshold but flipped sign out-of-sample — a selection artifact (the breakout study's failure mode), not an edge.

## Primary: pooled rank IC

| period | pooled IC | cross-sectional IC | n |
|---|---|---|---|
| design | 0.0248 | 0.0064 | 25920 |
| test | -0.0040 | -0.0037 | 33205 |

test IC 95% bootstrap CI: [-0.0239, 0.0170]

## Verdict: **NO EDGE (H0)**

| # | criterion | value | pass |
|---|---|---|---|
| C1 | | test IC -0.0040, CI [-0.0239, 0.0170] | ❌ |
| C2 | | design 0.0248 / test -0.0040 | ❌ |
| C3 | | decile monotonicity -0.067 | ❌ |
| C5 | | top-bottom decile spread -11.84 bps | ❌ |
| C6 | | min leave-one-out IC -0.0070 | ❌ |
| C7 | | IC ex-BTC/ETH -0.0068 | ❌ |

## Test deciles (D → forward return, market-neutral)

| decile | n | mean | median |
|---|---|---|---|
| 0 | 3301 | 0.00074 | -0.00031 |
| 1 | 3300 | -0.00046 | -0.00059 |
| 2 | 3300 | 0.00004 | -0.00030 |
| 3 | 3300 | -0.00046 | -0.00053 |
| 4 | 3301 | -0.00014 | -0.00030 |
| 5 | 3300 | 0.00011 | -0.00018 |
| 6 | 3300 | 0.00079 | 0.00003 |
| 7 | 3300 | 0.00001 | -0.00025 |
| 8 | 3300 | -0.00015 | -0.00071 |
| 9 | 3301 | -0.00044 | -0.00048 |

## Test event study (|D| quintiles)

| bucket | n | mean | median |
|---|---|---|---|
| 0 | 6601 | 0.00014 | -0.00045 |
| 1 | 6600 | -0.00021 | -0.00041 |
| 2 | 6601 | -0.00002 | -0.00025 |
| 3 | 6600 | 0.00040 | -0.00012 |
| 4 | 6601 | -0.00030 | -0.00061 |

## Test leave-one-symbol-out pooled IC

| dropped | IC |
|---|---|
| BTCUSDT | -0.0070 |
| ETHUSDT | -0.0037 |
| SOLUSDT | -0.0050 |
| ZECUSDT | -0.0011 |
| NEARUSDT | -0.0057 |
| XRPUSDT | -0.0048 |
| OGNUSDT | -0.0024 |
| HYPEUSDT | -0.0040 |
| BNBUSDT | -0.0022 |
| DOGEUSDT | -0.0037 |
