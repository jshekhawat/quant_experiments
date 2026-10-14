# EMA9/20 crosses as a directional signal — 10 coins, 1h

- market: futures · timeframe: 1h · signal: EMA9 vs EMA20
- sample: 2025-01-01 → 2026-10-01 · design/test split at 2026-01-01
- bars: 149,533 · cross events: 6,948 (↑ 3,475 cross_above / ↓ 3,473 cross_below)

## Notes
- pooled rank IC(ema_sign, y_24h) = -0.0361
- cross_above hit rate: design 46.6% → test 51.1%

## Executive summary

**Question.** On the 10 top coins, 1h, does an EMA9/20 cross predict the direction of the next move, and how long does that move last?

**Answer.** NO EDGE (H0). At the primary 24h horizon a cross_above is close to a coin flip (48.5% directional hits vs a 49.5% unconditional up-rate) and a cross_below is only marginally better (50.9% directional hits). The bootstrap confidence intervals on the oriented return straddle zero for both directions, so neither cross supports a durable directional edge. Moves do have a characteristic length (median ≈ 15h from a cross to the opposite cross) but the net run is small relative to the noise and positive-skewed: more than half of all regimes finish underwater in the cross direction.

**Headline numbers (24h)**:

| metric | cross_above (↑) | cross_below (↓) |
|---|---|---|
| n | 3,464 | 3,465 |
| directional hit rate | 48.5% | 50.9% |
| mean oriented return | -0.034% | +0.077% |
| median oriented return | -0.094% | +0.076% |
| 95% CI, mean oriented return | [-0.332%, +0.247%] | [-0.179%, +0.322%] |

## Verdict: **NO EDGE (H0)**

| # | criterion | pass |
|---|---|---|
| cross_above: test hit rate > baseline + 2pp | | ❌ |
| cross_above: oriented mean 95% CI low > 0 | | ❌ |
| cross_above: same sign design vs test | | ❌ |
| cross_below: test hit rate > baseline + 2pp | | ❌ |
| cross_below: oriented mean 95% CI low > 0 | | ❌ |
| pooled rank IC > 0.02 | | ❌ |

## Cross frequency per symbol

| symbol | bars | cross_above | cross_below | events/mo | median up bars | median down bars |
|---|---|---|---|---|---|---|
| BNBUSDT | 15,312 | 368 | 368 | 34.6 | 15 | 14 |
| BTCUSDT | 15,312 | 354 | 354 | 33.3 | 14 | 14 |
| DOGEUSDT | 15,312 | 354 | 354 | 33.3 | 14 | 17 |
| ETHUSDT | 15,312 | 369 | 370 | 34.7 | 15 | 14 |
| HYPEUSDT | 11,725 | 264 | 264 | 32.4 | 15 | 16 |
| NEARUSDT | 15,312 | 359 | 358 | 33.7 | 13 | 15 |
| OGNUSDT | 15,312 | 391 | 390 | 36.7 | 13 | 14 |
| SOLUSDT | 15,312 | 339 | 339 | 31.9 | 15 | 17 |
| XRPUSDT | 15,312 | 344 | 344 | 32.4 | 14 | 19 |
| ZECUSDT | 15,312 | 333 | 332 | 31.3 | 15 | 16 |

## Predictive power — oriented forward return (design + test pooled)

`y` is the forward return in the cross direction; `hit` = share with `y > 0`.

| group | horizon | n | hit rate | mean y | median y |
|---|---|---|---|---|---|
| all | 1h | 6,947 | 46.5% | +0.010% | -0.025% |
| all | 4h | 6,946 | 48.4% | +0.022% | -0.023% |
| all | 12h | 6,939 | 49.5% | +0.018% | +0.000% |
| all | 24h | 6,929 | 49.7% | +0.021% | +0.000% |
| all | 3d | 6,910 | 49.9% | +0.057% | +0.000% |
| up | 1h | 3,475 | 46.6% | +0.018% | -0.015% |
| up | 4h | 3,475 | 48.5% | +0.017% | -0.021% |
| up | 12h | 3,472 | 48.4% | -0.044% | -0.045% |
| up | 24h | 3,464 | 48.5% | -0.034% | -0.094% |
| up | 3d | 3,455 | 46.4% | +0.070% | -0.417% |
| down | 1h | 3,472 | 46.3% | +0.002% | -0.032% |
| down | 4h | 3,471 | 48.2% | +0.026% | -0.024% |
| down | 12h | 3,467 | 50.6% | +0.079% | +0.026% |
| down | 24h | 3,465 | 50.9% | +0.077% | +0.076% |
| down | 3d | 3,455 | 53.3% | +0.044% | +0.436% |

Unconditional baseline (all bars): P(up move) at 24h = 49.5%, mean |move| = 3.026%, bars = 149,533.

### Bootstrap 95% CIs on `24h` (block-resampled by timestamp)

| group | mean y CI | hit rate CI |
|---|---|---|
| up | [-0.332%, +0.247%] | [45.2%, 51.7%] |
| down | [-0.179%, +0.322%] | [47.9%, 54.0%] |

## Design vs test consistency

| group | period | n | hit rate | mean y |
|---|---|---|---|---|
| all | design | 3,988 | 49.1% | -0.072% |
| up | design | 1,993 | 46.6% | -0.229% |
| down | design | 1,995 | 51.5% | +0.085% |
| all | test | 2,941 | 50.5% | +0.148% |
| up | test | 1,471 | 51.1% | +0.229% |
| down | test | 1,470 | 50.0% | +0.067% |

## How long the move lasted (cross → opposite cross)

The move is held until the opposite cross fires, so `bars` is the regime length.

| group | n | mean bars | median | p25 | p75 | mean run | median run | share run>0 | median bars→peak | mean MFE | mean MAE | censored |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 6,948 | 21.5 | 15 | 6 | 30 | +0.80% | -0.21% | 35.9% | 5 | 3.62% | 1.52% | 0.1% |
| up | 3,475 | 21.3 | 14 | 6 | 30 | +0.88% | -0.25% | 34.7% | 5 | 3.74% | 1.52% | 0.1% |
| down | 3,473 | 21.7 | 15 | 6 | 31 | +0.72% | -0.18% | 37.1% | 5 | 3.49% | 1.52% | 0.1% |

### Duration by outcome (winning vs losing regimes)

| group | outcome | share | n | median bars | mean bars | mean run | median bars→peak |
|---|---|---|---|---|---|---|---|
| up | win | 35% | 1,205 | 38 | 42.8 | +4.37% | 23 |
| up | loss | 65% | 2,270 | 8 | 9.9 | -0.98% | 2 |
| down | win | 37% | 1,290 | 37 | 41.5 | +3.55% | 23 |
| down | loss | 63% | 2,183 | 8 | 10.0 | -0.96% | 1 |

## Per-symbol breakdown

| symbol | up n | down n | up hit | down hit | up mean y | down mean y | median up bars | median down bars |
|---|---|---|---|---|---|---|---|---|
| BNBUSDT | 368 | 368 | 51.9% | 46.7% | +0.15% | -0.05% | 15 | 14 |
| BTCUSDT | 354 | 354 | 53.4% | 48.9% | +0.11% | -0.07% | 14 | 14 |
| DOGEUSDT | 354 | 354 | 42.9% | 55.4% | -0.65% | +0.75% | 14 | 17 |
| ETHUSDT | 369 | 370 | 50.1% | 48.4% | +0.06% | -0.11% | 15 | 14 |
| HYPEUSDT | 264 | 264 | 54.2% | 50.8% | +0.36% | -0.15% | 15 | 16 |
| NEARUSDT | 359 | 358 | 47.6% | 51.1% | -0.07% | +0.00% | 13 | 15 |
| OGNUSDT | 391 | 390 | 44.2% | 47.7% | -0.40% | +0.08% | 13 | 14 |
| SOLUSDT | 339 | 339 | 47.8% | 53.1% | -0.26% | +0.42% | 15 | 17 |
| XRPUSDT | 344 | 344 | 41.3% | 53.8% | -0.30% | +0.30% | 14 | 19 |
| ZECUSDT | 333 | 332 | 51.7% | 52.7% | +0.83% | -0.45% | 15 | 16 |
