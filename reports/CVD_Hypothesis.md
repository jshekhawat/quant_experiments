# Task: Research and Backtest Historical CVD–Price Relationships

You are working on my quantitative crypto research project. Your task is to investigate whether historical Cumulative Volume Delta (CVD) contains useful information about future price behaviour, beyond what price action alone can provide.

Do not assume CVD has predictive power. The objective is to formulate falsifiable hypotheses, implement rigorous experiments, and retain negative results.

## 1. Project context

I am building quantitative research infrastructure for crypto markets, with an emphasis on order flow, market microstructure, and trading perpetual futures.

Existing infrastructure includes:

- Python-based research and analytics.
- Historical Binance spot and perpetual futures data.
- One-minute OHLCV klines with taker-buy volume and taker-buy quote volume.
- Derived spot and futures CVD series.
- Funding-rate and open-interest datasets.
- Existing resampling, walk-forward splits, regime classification, and statistical evaluation utilities.
- A Rust-based charting and market-data project called AlphaFlow.

Relevant research documents:
- `spot_cvd_divergence_study.md`
- `spot_cvd_divergence_results.md`

The previous study tested whether standardized spot-flow minus perp-flow predicted market-neutral forward returns.

Its primary result was negative:
- Design-period pooled rank IC: +0.0248.
- Test-period pooled rank IC: -0.0040.
- Test-period 95% bootstrap CI: [-0.0239, 0.0170].
- Test-period top–bottom decile spread: -11.84 bps.
- The preregistered criteria for declaring an edge were not met.

The study used ten crypto assets, one-hour bars, a four-hour primary prediction horizon, and an exploratory 2026 sample.

**Do not try to rescue the failed hypothesis by changing parameters until the results look profitable.** Treat it as a completed negative result. The new experiments must test materially different hypotheses.

First inspect the repository, the two research documents, and the existing data/analytics implementations. Reuse existing infrastructure wherever appropriate. Do not assume that files or APIs exist merely because they are mentioned here.

## 2. Research objective

Investigate the relationship between aggressive buying/selling and the resulting price response.

The central question is:

**Does the effectiveness of aggressive order flow in moving price contain information about subsequent price behaviour?**

CVD measures net taker flow, not the full order book. We cannot directly observe hidden liquidity, passive-order intent, or the identity of participants from CVD alone. Treat absorption and exhaustion as observable proxies, not established explanations of market behaviour.

## 3. Candidate hypotheses

Evaluate these hypotheses for feasibility, distinctness, data requirements, and potential leakage before implementing them.

### H1 — Price absorption

When aggressive selling produces significantly less downward price movement than expected, does price subsequently outperform comparable high-selling-pressure events?

Construct an absorption proxy using signed quote delta, normalized flow intensity, and price displacement. Estimate expected price impact using past data only, ideally controlling for volatility, recent returns, time horizon, and prevailing market regime.

Compare:
- Strong selling with unusually weak downside price response.
- Comparable strong selling with a substantial downside response.
- Matched control periods with similar volatility, trend, and liquidity proxies.

Test forward returns and maximum favourable/adverse excursions.

Repeat symmetrically for aggressive buying with unusually weak upward price response.

Important: A near-zero flow denominator makes simple price-impact ratios unstable. Handle this explicitly. Prefer conditional or regression-based estimates of expected price impact over arbitrary ratios.

### H2 — Failed liquidity sweeps

When price sweeps a previously established swing high or low and closes back inside the prior range, does the magnitude and direction of aggressive flow predict reversal strength?

Examples:
- Aggressive buying during a failed upside breakout.
- Aggressive selling during a failed downside breakdown.

Compare against otherwise similar failed sweeps without extreme directional flow.

Define swing levels and events using only information available at signal time. Avoid look-ahead when identifying swing points.

### H3 — Flow-confirmed breakout continuation

Do breakouts supported by persistent same-direction aggressive flow exhibit better continuation than breakouts with weak flow or immediate flow reversal?

Define the breakout mechanically using historical price ranges. Measure flow persistence, breakout displacement, failure rates, subsequent returns, and adverse excursions.

Compare against a price-only breakout baseline.

### H4 — Aggressive-flow exhaustion

After a substantial directional price move, does declining price progress per unit of aggressive flow predict weakening momentum or reversal?

Test whether changes in flow effectiveness add information beyond:
- Past returns and trend strength.
- Volatility.
- Recent flow magnitude.
- Distance from established price levels.

Distinguish momentum deceleration from an actual directional reversal.

### H5 — Directional price-impact asymmetry

Does comparable normalized aggressive buying versus selling produce different price impacts, conditional on market regime?

Estimate the relationship between signed quote delta and contemporaneous and subsequent price displacement. Separate the explanatory, same-bar relationship from genuine predictive relationships.

Test whether the asymmetry remains after controlling for volatility, returns, and flow magnitude.

This hypothesis may be valuable as a market-state or execution-cost model even if it is not a profitable directional signal.

### H6 — Spot-flow lead–lag

Does lagged spot taker flow contain incremental predictive information about subsequent perpetual-futures returns after controlling for contemporaneous and lagged perp flow?

Test prespecified short lags and horizons. Consider whether any relationship depends on:
- Funding rates.
- Open-interest changes.
- Trend and volatility regimes.
- Spot/perp flow agreement or disagreement.

Avoid merely recreating the failed spot–perp divergence study with slightly different normalization.

## 4. Recommended priority

Begin by investigating H1, price absorption, as the first experiment. It appears to provide a distinct and testable question compared with the failed broad divergence study.

Before writing significant code, produce a short design document specifying:
1. The exact hypothesis and economic rationale.
2. The event definitions and formulas.
3. The required input data.
4. The baseline and matched-control construction.
5. The primary outcome and prediction horizon.
6. Potential confounders and sources of leakage.
7. The statistical evaluation and cost assumptions.
8. The preregistered pass/fail criteria.
9. Whether the hypothesis can be tested with the existing data.

Select a primary specification in advance. Treat other horizons and thresholds as robustness tests, report the full sweep, and account for multiple testing. Do not choose the specification based on test-period performance.

After documenting the design, implement and test H1. Do not implement all six hastily or combine them into a single opaque signal.

## 5. Data and feature requirements

The existing one-minute Binance klines provide:

- Total volume.
- Taker-buy base volume.
- Total quote volume.
- Taker-buy quote volume.

For each bar, signed taker flow can be reconstructed as:

`delta = 2 * taker_buy_volume - volume`

`quote_delta = 2 * taker_buy_quote_volume - quote_volume`

The same calculations can be made in quote currency.

Aggregate signed bar-level flow over trailing windows as needed. Normalize using trailing distributions or volatility/activity-adjusted estimates. Every feature at time t must be computable using information available at t.

Do not use the anchored absolute CVD value as a stationary predictor without justification.

Tick-level trade downloads are not initially authorized or required. Determine whether the existing one-minute bars suffice before proposing a more expensive data collection exercise.

Be careful with:
- Base-volume versus quote-volume units.
- Spot versus perp market differences.
- Bar-boundary alignment.
- Stablecoin/quote-currency consistency.
- Missing bars and abnormal volume.
- The distinction between contemporaneous price impact and subsequent predictive power.

## 6. Backtesting and statistical requirements

Use the existing project conventions where sound.

Minimum requirements:

**Causal feature construction**
- Use trailing windows only.
- No future information in swing detection, normalization, matched controls, or regime assignment.
- Enter no earlier than the bar following signal confirmation.
- Verify forward-return alignment explicitly.

**Baselines**
- Compare every candidate against a price-action-only baseline.
- Where appropriate, compare against a simple flow-only baseline.
- Include matched-event comparisons to distinguish the proposed effect from ordinary trend and volatility effects.

**Evaluation**
- Report sample size and event frequency.
- Report mean and median forward returns.
- Report forward-return distributions and confidence intervals.
- Measure MFE and MAE.
- Evaluate performance after fees and slippage using defensible assumptions.
- Account for overlapping outcome windows and serial dependence, using block bootstrap or another appropriate method.
- Include symbol-level and regime-level stability checks.
- Use chronological development and test periods.
- Reserve genuine out-of-sample data whenever available; determine the existing history coverage before deciding the split.

If data only covers one market regime or one year, label the result exploratory. Do not claim generality across regimes that were not tested.

**Incremental value**

The principal question is not whether the raw feature correlates with returns. It is whether it improves prediction, trade selection, or risk-adjusted outcomes relative to a reasonable baseline after costs.

Define the economic and statistical pass/fail criteria before examining test results. Report negative or inconclusive results honestly.

## 7. Implementation expectations

Inspect the current repository structure before changing anything.

- Reuse existing data loaders, resampling logic, statistics, event-study functions, walk-forward utilities, and regime analytics where appropriate.
- Avoid duplicating established functionality.
- Follow existing naming, typing, logging, error-handling, and test conventions.
- Add deterministic tests for feature construction, causality, forward-return alignment, edge cases, and synthetic positive/negative examples where suitable.
- Do not silently change existing research code or overwrite prior reports.
- Keep the research implementation separate from the live trading or execution path.
- Make the experiment reproducible from a documented CLI command or script.
- Generate a Markdown report and machine-readable outputs for further analysis.

Inspect existing test and report conventions before choosing module names or output paths.

## 8. Deliverables

Produce the following in order:

1. **Repository assessment:** Relevant existing code, datasets, reusable utilities, and data limitations.
2. **Research design:** A concise, frozen H1 specification with causal definitions and explicit pass/fail criteria.
3. **Implementation:** Reusable Python functions for constructing the absorption proxy, identifying events, measuring outcomes, and comparing against baselines.
4. **Automated tests:** Including causality and alignment checks.
5. **Experiment results:** A reproducible report with descriptive statistics, uncertainty estimates, baseline comparisons, robustness checks, costs, and a clear verdict.
6. **Next-step recommendation:** Based on the evidence, recommend whether to investigate H2–H6, improve the price-impact model, acquire additional data, or abandon this line of research.

Do not fabricate test results. If datasets are missing, APIs are unclear, or a required experiment cannot run, report the blocker precisely rather than substituting illustrative results.

## 9. Working principle

Approach this as quantitative research, not indicator development.

The objective is not to create a visually appealing CVD signal. It is to determine whether measurable differences between aggressive order flow and its price impact provide robust incremental information about future market behaviour.

Start by inspecting the actual codebase and both research documents. Present your findings and proposed H1 specification before making substantial implementation changes.