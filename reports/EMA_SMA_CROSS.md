# DeepSeek Research Prompt: Bollinger Basis SMA vs EMA Crossover — Quantitative Study

## Objective

Conduct a rigorous quantitative study to determine whether crossovers between the **20-period Simple Moving Average (SMA), used as the Bollinger Bands middle line, and different Exponential Moving Averages (EMAs)** contain predictive information or produce a profitable trading strategy on 1-hour crypto perpetual futures.

The purpose is to discover whether the relative behaviour of the SMA and EMA provides a useful trading signal, whether that signal depends on market conditions, and whether any observed edge survives realistic trading costs and out-of-sample validation.

Do not assume that EMA crossovers work. Treat every proposed signal as a hypothesis to test.

## 1. Inspect the Existing Research Environment

Before writing new code:

1. Inspect the existing project structure, Binance datasets, previous EMA crossover study, backtesting engine, research scripts, and generated reports.
2. Reuse the existing data pipeline and backtesting infrastructure wherever possible.
3. Review the previous EMA crossover study's implementation and methodology so that this experiment remains comparable with its results.
4. Identify reusable functions for OHLCV processing, EMA calculations, crossovers, funding, transaction costs, trade execution, performance metrics, walk-forward testing, and reporting.
5. Verify the exact data schemas and available fields rather than assuming them.

The existing research used Binance USD-M perpetual data for BTCUSDT, ETHUSDT, SOLUSDT, ZECUSDT, NEARUSDT, XRPUSDT, OGNUSDT, HYPEUSDT, BNBUSDT, and DOGEUSDT. The available period was January 2025 through September 2026, with 1-hour candles constructed from 1-minute data.

Use the same dataset and universe where available. Document any differences.

Do not rewrite working infrastructure unnecessarily.

## 2. Core Hypothesis

The Bollinger Bands middle line is an SMA, conventionally calculated over 20 periods.

An EMA gives more weight to recent prices than an SMA of the same period. Their relative positions may contain information about short-term momentum or changing market direction.

Test the following hypotheses:

**H1 — Crossover information:** An EMA crossing above or below SMA(20) contains information about subsequent price direction.

**H2 — Period sensitivity:** The predictive or trading performance depends on the EMA period relative to the fixed SMA(20).

**H3 — Directional asymmetry:** Bullish and bearish crossovers have different predictive value and profitability.

**H4 — Market regimes:** The usefulness of these crossovers depends on whether the market is trending, ranging, or experiencing changing volatility.

**H5 — Bollinger volatility information:** Incorporating Bollinger Band width or changes in band width improves crossover performance.

**H6 — Conditional price information:** Combining crossover signals with price relative to the Bollinger basis provides incremental value beyond the crossover alone.

**H7 — Economic viability:** Any apparent edge survives transaction costs, slippage, funding, and out-of-sample testing.

Evaluate each hypothesis separately. Do not treat correlation, directional accuracy, and profitability as equivalent.

## 3. Indicator Definitions

Use closing prices to calculate the indicators.

### Reference indicator

- Bollinger middle line: SMA(20).
- Bollinger upper band: SMA(20) + 2 × rolling standard deviation over 20 periods.
- Bollinger lower band: SMA(20) − 2 × rolling standard deviation over 20 periods.
- Bollinger bandwidth: (upper band − lower band) / SMA(20).

Use the standard deviation convention employed by the existing implementation if it already has a clearly documented convention, and report it.

Calculate the following EMAs:

`EMA(5), EMA(8), EMA(10), EMA(12), EMA(15), EMA(20), EMA(25), EMA(30), EMA(40), EMA(50), EMA(75), EMA(100), EMA(150), EMA(200)`

Compare each EMA against the fixed SMA(20).

Do not substitute a different SMA period for different EMA candidates in the main experiment.

EMA(20) versus SMA(20) is an important special case: it isolates the effect of exponential versus simple smoothing at the same nominal period.

Warm-up requirements must be handled consistently, especially for EMA(150) and EMA(200).

## 4. Experiment A — Pure SMA/EMA Crossover

This is the primary experiment. Do not introduce Bollinger bandwidth filters or price-position filters into its base signals.

For each EMA period:

**Bullish crossover**

EMA(n) crosses from below SMA(20) to above SMA(20).

**Bearish crossover**

EMA(n) crosses from above SMA(20) to below SMA(20).

Detect actual crossing events, not merely conditions that remain true across several candles.

Evaluate three strategy variants:

### A1. Long-only

Enter long on a bullish crossover. Exit on a bearish crossover.

### A2. Short-only

Enter short on a bearish crossover. Exit on a bullish crossover.

### A3. Long-short

A bullish crossover establishes a long position. A bearish crossover establishes a short position, reversing an existing position when applicable.

Specify what happens to an existing position when a signal occurs, how reversals are executed, and how fees are charged.

Do not assume long and short signals have symmetric predictive power.

For the base experiment, compare candidates using the same execution and accounting assumptions.

## 5. Experiment B — Predictive Analysis Before Trading

Before assessing trading profitability, measure whether the signal contains any information about future returns.

At each crossover event, calculate forward returns over:

- 1 hour
- 4 hours
- 8 hours
- 12 hours
- 24 hours
- 48 hours
- 72 hours

Orient returns in the direction of the crossover. For example, a negative price return after a bearish crossover is a positive direction-aligned outcome.

Report, by EMA period and signal direction:

- Number of events.
- Directional hit rate.
- Mean and median direction-aligned return.
- Return dispersion.
- Confidence intervals.
- Results versus the unconditional market baseline.
- Results versus a suitable no-signal baseline.
- Cross-symbol and cross-period consistency.

Crossover events may overlap in their forward-return windows. Account for temporal dependence and avoid treating overlapping observations as independent samples when estimating uncertainty.

Use these results to distinguish whether the signal predicts direction, predicts the magnitude of a move, or provides little measurable information.

Do not conclude that a strategy is tradable simply because its directional hit rate exceeds 50%.

## 6. Experiment C — Bollinger Bandwidth and Volatility Filters

Determine whether the crossover is more useful under particular volatility conditions.

Test the following conditions separately:

1. High versus low Bollinger bandwidth.
2. Expanding versus contracting bandwidth.
3. Bandwidth breakout from a historically compressed range.
4. Price movement relative to the Bollinger basis.
5. Trending versus ranging market conditions.

Use causal definitions based only on information available at the signal time.

For example, define expanding bandwidth as bandwidth increasing relative to the previous candle. Define high and low bandwidth using rolling historical thresholds, not full-sample quantiles that use future information.

For price-position filtering, compare:

- Longs only when price is above SMA(20).
- Shorts only when price is below SMA(20).
- No price-position restriction.

Explicitly distinguish the market-condition filter from the trade-entry condition.

Do not combine every possible filter into one enormous parameter search immediately.

First establish the base crossover results. Then introduce one filter at a time and measure its incremental contribution.

For each filter, answer:

- Does it improve out-of-sample risk-adjusted returns?
- Does it reduce false signals or merely reduce trade frequency?
- Does it improve expectancy after costs?
- Does it reduce drawdown?
- Does its benefit remain when tested on other coins and periods?

## 7. Experiment D — Trend and Range Classification

Independently investigate whether SMA/EMA crossovers behave differently across market regimes.

Classify conditions using causal information, including:

- Directional price movement over trailing periods.
- Realised volatility.
- Bollinger bandwidth.
- Changes in bandwidth.

Possible regimes include trending upward, trending downward, ranging with low volatility, and ranging with high volatility.

Document precise regime definitions and thresholds before evaluating strategy performance.

Do not use future returns to determine which regime a historical trade belonged to.

Report results separately for each regime, and distinguish results used to formulate hypotheses from results used to validate them.

The objective is to determine whether the crossover has conditional usefulness rather than assuming it should perform equally well in all markets.

## 8. Backtesting and Execution Assumptions

Use the existing Binance USD-M perpetual backtest infrastructure where possible.

Preserve the previous study's execution conventions for comparability:

- Generate signals at candle close t.
- Execute no earlier than candle open t+1.
- Mark portfolio value consistently.
- Apply 6 basis points per side as the initial combined fee-plus-slippage assumption, while retaining configurable costs.
- Account for funding using the available funding dataset and actual applicable timestamps.
- Apply costs to entries, exits, and position reversals as appropriate.
- Handle position sizing, exposure, equity, and returns consistently.
- Prevent duplicate entries from a crossover condition that persists across multiple candles.
- Ensure no future information enters indicator calculations, signal generation, filters, or parameter selection.

The base 6 bps assumption is a starting point, not a claim that it matches every real execution.

Run sensitivity analysis at 0, 3, 6, 12, and 20 bps per side.

Also report break-even transaction costs for promising candidates.

Verify the funding application and cash-flow accounting. If the existing engine approximates funding rather than maintaining a complete cash-flow ledger, document that limitation.

Do not silently change leverage, sizing, accounting, or execution assumptions between candidates.

## 9. Performance Metrics

Report the following metrics consistently:

- Net total return and annualised return.
- Sharpe and Sortino ratios.
- Maximum drawdown and drawdown duration.
- Profit factor.
- Win rate.
- Number of trades.
- Average trade return and expectancy.
- Average holding period.
- Time in market and turnover.
- Long and short contribution.
- Fees, slippage, and funding.
- Break-even transaction cost.
- Results by symbol and calendar period.

Use the same annualisation conventions as the previous study where practical, documenting any differences.

A low win rate does not automatically imply that a strategy is bad, and a high win rate does not automatically imply profitability. Interpret win rate alongside payoff distribution, expectancy, costs, and drawdowns.

Include buy-and-hold and no-trade baselines where appropriate.

## 10. Train, Validation, and Out-of-Sample Testing

Use chronological splits.

Maintain comparability with the previous study's initial design/test split and walk-forward procedure, while documenting the exact date ranges used.

Prefer the following process:

1. Use training data to develop hypotheses and select candidate parameters.
2. Use validation data for limited candidate selection.
3. Freeze the specification before the final untouched test.
4. Roll forward through chronological walk-forward folds.
5. Aggregate out-of-sample performance across folds.
6. Report each fold separately to expose changes in performance.

Test both individual symbols and pooled results.

Do not randomly shuffle observations.

Because many EMA combinations and filters are being investigated, account for multiple testing and selection bias. The more configurations tested, the less meaningful the single best historical result becomes.

If all available data has already been inspected during earlier research, acknowledge that it is not a pristine final holdout.

Do not describe a configuration as validated solely because it has the highest out-of-sample Sharpe among the candidates examined.

## 11. Robustness Testing

For the strongest candidates:

- Test whether nearby EMA periods provide similar performance.
- Compare performance across multiple symbols.
- Examine long-only and short-only performance independently.
- Measure sensitivity to trading costs and funding.
- Determine whether returns are concentrated in a handful of large trades.
- Inspect performance across individual calendar months and market regimes.
- Use block bootstrap or another appropriate statistical method to preserve temporal dependence.
- Quantify uncertainty where feasible.

Prefer a broad region of reasonably performing parameters over a single isolated optimum.

A candidate whose performance disappears after a small parameter change should be treated as fragile.

## 12. Deliverables

Produce a reproducible research report and supporting artefacts.

### A. Main comparison table

For every EMA period and strategy variant, report in-sample and out-of-sample performance using a common evaluation period.

### B. Heatmaps

Generate separate heatmaps for:

- Out-of-sample net return.
- Out-of-sample Sharpe ratio.
- Maximum drawdown.
- Trade frequency.
- Long-only versus short-only results.

### C. Crossover diagnostics

Show forward-return distributions after bullish and bearish crossovers and compare them with unconditional market returns.

### D. Bollinger filter analysis

Quantify how each bandwidth, volatility, or price-position filter changes the baseline strategy's results.

### E. Equity curves

Plot strategy equity alongside the appropriate benchmark, with drawdown curves and clear train/test boundaries.

### F. Regime analysis

Show whether performance is concentrated in specific market regimes, symbols, or periods.

### G. Reproducible implementation

Provide reusable code that can regenerate the results, parameter tables, plots, and experiment logs.

Save the generated report and artefacts using the existing project conventions. Avoid overwriting prior research results without an explicit versioned output location.

## 13. Final Questions the Report Must Answer

Conclude by answering these questions directly:

1. Does an EMA/SMA(20) crossover demonstrate predictive information beyond unconditional market behaviour?
2. Which EMA periods appear most informative, and are they consistent across symbols?
3. Is the signal useful for long-only trading, short-only trading, or both?
4. Does EMA(20) versus SMA(20) behave differently enough to justify treating it as a separate candidate?
5. Does Bollinger bandwidth improve signal quality or trading profitability?
6. Does price relative to the Bollinger basis add incremental predictive value?
7. Are crossovers useful only in specific market regimes?
8. Does any candidate remain profitable after realistic costs and funding?
9. Are positive results robust across nearby periods, symbols, and walk-forward folds?
10. Is there enough evidence to justify further strategy development, or should the crossover hypothesis be rejected?

## Standard of Evidence

Do not optimise for the highest historical return.

A promising candidate should demonstrate positive out-of-sample performance after costs, acceptable drawdowns, consistency across independent periods, and robustness to nearby parameter choices. Confidence intervals and multiple-testing risks must be taken seriously.

If the evidence does not establish a reliable edge, state that clearly.

The objective is to determine whether the relationship between an EMA and the Bollinger basis contains actionable information, not to force a profitable strategy out of historical data.