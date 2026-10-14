# DeepSeek Research Prompt: Multi-Factor Alpha Discovery Using EMA Momentum, CVD, Volatility, and Price Response

## Objective

Conduct a rigorous quantitative research study to determine whether combinations of price momentum, cumulative volume delta (CVD), volatility, and price response contain predictive information that can be converted into profitable crypto perpetual-futures strategies.

The study must investigate five distinct hypotheses:

1. **Momentum + flow confirmation:** Does aggressive order flow confirm or improve price-momentum signals?
2. **Price–CVD divergence:** Does disagreement between price and order flow reveal potential exhaustion or reversal?
3. **Trend + pullback + flow reversal:** Can order flow improve entry timing during pullbacks within an established trend?
4. **Volatility compression + flow impulse:** Does order flow distinguish meaningful breakouts from failed breakouts?
5. **Order-flow pressure + price response:** Does strong aggressive buying or selling that produces little price movement indicate absorption, exhaustion, or impending continuation?

The goal is not to combine as many indicators as possible or find the highest historical return. It is to establish whether each factor contributes incremental predictive information, whether combinations outperform their individual components, and whether any resulting edge survives realistic execution costs and out-of-sample validation.

Treat each hypothesis as falsifiable. Report negative results honestly.

## 1. Inspect the Existing Research Environment

Before implementing anything, inspect the existing project and research infrastructure.

Previous studies have investigated:

- EMA crossover strategies.
- EMA versus SMA(20), the Bollinger Bands middle line.
- EMA rate of change, acceleration, and moving-average spread velocity.
- A causal trend/volatility regime classifier.
- Binance perpetual-futures backtesting with trading costs and funding.

Reuse the existing loaders, indicator calculations, strategy engine, execution model, funding treatment, performance metrics, and validation infrastructure wherever practical.

The existing dataset has covered the following Binance USD-M perpetual markets:

`BTCUSDT, ETHUSDT, SOLUSDT, ZECUSDT, NEARUSDT, XRPUSDT, OGNUSDT, HYPEUSDT, BNBUSDT, DOGEUSDT`

The historical period runs from January 2025 through September 2026 for most symbols. HYPE has a shorter history. One-hour candles are constructed from one-minute data.

Previously reported execution assumptions included signals evaluated at candle close, fills at the following candle open, 6 basis points per side for combined fees and slippage, and funding enabled. Verify these assumptions against the actual implementation.

### Critical data requirement

Inspect the actual available datasets and schemas before choosing the CVD calculation.

Determine whether historical data contains:

- Trade-level prices, quantities, and aggressor-side information.
- Binance kline taker-buy base volume and quote volume.
- Total base volume and quote volume.
- Order-book snapshots or depth updates.
- Funding rates and open interest.

Do not assume trade-level data or order-book data exists.

If the available Binance candles contain taker-buy volume, derive aggressive sell volume consistently from total volume minus taker-buy volume, subject to verification of the field definitions and units. If only OHLCV is available, document that limitation. Do not manufacture order-flow data from candle direction and present it as actual CVD.

If reliable aggressor-side volume is unavailable, run the price-only experiments where possible and clearly mark the order-flow experiments as blocked or proxy-based.

## 2. Define the Core Features

Calculate features using information available at the end of each completed hourly candle.

### A. Price momentum

Calculate log returns over multiple horizons:

\[
R_{k,t}=\log\left(\frac{C_t}{C_{t-k}}\right)
\]

Test initial horizons of:

`1, 3, 6, 12, 24 hours`

Also calculate volatility-normalized price momentum using a trailing volatility estimate.

### B. EMA momentum

Reuse the existing EMA ROC implementation where possible.

Test a limited reference set of EMA configurations, including:

- EMA(10) versus SMA(20).
- EMA(12) versus SMA(20).
- EMA(10) versus EMA(30).
- EMA(20) versus EMA(50).

Calculate EMA rate of change, acceleration, and normalized spread velocity where relevant. These are candidate price features, not presumed sources of edge.

Avoid expanding the EMA search grid unnecessarily before testing whether order flow adds information.

### C. CVD and order-flow imbalance

Where reliable aggressor-side volume exists, calculate:

\[
\Delta V_t=V_{\text{aggressive buy},t}
-V_{\text{aggressive sell},t}
\]

For each rolling horizon \(k\), calculate normalized flow imbalance:

\[
FI_{k,t}=
\frac{\sum_{i=t-k+1}^{t}\Delta V_i}
{\sum_{i=t-k+1}^{t}
\left(V_{\text{aggressive buy},i}
+V_{\text{aggressive sell},i}\right)+\epsilon}
\]

The value should lie approximately between −1 and +1.

Test rolling horizons of:

`1, 4, 12, 24 hours`

Also calculate the cumulative sum of trade delta where useful, but do not use an unadjusted cumulative CVD level as the primary cross-symbol feature. Its magnitude and drift can make comparisons misleading.

Test flow acceleration and changes in imbalance only where they have a clear definition.

Document whether volume is measured in base units, quote units, or notional terms. Keep the chosen convention consistent when comparing symbols.

### D. Volatility and compression

Calculate:

- Bollinger bandwidth based on SMA(20) and a documented 20-bar standard-deviation convention.
- Rolling realised volatility.
- Historical bandwidth percentile.
- Changes in bandwidth.
- Price displacement relative to recent volatility.

Use trailing distributions rather than full-sample quantiles when generating live-usable states.

### E. Price response to aggressive flow

For each rolling horizon, compare aggressive flow with the corresponding price movement.

Define normalized price response:

\[
PR_{k,t}=\frac{R_{k,t}}{RV_{k,t}+\epsilon}
\]

where \(RV_{k,t}\) is a compatible trailing volatility scale.

Investigate conditions where aggressive buying or selling is unusually strong relative to recent history, but normalized price response is unexpectedly small.

Treat this as a proxy for potential absorption, not proof of hidden liquidity or passive market-maker behaviour. Historical candles cannot reveal all resting orders or the intentions of market participants.

## 3. Establish Shared Baselines

Before evaluating combined factors, measure the performance of their individual components.

For each research track, establish:

1. A price-only baseline.
2. An order-flow-only baseline where meaningful.
3. A combined model.
4. An appropriate no-trade baseline.
5. Buy-and-hold or another suitable benchmark when relevant.

Measure predictive performance before simulating trades.

Use the same data windows, symbols, execution assumptions, cost conventions, and portfolio accounting for comparisons.

The key measure is incremental value: does adding CVD improve the price-only baseline, or is it merely duplicating information already contained in price?

## 4. Research Track A — Momentum + Flow Confirmation

### Hypothesis

Price momentum is more informative when supported by aggressive order flow in the same direction.

### Example bullish condition

- Price ROC is positive.
- EMA ROC or another predefined trend feature is positive.
- Rolling CVD imbalance is positive.
- Aggressive buying is elevated relative to its own trailing distribution.

Define the bearish counterpart symmetrically, then allow the data to demonstrate whether the two sides are actually symmetric.

Test the following models separately:

1. Price ROC alone.
2. CVD imbalance alone.
3. Price ROC plus CVD imbalance.
4. Price ROC plus CVD imbalance and a predefined trend condition.
5. Price ROC plus CVD imbalance with an additional flow-strength threshold.

Evaluate both continuous features and a small number of documented threshold-based rules.

Questions to answer:

- Does CVD improve forward-return prediction beyond ROC alone?
- Does it improve trade expectancy, profit factor, or risk-adjusted returns?
- Does confirmation reduce false signals at the expense of missing profitable trades?
- Does the incremental contribution persist across symbols, horizons, and market regimes?

Do not assume that buying pressure confirming upward momentum necessarily predicts continuation. Test it.

## 5. Research Track B — Price–CVD Divergence

### Hypothesis

A new price extreme accompanied by weaker aggressive flow may signal momentum exhaustion or a potential reversal.

### Bullish-exhaustion example

- Price reaches a new low relative to a documented trailing window.
- Aggressive selling or negative CVD momentum is weaker than during an earlier comparable low.
- Price momentum is no longer deteriorating as quickly.

### Bearish-exhaustion example

- Price reaches a new high.
- Aggressive buying or positive CVD momentum is weaker than during an earlier comparable high.
- Price momentum is weakening.

Implement divergence using strictly causal comparisons. If the method compares successive peaks or troughs, the algorithm must not use future candles to identify a pivot at the time the signal is supposedly generated.

Test a simple rolling-window formulation first, then a separately documented swing-point formulation if useful.

Compare:

1. Price reversal signals alone.
2. CVD weakening alone.
3. Price extremes plus CVD divergence.
4. Price extremes plus divergence and EMA momentum deceleration.

Measure subsequent reversal magnitude, continuation probability, adverse movement, and time to any subsequent reversal.

Explicitly test the competing hypothesis that divergence merely identifies temporary pauses during powerful trends.

Do not treat every visual divergence as a tradable event.

## 6. Research Track C — Trend + Pullback + Flow Reversal

### Hypothesis

Order flow can improve entry timing during a pullback within an established trend.

### Example long setup

1. The broader trend is upward, according to a causal and predefined trend classifier.
2. Short-term price momentum weakens during a pullback.
3. CVD improves, suggesting aggressive selling is easing or aggressive buying is returning.
4. A predefined price or momentum confirmation triggers entry.

Construct the bearish mirror case.

Compare the following:

- Trend-following entry without a pullback condition.
- Trend plus price pullback.
- Trend plus CVD reversal.
- Trend plus price pullback and CVD reversal.
- The combined setup with an additional confirmation condition.

Use trailing trend and volatility features only. The classifier must not use future information.

Questions to answer:

- Does CVD improve entry timing relative to price-only pullback strategies?
- Does it reduce maximum adverse excursion?
- Does it improve trade expectancy after costs?
- Does it help across multiple trend regimes or only in certain conditions?
- How much return is sacrificed by waiting for confirmation?

Do not tune the regime classifier and entry thresholds simultaneously against the same evaluation window.

## 7. Research Track D — Volatility Compression + Flow Impulse

### Hypothesis

Order flow helps distinguish breakouts that attract sustained aggressive participation from breakouts that fail quickly.

### Example bullish setup

1. Bollinger bandwidth is unusually low relative to its trailing distribution.
2. Price breaks above a previously defined range.
3. CVD imbalance shifts positively or exhibits an unusually strong positive impulse.

Test the bearish counterpart separately.

Define range boundaries using only information available before the breakout. Specify whether a breakout is based on candle close, high/low, or another explicit rule.

Compare:

- Breakout without a CVD filter.
- CVD impulse without a compression filter.
- Compression plus price breakout.
- Compression plus breakout plus CVD confirmation.
- The same strategies under different volatility conditions.

Measure breakout follow-through, failed-breakout frequency, MFE, MAE, time to failure, and net trading performance.

Investigate whether flow confirmation adds value or merely delays entry until after much of the breakout has already occurred.

Avoid using a future candle to determine whether the initial breakout was genuine.

## 8. Research Track E — Order-Flow Pressure + Price Response

### Hypothesis

The relationship between aggressive flow and resulting price displacement contains information about liquidity conditions, potential absorption, or momentum continuation.

This is the preferred initial research priority because it tests the interaction between two distinct observables: executed aggressive flow and the market's price response.

Test two competing explanations.

### E1. Absorption or exhaustion

- Aggressive buying is unusually strong.
- Price makes relatively little upward progress.
- Alternatively, aggressive selling is unusually strong but price declines very little.
- The subsequent move may reverse if opposing passive liquidity continues to absorb the aggressive orders.

### E2. Flow continuation

- Aggressive buying is strong and price responds positively.
- Aggressive selling is strong and price responds negatively.
- The flow-price relationship indicates that aggressive participants are successfully moving price.

Create a small, predeclared feature set containing:

- Signed flow imbalance.
- Absolute flow imbalance.
- Normalized price response.
- Flow strength relative to its trailing distribution.
- Price momentum over matching windows.

Do not equate strong flow with absorption automatically. Define the absorption candidate through strong aggressive flow combined with unusually small price displacement, and evaluate the subsequent outcome.

Compare:

1. Price response alone.
2. Flow imbalance alone.
3. Flow imbalance plus price response.
4. A model that explicitly separates high-flow/low-impact conditions from high-flow/high-impact conditions.

Measure conditional forward returns, reversal probability, continuation probability, MFE, MAE, and risk-adjusted strategy performance.

A critical comparison is whether high-flow/low-impact conditions have different subsequent returns from high-flow/high-impact conditions after controlling for recent trend, volatility, and direction.

Do not claim direct evidence of passive liquidity, market-maker inventory, or intent without order-book data supporting those conclusions.

## 9. Experiment Sequencing and Factor Ablation

Do not combine all five tracks into one large parameter search.

Implement them as separate, reproducible experiments sharing the same data and execution framework.

Prioritise the order-flow pressure versus price-response experiment first, but run baseline diagnostics for all five tracks.

For each track:

1. Validate data availability and feature construction.
2. Evaluate the individual component features.
3. Evaluate the predefined interactions.
4. Test predictive information on forward returns.
5. Evaluate trading strategies only after the predictive analysis.
6. Perform walk-forward validation.
7. Investigate regime dependence only after the base results are understood.

Perform ablation tests: remove one component from the combined model and measure the change in predictive performance and net trading outcomes.

Use a limited, documented search space. Add parameters only when the earlier results justify further research.

## 10. Predictive Evaluation

At minimum, evaluate forward returns over:

`1h, 4h, 8h, 12h, 24h, 48h, 72h`

Report:

- Number of observations and eligible events.
- Mean and median forward returns.
- Directional hit rate.
- Rank information coefficient or another appropriate dependence measure.
- Return dispersion and confidence intervals.
- MFE and MAE for event-based hypotheses.
- Results versus unconditional price behaviour.
- Incremental predictive information over the price-only baseline.
- Results by symbol, calendar period, and market regime.

Use temporal block bootstrap, heteroskedasticity/autocorrelation-consistent errors, or other suitable methods to account for serial dependence and overlapping forward-return windows.

Do not treat thousands of overlapping hourly observations as thousands of independent experiments.

Distinguish statistical significance from economic significance. A measurable information coefficient does not, by itself, establish a profitable trading edge.

## 11. Trading Strategy and Execution Requirements

Reuse the existing Binance USD-M perpetual backtesting engine and its documented accounting conventions.

Preserve the following assumptions unless source-code inspection identifies a concrete defect:

- Hourly candles resampled from one-minute data.
- Signals evaluated at the candle close.
- Earliest ordinary execution at the next candle open.
- Initial fee-plus-slippage assumption of 6 basis points per side.
- Funding enabled.
- Costs applied to entries, exits, and reversals.
- Consistent equity, exposure, and return accounting.
- Proper feature warm-up and handling of missing values.
- No look-ahead bias.

Run additional cost scenarios at:

`0, 3, 6, 12, 20 bps per side`

If a strategy uses stops, confirm their sequencing and price assumptions can be faithfully simulated with the available data.

Order-book impact, queue position, partial fills, latency, and full liquidation mechanics must not be assumed to be accurately modelled if the engine does not support them.

For each trading strategy, evaluate long-only, short-only, and long-short versions separately.

Report:

- Net total and annualised return.
- Sharpe and Sortino ratios.
- Maximum drawdown and duration.
- Profit factor and win rate.
- Average trade expectancy and holding duration.
- Trade count, turnover, and exposure.
- Fees, slippage, funding, and break-even transaction costs.
- Performance per symbol and per chronological test window.

## 12. Walk-Forward Validation and Multiple Testing

The existing 2025–2026 sample has already been repeatedly examined in previous studies. Its final portion must not be presented as a pristine holdout if it informed research decisions.

Use chronological training, validation, and testing. Freeze feature definitions, thresholds, and selection rules before each held-out window.

Prefer a walk-forward procedure that selects candidates using training data, applies limited validation, and then evaluates the frozen candidate on the next unseen period.

For each track, report:

- Aggregate walk-forward results.
- Results for every individual fold.
- The configuration selected in each fold.
- How parameters and performance change across time.
- The fraction of positive folds.
- Performance versus the matching baseline.

Account for multiple testing across features, horizons, symbols, thresholds, and strategy variants.

A positive result from the best configuration in a large search is not independent evidence of predictive ability.

If no untouched historical data remains, clearly identify the need for future paper trading or genuinely new observations to confirm the result.

## 13. Market-Regime and Cross-Symbol Analysis

Use the existing causal regime classifier where suitable. Regimes should rely only on trailing price movement and volatility information.

Evaluate whether the factor combinations behave differently in:

- Uptrends.
- Downtrends.
- Low-volatility ranges.
- High-volatility ranges.
- Volatility compression and expansion.
- Market-aligned versus market-divergent conditions.

Keep the asset's own regime separate from broader crypto-market context.

Examine each symbol independently as well as the pooled universe. Avoid selecting coins solely because they performed well in the same historical data used to develop the model.

The goal is to determine whether a candidate captures a broadly repeatable effect or a narrow relationship specific to one coin, one regime, or one time window.

## 14. Deliverables

Create a versioned research output without overwriting previous studies.

Produce:

1. **Research report:** Hypotheses, data audit, methodology, results, limitations, and conclusions.
2. **Feature diagnostics:** Distributions, correlations, conditional returns, and incremental information.
3. **Track comparison:** Results for continuation, divergence, trend pullback, compression breakout, and flow-versus-price response.
4. **Ablation table:** The incremental contribution of each feature in the combined hypotheses.
5. **Trade diagnostics:** MFE, MAE, failed signals, holding periods, costs, and representative examples.
6. **Equity curves and drawdowns:** Include matched baselines and clearly marked test periods.
7. **Parameter and regime analysis:** Show whether results generalize across neighbouring configurations, symbols, and conditions.
8. **Reproducible code:** Reuse existing infrastructure and document all feature and execution assumptions.
9. **Experiment log:** Record every configuration, window, threshold, and result examined.

Do not hide negative outcomes, selectively report profitable symbols, or silently omit failed experiments.

## 15. Questions the Final Report Must Answer

Answer these questions directly:

1. Does CVD add predictive information beyond price momentum?
2. Does price–CVD divergence predict genuine reversals or merely temporary pauses?
3. Does CVD improve entries during pullbacks in established trends?
4. Does CVD confirmation improve breakout follow-through?
5. Do high-flow/low-price-impact conditions predict reversals differently from high-flow/high-price-impact conditions?
6. Which factor interactions provide incremental information beyond their individual components?
7. Does the information translate into positive net trading expectancy after fees, slippage, and funding?
8. Are long and short results meaningfully different?
9. Are the findings stable across symbols, market regimes, and walk-forward periods?
10. Which hypotheses deserve further research, which should be rejected, and is there sufficient evidence to justify strategy development?

## Standard of Evidence

The purpose is to discover repeatable relationships between price, aggressive order flow, volatility, and market response.

Prefer simple, interpretable factor combinations that provide incremental out-of-sample value over more elaborate models that merely fit the historical data.

A candidate should not be considered successful solely because of a high historical return, a high hit rate, a plausible market narrative, or a statistically significant but economically negligible correlation.

The strongest evidence would be a reproducible improvement over appropriate baselines, positive net expectancy under realistic execution assumptions, and stable walk-forward performance across independent periods.

If no candidate demonstrates a reliable edge, report that conclusion explicitly. The objective is robust alpha discovery, not forcing a profitable strategy out of the available data.