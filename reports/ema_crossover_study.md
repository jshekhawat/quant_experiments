# Research Study: EMA Crossover Strategies on Binance Data

## Objective

Conduct a rigorous quantitative study to determine whether EMA crossover strategies generate a statistically and economically meaningful trading edge on **1-hour crypto candles**, accounting for transaction costs, market regimes, and out-of-sample performance.

The goal is not to find the parameter combination with the highest historical returns. The goal is to understand **which EMA configurations, if any, generalize to unseen data and under what market conditions**.

## 1. Inspect the Data

Before implementing anything:

- Inspect the available Binance datasets, directory structure, schemas, symbols, date ranges, candle intervals, and storage formats.
- Determine whether the data contains OHLCV only or also trades, order books, funding rates, and other useful fields.
- Check for missing candles, duplicate timestamps, malformed OHLCV values, timezone inconsistencies, and discontinuities.
- Determine whether native 1-hour candles are available. If only lower-timeframe candles exist, construct 1-hour candles carefully.
- Document all data-quality issues and any assumptions required.

Do not silently discard problematic data or assume fields exist when they do not.

## 2. Define the Hypotheses

Test the following hypotheses independently:

**H1 — Crossover profitability:** EMA crossovers produce positive net returns after transaction costs.

**H2 — Parameter sensitivity:** Some EMA period pairs deliver more robust results than others, and their performance is not confined to one narrow parameter combination.

**H3 — Trend filter:** Filtering crossover trades using EMA 200 improves risk-adjusted returns or reduces drawdowns.

**H4 — Directional asymmetry:** Long-only, short-only, and long-short strategies exhibit materially different performance.

**H5 — Regime dependency:** EMA crossovers perform differently during trending, ranging, high-volatility, and low-volatility markets.

Do not assume these hypotheses are true.

## 3. Strategy Definitions

Use the EMA calculated from candle closing prices.

Evaluate these initial candidate pairs:

- 9 / 21
- 12 / 26
- 20 / 50
- 21 / 55
- 50 / 200

Then evaluate a broader parameter grid:

- Fast EMA: 5–40 periods.
- Slow EMA: 20–200 periods.
- Require fast EMA < slow EMA.
- Use sensible increments to control the number of experiments, and document the search space before running it.

For each parameter pair, test the following strategy variants:

**A. Long-only**
- Enter long when the fast EMA crosses above the slow EMA.
- Exit when the fast EMA crosses below the slow EMA.

**B. Short-only**
- Enter short when the fast EMA crosses below the slow EMA.
- Exit when the fast EMA crosses above the slow EMA.

**C. Long-short**
- Cross above: enter long or reverse from short to long.
- Cross below: enter short or reverse from long to short.

**D. Trend-filtered variants**
- Permit longs only when price is above EMA 200.
- Permit shorts only when price is below EMA 200.
- Test the filter both as a trade-entry condition and as a market-regime filter. Clearly distinguish these implementations.

Document precisely how positions behave when a crossover occurs while a trend filter blocks the signal.

## 4. Backtest Correctly

Prevent look-ahead bias and unrealistic execution.

- Calculate signals using information available at the candle close.
- Execute trades no earlier than the next candle's open, unless actual finer-grained execution data supports another method.
- Include trading fees and configurable slippage.
- If funding-rate data is available and the strategy trades perpetuals, account for funding using timestamps and applicable rates. If unavailable, report that perpetual-return estimates exclude funding.
- Model position reversals, position sizing, cash and equity accounting consistently.
- Avoid accidental repeated entries when a crossover signal remains true.
- Account for warm-up periods required to initialize longer EMAs.
- Handle missing candles and market closures explicitly.
- Ensure that indicators and trade decisions never use future information.

Test several configurable transaction-cost scenarios. Record the fee and slippage assumptions and calculate the break-even cost level at which each strategy stops being profitable.

## 5. Validation Methodology

Use chronological train, validation, and test periods. Never randomly shuffle time-series observations.

Prefer walk-forward evaluation:

1. Select parameters using only the training window.
2. Evaluate or select between candidates using the validation window.
3. Freeze the chosen configuration and evaluate it on the next unseen test window.
4. Roll forward and repeat.
5. Aggregate the out-of-sample results.

Keep the final test data untouched during parameter selection. If the dataset permits, report results across multiple distinct market regimes.

Because a large parameter grid can generate false discoveries, account for multiple testing and parameter-selection bias. Do not present the best in-sample configuration as evidence of a genuine edge.

## 6. Performance Metrics

Report, at minimum:

- Total net return and annualized return.
- Sharpe ratio and Sortino ratio, with calculation conventions documented.
- Maximum drawdown and drawdown duration.
- Profit factor and win rate.
- Number of trades and average holding duration.
- Average return per trade and expectancy.
- Exposure time and turnover.
- Long versus short contribution.
- Fees, slippage, and funding costs, where available.
- Performance across different cost assumptions.
- In-sample versus out-of-sample performance.

Use consistent return and annualization conventions. Explain limitations when the available history is too short for reliable annualized statistics.

Include buy-and-hold as a baseline and, where appropriate, cash or no-trade as a baseline. Compare strategies over matching evaluation periods.

## 7. Market-Regime Analysis

Investigate whether crossover performance depends on market conditions.

Evaluate performance during:

- Trending versus ranging markets.
- High- versus low-volatility periods.
- Bull versus bear markets.
- Large directional moves versus choppy periods.

Use objective, explicitly documented regime definitions. Avoid defining regimes using future price information.

Show which regimes contribute profits and which generate losses. Determine whether the apparent edge survives when evaluated across different periods rather than being driven by a few exceptional trades.

## 8. Robustness and Statistical Analysis

For the most promising candidates:

- Check whether nearby EMA period combinations perform similarly.
- Measure how sensitive results are to fees and slippage.
- Evaluate performance across available symbols, while separating per-symbol findings from pooled results.
- Inspect the distribution of trade returns and the concentration of profits in a small number of trades.
- Use appropriate bootstrap or other statistical procedures where feasible, respecting temporal dependence.
- Report uncertainty and limitations, not just point estimates.
- Flag strategies whose results depend heavily on one symbol, one time period, or one exact parameter pair.

Do not claim statistical significance unless the methodology supports that claim.

## 9. Outputs

Produce the following deliverables:

1. **Research report:** Hypotheses, methodology, data-quality findings, results, limitations, and conclusions.
2. **Candidate comparison table:** EMA pair, strategy variant, in-sample metrics, out-of-sample metrics, trading costs, and trade count.
3. **Parameter heatmaps:** Out-of-sample performance and drawdown across the EMA search grid. Clearly distinguish in-sample from out-of-sample results.
4. **Equity curves:** Net strategy returns alongside the relevant baselines, including drawdowns.
5. **Regime breakdown:** Performance by market condition and calendar period.
6. **Reproducible code:** A configurable backtest that can be rerun without manual intervention.
7. **Experiment log:** Parameters, assumptions, data windows, and results for every candidate tested.

Use the existing project structure and data infrastructure where practical. Do not create a separate data pipeline unnecessarily.

## 10. Final Research Questions

Conclude by answering these questions directly:

1. Do any EMA crossover configurations generate positive out-of-sample returns after costs?
2. Which configurations are robust rather than merely optimal in-sample?
3. Does EMA 200 filtering improve results, and at what cost in missed trades?
4. Are long and short signals independently useful?
5. Which market regimes are suitable or unsuitable for this strategy?
6. How sensitive are results to fees, slippage, and funding?
7. Does any candidate outperform the relevant baseline on a risk-adjusted basis?
8. What evidence would justify further research, and what findings would justify rejecting the strategy?

**Standard of evidence:** A strategy is not successful merely because its historical equity curve rises. Prefer stable out-of-sample performance, reasonable drawdowns, nearby parameter robustness, and positive net expectancy under realistic execution assumptions. If no candidate demonstrates a robust edge, report that conclusion rather than forcing a winner.