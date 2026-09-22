# Build a Personal Crypto Quant Research Lab

Build a standalone Python-based quantitative research and backtesting environment for personal crypto strategy research.

This project has nothing to do with any existing application or product. It should be a self-contained research lab whose primary purpose is:

* obtaining historical crypto market data;
* storing and querying it efficiently;
* computing indicators and market features;
* writing and testing trading strategies;
* running fast backtests;
* running parameter sweeps;
* evaluating robustness;
* eventually running more realistic event-driven execution simulations.

Do not build a trading bot, exchange connector for live trading, or production execution system at this stage.

The priority is making experimentation extremely fast and reproducible.

---

# 1. Inspect the repository first

Before implementing anything:

* inspect the repository structure;
* identify the existing Python tooling and package manager;
* reuse existing conventions where sensible;
* avoid unnecessary architectural complexity;
* identify reusable utilities before creating new ones.

Do not blindly implement the entire specification at once.

First establish the smallest working vertical slice:

```text
historical data
→ Parquet
→ DuckDB
→ Polars
→ features
→ strategy
→ VectorBT
→ performance report
```

Once that works, extend the system incrementally.

---

# 2. Recommended technology stack

Use current, actively maintained libraries.

Verify the latest stable compatible versions during implementation rather than blindly copying old version numbers.

## Core data stack

Use:

* Python
* `uv`
* Polars
* DuckDB
* PyArrow
* NumPy
* SciPy
* pandas only where required for ecosystem compatibility

Prefer Polars for transformation and analysis.

Use pandas primarily at library boundaries where another library expects pandas objects.

## Fast research/backtesting

Use **VectorBT** as the primary research and vectorized backtesting engine.

Use it for:

* signal generation;
* portfolio simulation;
* fast parameter sweeps;
* large numbers of strategy variations;
* exploratory research.

Do not implement a custom vectorized backtesting framework unless there is a specific limitation that requires it.

## Event-driven simulation

Evaluate **NautilusTrader** as the secondary backtesting engine for more realistic execution simulation.

Use it for eventual support of:

* event-driven simulation;
* bar, quote and trade data;
* order-book simulation;
* multiple instruments;
* multiple venues;
* fees;
* slippage;
* order execution;
* latency;
* more realistic fill modeling.

Do not make NautilusTrader a hard dependency for the basic research workflow.

The lab should still function perfectly well with only:

```text
Polars + DuckDB + VectorBT
```

## Performance analysis

Use **QuantStats** where useful for portfolio and performance reporting.

Use:

* SciPy for numerical/statistical calculations;
* statsmodels for statistical analysis where appropriate;
* scikit-learn only for actual research problems that benefit from ML.

Do not add libraries just because they are popular.

## Parameter optimization

Use **Optuna** when parameter-search requirements go beyond straightforward VectorBT sweeps.

Do not begin with expensive optimization infrastructure.

---

# 3. Project structure

Create approximately:

```text
quant_lab/
├── README.md
├── pyproject.toml
├── uv.lock
│
├── configs/
│   └── default.toml
│
├── data/
│   ├── raw/
│   ├── normalized/
│   └── catalogs/
│
├── notebooks/
│
├── scripts/
│   ├── download_binance.py
│   ├── build_dataset.py
│   └── run_backtest.py
│
├── src/
│   └── quant_lab/
│       ├── data/
│       ├── features/
│       ├── strategies/
│       ├── backtest/
│       ├── analytics/
│       ├── optimization/
│       └── cli/
│
├── tests/
│
└── experiments/
    ├── configs/
    ├── results/
    └── reports/
```

Use a real Python package rather than a loose collection of scripts and notebooks.

---

# 4. Historical data

Use **Binance Public Data** as the initial historical data source.

Start with:

```text
market: USD-M perpetual futures
symbol: BTCUSDT
timeframe: 1m
```

The downloader should support:

```text
exchange
market type
symbol
dataset type
interval
start date
end date
```

Start with a relatively small date range so the complete workflow can be verified quickly.

Then support expansion to:

```text
ETHUSDT
SOLUSDT
BNBUSDT
XRPUSDT
```

and additional symbols later.

Prefer bulk historical archives over repeatedly requesting old data through exchange REST APIs.

The downloader should be:

* resumable;
* idempotent;
* safe to run repeatedly;
* capable of detecting already-downloaded files;
* capable of verifying checksums where available.

---

# 5. Data types

Initially support:

```text
OHLCV / klines
trades
aggTrades
```

Design interfaces so the following can be added later:

```text
funding rates
mark price
index price
premium / basis
open interest
liquidations
order book
quotes
```

Only use data that has a reliable historical source.

Never fabricate missing history.

---

# 6. Storage architecture

Use:

```text
Parquet
+
DuckDB
```

as the fundamental storage architecture.

Parquet is the persistent data format.

DuckDB is the query and analytical layer.

Do not introduce:

```text
Postgres
ClickHouse
Kafka
Spark
Flink
Airflow
Kubernetes
```

at this stage.

They are unnecessary for the initial research lab.

The lab should be able to query large datasets directly from Parquet using DuckDB.

Example:

```sql
SELECT *
FROM read_parquet('data/normalized/binance/futures/BTCUSDT/1m/*.parquet')
WHERE timestamp >= '2024-01-01'
  AND timestamp < '2025-01-01';
```

---

# 7. Partitioning

Use sensible Parquet partitioning, for example:

```text
exchange/
market_type/
symbol/
dataset/
year/
month/
```

Avoid creating thousands of tiny files.

Maintain metadata describing each dataset:

```text
exchange
market_type
symbol
dataset_type
timeframe
start
end
source
downloaded_at
schema_version
```

---

# 8. Canonical schemas

Normalize timestamps to UTC.

Use a canonical representation independent of the source exchange.

### Bar

```python
timestamp
open
high
low
close
volume
quote_volume
trade_count
taker_buy_volume
taker_buy_quote_volume
```

### Trade

```python
timestamp
trade_id
price
quantity
quote_quantity
side
```

### Funding

```python
timestamp
symbol
funding_rate
```

### Open Interest

```python
timestamp
symbol
open_interest
```

### Liquidation

```python
timestamp
symbol
side
price
quantity
notional
```

Keep different market-data domains in separate datasets rather than forcing everything into one enormous table.

---

# 9. Data catalog

Create a lightweight catalog/manifest.

It should be possible to answer:

```text
What datasets do I have?
Which symbols?
Which exchanges?
Which date ranges?
Which timeframes?
Which files are complete?
Where did the data come from?
```

The catalog should be machine-readable.

Do not require a database server for this.

---

# 10. Data validation

Implement validation for:

```text
missing timestamps
duplicate timestamps
non-monotonic timestamps
invalid prices
invalid quantities
unexpected gaps
obvious outliers
timezone inconsistencies
duplicate trades
```

Generate a data-quality report.

Do not silently repair suspicious data.

Any normalization or repair should be explicit and documented.

---

# 11. Feature engine

Create a reusable feature API.

Basic features:

```python
ema(...)
sma(...)
atr(...)
rsi(...)
vwap(...)
returns(...)
rolling_volume(...)
realized_volatility(...)
zscore(...)
```

Market features:

```python
volume_delta(...)
cvd(...)
open_interest_change(...)
funding_rate(...)
basis(...)
liquidation_intensity(...)
relative_volume(...)
range_expansion(...)
```

The API should make it easy to add new features without modifying the backtesting engine.

All feature calculations must respect temporal causality.

No feature may accidentally use future information.

---

# 12. Strategy interface

Create a deliberately simple strategy abstraction.

For example:

```python
class Strategy:
    def features(self, data):
        ...

    def signals(self, data):
        ...

    def portfolio(self, data):
        ...
```

Do not overengineer the abstraction.

I want to be able to write:

```python
class VolatilityBreakout(Strategy):
    def signals(self, data):
        ...
```

without understanding the internals of the entire lab.

Strategies must be independent from the historical-data downloader.

---

# 13. First example strategy

Implement one simple strategy purely to demonstrate the full pipeline.

Example:

```text
close > EMA(100)

AND

price breaks a configurable volatility threshold

AND

volume > 1.5 × rolling_mean(volume, 20)
```

Make all parameters configurable.

Do not assume this strategy is profitable.

It exists only as a reference implementation.

The complete flow must work:

```text
data
→ features
→ signals
→ positions
→ portfolio
→ metrics
→ report
```

---

# 14. Backtest assumptions

Every backtest must explicitly define:

```text
initial capital
fees
maker fee
taker fee
slippage
position sizing
leverage
maximum position size
funding
execution delay
```

Do not silently use:

```text
zero fees
zero slippage
instant fills
```

unless explicitly configured.

Every report must display the execution assumptions.

---

# 15. Lookahead bias

Treat lookahead prevention as a first-class concern.

Test for:

* future-data leakage;
* incorrectly centered rolling calculations;
* improperly shifted signals;
* using candle close information before the close;
* same-bar stop/target ambiguity;
* survivorship bias;
* data availability bias;
* optimization leakage.

Clearly define:

```text
signal timestamp
order timestamp
fill timestamp
position activation timestamp
```

---

# 16. Intrabar execution

Do not assume OHLC candles tell us the order in which prices occurred.

For example, when a single candle touches both:

```text
take profit
stop loss
```

the simulator needs an explicit execution policy.

Document the policy.

Design the system so higher-resolution data can later be used for execution validation.

Example:

```text
strategy timeframe = 1h
execution timeframe = 1m
```

The eventual event-driven validation layer should make this more realistic.

---

# 17. Performance metrics

Every backtest should calculate at least:

```text
total return
CAGR
net PnL
gross PnL
maximum drawdown
Sharpe ratio
Sortino ratio
Calmar ratio
volatility
win rate
profit factor
expectancy
average win
average loss
trade count
average holding period
turnover
fees paid
funding paid
```

Generate:

```text
equity curve
drawdown curve
monthly returns
trade distribution
PnL distribution
```

Use QuantStats when useful rather than manually recreating its functionality.

---

# 18. Experiment tracking

Every experiment must be reproducible.

Persist:

```text
strategy name
git commit
dataset identifier
data range
parameters
execution assumptions
random seed
metrics
```

Store machine-readable experiment results:

```text
JSON
Parquet
```

and human-readable reports where useful.

An experiment result should always be traceable back to the exact configuration and dataset that produced it.

---

# 19. Train / validation / test

Build the architecture around chronological splits.

Support:

```text
train
validation
test
```

and eventually:

```text
walk-forward analysis
```

Never optimize parameters over the entire dataset and then present performance on that same dataset as out-of-sample evidence.

Provide utilities for chronological dataset splitting.

Do not hardcode specific calendar ranges; use ranges that actually exist in the downloaded data.

---

# 20. Parameter sweeps

Make it trivial to test grids such as:

```text
EMA:
50
100
150
200

ATR:
10
14
20

volume multiplier:
1.0
1.25
1.5
2.0
```

Use VectorBT's parameterization capabilities first.

Generate result matrices that make it possible to inspect:

```text
return
drawdown
Sharpe
trade count
profit factor
```

across parameter combinations.

Do not immediately build distributed computation.

Benchmark first.

---

# 21. Statistical robustness

Add utilities for:

```text
bootstrap analysis
Monte Carlo trade reshuffling
parameter sensitivity
drawdown distributions
return distributions
```

The purpose is to determine whether a strategy's result is robust rather than simply finding the highest historical return.

Do not turn these into opaque "strategy quality scores."

Expose the underlying statistics.

---

# 22. Cross-sectional research

Design the data model for multiple instruments from the beginning.

Eventually support:

```text
BTC
ETH
SOL
BNB
XRP
```

This should allow research into:

```text
relative strength
cross-sectional momentum
correlation
beta
BTC-relative returns
market breadth
capital rotation
cross-asset signals
```

The strategy layer should be able to operate on one or many instruments.

---

# 23. Multi-venue support

Do not hardcode Binance into strategy logic.

Design the data layer so additional venues can eventually be added:

```text
Binance
Hyperliquid
Bybit
OKX
Lighter
```

Use exchange adapters or source-specific loaders.

For example:

```python
source.fetch_bars(...)
source.fetch_trades(...)
source.fetch_funding(...)
source.fetch_open_interest(...)
```

Strategies should consume canonical data rather than exchange-specific schemas.

---

# 24. Composite market datasets

Design the architecture to support derived cross-venue datasets such as:

```text
composite funding
composite volume
composite open interest
cross-venue basis
cross-venue spread
```

These should be separate derived datasets.

Example:

```python
composite_funding(...)
```

must have an explicit methodology.

Do not silently combine instruments whose contract specifications differ.

For every composite dataset, record:

```text
venues included
weighting methodology
timestamp alignment methodology
missing-data treatment
```

---

# 25. CLI

Provide a CLI similar to:

```bash
quantlab data download ...
quantlab data inspect ...
quantlab data build ...
quantlab backtest ...
quantlab optimize ...
quantlab report ...
```

For example:

```bash
quantlab data download binance \
    --market futures \
    --symbol BTCUSDT \
    --interval 1m \
    --start 2024-01-01 \
    --end 2025-01-01
```

and:

```bash
quantlab backtest \
    --strategy volatility_breakout \
    --symbol BTCUSDT \
    --timeframe 1m \
    --start 2024-01-01 \
    --end 2025-01-01
```

Commands should provide useful progress and failure messages.

---

# 26. Notebook

Create a single canonical research notebook demonstrating:

```text
load BTC data
↓
inspect dataset
↓
validate data quality
↓
calculate features
↓
generate signals
↓
run VectorBT backtest
↓
display metrics
↓
plot equity curve
↓
plot drawdown
↓
run parameter sweep
↓
compare results
```

This notebook should become the template for future experiments.

---

# 27. Event-driven validation

Once the VectorBT pipeline works, integrate NautilusTrader behind a clean interface.

The goal is:

```text
VectorBT
    ↓
fast hypothesis testing

NautilusTrader
    ↓
execution realism
```

The same strategy concept should be expressible in both systems where practical.

The two engines do not need to share every internal abstraction.

Avoid forcing one framework to imitate the other.

---

# 28. Testing

Write automated tests for:

```text
data parsing
normalization
timestamps
feature calculations
strategy signals
position sizing
fees
funding
PnL
lookahead prevention
experiment serialization
```

Create at least one deterministic end-to-end test using a small synthetic dataset:

```text
synthetic market data
→ strategy
→ backtest
→ known expected result
```

The test should fail if a future change alters the expected calculation.

---

# 29. Documentation

README must explain:

1. installation;
2. project structure;
3. downloading historical data;
4. dataset schema;
5. querying with DuckDB;
6. creating features;
7. creating strategies;
8. running VectorBT backtests;
9. running parameter sweeps;
10. interpreting reports;
11. using event-driven validation;
12. running tests.

Also explicitly document what the system does not model yet.

---

# 30. Non-goals

Do NOT build:

```text
live trading
exchange order submission
authentication systems
web dashboard
microservices
Kafka
Spark
Flink
Kubernetes
distributed data infrastructure
```

unless the existing repository genuinely requires one of these for some unrelated reason.

The first version should remain a local research environment.

---

# 31. Implementation strategy

Implement incrementally.

### Phase 1

Get this working:

```text
Binance BTCUSDT 1m
→ Parquet
→ DuckDB
→ Polars
→ VectorBT
→ example backtest
```

### Phase 2

Add:

```text
data validation
feature library
metrics
experiment tracking
parameter sweeps
notebook
CLI
```

### Phase 3

Add:

```text
funding
open interest
trades
liquidations
multi-symbol research
walk-forward testing
robustness analysis
```

### Phase 4

Add:

```text
multi-venue datasets
composite market data
event-driven execution validation
NautilusTrader
```

Do not jump directly to Phase 4.

---

# 32. Definition of done

I should be able to clone the repository and do something equivalent to:

```bash
uv sync

quantlab data download binance \
    --market futures \
    --symbol BTCUSDT \
    --interval 1m \
    --start 2024-01-01 \
    --end 2024-03-01

quantlab backtest \
    --strategy volatility_breakout \
    --symbol BTCUSDT \
    --timeframe 1m \
    --start 2024-01-01 \
    --end 2024-03-01
```

and receive:

```text
backtest summary
trade statistics
risk statistics
equity curve
drawdown
execution assumptions
experiment identifier
```

with no manual wiring.

The entire system should be understandable by one developer.

Prefer boring, explicit code over clever abstractions.

Keep the architecture open for future quantitative research, but make the initial implementation as small as possible.
