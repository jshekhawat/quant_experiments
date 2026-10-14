# quant-lab

A standalone, personal **crypto quantitative research and backtesting lab**.

It is a local research environment built around one fast, reproducible vertical
slice:

```
historical data -> Parquet -> DuckDB -> Polars -> features
    -> strategy -> VectorBT -> performance report
```

**This is not a trading bot.** It has no live-trading connector, no order
submission, no authentication, and no production execution system. See
[What this does not model yet](#what-this-does-not-model-yet).

---

## Table of contents

1. [Installation](#1-installation)
2. [Project structure](#2-project-structure)
3. [Downloading historical data](#3-downloading-historical-data)
4. [Dataset schema](#4-dataset-schema)
5. [Querying with DuckDB](#5-querying-with-duckdb)
6. [Creating features](#6-creating-features)
7. [Creating strategies](#7-creating-strategies)
8. [Running VectorBT backtests](#8-running-vectorbt-backtests)
9. [Running parameter sweeps](#9-running-parameter-sweeps)
10. [Interpreting reports](#10-interpreting-reports)
11. [Experiment tracking](#11-experiment-tracking)
12. [Train / validation / test and walk-forward](#12-train--validation--test-and-walk-forward)
13. [Robustness analysis](#13-robustness-analysis)
14. [Multi-symbol and cross-sectional research](#14-multi-symbol-and-cross-sectional-research)
15. [Multi-venue and composite datasets](#15-multi-venue-and-composite-datasets)
16. [Event-driven validation (NautilusTrader)](#16-event-driven-validation-nautilustrader)
17. [Lookahead bias and the timing contract](#17-lookahead-bias-and-the-timing-contract)
18. [Intrabar execution policy](#18-intrabar-execution-policy)
19. [Running tests](#19-running-tests)
20. [What this does not model yet](#what-this-does-not-model-yet)

---

## 1. Installation

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync                      # core: Polars + DuckDB + VectorBT + QuantStats
uv sync --extra optimization # + Optuna
uv sync --extra notebook     # + JupyterLab
uv sync --extra dashboard    # + Streamlit (optional read-only viewer)
uv sync --extra event        # + NautilusTrader (optional, heavy)
```

The lab works perfectly with only the core stack (`Polars + DuckDB +
VectorBT`). Everything else is optional.

Run the CLI:

```bash
uv run quantlab --help
```

## 2. Project structure

```
quant_lab/
├── README.md
├── pyproject.toml
├── configs/default.toml          # paths + execution assumptions
├── data/
│   ├── raw/                      # downloaded archives (gitignored)
│   ├── normalized/               # canonical Parquet (gitignored)
│   └── catalogs/catalog.json     # machine-readable dataset manifest
├── notebooks/research_template.ipynb
├── scripts/                      # thin wrappers (download/build/backtest)
├── src/quant_lab/
│   ├── config.py                 # config loading
│   ├── timeframes.py             # timeframe/annualization helpers
│   ├── synthetic.py              # deterministic test data
│   ├── data/
│   │   ├── schemas.py            # canonical schemas
│   │   ├── storage.py            # Parquet partitioning + reads
│   │   ├── query.py              # DuckDB query layer
│   │   ├── catalog.py            # dataset catalog/manifest
│   │   ├── normalize.py          # raw -> canonical Parquet
│   │   ├── build.py              # download + normalize orchestration
│   │   ├── validate.py           # data-quality reporting
│   │   ├── composite.py          # cross-venue composite datasets
│   │   └── sources/              # exchange adapters (Binance)
│   ├── features/                 # feature API (core, market, cross_section)
│   ├── strategies/               # Strategy interface + reference strategy
│   ├── backtest/                 # VectorBT engine, metrics, assumptions, event
│   ├── analytics/                # chronological splits, robustness
│   ├── optimization/             # parameter sweeps + Optuna
│   ├── reporting/                # reports, plots, experiment tracking
│   ├── dashboard/                # optional read-only Streamlit viewer
│   └── cli/                      # `quantlab` CLI
├── tests/
└── experiments/
    ├── configs/                  # saved run configs
    ├── results/<experiment_id>/  # result.json, trades.parquet, equity.parquet
    └── reports/<experiment_id>/  # summary.md, plots, tearsheet.html
```

The package is a real installable Python package; scripts and notebooks are
thin clients of it.

## 3. Downloading historical data

Data comes from **Binance Public Data** bulk archives
(<https://data.binance.vision>), not repeated REST pagination. The downloader
is **resumable, idempotent, checksum-verified and safe to run repeatedly**.

```bash
# Download raw archives only
quantlab data download binance \
    --market futures --symbol BTCUSDT --interval 1m \
    --start 2024-01-01 --end 2024-03-01

# Download + normalize to canonical Parquet + update the catalog
quantlab data build binance \
    --market futures --symbol BTCUSDT --interval 1m \
    --start 2024-01-01 --end 2024-03-01

# Normalize existing raw archives only (no network)
quantlab data build binance --market futures --symbol BTCUSDT --interval 1m \
    --start 2024-01-01 --end 2024-03-01 --no-download
```

Supported datasets and archive granularities:

| dataset              | flag                 | granularity      | notes                          |
|----------------------|----------------------|------------------|--------------------------------|
| OHLCV / klines       | `bars`               | monthly + daily  | requires `--interval`          |
| trades               | `trades`             | monthly + daily  |                                |
| aggTrades            | `agg_trades`         | monthly + daily  |                                |
| funding rates        | `funding`            | monthly          |                                |
| open interest        | `open_interest`      | daily (derived)  | derived from `metrics`         |
| mark price klines    | `mark_price_bars`    | monthly + daily  | requires `--interval`          |
| index price klines   | `index_price_bars`   | monthly + daily  | requires `--interval`          |
| premium index klines | `premium_index_bars` | monthly + daily  | requires `--interval`          |
| metrics              | `metrics`            | daily            | OI, long/short ratios          |

Not available, and **never fabricated**: liquidations (no reliable bulk source)
and historical open interest outside the `metrics` archive.

Inspect what you have:

```bash
quantlab data inspect
quantlab data inspect --symbol BTCUSDT --dataset bars --json
quantlab data validate --symbol BTCUSDT --dataset bars --interval 1m
```

Note: monthly archives cover whole calendar months. If you request
`--end 2024-02-01`, the February archive is downloaded; query-time filtering
(`--start/--end` on reads) still bounds the data you actually use.

## 4. Dataset schema

Timestamps are always UTC (`Datetime("us", "UTC")`). Schemas are
exchange-independent; strategies only ever consume canonical data.

**Bar**

```python
timestamp, open, high, low, close, volume, quote_volume,
trade_count, taker_buy_volume, taker_buy_quote_volume
```

**Trade**

```python
timestamp, trade_id, price, quantity, quote_quantity, side
```

**AggTrade**

```python
timestamp, agg_trade_id, price, quantity, first_trade_id, last_trade_id, side
```

**Funding**

```python
timestamp, symbol, funding_rate, funding_interval_hours
```

**OpenInterest**

```python
timestamp, symbol, open_interest, open_interest_value
```

**Metrics**

```python
timestamp, symbol, open_interest, open_interest_value,
toptrader_account_long_short_ratio, toptrader_position_long_short_ratio,
global_account_long_short_ratio, taker_long_short_vol_ratio
```

**Liquidation** (schema defined; no reliable source yet)

```python
timestamp, symbol, side, price, quantity, notional
```

Parquet layout (one file per month, avoiding thousands of tiny files):

```
data/normalized/{exchange}/{market_type}/{dataset}/{symbol}/{timeframe}/{year}/{month}.parquet
```

## 5. Querying with DuckDB

DuckDB queries Parquet directly — no ingestion, no server.

```python
import duckdb

duckdb.sql("""
    SELECT date_trunc('day', timestamp) AS day,
           avg(close) AS avg_close,
           sum(volume) AS volume
    FROM read_parquet('data/normalized/binance/futures/bars/BTCUSDT/1m/*/*.parquet')
    WHERE timestamp >= '2024-01-01' AND timestamp < '2024-02-01'
    GROUP BY 1 ORDER BY 1
""").pl()
```

Or via the Python API:

```python
from quant_lab.config import get_config
from quant_lab.data.query import query
from quant_lab.data.schemas import DatasetType
from quant_lab.data.storage import DatasetKey

cfg = get_config()
key = DatasetKey("binance", "futures", DatasetType.BARS, "BTCUSDT", "1m")
query(cfg.paths.data_normalized, key,
      select="count(*) AS n, avg(close) AS mean",
      where="timestamp >= '2024-01-01'")
```

`read_dataset(...)` returns Polars and pushes down predicates:

```python
from quant_lab.data.storage import read_dataset
df = read_dataset(cfg.paths.data_normalized, key, start="2024-01-01", end="2024-02-01")
```

## 6. Creating features

Features are functions `(frame, **params) -> pl.Expr`. All windows are trailing
(causal). Add a new feature in one line and the backtest engine never changes:

```python
from quant_lab.features import feature
import polars as pl

@feature("close_ema_ratio")
def close_ema_ratio(frame, *, span=20):
    return pl.col("close") / pl.col("close").ewm_mean(span=span, adjust=False)
```

Use features directly or by spec:

```python
from quant_lab.features import FeatureSpec, compute_features

df = compute_features(df, [
    FeatureSpec("ema", {"span": 100}),
    FeatureSpec("atr", {"period": 14}),
    FeatureSpec("rsi", {"period": 14}),
    FeatureSpec("realized_volatility", {"window": 60}),
    FeatureSpec("relative_volume", {"window": 20}),
])
```

Built-ins: `sma, ema, atr, rsi, vwap, returns, rolling_volume,
realized_volatility, zscore, roc, stdev, rolling_max, rolling_min, momentum,
true_range, price_range` and market features `volume_delta, cvd, cvd_change,
open_interest_change, open_interest_zscore, funding_rate, funding_cumulative,
basis, premium, liquidation_intensity, relative_volume, range_expansion,
trade_imbalance`.

List them: `quantlab features list`.

### Recreating historical CVD

CVD is accumulated net taker flow, and Binance klines already carry the taker-buy
split, so it is an exact per-bar reconstruction — no tick data required:

```bash
# Top 100 liquid symbols, spot + USD-M futures, 2026 to date
python scripts/recreate_cvd.py --fetch-top 100 \
    --markets futures,spot --start 2026-01-01 --end 2026-10-01
```

This ranks live 24h quote volume (stable/stable and fiat pairs excluded),
downloads and normalizes 1m bars, then writes continuous CVD partitions to
`data/derived/cvd/{market}/{symbol}/1m/{year}/{month}.parquet` with columns
`timestamp, symbol, market, volume_delta, cvd, quote_volume_delta, cvd_quote`.
The cumulative sum is continuous across months and anchored at the first bar of
the range; spot and futures are separate series and are never summed.

Reuse a cached universe (`--universe-file configs/cvd_universe.json`) or explicit
symbols (`--symbols BTCUSDT,ETHUSDT`), and pass `--no-download` to rebuild from
bars you already have. Cost is small: ~17 MB of klines per symbol for nine months
of both markets, so the whole top 100 is a few GB and a couple of minutes.

## 7. Creating strategies

A strategy declares parameters, optionally computes features, and returns four
boolean signal columns.

```python
import polars as pl
from quant_lab.strategies.base import Strategy, register_strategy
from quant_lab.features.core import ema

@register_strategy
class EmaCross(Strategy):
    name = "ema_cross"
    default_params = {"span": 50}
    param_grid = {"span": [20, 50, 100]}

    def features(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.with_columns(ema(data, span=self.params["span"]).alias("ema"))

    def signals(self, data: pl.DataFrame) -> pl.DataFrame:
        close, ema_ = pl.col("close"), pl.col("ema")
        return data.with_columns(
            (close > ema_).fill_null(False).alias("entries"),
            (close < ema_).fill_null(False).alias("exits"),
            pl.lit(False).alias("short_entries"),
            pl.lit(False).alias("short_exits"),
        )
```

Strategies are independent of the downloader. The bundled reference strategy is
`volatility_breakout` (see `src/quant_lab/strategies/volatility_breakout.py`).
List them: `quantlab strategies list`.

## 8. Running VectorBT backtests

```bash
quantlab backtest \
    --strategy volatility_breakout \
    --symbol BTCUSDT --timeframe 1m \
    --start 2024-01-01 --end 2024-04-01 \
    --set ema_span=100 --set volume_multiplier=1.5
```

Output: backtest summary, trade statistics, risk statistics, execution
assumptions and an experiment identifier.

Execution assumptions are explicit and recorded with every run (see
`configs/default.toml`): initial capital, fees, maker/taker fees, slippage,
position sizing, leverage, max position size, funding, execution delay and
intrabar policy. Costs default to **non-zero** — a backtest never silently
assumes free, frictionless trading.

```python
from quant_lab.backtest import run_backtest
from quant_lab.strategies import build_strategy

result = run_backtest(df, build_strategy("volatility_breakout"), symbol="BTCUSDT", timeframe="1m")
print(result.metrics["sharpe"], result.trades.head())
```

## 9. Running parameter sweeps

```bash
# Grid sweep using the strategy's declared grid, with overrides
quantlab optimize --strategy volatility_breakout --symbol BTCUSDT --timeframe 1m \
    --start 2024-01-01 --end 2024-03-01 --metric sharpe \
    --grid ema_span=50,100,150,200 --grid volume_multiplier=1.0,1.5,2.0

# Optuna search (requires --extra optimization)
quantlab optimize --strategy volatility_breakout --symbol BTCUSDT --timeframe 1m \
    --start 2024-01-01 --end 2024-03-01 --method optuna --trials 100 \
    --grid ema_span=int:20:300 --grid volume_multiplier=float:0.5:3.0
```

The result matrix contains return, drawdown, Sharpe, trade count, profit
factor, etc. per combination. VectorBT's native parameterization is also
available for single-parameter sweeps (`quant_lab.optimization.vectorbt_sweep`).

```python
from quant_lab.optimization import parameter_grid, run_sweep, best_by
grid = parameter_grid(type(strategy), overrides={"ema_span": [50, 100, 150]})
results = run_sweep(df, type(strategy), grid, symbol="BTCUSDT", timeframe="1m")
print(best_by(results, "sharpe"))
```

## 10. Interpreting reports

Each run writes `experiments/reports/<id>/`:

- `summary.md` / `summary.json` — metrics, trade stats, risk stats, execution
  assumptions, experiment id;
- `equity.png`, `drawdown.png`, `monthly_returns.png`,
  `trade_pnl_distribution.png`, `returns_distribution.png`;
- `tearsheet.html` — QuantStats tearsheet (when available).

Regenerate any saved report:

```bash
quantlab report latest
quantlab report <experiment_id>
```

### Optional read-only dashboard

A minimal, local-only Streamlit viewer over saved experiments is available as
an optional extra. It is **read-only**: no writes, no authentication, no live
data, no exchange connectivity, and it binds to `127.0.0.1` by default.

```bash
uv sync --extra dashboard
quantlab dashboard --host 127.0.0.1 --port 8501
```

It browses the experiment list, metrics, equity/drawdown, trades, execution
assumptions and the QuantStats tearsheet. It is a convenience layer over the
same artifacts described above — not a web platform. (The original brief lists
"web dashboard" as a non-goal; this deliberately stays a small local viewer.)

Reported metrics include: total return, CAGR, net/gross PnL, maximum drawdown,
Sharpe, Sortino, Calmar, volatility, win rate, profit factor, expectancy,
average win/loss, trade count, average holding period, turnover, fees paid and
funding paid.

## 11. Experiment tracking

Every experiment persists (machine-readable JSON + Parquet):

```
strategy, parameters, git commit, dataset identifier, data range,
execution assumptions, random seed, metrics, command
```

Trace any result back to the exact configuration and dataset:

```bash
quantlab experiments list
quantlab experiments list --json
```

## 12. Train / validation / test and walk-forward

Splits are always chronological and derived from ranges that actually exist in
the data — never hardcoded calendar dates.

```python
from quant_lab.analytics import chronological_split, walk_forward_splits, split_by_dates

splits = chronological_split(df, train=0.6, validation=0.2, test=0.2)
for train, test in walk_forward_splits(df, train_rows=20_000, test_rows=5_000):
    ...
```

Never optimize on the full sample and present it as out-of-sample evidence.

## 13. Robustness analysis

Underlying distributions are exposed; there is no opaque "strategy quality
score".

```python
from quant_lab.analytics import bootstrap_returns, monte_carlo_trades, parameter_sensitivity

boot = bootstrap_returns(result.returns, n=1000, block=10)
print(boot.format())                       # total return / Sharpe / max-DD distributions
mc = monte_carlo_trades(result.trades["pnl"], n=1000)
print(mc.format())                         # path-dependence of the trade sequence
```

Also available: `drawdown_distribution`, `return_distribution`,
`parameter_sensitivity`.

## 14. Multi-symbol and cross-sectional research

The data model is multi-instrument from the start. Build each symbol's dataset,
then work on a wide frame:

```python
from quant_lab.features.cross_section import (
    wide_close, wide_returns, relative_strength, cross_sectional_momentum,
    correlation_matrix, beta, market_breadth,
)
```

Supports relative strength, cross-sectional momentum, correlation, beta,
BTC-relative returns and market breadth.

## 15. Multi-venue and composite datasets

Source adapters implement a common interface, so Binance is not hardcoded into
strategy logic:

```python
source.fetch_bars(...)
source.fetch_trades(...)
source.fetch_funding(...)
source.fetch_open_interest(...)
```

Derived cross-venue datasets live in `quant_lab.data.composite`:
`composite_funding`, `composite_volume`, `composite_open_interest`,
`cross_venue_spread`. Each records its methodology: venues included, weighting,
timestamp alignment and missing-data treatment. Contracts with different
specifications are never silently combined.

## 16. Event-driven validation (NautilusTrader)

VectorBT is the fast hypothesis-testing engine. NautilusTrader is the optional
secondary engine for realistic, event-driven execution simulation. The
interface lives in `quant_lab.backtest.event`:

```python
from quant_lab.backtest.event import signal_frame_to_intents, NautilusEngine

intents = signal_frame_to_intents(signals, symbol="BTCUSDT", execution_delay_bars=1)
```

`signal_frame_to_intents` produces an explicit, inspectable order schedule using
the same timing contract as the VectorBT engine. The `NautilusEngine` adapter is
import-guarded (install `--extra event`); it is a Phase-4 scaffold and is **not**
a live-trading connector. The lab does not require NautilusTrader to function.

## 17. Lookahead bias and the timing contract

Lookahead prevention is a first-class concern.

- Features only use trailing windows; tests assert that appending future bars
  does not change past feature values.
- Signals are evaluated at the **close** of bar `t`.
- Explicit timestamps:

| stage               | definition                                   |
|---------------------|----------------------------------------------|
| signal timestamp    | close of bar `t`                             |
| order timestamp     | bar `t + execution_delay_bars`               |
| fill timestamp      | open of `t + execution_delay_bars` (market)  |
| position activation | the same bar as the fill                     |

With `execution_delay_bars = 0`, fills happen at the close of bar `t`.
`execution_delay_bars` defaults to `1`. Tests cover future-data leakage,
centered-window mistakes, shifted signals, fees/PnL invariants and
signal causality.

## 18. Intrabar execution policy

OHLC candles do not reveal the order in which prices occurred. When a single
bar touches both a stop and a target, the simulator needs an explicit policy.
The configured `intrabar_policy` (`worst_case` by default; also `best_case`,
`stop_first`, `target_first`) documents the assumption. Higher-resolution
execution validation (e.g. strategy on `1h`, execution on `1m`) is the intended
role of the event-driven layer.

## 19. Running tests

```bash
uv run pytest
```

Tests cover data parsing, normalization, timestamps, feature calculations,
strategy signals, position sizing, fees, funding, PnL, lookahead prevention,
experiment serialization, validation, splits, composites, cross-section, and a
deterministic end-to-end test (`tests/test_e2e.py`) with a known expected
result that fails if the calculation changes.

Lint: `uv run ruff check src tests`.

## What this does not model yet

- **No live trading** and no exchange order submission or authentication.
- **No order-book / quote-level simulation** in the default path; VectorBT
  works on bars. The event-driven adapter is a scaffold.
- **Funding** is modelled as a periodic drag on open notional when a
  `funding_rate` series is present; it is not a full cash-flow ledger.
- **Liquidations** are unsupported (no reliable bulk historical source).
- **Borrow / margin mechanics, liquidation of the strategy's own position,
  partial fills and queue position** are not modelled.
- **Intrabar path** is approximated by an explicit policy, not reconstructed
  from ticks.
- **Slippage** is a fixed per-side fraction, not a market-impact model.
- **Composite datasets** are implemented for a small set of metrics with
  explicit methodology; not all instruments/contracts are interchangeable.
- No Postgres, ClickHouse, Kafka, Spark, Flink, Airflow, Kubernetes or any
  distributed data infrastructure. The lab is intentionally local.
- **No web dashboard as a platform.** The optional Streamlit viewer is a small,
  read-only, localhost-only convenience over saved artifacts; there is no
  authentication, no persistence, no multi-user support and no write path.

The goal is fast, reproducible, honest research — not a production trading
system.
