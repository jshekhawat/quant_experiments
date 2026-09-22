"""Cross-sectional research over multiple instruments.

Works on a *wide* frame: one ``timestamp`` column plus one column per symbol.
This is the data model that enables relative strength, cross-sectional
momentum, correlation, beta, BTC-relative returns and market breadth.
"""

from __future__ import annotations

import polars as pl


def wide_close(frames: dict[str, pl.DataFrame], *, value: str = "close") -> pl.DataFrame:
    """Full-outer join per-symbol frames into a wide price frame.

    Missing observations stay null; no forward-filling is done implicitly.
    """
    if not frames:
        return pl.DataFrame(schema={"timestamp": pl.Datetime("us", "UTC")})
    out: pl.DataFrame | None = None
    for symbol, df in frames.items():
        sub = df.select(pl.col("timestamp"), pl.col(value).alias(symbol)).sort("timestamp")
        out = sub if out is None else out.join(sub, on="timestamp", how="full", coalesce=True)
    assert out is not None
    return out.sort("timestamp")


def wide_returns(prices: pl.DataFrame, *, period: int = 1, log: bool = False) -> pl.DataFrame:
    exprs = []
    for col in prices.columns:
        if col == "timestamp":
            continue
        if log:
            exprs.append((pl.col(col) / pl.col(col).shift(period)).log().alias(col))
        else:
            exprs.append(pl.col(col).pct_change(n=period).alias(col))
    return prices.select(pl.col("timestamp"), *exprs)


def relative_strength(prices: pl.DataFrame, benchmark: str, *, window: int = 20) -> pl.DataFrame:
    """Rolling return of each symbol minus the benchmark's rolling return."""
    rets = wide_returns(prices, period=1)
    b = rets[benchmark].rolling_sum(window_size=window, min_periods=window)
    symbols = [c for c in rets.columns if c != "timestamp"]
    exprs = [
        (pl.col(col).rolling_sum(window_size=window, min_periods=window) - b).alias(col)
        for col in symbols
    ]
    return rets.with_columns(exprs).select("timestamp", *symbols)


def cross_sectional_momentum(prices: pl.DataFrame, *, window: int = 20) -> pl.DataFrame:
    """Rank symbols by trailing return each bar (1 = strongest).

    Ranks are computed *across symbols within each timestamp*, not across time.
    """
    symbols = [c for c in prices.columns if c != "timestamp"]
    if not symbols:
        return prices
    long = prices.unpivot(index="timestamp", on=symbols, variable_name="symbol", value_name="value")
    long = long.sort("timestamp", "symbol").with_columns(
        (pl.col("value") / pl.col("value").shift(window).over("symbol") - 1.0).alias("momentum")
    )
    long = long.with_columns(
        pl.col("momentum").rank(descending=True).over("timestamp").alias("rank")
    )
    ranked = long.pivot(values="rank", index="timestamp", on="symbol").sort("timestamp")
    return ranked.select("timestamp", *symbols)


def correlation_matrix(returns: pl.DataFrame) -> pl.DataFrame:
    symbols = [c for c in returns.columns if c != "timestamp"]
    data = {s: returns[s].to_numpy() for s in symbols}
    import numpy as np

    matrix = np.full((len(symbols), len(symbols)), np.nan)
    for i, a in enumerate(symbols):
        for j, b in enumerate(symbols):
            x, y = data[a], data[b]
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() > 1:
                matrix[i, j] = np.corrcoef(x[mask], y[mask])[0, 1]
    return pl.DataFrame(matrix.tolist(), schema=symbols).with_columns(
        pl.Series("symbol", symbols)
    ).select("symbol", *symbols)


def beta(returns: pl.DataFrame, benchmark: str, *, window: int | None = None) -> pl.DataFrame:
    """Per-symbol beta to the benchmark (rolling if ``window`` given)."""
    import numpy as np

    b = returns[benchmark]
    symbols = [c for c in returns.columns if c != "timestamp" and c != benchmark]
    exprs = []
    for col in symbols:
        if window is None:
            mask = b.is_not_null() & returns[col].is_not_null()
            x = b.filter(mask).to_numpy()
            y = returns[col].filter(mask).to_numpy()
            value = float(np.cov(x, y)[0, 1] / np.var(x)) if np.var(x) > 0 else None
            exprs.append(pl.lit(value).alias(col))
        else:
            cov = pl.rolling_cov(pl.col(col), b, window_size=window, min_periods=window)
            var = b.rolling_var(window_size=window, min_periods=window)
            exprs.append((cov / var).alias(col))
    return returns.with_columns(exprs).select("timestamp", *symbols)


def btc_relative_returns(prices: pl.DataFrame, benchmark: str = "BTCUSDT", *, window: int = 1) -> pl.DataFrame:
    return relative_strength(prices, benchmark, window=window)


def market_breadth(prices: pl.DataFrame, *, window: int = 20) -> pl.DataFrame:
    """Fraction of symbols with a positive trailing return each bar."""
    rets = wide_returns(prices, period=window)
    symbols = [c for c in rets.columns if c != "timestamp"]
    if not symbols:
        return rets.select("timestamp")
    positives = pl.sum_horizontal([(pl.col(s) > 0).cast(pl.Int32) for s in symbols])
    valid = pl.sum_horizontal([pl.col(s).is_not_null().cast(pl.Int32) for s in symbols])
    breadth = pl.when(valid > 0).then(positives / valid).otherwise(None)
    return rets.with_columns(breadth.alias("breadth")).select("timestamp", "breadth")
