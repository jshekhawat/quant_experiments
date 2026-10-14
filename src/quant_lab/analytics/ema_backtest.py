"""Vectorised EMA-crossover backtest engine and evaluation helpers.

This is a deliberately transparent, dependency-light engine (NumPy/Pandas) used
by the EMA-crossover research study. It exists alongside the VectorBT engine in
``quant_lab.backtest`` because the study needs, for thousands of parameter
combinations, (a) next-open execution, (b) explicit per-side costs, (c) perp
funding, and (d) reversal/position accounting that can be inspected exactly.

Timing contract (same as the rest of the lab):

* signal is evaluated on the **close of bar ``t``**;
* the resulting target position is held from the **open of bar ``t+1``**;
* equity is marked at the **close of every bar**.

Bar return with a position change is split at the bar open so a fill at the open
is modelled exactly::

    r_t = pos_{t-1} * (open_t / close_{t-1} - 1)
        + pos_t     * (close_t / open_t - 1)

Per-side cost ``c`` (fee + slippage) is charged on the traded notional
``|pos_t - pos_{t-1}|`` and perp funding ``f`` on the held notional ``pos_t * f``.

Nothing in this module is a live-trading system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ..data.resample import resample_bars
from ..data.schemas import DatasetType
from ..data.storage import DatasetKey, read_dataset
from ..timeframes import periods_per_year, timeframe_seconds

# --- variants ----------------------------------------------------------------

#: side x (no trend filter | EMA200 entry filter | EMA200 regime gate)
VARIANTS: tuple[str, ...] = (
    "long",
    "short",
    "both",
    "long_trend_entry",
    "both_trend_entry",
    "both_trend_regime",
)


def _ffill(a: np.ndarray) -> np.ndarray:
    idx = np.where(~np.isnan(a), np.arange(a.size), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


def desired_positions(
    close: np.ndarray,
    ema_fast: np.ndarray,
    ema_slow: np.ndarray,
    ema200: np.ndarray,
    variant: str,
) -> np.ndarray:
    """Target position (``-1``, ``0``, ``+1``) decided at each bar's close.

    * ``long`` / ``short`` / ``both`` — raw crossover direction, no filter.
    * ``*_trend_entry`` — a cross may only open a position if price is on the
      correct side of EMA200 **at the cross bar**; otherwise the signal is
      skipped and the book stays flat until the next *cross* (a later filter
      flip does **not** re-enter).
    * ``*_trend_regime`` — the raw direction is held only while price stays on
      the correct side of EMA200; the position is flattened when the regime
      turns and re-opens automatically if the raw direction still agrees when
      the regime turns back.
    """
    n = close.size
    valid = np.isfinite(ema_fast) & np.isfinite(ema_slow)
    sign = np.zeros(n, dtype=np.int8)
    above = valid & (ema_fast > ema_slow)
    sign[valid & above] = 1
    sign[valid & ~above] = -1

    if variant in ("long", "short", "both"):
        if variant == "long":
            return np.where(sign == 1, 1, 0).astype(np.int8)
        if variant == "short":
            return np.where(sign == -1, -1, 0).astype(np.int8)
        return sign.copy()

    tvalid = np.isfinite(ema200)
    filter_long = tvalid & (close > ema200)
    filter_short = tvalid & (close < ema200)

    if variant.endswith("regime"):
        base = sign if variant.startswith("both") else np.where(sign == 1, 1, 0).astype(np.int8)
        allowed = ((base == 1) & filter_long) | ((base == -1) & filter_short)
        return np.where(allowed, base, 0).astype(np.int8)

    # entry-condition variants: latch the direction at each cross.
    changed = np.zeros(n, dtype=bool)
    changed[1:] = (sign[1:] != sign[:-1]) & (sign[1:] != 0)
    changed |= valid & ~np.concatenate([[False], valid[:-1]])
    if variant == "long_trend_entry":
        blocked = ~filter_long
    else:  # both_trend_entry
        blocked = ((sign == 1) & ~filter_long) | ((sign == -1) & ~filter_short)
    resolved = np.where(changed, np.where(blocked, 0, sign), np.nan)
    return np.nan_to_num(_ffill(resolved)).astype(np.int8)


# --- engine ------------------------------------------------------------------


@dataclass(slots=True)
class BtResult:
    symbol: str
    variant: str
    fast: int
    slow: int
    timestamps: np.ndarray
    positions: np.ndarray
    returns: np.ndarray
    turnover: float
    funding_cost: float
    n_trades: int


def simulate(
    timestamps: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    target: np.ndarray,
    *,
    cost_per_side: float,
    funding: np.ndarray | None = None,
    symbol: str = "",
    variant: str = "",
    fast: int = 0,
    slow: int = 0,
) -> BtResult:
    """Run one backtest from a target-position array (decided at bar close)."""
    n = close.size
    pos = np.zeros(n)
    pos[1:] = target[:-1]  # held during bar t = target decided at t-1
    prev = np.zeros(n)
    prev[1:] = pos[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        cc = np.zeros(n)
        cc[1:] = close[1:] / close[:-1] - 1.0
        co = np.zeros(n)
        co[1:] = open_[1:] / close[:-1] - 1.0
        oc = np.where(open_ != 0, close / open_ - 1.0, 0.0)

    same = pos == prev
    gross = np.where(same, pos * cc, prev * co + pos * oc)
    delta = np.abs(pos - prev)
    cost = cost_per_side * delta
    fund_cost = pos * funding if funding is not None else 0.0
    net = gross - cost - fund_cost
    net = np.nan_to_num(net)

    n_trades = int(np.count_nonzero(np.diff(np.concatenate([[0.0], pos]))))
    return BtResult(
        symbol=symbol,
        variant=variant,
        fast=fast,
        slow=slow,
        timestamps=timestamps,
        positions=pos.astype(np.int8),
        returns=net,
        turnover=float(delta.sum()),
        funding_cost=float(np.sum(fund_cost)),
        n_trades=n_trades,
    )


def extract_trades(
    timestamps: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    positions: np.ndarray,
    *,
    cost_per_side: float,
    funding: np.ndarray | None = None,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """Per-trade records from a held-position array (costs and funding included)."""
    n = positions.size
    rows: list[dict] = []
    t = 0
    while t < n:
        p = positions[t]
        if p == 0:
            t += 1
            continue
        s = t
        while t + 1 < n and positions[t + 1] == p:
            t += 1
        e = t  # last bar held at p
        entry_price = open_[s] if open_[s] != 0 else close[s]
        if e + 1 < n and open_[e + 1] != 0:
            exit_price = open_[e + 1]
            exit_time = timestamps[e + 1]
        else:
            exit_price = close[e]
            exit_time = timestamps[e]
        gross = p * (exit_price / entry_price - 1.0)
        costs = (cost_per_side * 2.0) + (cost_per_side if (e + 1) >= n else 0.0)
        fund = float(np.sum(funding[s : e + 1]) * p) if funding is not None else 0.0
        net = gross - costs - fund
        rows.append(
            {
                "direction": "long" if p > 0 else "short",
                "entry_time": pd.Timestamp(timestamps[s]),
                "exit_time": pd.Timestamp(exit_time),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "ret": float(net),
                "pnl": float(net) * initial_capital,
                "fees": float(cost_per_side * 2.0) * initial_capital,
                "funding": float(fund) * initial_capital,
            }
        )
        t += 1
    df = pd.DataFrame(rows)
    if not df.empty:
        df["holding_period"] = df["exit_time"] - df["entry_time"]
    return df


# --- metrics -----------------------------------------------------------------


def drawdown_stats(equity: np.ndarray) -> tuple[float, float]:
    """(max drawdown, longest peak-to-recovery duration in bars)."""
    if equity.size == 0:
        return float("nan"), float("nan")
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    max_dd = float(dd.min())
    underwater = dd < -1e-12
    longest = 0
    cur = 0
    for u in underwater:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return max_dd, float(longest)


def returns_metrics(
    returns: np.ndarray,
    *,
    ppy: float,
    positions: np.ndarray | None = None,
    turnover: float | None = None,
    funding_cost: float | None = None,
    init_capital: float = 1.0,
) -> dict[str, float]:
    ret = np.asarray(returns, dtype=float)
    ret = ret[np.isfinite(ret)]
    out: dict[str, float] = {}
    if ret.size == 0:
        return {k: float("nan") for k in (
            "n_bars", "total_return", "cagr", "sharpe", "sortino", "volatility",
            "max_drawdown", "dd_duration_bars", "exposure", "turnover", "funding_cost",
        )}
    eq = np.cumprod(1.0 + ret)
    total = float(eq[-1] - 1.0)
    n = ret.size
    years = n / ppy
    cagr = float(eq[-1] ** (1.0 / years) - 1.0) if years > 1e-9 and eq[-1] > 0 else float("nan")
    max_dd, dd_bars = drawdown_stats(eq)
    sd = float(ret.std(ddof=1)) if n > 1 else 0.0
    mean = float(ret.mean())
    sharpe = mean / sd * np.sqrt(ppy) if sd > 0 else float("nan")
    downside = ret[ret < 0]
    dsd = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
    sortino = mean / dsd * np.sqrt(ppy) if dsd > 0 else float("nan")
    out.update(
        n_bars=float(n),
        total_return=total,
        cagr=cagr,
        sharpe=float(sharpe),
        sortino=float(sortino),
        volatility=sd * np.sqrt(ppy),
        max_drawdown=max_dd,
        dd_duration_bars=dd_bars,
        exposure=float(np.mean(positions != 0)) if positions is not None else float("nan"),
        turnover=float(turnover) if turnover is not None else float("nan"),
        funding_cost=float(funding_cost) if funding_cost is not None else 0.0,
    )
    return out


def trade_stats(trades: pd.DataFrame, *, init_capital: float = 1.0) -> dict[str, float]:
    out = {
        "n_trades": 0.0, "win_rate": float("nan"), "profit_factor": float("nan"),
        "expectancy": float("nan"), "avg_trade_ret": float("nan"),
        "avg_holding_bars": float("nan"), "long_share": float("nan"),
        "long_pnl": float("nan"), "short_pnl": float("nan"),
    }
    if trades is None or trades.empty:
        return out
    pnl = trades["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gw, gl = float(wins.sum()), float(losses.sum())
    hold = trades["holding_period"].dt.total_seconds() / 3600.0
    longs = trades[trades["direction"] == "long"]
    out.update(
        n_trades=float(len(pnl)),
        win_rate=float(len(wins) / len(pnl)),
        profit_factor=float(gw / abs(gl)) if gl < 0 else float("nan"),
        expectancy=float(pnl.mean()),
        avg_trade_ret=float(trades["ret"].astype(float).mean()),
        avg_holding_bars=float(hold.mean()),
        long_share=float(len(longs) / len(pnl)),
        long_pnl=float(longs["pnl"].sum()),
        short_pnl=float(trades[trades["direction"] == "short"]["pnl"].sum()),
    )
    return out


# --- statistics --------------------------------------------------------------


def block_bootstrap(
    returns: np.ndarray, stat, *, block: int = 24, n: int = 1000, seed: int = 0
) -> np.ndarray:
    """Stationary contiguous-block bootstrap of a return series."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < block + 1:
        return np.array([])
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(r.size / block))
    out = np.full(n, np.nan)
    for i in range(n):
        starts = rng.integers(0, r.size - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[: r.size]
        out[i] = stat(r[idx])
    return out[np.isfinite(out)]


def sharpe_of(ret: np.ndarray, ppy: float) -> float:
    sd = ret.std(ddof=1)
    return float(ret.mean() / sd * np.sqrt(ppy)) if sd > 0 else float("nan")


# --- data --------------------------------------------------------------------


@dataclass(slots=True)
class SymbolBundle:
    symbol: str
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    funding: np.ndarray
    ema: dict[int, np.ndarray]

    def target(self, variant: str, fast: int, slow: int, *, trend: int = 200) -> np.ndarray:
        return desired_positions(
            self.close, self.ema[fast], self.ema[slow], self.ema[trend], variant
        )


def load_symbol(
    symbol: str,
    *,
    start: str,
    end: str,
    spans: set[int],
    timeframe: str = "1h",
    source_timeframe: str = "1m",
    market: str = "futures",
    exchange: str = "binance",
    root: Path | None = None,
    with_funding: bool = True,
) -> SymbolBundle | None:
    """Load 1h bars (resampled from 1m) plus EMA values and perp funding."""
    from ..config import get_config

    cfg = get_config()
    data_root = Path(root) if root is not None else cfg.paths.data_normalized
    bars = read_dataset(
        data_root,
        DatasetKey(exchange, market, DatasetType.BARS, symbol, source_timeframe),
        start=start,
        end=end,
    )
    if bars.is_empty():
        return None
    bars = resample_bars(bars, timeframe=timeframe, source_timeframe=source_timeframe)
    if bars.height < max(spans) + 5:
        return None

    ema: dict[int, np.ndarray] = {}
    for span in sorted(spans):
        ema[span] = (
            bars.select(pl.col("close").ewm_mean(span=span, adjust=False, min_periods=span))
            .to_series()
            .to_numpy()
            .astype(float)
        )

    funding = np.zeros(bars.height)
    if with_funding:
        fr = read_dataset(
            data_root,
            DatasetKey(exchange, market, DatasetType.FUNDING, symbol),
            start=start,
            end=end,
        )
        if not fr.is_empty():
            fr = fr.with_columns(pl.col("timestamp").dt.truncate("1h")).group_by("timestamp").agg(
                pl.col("funding_rate").sum()
            )
            fdf = fr.to_pandas().set_index("timestamp")["funding_rate"]
            idx = pd.DatetimeIndex(bars["timestamp"].to_pandas())
            funding = fdf.reindex(idx).fillna(0.0).to_numpy()

    return SymbolBundle(
        symbol=symbol,
        timestamps=bars["timestamp"].to_numpy(),
        open=bars["open"].to_numpy().astype(float),
        high=bars["high"].to_numpy().astype(float),
        low=bars["low"].to_numpy().astype(float),
        close=bars["close"].to_numpy().astype(float),
        funding=funding,
        ema=ema,
    )


def periods_per_year_for(timeframe: str) -> float:
    try:
        return periods_per_year(timeframe)
    except ValueError:
        return 365.0 * 24.0


def timeframe_hours(timeframe: str) -> float:
    return timeframe_seconds(timeframe) / 3600.0


__all__ = [
    "BtResult",
    "SymbolBundle",
    "VARIANTS",
    "block_bootstrap",
    "desired_positions",
    "drawdown_stats",
    "extract_trades",
    "load_symbol",
    "periods_per_year_for",
    "returns_metrics",
    "sharpe_of",
    "simulate",
    "timeframe_hours",
    "trade_stats",
]
