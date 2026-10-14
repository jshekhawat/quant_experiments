"""Multi-factor alpha: EMA momentum, CVD/order flow, volatility and price response.

Supports ``reports/EMA_CVD_MFA.md``. Combines, per symbol and per hourly bar:

* **price momentum** — log returns ``R_k`` and volatility-normalised momentum
  ``PR_k = R_k / (RV_k sqrt(k))``;
* **EMA momentum** — rate of change, acceleration and normalised spread velocity
  from :mod:`quant_lab.analytics.ema_momentum`;
* **order flow** — signed taker quote flow ``qd = 2*taker_buy_quote - quote_volume``
  and the normalised flow imbalance ``FI_k = sum(qd,k) / sum(quote_volume,k)``
  (in ``[-1, 1]``), plus its trailing z-score ``Fz_k``;
* **volatility / compression** — Bollinger bandwidth, its trailing z-score and a
  causal ``compress`` flag, realized volatility;
* **price response / absorption** — strong flow with unusually small displacement
  (absorption candidate) versus strong flow with large displacement (continuation).

Everything is trailing/causal and aligned to the close of bar ``t``. Forward
outcomes start at ``t+1`` (one-bar execution delay), matching the previous
studies. Backtesting reuses :mod:`quant_lab.analytics.ema_backtest`.

Data note: order flow is reconstructed *exactly* from Binance kline taker fields
(no tick data), in **quote (USDT)** units so symbols are comparable. There is no
order book, so "absorption" is an observable proxy, never a claim about hidden
liquidity or market-maker intent.

Research code only — no live trading.
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
from . import ema_backtest as eb
from . import ema_momentum as em
from .bollinger_cross import bollinger

# --- constants ---------------------------------------------------------------

PRIMARY_TIMEFRAME = "1h"
SOURCE_TIMEFRAME = "1m"
HORIZONS: dict[str, int] = {"1h": 1, "4h": 4, "8h": 8, "12h": 12, "24h": 24, "48h": 48, "72h": 72}
PRIMARY_HORIZON = "4h"

FLOW_HORIZONS: tuple[int, ...] = (1, 4, 6, 12, 24)
PRICE_HORIZONS: tuple[int, ...] = (1, 3, 4, 6, 12, 24)
NORM_WINDOW = 720        # ~30 days of hourly bars for trailing normalisation
TREND_SPAN = 200
BREAK_LOOKBACK = 24
EXTREME_LOOKBACK = 24
RV_WINDOW = 24
STRONG_Q = 0.90          # trailing quantile defining "strong flow"
LOW_IMPACT_Q = 0.25
HIGH_IMPACT_Q = 0.75
COMPRESS_Q = 0.25

EMA_CONFIGS: tuple[str, ...] = ("ema10_sma20", "ema12_sma20", "ema10_ema30", "ema20_ema50")

#: The predeclared flows constant used by the absorption / continuation track.
PRIMARY_FLOW_HORIZON = 4


# --- per-symbol bundle -------------------------------------------------------


@dataclass(slots=True)
class MfaBundle:
    symbol: str
    frame: pl.DataFrame                    # causal features, one row per bar
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    funding: np.ndarray

    @property
    def n(self) -> int:
        return int(self.close.size)

    def col(self, name: str) -> np.ndarray:
        return self.frame[name].to_numpy().astype(float)


def _ema(close: pl.Expr, span: int) -> pl.Expr:
    return close.ewm_mean(span=span, adjust=False, min_periods=span)


def _zscore(col: str, window: int) -> pl.Expr:
    mean = pl.col(col).rolling_mean(window_size=window, min_periods=max(30, window // 3))
    std = pl.col(col).rolling_std(window_size=window, min_periods=max(30, window // 3))
    return ((pl.col(col) - mean) / std).alias(f"{col}_z")


def _quantile(col: str, q: float, window: int) -> pl.Expr:
    return pl.col(col).rolling_quantile(
        q, interpolation="linear", window_size=window, min_periods=max(30, window // 3)
    )


def load_mfa_bundle(
    symbol: str,
    *,
    start: str,
    end: str,
    timeframe: str = PRIMARY_TIMEFRAME,
    source_timeframe: str = SOURCE_TIMEFRAME,
    market: str = "futures",
    exchange: str = "binance",
    root: Path | None = None,
    with_funding: bool = True,
) -> MfaBundle | None:
    """Load 1h bars and attach every causal multi-factor feature for one symbol."""
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
    if source_timeframe != timeframe:
        bars = resample_bars(bars, timeframe=timeframe, source_timeframe=source_timeframe)
    if bars.height < TREND_SPAN + 5:
        return None
    bars = bars.sort("timestamp")

    # --- order flow (quote units) ---
    frame = bars.with_columns(
        (pl.col("taker_buy_quote_volume")).alias("_tbq"),
        (pl.col("quote_volume") - pl.col("taker_buy_quote_volume")).alias("_tsq"),
    ).with_columns(
        (pl.col("_tbq") - pl.col("_tsq")).alias("qd"),
    )
    # normalised flow imbalance over each rolling horizon
    flow_exprs = []
    for k in FLOW_HORIZONS:
        num = pl.col("qd").rolling_sum(window_size=k, min_periods=k)
        den = pl.col("quote_volume").rolling_sum(window_size=k, min_periods=k)
        flow_exprs.append(pl.when(den > 0).then(num / den).otherwise(None).alias(f"FI_{k}"))
    frame = frame.with_columns(flow_exprs)
    for k in FLOW_HORIZONS:
        frame = frame.with_columns(_zscore(f"FI_{k}", NORM_WINDOW).alias(f"Fz_{k}"))
    frame = frame.with_columns((pl.col("FI_4") - pl.col("FI_4").shift(1)).alias("FI4_chg"))

    # --- price momentum and volatility-normalised momentum ---
    logret = (pl.col("close").log() - pl.col("close").log().shift(1)).alias("logret")
    frame = frame.with_columns(logret).with_columns(
        pl.col("logret").rolling_std(window_size=RV_WINDOW, min_periods=RV_WINDOW).alias("rv")
    )
    price_exprs = []
    for k in PRICE_HORIZONS:
        r = (pl.col("close").log() - pl.col("close").log().shift(k))
        price_exprs.append(r.alias(f"R_{k}"))
        norm = (pl.col("rv") * np.sqrt(k))
        price_exprs.append(
            pl.when(norm > 0).then(r / norm).otherwise(None).alias(f"PR_{k}")
        )
    frame = frame.with_columns(price_exprs)

    # --- Bollinger bandwidth / compression ---
    bands = bollinger(bars["close"].to_numpy().astype(float))
    frame = frame.with_columns(
        pl.Series("bw", bands["width"]),
        pl.Series("bb_mid", bands["mid"]),
    ).with_columns(
        _zscore("bw", NORM_WINDOW),
        (pl.col("bw") - pl.col("bw").shift(1)).alias("bw_chg"),
    ).with_columns(
        (pl.col("bw") <= _quantile("bw", COMPRESS_Q, NORM_WINDOW)).alias("compress")
    )

    # --- EMA momentum features (four reference configs) ---
    ema_spans = sorted({c.fast for c in em.candidate_configs() if c.name in EMA_CONFIGS} |
                       {c.slow for c in em.candidate_configs() if c.name in EMA_CONFIGS}) or [12, 20, 50]
    ema_cols = {span: _ema(pl.col("close"), span) for span in ema_spans}
    frame = frame.with_columns([expr.alias(f"ema_{span}") for span, expr in ema_cols.items()])
    frame = frame.with_columns(
        pl.col("close").rolling_mean(window_size=20, min_periods=20).alias("sma_20")
    )
    ema_exprs = []
    for name in EMA_CONFIGS:
        cfg_ma = em.config_by_name(name)
        fast_col = f"ema_{cfg_ma.fast}" if cfg_ma.fast_kind == "ema" else "sma_20"
        slow_col = f"ema_{cfg_ma.slow}" if cfg_ma.slow_kind == "ema" else "sma_20"
        fast, slow = pl.col(fast_col), pl.col(slow_col)
        roc = (fast / fast.shift(1) - 1.0)
        ema_exprs.append(roc.alias(f"{name}_roc1"))
        spread = pl.when(pl.col("rv") > 0).then((fast - slow) / (pl.col("rv") * pl.col("close"))).otherwise(None)
        ema_exprs.append((spread - spread.shift(1)).alias(f"{name}_svel1"))
        ema_exprs.append((roc - roc.shift(1)).alias(f"{name}_accel1"))
    frame = frame.with_columns(ema_exprs)

    # --- trend / extremes / breakouts (causal) ---
    frame = frame.with_columns(
        _ema(pl.col("close"), TREND_SPAN).alias("ema_trend"),
    ).with_columns(
        (pl.col("close") > pl.col("ema_trend")).alias("trend_up"),
        (pl.col("close") < pl.col("ema_trend")).alias("trend_dn"),
        (pl.col("close") < pl.col("close").rolling_min(window_size=EXTREME_LOOKBACK, min_periods=EXTREME_LOOKBACK).shift(1)).alias("new_low"),
        (pl.col("close") > pl.col("close").rolling_max(window_size=EXTREME_LOOKBACK, min_periods=EXTREME_LOOKBACK).shift(1)).alias("new_high"),
        (pl.col("close") > pl.col("close").rolling_max(window_size=BREAK_LOOKBACK, min_periods=BREAK_LOOKBACK).shift(1)).alias("break_up"),
        (pl.col("close") < pl.col("close").rolling_min(window_size=BREAK_LOOKBACK, min_periods=BREAK_LOOKBACK).shift(1)).alias("break_dn"),
        (pl.col("close") > pl.col("close").shift(1)).alias("up_bar"),
    )

    # --- absorption / continuation gates and divergence ---
    mp = max(30, NORM_WINDOW // 3)
    frame = frame.with_columns(
        (pl.col("Fz_4").abs() >= pl.col("Fz_4").abs().rolling_quantile(
            STRONG_Q, interpolation="linear", window_size=NORM_WINDOW, min_periods=mp
        )).alias("flow_strong"),
        (pl.col("PR_4").abs() <= pl.col("PR_4").abs().rolling_quantile(
            LOW_IMPACT_Q, interpolation="linear", window_size=NORM_WINDOW, min_periods=mp
        )).alias("low_impact"),
        (pl.col("PR_4").abs() >= pl.col("PR_4").abs().rolling_quantile(
            HIGH_IMPACT_Q, interpolation="linear", window_size=NORM_WINDOW, min_periods=mp
        )).alias("high_impact"),
        (-pl.col("R_6").sign() * pl.col("FI_6")).alias("DIV_6"),
    )

    frame = frame.drop(["_tbq", "_tsq", "ema_trend", "bb_mid"] +
                       [f"ema_{s}" for s in ema_spans] + ["sma_20"], strict=False)

    ts = frame["timestamp"].to_numpy()
    funding = np.zeros(frame.height)
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
            idx = pd.DatetimeIndex(frame["timestamp"].to_pandas())
            funding = fdf.reindex(idx).fillna(0.0).to_numpy()

    return MfaBundle(
        symbol=symbol,
        frame=frame,
        timestamps=ts,
        open=frame["open"].to_numpy().astype(float),
        high=frame["high"].to_numpy().astype(float),
        low=frame["low"].to_numpy().astype(float),
        close=frame["close"].to_numpy().astype(float),
        funding=funding,
    )


# --- forward outcomes --------------------------------------------------------


def add_forward_outcomes(frame: pl.DataFrame, horizons: dict[str, int] | None = None) -> pl.DataFrame:
    """Attach forward returns (entry close ``t+1``) and unoriented MFE/MAE."""
    horizons = horizons or HORIZONS
    out = frame
    for name, bars in horizons.items():
        out = out.with_columns(
            (pl.col("close").shift(-(1 + bars)) / pl.col("close").shift(-1) - 1.0).alias(f"fwd_{name}")
        )
    high = frame["high"].to_numpy().astype(float)
    low = frame["low"].to_numpy().astype(float)
    close = frame["close"].to_numpy().astype(float)
    for name, bars in horizons.items():
        mfe = np.full(close.size, np.nan)
        mae = np.full(close.size, np.nan)
        for i in range(close.size):
            j0, j1 = i + 1, min(close.size, i + 1 + bars)
            if j1 <= j0 or not close[i]:
                continue
            mfe[i] = (np.max(high[j0:j1]) - close[i]) / close[i]
            mae[i] = (close[i] - np.min(low[j0:j1])) / close[i]
        out = out.with_columns(pl.Series(f"mfe_{name}", mfe), pl.Series(f"mae_{name}", mae))
    return out


def build_mfa_panel(
    symbols,
    *,
    start: str,
    end: str,
    timeframe: str = PRIMARY_TIMEFRAME,
    source_timeframe: str = SOURCE_TIMEFRAME,
    market: str = "futures",
    root: Path | None = None,
    with_funding: bool = True,
    bundles: dict[str, MfaBundle] | None = None,
) -> pl.DataFrame:
    """Long panel with features + forward outcomes for a symbol list."""
    frames: list[pl.DataFrame] = []
    for symbol in symbols:
        b = (bundles or {}).get(symbol)
        if b is None:
            b = load_mfa_bundle(
                symbol, start=start, end=end, timeframe=timeframe,
                source_timeframe=source_timeframe, market=market, root=root, with_funding=with_funding,
            )
        if b is None:
            continue
        frame = add_forward_outcomes(b.frame)
        frames.append(frame.with_columns(pl.lit(symbol).alias("symbol")))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort(["timestamp", "symbol"])


# --- estimators --------------------------------------------------------------


def boot_1d(x: np.ndarray, stat, *, block: int = 24, n: int = 200, seed: int = 0) -> np.ndarray:
    """Contiguous-block bootstrap of a one-dimensional statistic."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    N = v.size
    if N < block + 1:
        return np.array([])
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(N / block))
    out = np.full(n, np.nan)
    for i in range(n):
        starts = rng.integers(0, N - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:N]
        out[i] = stat(v[idx])
    return out[np.isfinite(out)]


def oriented_summary(y: np.ndarray, *, block: int = 24, n: int = 200, seed: int = 0) -> dict[str, float]:
    """n, hit rate, mean/median, and a block-bootstrap CI on the mean."""
    v = np.asarray(y, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "hit": np.nan, "mean": np.nan, "median": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    vals = boot_1d(v, lambda a: float(a.mean()), block=block, n=n, seed=seed)
    return {
        "n": int(v.size), "hit": float((v > 0).mean()), "mean": float(v.mean()),
        "median": float(np.median(v)),
        "ci_lo": float(np.percentile(vals, 2.5)) if vals.size else np.nan,
        "ci_hi": float(np.percentile(vals, 97.5)) if vals.size else np.nan,
    }


def demean_by_symbol(x: np.ndarray, sid: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    for s in np.unique(sid):
        m = sid == s
        out[m] = out[m] - np.nanmean(out[m])
    return out


def hac_regression(y, base, add, *, maxlags: int = 24) -> dict[str, float]:
    """OLS of ``y`` on ``base`` vs ``base + add`` with HAC errors; incremental R² and t."""
    import statsmodels.api as sm

    X = np.column_stack([base, add])
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if int(m.sum()) < 50:
        return {"n": 0, "r2_base": np.nan, "r2_full": np.nan, "delta_r2": np.nan, "t_add": np.nan}
    yv = y[m]
    rb = sm.OLS(yv, sm.add_constant(base[m])).fit()
    rf = sm.OLS(yv, sm.add_constant(X[m])).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {
        "n": int(m.sum()), "r2_base": float(rb.rsquared), "r2_full": float(rf.rsquared),
        "delta_r2": float(rf.rsquared - rb.rsquared), "t_add": float(rf.tvalues[1 + base.shape[1]]),
    }


# --- strategies --------------------------------------------------------------

STRATEGIES: tuple[str, ...] = (
    "momentum", "flow", "momentum_flow", "absorption_rev", "continuation",
    "trend_pullback_flow", "compression_break_flow",
)
SIDES: tuple[str, ...] = ("long", "short", "both")


def state_positions(bull: np.ndarray, bear: np.ndarray, side: str) -> np.ndarray:
    """``+1`` while ``bull`` holds, ``-1`` while ``bear`` holds, filtered by side."""
    pos = np.where(bull, 1, np.where(bear, -1, 0)).astype(np.int8)
    if side == "long":
        return np.where(pos == 1, 1, 0).astype(np.int8)
    if side == "short":
        return np.where(pos == -1, -1, 0).astype(np.int8)
    return pos


def strategy_target(bundle: MfaBundle, strategy: str, side: str) -> np.ndarray:
    """Target position (decided at bar close) for one multi-factor strategy."""
    c = {name: bundle.col(name) for name in bundle.frame.columns}
    r24 = c["R_24"]
    fi = c[f"FI_{PRIMARY_FLOW_HORIZON}"]
    fz = c[f"Fz_{PRIMARY_FLOW_HORIZON}"]
    trend_up, trend_dn = c["trend_up"] > 0, c["trend_dn"] > 0
    strong = c["flow_strong"] > 0
    low_impact = c["low_impact"] > 0
    high_impact = c["high_impact"] > 0
    pull_up = np.isfinite(c["R_6"]) & (c["R_6"] < 0)
    pull_dn = np.isfinite(c["R_6"]) & (c["R_6"] > 0)
    flowturn_up = c["FI4_chg"] > 0
    flowturn_dn = c["FI4_chg"] < 0
    up_bar = c["up_bar"] > 0
    brk_up, brk_dn = c["break_up"] > 0, c["break_dn"] > 0
    comp = c["compress"] > 0

    if strategy == "momentum":
        return state_positions(r24 > 0, r24 < 0, side)
    if strategy == "flow":
        return state_positions(fi > 0, fi < 0, side)
    if strategy == "momentum_flow":
        return state_positions((r24 > 0) & (fi > 0), (r24 < 0) & (fi < 0), side)
    if strategy == "absorption_rev":            # E1: strong flow, weak price response → reversal
        return state_positions(strong & low_impact & (fi < 0), strong & low_impact & (fi > 0), side)
    if strategy == "continuation":               # E2: strong flow, large price response → continuation
        return state_positions(strong & high_impact & (fi > 0), strong & high_impact & (fi < 0), side)
    if strategy == "trend_pullback_flow":        # C: trend + pullback + flow reversal + confirmation
        return state_positions(
            trend_up & pull_up & flowturn_up & up_bar,
            trend_dn & pull_dn & flowturn_dn & ~up_bar, side)
    if strategy == "compression_break_flow":     # D: compression + breakout + flow impulse
        return state_positions(brk_up & comp & (fz > 0), brk_dn & comp & (fz < 0), side)
    raise KeyError(strategy)


def simulate(bundle: MfaBundle, target: np.ndarray, *, cost_per_side: float) -> eb.BtResult:
    return eb.simulate(
        bundle.timestamps, bundle.open, bundle.close, target,
        cost_per_side=cost_per_side, funding=bundle.funding, symbol=bundle.symbol,
    )


__all__ = [
    "BREAK_LOOKBACK",
    "EMA_CONFIGS",
    "FLOW_HORIZONS",
    "HORIZONS",
    "MfaBundle",
    "NORM_WINDOW",
    "PRICE_HORIZONS",
    "PRIMARY_FLOW_HORIZON",
    "PRIMARY_HORIZON",
    "PRIMARY_TIMEFRAME",
    "SIDES",
    "STRATEGIES",
    "add_forward_outcomes",
    "boot_1d",
    "build_mfa_panel",
    "demean_by_symbol",
    "hac_regression",
    "load_mfa_bundle",
    "oriented_summary",
    "simulate",
    "state_positions",
    "strategy_target",
]
