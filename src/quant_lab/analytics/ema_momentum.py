"""EMA momentum (rate of change), acceleration and spread dynamics.

This module supports the ``reports/EMA_MOMENTUM.md`` study: does an EMA's
**rate of change** (ROC), its **acceleration** (change in ROC) and the
**normalised distance / velocity between two moving averages** anticipate a
conventional crossover, and does that anticipation carry tradable information?

Everything here is causal and aligned to the close of bar ``t``:

* Feature A — EMA slope / rate of change: ``(MA_t / MA_{t-k} - 1) / k``;
* Feature B — EMA acceleration: ``ROC_{k,t} - ROC_{k,t-1}``;
* Feature C — normalised spread: ``(MA_fast - MA_slow) / ATR`` with Wilder ATR;
* Feature D — spread velocity: ``(D_t - D_{t-k}) / k``.

The module also detects *early-warning* conditions (momentum deceleration,
momentum reversal, spread convergence, their combinations), measures the
**lead time** to the reference crossover and **false-turn** rates, and builds
latch-based target positions for the strategy variants. Backtesting itself
reuses :mod:`quant_lab.analytics.ema_backtest` so execution, costs and perp
funding stay identical to the earlier EMA studies.

Sign conventions
----------------
``sign = +1`` while fast MA > slow MA, ``-1`` while fast MA < slow MA and ``0``
during warm-up. For the normalised spread ``D`` the sign matches ``sign`` (ATR
is positive), so ``D < 0`` means the fast average is below the slow average.

Research code only — no live trading, no order submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ..features.core import atr as atr_expr
from . import ema_backtest as eb
from .bollinger_cross import cross_events, cross_sign, forward_returns

# --- pre-registered constants ------------------------------------------------

ATR_PERIOD = 14
ROC_LOOKBACKS: tuple[int, ...] = (1, 3, 6, 12)
HORIZONS: dict[str, int] = {"1h": 1, "3h": 3, "6h": 6, "12h": 12, "24h": 24, "48h": 48, "72h": 72}
REF_SMA_PERIOD = 20

#: Family A — EMA(n) crossed against the fixed Bollinger basis SMA(20).
FAMILY_A_EMA_PERIODS: tuple[int, ...] = (5, 8, 10, 12, 15, 20, 25, 30, 40, 50)
#: Family B — fast EMA vs slow EMA (fast < slow).
FAMILY_B_PAIRS: tuple[tuple[int, int], ...] = (
    (5, 20), (8, 20), (10, 20), (10, 30), (12, 26), (12, 30), (15, 30), (20, 50),
)

#: Pre-registered reference candidates (chosen from the prior study, not here).
PRIMARY_CONFIG = "ema12_sma20"
SECONDARY_CONFIG = "ema10_ema30"

LEAD_WINDOWS_H: tuple[int, ...] = (6, 12, 24, 48)
FALSE_TURN_H = 24            # horizon (hours) for the volatility-scaled false-turn test
FALSE_TURN_MULTS: tuple[float, ...] = (0.5, 1.0, 1.5)

STRATEGIES: tuple[str, ...] = ("cross", "roc", "velocity", "combined", "confirmed")
SIDES: tuple[str, ...] = ("long", "short", "both")


# --- small numeric helpers ---------------------------------------------------


def _ffill(a: np.ndarray) -> np.ndarray:
    idx = np.where(~np.isfinite(a), 0, np.arange(a.size))
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(x.size, np.nan, dtype=float)
    if k <= 0:
        return x.astype(float, copy=True)
    if k < x.size:
        out[k:] = x[:-k]
    return out


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(den) > 0, num / den, np.nan)
    return out


def ema_roc(ma: np.ndarray, k: int) -> np.ndarray:
    """Average proportional change per bar of a moving average over ``k`` bars."""
    prev = _shift(ma, k)
    return _safe_div(ma - prev, prev) / float(k)


def ema_acceleration(roc_k: np.ndarray) -> np.ndarray:
    """Change in the ROC between consecutive bars (``ROC_t - ROC_{t-1}``)."""
    return roc_k - _shift(roc_k, 1)


def normalized_spread(fast: np.ndarray, slow: np.ndarray, atr: np.ndarray) -> np.ndarray:
    """``(fast - slow) / ATR`` — a volatility-normalised EMA spread."""
    return _safe_div(fast - slow, atr)


def spread_velocity(spread: np.ndarray, k: int) -> np.ndarray:
    """Average change per bar of the normalised spread over ``k`` bars."""
    return (_shift(spread, 0) - _shift(spread, k)) / float(k)


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, *, period: int = ATR_PERIOD) -> np.ndarray:
    """Wilder's ATR (RMA of true range) matching ``features.core.atr``."""
    frame = pl.DataFrame({"high": high, "low": low, "close": close})
    return frame.select(atr_expr(frame, period=period)).to_series().to_numpy().astype(float)


def sma_series(close: np.ndarray, window: int) -> np.ndarray:
    return (
        pl.Series("close", np.asarray(close, dtype=float))
        .rolling_mean(window_size=window, min_periods=window)
        .to_numpy()
        .astype(float)
    )


def sign_events(x: np.ndarray) -> np.ndarray:
    """``+1`` when ``x`` crosses above 0, ``-1`` when it crosses below, else 0."""
    prev = _shift(x, 1)
    up = (x > 0) & (prev <= 0)
    dn = (x < 0) & (prev >= 0)
    ev = np.zeros(x.size, dtype=np.int8)
    ev[up] = 1
    ev[dn] = -1
    return ev


# --- configuration -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MAConfig:
    """A fast/slow moving-average pair (each leg simple or exponential)."""

    name: str
    family: str          # "ema_sma" | "ema_ema"
    fast_kind: str       # "ema" | "sma"
    fast: int
    slow_kind: str
    slow: int

    def describe(self) -> str:
        fk = self.fast_kind.upper()
        sk = self.slow_kind.upper()
        return f"{fk}({self.fast}) vs {sk}({self.slow})"


def candidate_configs() -> list[MAConfig]:
    """The limited candidate set: Family A (EMA vs SMA20) + Family B (EMA/EMA)."""
    cfgs: list[MAConfig] = []
    for p in FAMILY_A_EMA_PERIODS:
        cfgs.append(MAConfig(f"ema{p}_sma{REF_SMA_PERIOD}", "ema_sma", "ema", p, "sma", REF_SMA_PERIOD))
    for fast, slow in FAMILY_B_PAIRS:
        cfgs.append(MAConfig(f"ema{fast}_ema{slow}", "ema_ema", "ema", fast, "ema", slow))
    return cfgs


def config_by_name(name: str) -> MAConfig:
    for cfg in candidate_configs():
        if cfg.name == name:
            return cfg
    raise KeyError(name)


@dataclass(slots=True)
class FeatureSet:
    """All four feature families for one moving-average configuration."""

    config: MAConfig
    roc: dict[int, np.ndarray]
    accel: dict[int, np.ndarray]
    spread: np.ndarray
    spread_vel: dict[int, np.ndarray]
    sign: np.ndarray

    def feature(self, name: str) -> np.ndarray:
        if name == "spread":
            return self.spread
        if name == "sign":
            return self.sign.astype(float)
        if name.startswith("roc_"):
            return self.roc[int(name.split("_")[1])]
        if name.startswith("accel_"):
            return self.accel[int(name.split("_")[1])]
        if name.startswith("svel_"):
            return self.spread_vel[int(name.split("_")[1])]
        raise KeyError(name)

    def feature_names(self) -> list[str]:
        names = ["spread"]
        for k in self.roc:
            names.append(f"roc_{k}")
        for k in self.accel:
            names.append(f"accel_{k}")
        for k in self.spread_vel:
            names.append(f"svel_{k}")
        return names


@dataclass(slots=True)
class Signals:
    """Early-warning conditions for one configuration (boolean per bar).

    ``*_bull``/``*_bear`` are mirror conditions. ``reversal_*`` and
    ``converge_*``/``combined_*`` are *events* (the condition became true on
    this bar); the deceleration arrays are states (the condition is currently
    true), because deceleration is a condition, not a crossing.
    """

    decel_bull: np.ndarray
    decel_bear: np.ndarray
    reversal_bull: np.ndarray
    reversal_bear: np.ndarray
    converge_bull: np.ndarray
    converge_bear: np.ndarray
    combined_bull: np.ndarray
    combined_bear: np.ndarray
    confirmed_bull: np.ndarray
    confirmed_bear: np.ndarray


def build_signals(fs: FeatureSet, *, k: int = 1, confirm_window: int = 12) -> Signals:
    """Detect the B1–B4 early-warning conditions (bearish mirror included)."""
    roc = fs.roc[k]
    accel = fs.accel[k]
    vel = fs.spread_vel[k]
    sign = fs.sign
    below = sign < 0          # fast MA below slow MA
    above = sign > 0

    decel_bull = (roc < 0) & (accel > 0) & below
    decel_bear = (roc > 0) & (accel < 0) & above

    roc_ev = sign_events(roc)
    reversal_bull = (roc_ev == 1) & below
    reversal_bear = (roc_ev == -1) & above

    vel_ev = sign_events(vel)
    converge_bull = (vel_ev == 1) & below       # spread rising toward 0 while negative
    converge_bear = (vel_ev == -1) & above      # spread falling toward 0 while positive

    combined_bull = reversal_bull & (vel > 0)
    combined_bear = reversal_bear & (vel < 0)

    confirmed_bull = confirmation_events(converge_bull, reversal_bull, confirm_window)
    confirmed_bear = confirmation_events(converge_bear, reversal_bear, confirm_window)

    return Signals(
        decel_bull=decel_bull, decel_bear=decel_bear,
        reversal_bull=reversal_bull, reversal_bear=reversal_bear,
        converge_bull=converge_bull, converge_bear=converge_bear,
        combined_bull=combined_bull, combined_bear=combined_bear,
        confirmed_bull=confirmed_bull, confirmed_bear=confirmed_bear,
    )


def confirmation_events(early_ev: np.ndarray, confirm_ev: np.ndarray, window: int) -> np.ndarray:
    """Mark the first ``confirm_ev`` within ``window`` bars after each ``early_ev``."""
    out = np.zeros(early_ev.size, dtype=bool)
    n = early_ev.size
    for i in np.flatnonzero(early_ev):
        lo, hi = i + 1, min(n, i + 1 + window)
        if hi <= lo:
            continue
        j = np.flatnonzero(confirm_ev[lo:hi])
        if j.size:
            out[lo + j[0]] = True
    return out


# --- bundle ------------------------------------------------------------------


@dataclass(slots=True)
class MomentumBundle:
    symbol: str
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    funding: np.ndarray
    atr: np.ndarray
    sma: dict[int, np.ndarray]
    ema: dict[int, np.ndarray]
    features: dict[str, FeatureSet]
    signals: dict[int, dict[str, Signals]] = field(default_factory=dict)

    def ma(self, kind: str, period: int) -> np.ndarray:
        return self.sma[period] if kind == "sma" else self.ema[period]

    @property
    def n(self) -> int:
        return int(self.close.size)


def needed_spans() -> set[int]:
    spans: set[int] = {REF_SMA_PERIOD}
    for cfg in candidate_configs():
        spans.add(cfg.fast)
        spans.add(cfg.slow)
    return spans


def load_momentum_bundle(
    symbol: str,
    *,
    start: str,
    end: str,
    timeframe: str = "1h",
    source_timeframe: str = "1m",
    market: str = "futures",
    exchange: str = "binance",
    root: Path | None = None,
    with_funding: bool = True,
) -> MomentumBundle | None:
    """Load 1h bars (via the EMA engine loader), then compute all features."""
    spans = sorted(needed_spans())
    base = eb.load_symbol(
        symbol, start=start, end=end, spans=set(spans), timeframe=timeframe,
        source_timeframe=source_timeframe, market=market, exchange=exchange,
        root=root, with_funding=with_funding,
    )
    if base is None:
        return None

    atr = wilder_atr(base.high, base.low, base.close)
    sma = {REF_SMA_PERIOD: sma_series(base.close, REF_SMA_PERIOD)}

    features: dict[str, FeatureSet] = {}
    for cfg in candidate_configs():
        fast = base.ema[cfg.fast] if cfg.fast_kind == "ema" else sma[cfg.fast]
        slow = base.ema[cfg.slow] if cfg.slow_kind == "ema" else sma[cfg.slow]
        roc = {k: ema_roc(fast, k) for k in ROC_LOOKBACKS}
        accel = {k: ema_acceleration(roc[k]) for k in ROC_LOOKBACKS}
        d = normalized_spread(fast, slow, atr)
        svel = {k: spread_velocity(d, k) for k in ROC_LOOKBACKS}
        features[cfg.name] = FeatureSet(
            config=cfg, roc=roc, accel=accel, spread=d, spread_vel=svel,
            sign=cross_sign(fast, slow),
        )

    bundle = MomentumBundle(
        symbol=symbol, timestamps=base.timestamps, open=base.open, high=base.high,
        low=base.low, close=base.close, funding=base.funding, atr=atr, sma=sma,
        ema=base.ema, features=features,
    )
    for k in ROC_LOOKBACKS:
        bundle.signals[k] = {name: build_signals(fs, k=k) for name, fs in features.items()}
    return bundle


# --- lead-time / false-turn analysis -----------------------------------------


def _slice_stats(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, atr: np.ndarray, i: int, d: int, horizon: int
) -> dict[str, float]:
    """Oriented forward returns, MFE/MAE (ATR units) and false-turn flags."""
    n = close.size
    j0, j1 = i + 1, min(n, i + 1 + horizon)
    ref = close[i]
    unit = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else np.nan
    out: dict[str, float] = {"mfe_atr": np.nan, "mae_atr": np.nan}
    for m in FALSE_TURN_MULTS:
        out[f"false_turn_{m:g}"] = np.nan
    if j1 <= j0 or not np.isfinite(ref) or not np.isfinite(unit):
        return out
    hi, lo = high[j0:j1], low[j0:j1]
    if d > 0:
        fav_move = float(np.max(hi) - ref)
        adv_move = float(ref - np.min(lo))
    else:
        fav_move = float(ref - np.min(lo))
        adv_move = float(np.max(hi) - ref)
    out["mfe_atr"] = fav_move / unit
    out["mae_atr"] = adv_move / unit
    for m in FALSE_TURN_MULTS:
        fav_level = m * unit
        adv_level = m * unit
        if d > 0:
            fav_hit = np.flatnonzero(hi >= ref + fav_level)
            adv_hit = np.flatnonzero(lo <= ref - adv_level)
        else:
            fav_hit = np.flatnonzero(lo <= ref - fav_level)
            adv_hit = np.flatnonzero(hi >= ref + adv_level)
        ft = float(adv_hit.size > 0 and (fav_hit.size == 0 or adv_hit[0] < fav_hit[0]))
        out[f"false_turn_{m:g}"] = ft
    return out


def analyze_lead_time(
    bundle: MomentumBundle,
    config_name: str,
    *,
    k: int = 1,
    horizons: dict[str, int] | None = None,
    false_turn_h: int = FALSE_TURN_H,
) -> pd.DataFrame:
    """Per-event lead time, false-turn and forward-return metrics.

    One row per early-warning *event* (reversal / convergence / combined /
    confirmation) plus momentum-deceleration *onsets*. The reference event is
    the conventional crossover of the **same two moving averages**.
    """
    horizons = horizons or HORIZONS
    fs = bundle.features[config_name]
    sig = bundle.signals[k][config_name]

    fast = bundle.ma(fs.config.fast_kind, fs.config.fast)
    slow = bundle.ma(fs.config.slow_kind, fs.config.slow)
    cross_ev = cross_events(fast, slow)
    cross_idx = np.flatnonzero(cross_ev)
    cross_dir = cross_ev[cross_idx]
    same_idx = {1: cross_idx[cross_dir == 1], -1: cross_idx[cross_dir == -1]}
    ts = np.asarray(bundle.timestamps).astype("datetime64[ns]").astype(np.int64)
    fwd = forward_returns(bundle.close, horizons)

    decel_bull_on = sig.decel_bull & ~(_shift(sig.decel_bull.astype(float), 1) == 1)
    decel_bear_on = sig.decel_bear & ~(_shift(sig.decel_bear.astype(float), 1) == 1)

    def _dir(flag: np.ndarray, d: int) -> np.ndarray:
        return flag.astype(np.int8) * d

    event_sets: list[tuple[str, np.ndarray]] = [
        ("decel_bull", _dir(decel_bull_on, 1)),
        ("decel_bear", _dir(decel_bear_on, -1)),
        ("reversal_bull", _dir(sig.reversal_bull, 1)),
        ("reversal_bear", _dir(sig.reversal_bear, -1)),
        ("converge_bull", _dir(sig.converge_bull, 1)),
        ("converge_bear", _dir(sig.converge_bear, -1)),
        ("combined_bull", _dir(sig.combined_bull, 1)),
        ("combined_bear", _dir(sig.combined_bear, -1)),
        ("confirmed_bull", _dir(sig.confirmed_bull, 1)),
        ("confirmed_bear", _dir(sig.confirmed_bear, -1)),
    ]

    rows: list[dict] = []
    for label, ev in event_sets:
        for i in np.flatnonzero(ev):
            d = int(ev[i])
            same = same_idx[d]
            opp = same_idx[-d]
            pos_same = np.searchsorted(same, i, side="right")
            pos_opp = np.searchsorted(opp, i, side="right")
            j_same = int(same[pos_same]) if pos_same < same.size else -1
            j_opp = int(opp[pos_opp]) if pos_opp < opp.size else -1
            lead_same = float(j_same - i) if j_same >= 0 else np.nan
            lead_opp = float(j_opp - i) if j_opp >= 0 else np.nan
            wrong_first = bool(j_opp >= 0 and (j_same < 0 or j_opp < j_same))
            row = {
                "symbol": bundle.symbol, "config": config_name, "signal": label,
                "dir": d, "index": int(i), "timestamp": pd.Timestamp(ts[i]),
                "lead_same_h": lead_same, "lead_opp_h": lead_opp,
                "wrong_direction_first": wrong_first,
            }
            for w in LEAD_WINDOWS_H:
                row[f"within_{w}h"] = bool(np.isfinite(lead_same) and lead_same <= w)
            for name in horizons:
                y = fwd[name][i]
                row[f"fwd_{name}"] = float(d) * float(y) if np.isfinite(y) else np.nan
            row.update(_slice_stats(bundle.close, bundle.high, bundle.low, bundle.atr, i, d, false_turn_h))
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_lead_time(events: pd.DataFrame, horizons: dict[str, int] | None = None) -> pd.DataFrame:
    """Aggregate :func:`analyze_lead_time` rows by signal."""
    horizons = horizons or HORIZONS
    if events.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for label, grp in events.groupby("signal", sort=False):
        lead = grp["lead_same_h"].to_numpy(dtype=float)
        lead = lead[np.isfinite(lead)]
        row: dict[str, float | str | int] = {
            "signal": label, "n": int(grp.shape[0]),
            "median_lead_h": float(np.median(lead)) if lead.size else np.nan,
            "mean_lead_h": float(lead.mean()) if lead.size else np.nan,
            "pct_wrong_direction_first": float(grp["wrong_direction_first"].mean()),
        }
        for w in LEAD_WINDOWS_H:
            row[f"pct_within_{w}h"] = float(grp[f"within_{w}h"].mean())
        for name in horizons:
            col = grp[f"fwd_{name}"].to_numpy(dtype=float)
            col = col[np.isfinite(col)]
            row[f"hit_{name}"] = float((col > 0).mean()) if col.size else np.nan
            row[f"mean_{name}"] = float(col.mean()) if col.size else np.nan
            row[f"median_{name}"] = float(np.median(col)) if col.size else np.nan
        row["mfe_atr"] = float(grp["mfe_atr"].mean())
        row["mae_atr"] = float(grp["mae_atr"].mean())
        for m in FALSE_TURN_MULTS:
            row[f"false_turn_{m:g}"] = float(grp[f"false_turn_{m:g}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# --- positions / strategy variants -------------------------------------------


def latch_positions(bull: np.ndarray, bear: np.ndarray, side: str) -> np.ndarray:
    """Latch ``+1`` on a bull event and ``-1`` on a bear event until the next event."""
    raw = np.where(bull, 1.0, np.where(bear, -1.0, np.nan))
    state = np.nan_to_num(_ffill(raw)).astype(np.int8)
    if side == "long":
        return np.where(state == 1, 1, 0).astype(np.int8)
    if side == "short":
        return np.where(state == -1, -1, 0).astype(np.int8)
    return state


def position_from_sign(sign: np.ndarray, side: str) -> np.ndarray:
    if side == "long":
        return np.where(sign == 1, 1, 0).astype(np.int8)
    if side == "short":
        return np.where(sign == -1, -1, 0).astype(np.int8)
    return sign.astype(np.int8)


def strategy_target(
    bundle: MomentumBundle, config_name: str, strategy: str, side: str, *, k: int = 1
) -> np.ndarray:
    """Target position (decided at close ``t``) for one strategy variant."""
    fs = bundle.features[config_name]
    if strategy == "cross":
        return position_from_sign(fs.sign, side)
    sig = bundle.signals[k][config_name]
    if strategy == "roc":
        bull, bear = sig.reversal_bull, sig.reversal_bear
    elif strategy == "velocity":
        bull, bear = sig.converge_bull, sig.converge_bear
    elif strategy == "combined":
        bull, bear = sig.combined_bull, sig.combined_bear
    elif strategy == "confirmed":
        bull, bear = sig.confirmed_bull, sig.confirmed_bear
    else:
        raise KeyError(strategy)
    return latch_positions(bull, bear, side)


def apply_min_hold(target: np.ndarray, min_bars: int) -> np.ndarray:
    """Suppress reversals/flips that occur fewer than ``min_bars`` bars apart."""
    out = target.copy()
    last_change = -10**9
    cur = 0
    for t in range(target.size):
        want = int(target[t])
        if want != cur and (t - last_change) >= min_bars:
            cur = want
            last_change = t
        out[t] = cur
    return out.astype(np.int8)


def apply_atr_stop(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray,
    target: np.ndarray, *, mult: float = 2.0,
) -> np.ndarray:
    """Flatten a position once it runs ``mult`` × ATR against its entry.

    Re-entry is only allowed when the raw target changes sign (a fresh signal),
    which keeps the rule causal and inspectable.
    """
    n = target.size
    out = np.zeros(n, dtype=np.int8)
    pos = 0
    entry = np.nan
    unit = np.nan
    prev_raw = 0
    stopped = False
    for t in range(n):
        raw = int(target[t])
        if raw != prev_raw:
            stopped = False
        if raw != 0 and not stopped:
            if pos == 0:
                entry = close[t]
                unit = atr[t]
                pos = raw
            elif pos != raw:
                entry = close[t]
                unit = atr[t]
                pos = raw
        if pos != 0 and np.isfinite(unit) and unit > 0:
            adverse = (close[t] - entry) * pos
            if adverse <= -mult * unit:
                stopped = True
                pos = 0
        out[t] = pos
        prev_raw = raw
    return out


def simulate_positions(
    bundle: MomentumBundle, target: np.ndarray, *, cost_per_side: float
) -> eb.BtResult:
    return eb.simulate(
        bundle.timestamps, bundle.open, bundle.close, target,
        cost_per_side=cost_per_side, funding=bundle.funding, symbol=bundle.symbol,
    )


# --- statistics --------------------------------------------------------------


def rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation on the finite overlap."""
    from scipy.stats import rankdata

    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 20:
        return float("nan")
    rx = rankdata(x[m])
    ry = rankdata(y[m])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def block_boot_pair(
    x: np.ndarray, y: np.ndarray, stat, *, block: int = 24, n: int = 200, seed: int = 0
) -> np.ndarray:
    """Contiguous-block bootstrap of a paired statistic (respects overlap)."""
    m = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[m], y[m]
    N = xs.size
    if N < block + 1:
        return np.array([])
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(N / block))
    out = np.full(n, np.nan)
    for i in range(n):
        starts = rng.integers(0, N - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:N]
        out[i] = stat(xs[idx], ys[idx])
    return out[np.isfinite(out)]


def conditional_stats(x: np.ndarray, y: np.ndarray, *, cond: np.ndarray) -> dict[str, float]:
    """Directional hit rate and mean/median forward return under a condition."""
    m = np.isfinite(x) & np.isfinite(y) & cond
    yv = y[m]
    n = int(yv.size)
    if n == 0:
        return {"n": 0, "hit": float("nan"), "mean": float("nan"), "median": float("nan")}
    return {
        "n": n, "hit": float((yv > 0).mean()), "mean": float(yv.mean()),
        "median": float(np.median(yv)),
    }


def incremental_regression(
    y: np.ndarray, base: np.ndarray, add: np.ndarray, *, maxlags: int = 24
) -> dict[str, float]:
    """OLS of ``y`` on ``base`` (baseline) and ``base + add`` (augmented).

    Uses HAC (Newey–West) standard errors so heavy overlap in the forward-return
    windows does not produce spuriously small standard errors. Returns the
    incremental R² and the HAC t-statistic of the first added regressor.
    """
    import statsmodels.api as sm

    X = np.column_stack([base, add])
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if int(m.sum()) < 50:
        return {"n": 0, "r2_base": float("nan"), "r2_full": float("nan"), "delta_r2": float("nan"), "t_add": float("nan")}
    yv = y[m]
    Xb = sm.add_constant(base[m])
    Xf = sm.add_constant(X[m])
    rb = sm.OLS(yv, Xb).fit()
    rf = sm.OLS(yv, Xf).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    t_add = float(rf.tvalues[1 + base.shape[1]])
    return {
        "n": int(m.sum()),
        "r2_base": float(rb.rsquared),
        "r2_full": float(rf.rsquared),
        "delta_r2": float(rf.rsquared - rb.rsquared),
        "t_add": t_add,
    }


def pooled_arrays(
    bundles: list[MomentumBundle], config_name: str, feature: str
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Concatenate a feature and all forward returns across symbols.

    Returns ``(feature, {horizon: forward returns}, timestamps_ns)`` sorted by
    timestamp so block bootstraps remain time-ordered.
    """
    xs: list[np.ndarray] = []
    ys: dict[str, list[np.ndarray]] = {name: [] for name in HORIZONS}
    ts: list[np.ndarray] = []
    for b in bundles:
        fs = b.features[config_name]
        x = fs.feature(feature)
        xs.append(x)
        fwd = forward_returns(b.close, HORIZONS)
        for name in HORIZONS:
            ys[name].append(fwd[name])
        ts.append(np.asarray(b.timestamps).astype("datetime64[ns]").astype(np.int64))
    x_all = np.concatenate(xs)
    ts_all = np.concatenate(ts)
    order = np.argsort(ts_all, kind="stable")
    out_y = {name: np.concatenate(ys[name])[order] for name in HORIZONS}
    return x_all[order], out_y, ts_all[order]


__all__ = [
    "ATR_PERIOD",
    "FALSE_TURN_H",
    "FALSE_TURN_MULTS",
    "FAMILY_A_EMA_PERIODS",
    "FAMILY_B_PAIRS",
    "HORIZONS",
    "LEAD_WINDOWS_H",
    "PRIMARY_CONFIG",
    "REF_SMA_PERIOD",
    "ROC_LOOKBACKS",
    "SECONDARY_CONFIG",
    "SIDES",
    "STRATEGIES",
    "FeatureSet",
    "MAConfig",
    "MomentumBundle",
    "Signals",
    "analyze_lead_time",
    "apply_atr_stop",
    "apply_min_hold",
    "block_boot_pair",
    "build_signals",
    "candidate_configs",
    "conditional_stats",
    "config_by_name",
    "confirmation_events",
    "ema_acceleration",
    "ema_roc",
    "incremental_regression",
    "latch_positions",
    "load_momentum_bundle",
    "needed_spans",
    "normalized_spread",
    "pooled_arrays",
    "position_from_sign",
    "rank_ic",
    "sign_events",
    "simulate_positions",
    "sma_series",
    "spread_velocity",
    "strategy_target",
    "summarize_lead_time",
    "wilder_atr",
]
