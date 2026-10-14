"""H1 — price absorption from aggressive taker flow.

Hypothesis (see ``reports/cvd_absorption_study.md``): when aggressive selling
produces unusually *little* downward price movement, price subsequently
outperforms comparable strong-selling events; symmetrically for aggressive
buying. The signal is the residual of a **rolling price-impact regression**:

    f_t    = signed taker quote flow / trailing activity
    r_t    = bar log return
    r ~ alpha + beta * f           (rolling OLS over a trailing window)
    s_t    = r_t - (alpha + beta * f_t)
    A_t    = -sign(f_t) * s_t      (>0 = aggressor absorbed), z-scored

Everything is trailing/causal; forward outcomes start one bar after the signal.
The naive ``return / flow`` ratio is deliberately avoided (near-zero denominator).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ..config import Config, get_config
from ..data.resample import resample_bars
from ..data.schemas import DatasetType
from ..data.storage import DatasetKey, read_dataset
from ..features.core import safe_div
from ..timeframes import timeframe_seconds
from .divergence import decile_table, pooled_ic
from .regimes import RegimeConfig, attach_regimes, classify_regimes

DEFAULT_HORIZONS: dict[str, int] = {"1h": 1, "4h": 4, "24h": 24, "3d": 72}
PRIMARY_HORIZON = "4h"
PRIMARY_TIMEFRAME = "1h"
PRIMARY_BASELINES: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")

# Calendar horizons in hours; converted to bar counts for a given timeframe.
_CALENDAR_HOURS: dict[str, float] = {"1h": 1, "4h": 4, "24h": 24, "3d": 72}


def horizons_for_timeframe(timeframe: str) -> dict[str, int]:
    """Bar-count horizons at ``timeframe`` for the calendar horizons in hours."""
    secs = timeframe_seconds(timeframe)
    return {
        name: max(1, round(hours * 3600 / secs))
        for name, hours in _CALENDAR_HOURS.items()
        if hours * 3600 >= secs
    }


def _window_bars(days: float, timeframe: str) -> int:
    return max(30, round(days * 86400 / timeframe_seconds(timeframe)))


# --- signal construction -----------------------------------------------------


def normalized_flow(frame: pl.DataFrame, *, adv_window: int = 20) -> pl.DataFrame:
    """Signed taker quote flow, trailing activity and the normalized flow ``f``."""
    return (
        frame.with_columns(
            (2.0 * pl.col("taker_buy_quote_volume") - pl.col("quote_volume")).alias("qd"),
            pl.col("quote_volume")
            .rolling_mean(window_size=adv_window, min_periods=adv_window)
            .alias("adv"),
        )
        .with_columns((pl.col("close").log() - pl.col("close").log().shift(1)).alias("r"))
        .with_columns(safe_div(pl.col("qd"), pl.col("adv")).alias("f"))
    )


def rolling_impact_model(
    frame: pl.DataFrame,
    *,
    window: int = 720,
    min_periods: int | None = None,
    flow: str = "f",
    ret: str = "r",
) -> pl.DataFrame:
    """Rolling OLS of ``ret`` on ``flow`` (with intercept) → ``alpha``, ``beta``, ``s``.

    ``s`` is the realized minus expected same-bar return (the impact residual).
    """
    mp = min_periods or max(30, window // 3)
    f, r = pl.col(flow), pl.col(ret)
    out = frame.with_columns(
        f.rolling_mean(window_size=window, min_periods=mp).alias("_mf"),
        r.rolling_mean(window_size=window, min_periods=mp).alias("_mr"),
        (f * r).rolling_mean(window_size=window, min_periods=mp).alias("_mfr"),
        (f * f).rolling_mean(window_size=window, min_periods=mp).alias("_mf2"),
    )
    out = out.with_columns(
        (pl.col("_mfr") - pl.col("_mf") * pl.col("_mr")).alias("_cov"),
        (pl.col("_mf2") - pl.col("_mf") * pl.col("_mf")).alias("_var"),
    ).with_columns(safe_div(pl.col("_cov"), pl.col("_var")).alias("beta"))
    out = out.with_columns((pl.col("_mr") - pl.col("beta") * pl.col("_mf")).alias("alpha"))
    out = out.with_columns(
        (pl.col("alpha") + pl.col("beta") * f).alias("rhat"),
        (r - (pl.col("alpha") + pl.col("beta") * f)).alias("s"),
    )
    return out.drop("_mf", "_mr", "_mfr", "_mf2", "_cov", "_var")


def absorption_signal(
    frame: pl.DataFrame,
    *,
    adv_window: int = 20,
    impact_window: int = 720,
    impact_min_periods: int | None = None,
    norm_window: int | None = None,
    event_quantile: float = 0.90,
) -> pl.DataFrame:
    """Attach ``f``, impact model, absorption score ``A`` and strong-flow gate."""
    nw = norm_window or impact_window
    nmp = max(30, nw // 3)
    out = rolling_impact_model(
        normalized_flow(frame, adv_window=adv_window),
        window=impact_window,
        min_periods=impact_min_periods,
    )
    out = out.with_columns(
        (-(pl.col("f").sign()) * pl.col("s")).alias("_raw_abs"),
        pl.col("f").abs().alias("abs_f"),
    )
    out = out.with_columns(
        safe_div(
            pl.col("_raw_abs"),
            pl.col("_raw_abs").rolling_std(window_size=nw, min_periods=nmp),
        ).alias("A"),
        pl.col("r").rolling_std(window_size=24, min_periods=24).alias("realized_vol"),
        (pl.col("close") / pl.col("close").shift(24) - 1.0).alias("past_ret"),
    )
    out = out.with_columns(
        pl.col("abs_f")
        .rolling_quantile(
            event_quantile, interpolation="linear", window_size=nw, min_periods=nmp
        )
        .alias("f_q")
    ).with_columns((pl.col("abs_f") >= pl.col("f_q")).alias("strong_flow"))
    return out.drop("_raw_abs")


def _oriented_extremes(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, direction: np.ndarray, bars: int
) -> tuple[np.ndarray, np.ndarray]:
    """Favourable / adverse excursion (fraction) in the predicted direction."""
    n = close.size
    fav = np.full(n, np.nan)
    adv = np.full(n, np.nan)
    for i in range(n):
        d = direction[i]
        if not np.isfinite(d) or d == 0:
            continue
        j0, j1 = i + 1, min(n, i + 1 + bars)
        if j1 <= j0:
            continue
        ref = close[i]
        if not ref:
            continue
        hi = (np.max(high[j0:j1]) - ref) / ref
        lo = (ref - np.min(low[j0:j1])) / ref
        # d = +1 predicts up (fav = hi, adv = lo); d = -1 the mirror.
        fav[i] = hi if d > 0 else lo
        adv[i] = lo if d > 0 else hi
    return fav, adv


def add_outcomes(frame: pl.DataFrame, horizons: dict[str, int] | None = None) -> pl.DataFrame:
    """Attach predicted direction, oriented forward returns, hits and MFE/MAE."""
    horizons = horizons or DEFAULT_HORIZONS
    out = frame.with_columns((-(pl.col("f").sign())).alias("d"))
    for name, bars in horizons.items():
        fwd = pl.col("close").shift(-(1 + bars)) / pl.col("close").shift(-1) - 1.0
        out = out.with_columns(
            fwd.alias(f"fwd_{name}"),
            (pl.col("d") * fwd).alias(f"y_{name}"),
            ((pl.col("d") * fwd) > 0).cast(pl.Int8).alias(f"hit_{name}"),
        )
    high = frame["high"].to_numpy().astype(float)
    low = frame["low"].to_numpy().astype(float)
    close = frame["close"].to_numpy().astype(float)
    direction = out["d"].to_numpy().astype(float)
    for name, bars in horizons.items():
        fav, adv = _oriented_extremes(high, low, close, direction, bars)
        out = out.with_columns(pl.Series(f"mfe_{name}", fav), pl.Series(f"mae_{name}", adv))
    return out


def build_absorption_panel(
    symbols: Sequence[str],
    *,
    start: str,
    end: str,
    timeframe: str = PRIMARY_TIMEFRAME,
    source_timeframe: str = "1m",
    adv_window: int = 20,
    impact_window: int = 720,
    impact_min_periods: int | None = None,
    norm_window: int | None = None,
    event_quantile: float = 0.90,
    horizons: dict[str, int] | None = None,
    config: Config | None = None,
    root: Path | None = None,
) -> pl.DataFrame:
    """Build the perp absorption panel for a symbol list (returns long frame)."""
    cfg = config or get_config()
    data_root = Path(root) if root is not None else cfg.paths.data_normalized
    frames: list[pl.DataFrame] = []
    for symbol in symbols:
        bars = read_dataset(
            data_root,
            DatasetKey("binance", "futures", DatasetType.BARS, symbol, source_timeframe),
            start=start,
            end=end,
        )
        if bars.is_empty():
            continue
        if source_timeframe != timeframe:
            bars = resample_bars(bars, timeframe=timeframe, source_timeframe=source_timeframe)
        if bars.is_empty():
            continue
        sig = absorption_signal(
            bars,
            adv_window=adv_window,
            impact_window=impact_window,
            impact_min_periods=impact_min_periods,
            norm_window=norm_window,
            event_quantile=event_quantile,
        )
        regimes = classify_regimes(
            bars, config=RegimeConfig(regime_timeframe=timeframe), source_timeframe=timeframe
        )
        sig = attach_regimes(sig, regimes)
        sig = add_outcomes(sig, horizons)
        frames.append(sig.with_columns(pl.lit(symbol).alias("symbol")))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort(["timestamp", "symbol"])


def robustness_sweep(
    symbols: Sequence[str],
    *,
    start: str,
    end: str,
    design_end: str = "2026-05-01",
    timeframes: Sequence[str] = ("15m", "1h", "4h"),
    window_days: Sequence[float] = (15.0, 30.0, 60.0),
    quantiles: Sequence[float] = (0.90, 0.95),
    primary_horizon: str = PRIMARY_HORIZON,
    config: Config | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """Design/test pooled IC across timeframe × impact-window × event-quantile.

    IC-only (no bootstrap) so the sweep is cheap; the point is to check the sign
    and magnitude are stable, not to find a winning cell.
    """
    rows: list[dict[str, float | str]] = []
    cutoff = pd.Timestamp(design_end, tz="UTC")
    for tf in timeframes:
        hz = horizons_for_timeframe(tf)
        if primary_horizon not in hz:
            continue
        y_col = f"y_{primary_horizon}"
        for days in window_days:
            window = _window_bars(days, tf)
            for q in quantiles:
                panel = build_absorption_panel(
                    symbols,
                    start=start,
                    end=end,
                    timeframe=tf,
                    impact_window=window,
                    norm_window=window,
                    event_quantile=q,
                    horizons=hz,
                    config=config,
                    root=root,
                )
                if panel.is_empty():
                    continue
                pdf = panel.to_pandas()
                pdf = pdf[pdf[y_col].notna() & pdf["A"].notna()]
                de = pdf[pdf["timestamp"] < cutoff]
                te = pdf[pdf["timestamp"] >= cutoff]
                strong = te[te["strong_flow"].fillna(False)]
                rows.append(
                    {
                        "timeframe": tf,
                        "window_days": days,
                        "impact_window": window,
                        "event_quantile": q,
                        "design_ic": pooled_ic(de, "A", y_col) if not de.empty else float("nan"),
                        "test_ic": pooled_ic(te, "A", y_col) if not te.empty else float("nan"),
                        "strong_ic": pooled_ic(strong, "A", y_col) if not strong.empty else float("nan"),
                        "n_test": float(te.shape[0]),
                    }
                )
    return pd.DataFrame(rows)


# --- generic block bootstrap -------------------------------------------------


def block_bootstrap(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    *,
    n: int = 1000,
    block_timestamps: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Resample contiguous blocks of timestamps and evaluate ``stat_fn``."""
    d = df.sort_values("timestamp")
    times = d["timestamp"].to_numpy()
    if times.size < 2:
        return np.array([])
    change = np.ones(times.size, dtype=bool)
    change[1:] = times[1:] != times[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], times.size)
    t_count = starts.size
    if t_count < 2:
        return np.array([])
    rng = np.random.default_rng(seed)
    blocks = max(1, min(block_timestamps, t_count))
    n_blocks = int(np.ceil(t_count / blocks))
    offsets = np.arange(blocks)
    sig = d.to_numpy()
    out = np.full(n, np.nan)
    for i in range(n):
        s = rng.integers(0, t_count - blocks + 1, size=n_blocks)
        codes = (s[:, None] + offsets[None, :]).ravel()[:t_count]
        st = starts[codes]
        reps = ends[codes] - st
        total = int(reps.sum())
        if total < 3:
            continue
        idx = np.repeat(st, reps) + (np.arange(total) - np.repeat(np.cumsum(reps) - reps, reps))
        out[i] = stat_fn(pd.DataFrame(sig[idx], columns=d.columns))
    return out[np.isfinite(out)]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xs = pd.Series(x[mask]).rank().to_numpy()
    ys = pd.Series(y[mask]).rank().to_numpy()
    if xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


# --- matched controls and regression ----------------------------------------


def matched_spread(
    df: pd.DataFrame,
    y_col: str,
    *,
    signal: str = "A",
    strong: str = "strong_flow",
    top_q: float = 0.7,
    bottom_q: float = 0.3,
    n_bins: int = 3,
    strata: Sequence[str] = ("realized_vol", "past_ret", "abs_f"),
) -> float:
    """Weighted (absorbed − control) oriented return, matched on control strata."""
    d = df[df[strong].fillna(False) & df[signal].notna() & df[y_col].notna()].copy()
    if len(d) < 50:
        return float("nan")
    hi, lo = d[signal].quantile(top_q), d[signal].quantile(bottom_q)
    d = d[(d[signal] >= hi) | (d[signal] <= lo)].copy()
    d["grp"] = (d[signal] >= hi).astype(int)
    for c in strata:
        if c in d.columns:
            d[c + "_b"] = pd.qcut(d[c].rank(method="first"), n_bins, labels=False)
    key_cols = [c + "_b" for c in strata if c + "_b" in d.columns]
    if "regime" in d.columns:
        key_cols.append("regime")
    weights, diffs = [], []
    for _, g in d.groupby(key_cols, dropna=False):
        t = g[g["grp"] == 1][y_col]
        c = g[g["grp"] == 0][y_col]
        if len(t) >= 5 and len(c) >= 5:
            weights.append(len(t) + len(c))
            diffs.append(float(t.mean() - c.mean()))
    if not weights:
        return float("nan")
    w = np.array(weights, dtype=float)
    return float((w * np.array(diffs)).sum() / w.sum())


def absorption_regression(
    df: pd.DataFrame,
    y_col: str,
    *,
    signal: str = "A",
    controls: Sequence[str] = ("past_ret", "realized_vol", "abs_f"),
    regime_col: str = "regime",
) -> float:
    """OLS coefficient on ``signal`` with the specified controls and regime dummies."""
    cols = [signal, y_col, *controls]
    d = df.dropna(subset=[c for c in cols if c in df.columns]).copy()
    if len(d) < 100:
        return float("nan")
    x_parts = [np.ones((len(d), 1)), d[signal].to_numpy(dtype=float)[:, None]]
    for c in controls:
        if c in d.columns:
            x_parts.append(d[c].to_numpy(dtype=float)[:, None])
    if regime_col in d.columns:
        dummies = pd.get_dummies(d[regime_col], drop_first=True, dtype=float)
        if dummies.shape[1]:
            x_parts.append(dummies.to_numpy())
    x = np.hstack(x_parts)
    y = d[y_col].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(beta[1])


# --- orchestration -----------------------------------------------------------


@dataclass(slots=True)
class AbsorptionResult:
    symbols: list[str] = field(default_factory=list)
    timeframe: str = PRIMARY_TIMEFRAME
    primary_horizon: str = PRIMARY_HORIZON
    impact_window: int = 720
    event_quantile: float = 0.90
    design: dict[str, float] = field(default_factory=dict)
    test: dict[str, float] = field(default_factory=dict)
    test_ci: tuple[float, float] | None = None
    strong_flow_ic: float = float("nan")
    quintiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    matched: dict[str, float] = field(default_factory=dict)
    regression: dict[str, float] = field(default_factory=dict)
    loo: pd.DataFrame = field(default_factory=pd.DataFrame)
    ex_megacap_ic: float = float("nan")
    horizon_ic: dict[str, float] = field(default_factory=dict)
    by_symbol: pd.DataFrame = field(default_factory=pd.DataFrame)
    robustness: pd.DataFrame = field(default_factory=pd.DataFrame)
    criteria: dict[str, bool] = field(default_factory=dict)
    verdict: str = "not evaluated"
    notes: list[str] = field(default_factory=list)
    panel: pl.DataFrame = field(default_factory=pl.DataFrame)

    def to_markdown(self) -> str:
        lines = [
            "# H1 — Price Absorption — Results (exploratory, 2026 in-sample)",
            "",
            f"- universe ({len(self.symbols)}): " + ", ".join(self.symbols),
            f"- timeframe {self.timeframe} · horizon {self.primary_horizon} · "
            f"impact_window {self.impact_window} · event q{self.event_quantile:.2f}",
        ]
        if self.notes:
            lines += ["", "## Notes"] + [f"- {n}" for n in self.notes]
        lines += ["", "## Primary: pooled rank IC(A, oriented forward return)", ""]
        lines += ["| period | IC | n |", "|---|---|---|"]
        for name, block in (("design", self.design), ("test", self.test)):
            lines.append(f"| {name} | {block.get('ic', float('nan')):.4f} | {block.get('n', 0):.0f} |")
        if self.test_ci is not None:
            lines.append("")
            lines.append(f"test IC 95% bootstrap CI: [{self.test_ci[0]:.4f}, {self.test_ci[1]:.4f}]")
        lines.append(f"strong-flow-only test IC: {self.strong_flow_ic:.4f}")
        lines += ["", "## Horizon consistency (test IC)", ""]
        lines += ["| horizon | IC |", "|---|---|"]
        for h, v in self.horizon_ic.items():
            lines.append(f"| {h} | {v:.4f} |")
        if not self.quintiles.empty:
            lines += ["", "## Test quintiles (A → oriented return, strong-flow)", ""]
            lines += ["| quintile | n | mean | median |", "|---|---|---|---|"]
            for row in self.quintiles.itertuples(index=False):
                lines.append(f"| {row.decile} | {row.count} | {row.mean:.5f} | {row.median:.5f} |")
        if self.matched:
            lines += [
                "",
                "## Matched control (absorbed − control, strong-flow)",
                "",
                f"- spread: {self.matched.get('spread', float('nan')) * 1e4:.2f} bps "
                f"(CI [{self.matched.get('ci_low', float('nan')) * 1e4:.2f}, "
                f"{self.matched.get('ci_high', float('nan')) * 1e4:.2f}])",
                f"- n events: {self.matched.get('n', 0):.0f}",
            ]
        if self.regression:
            lines += [
                "",
                "## Control regression (A coefficient)",
                "",
                f"- coef: {self.regression.get('coef', float('nan')):.5f} "
                f"(CI [{self.regression.get('ci_low', float('nan')):.5f}, "
                f"{self.regression.get('ci_high', float('nan')):.5f}])",
            ]
        if not self.by_symbol.empty:
            lines += ["", "## Per-symbol test IC", "", "| symbol | IC | n |", "|---|---|---|"]
            for row in self.by_symbol.itertuples(index=False):
                lines.append(f"| {row.symbol} | {row.ic:.4f} | {row.n:.0f} |")
        if not self.robustness.empty:
            lines += [
                "",
                "## Robustness sweep (test IC; sign/magnitude stability)",
                "",
                "| timeframe | window_days | event_q | design_ic | test_ic | strong_ic | n_test |",
                "|---|---|---|---|---|---|---|",
            ]
            for row in self.robustness.itertuples(index=False):
                lines.append(
                    f"| {row.timeframe} | {row.window_days:g} | {row.event_quantile:.2f} | "
                    f"{row.design_ic:.4f} | {row.test_ic:.4f} | {row.strong_ic:.4f} | {row.n_test:.0f} |"
                )
        lines += ["", f"## Verdict: **{self.verdict}**", "", "| # | criterion | pass |", "|---|---|---|"]
        for key, passed in self.criteria.items():
            lines.append(f"| {key} | | {'✅' if passed else '❌'} |")
        return "\n".join(lines) + "\n"


def _ic(df: pd.DataFrame, y_col: str) -> float:
    return pooled_ic(df, "A", y_col)


def run_absorption_study(
    *,
    symbols: Sequence[str],
    start: str,
    end: str,
    design_end: str = "2026-05-01",
    timeframe: str = PRIMARY_TIMEFRAME,
    source_timeframe: str = "1m",
    adv_window: int = 20,
    impact_window: int = 720,
    event_quantile: float = 0.90,
    primary_horizon: str = PRIMARY_HORIZON,
    bootstrap: int = 1000,
    sweep: bool = True,
    config: Config | None = None,
    root: Path | None = None,
) -> AbsorptionResult:
    panel = build_absorption_panel(
        symbols,
        start=start,
        end=end,
        timeframe=timeframe,
        source_timeframe=source_timeframe,
        adv_window=adv_window,
        impact_window=impact_window,
        event_quantile=event_quantile,
        config=config,
        root=root,
    )
    result = AbsorptionResult(
        symbols=list(symbols),
        timeframe=timeframe,
        primary_horizon=primary_horizon,
        impact_window=impact_window,
        event_quantile=event_quantile,
    )
    if panel.is_empty():
        result.notes.append("panel is empty (no data for the universe)")
        return result
    result.panel = panel

    pdf = panel.to_pandas()
    y_col = f"y_{primary_horizon}"
    pdf = pdf[pdf[y_col].notna() & pdf["A"].notna()]
    design_end_ts = pd.Timestamp(design_end, tz="UTC")
    design = pdf[pdf["timestamp"] < design_end_ts]
    test = pdf[pdf["timestamp"] >= design_end_ts]
    block = DEFAULT_HORIZONS.get(primary_horizon, 4)

    result.design = {"ic": _ic(design, y_col), "n": float(design.shape[0])}
    result.test = {"ic": _ic(test, y_col), "n": float(test.shape[0])}
    strong = test[test["strong_flow"].fillna(False)]
    result.strong_flow_ic = _ic(strong, y_col)

    ci = block_bootstrap(
        test, lambda s: _ic(s, y_col), n=bootstrap, block_timestamps=block
    )
    if ci.size:
        result.test_ci = (float(np.percentile(ci, 2.5)), float(np.percentile(ci, 97.5)))

    result.horizon_ic = {
        name: _ic(test, f"y_{name}")
        for name in DEFAULT_HORIZONS
        if f"y_{name}" in test.columns and test[f"y_{name}"].notna().any()
    }
    result.quintiles = decile_table(strong, "A", y_col, n_deciles=5)

    if strong.shape[0] > 50:
        matched = matched_spread(strong, y_col)
        mci = block_bootstrap(strong, lambda s: matched_spread(s, y_col), n=bootstrap, block_timestamps=block)
        result.matched = {
            "spread": matched,
            "n": float(strong.shape[0]),
            "ci_low": float(np.percentile(mci, 2.5)) if mci.size else float("nan"),
            "ci_high": float(np.percentile(mci, 97.5)) if mci.size else float("nan"),
        }
        coef = absorption_regression(test, y_col)
        rci = block_bootstrap(test, lambda s: absorption_regression(s, y_col), n=bootstrap, block_timestamps=block)
        result.regression = {
            "coef": coef,
            "ci_low": float(np.percentile(rci, 2.5)) if rci.size else float("nan"),
            "ci_high": float(np.percentile(rci, 97.5)) if rci.size else float("nan"),
        }

    result.loo = pd.DataFrame(
        [
            {"dropped": s, "ic": _ic(test[test["symbol"] != s], y_col)}
            for s in symbols
        ]
    )
    result.ex_megacap_ic = _ic(
        test[~test["symbol"].isin(PRIMARY_BASELINES)], y_col
    )
    result.by_symbol = pd.DataFrame(
        [
            {"symbol": s, "ic": _ic(test[test["symbol"] == s], y_col),
             "n": float((test["symbol"] == s).sum())}
            for s in symbols
        ]
    )

    ic_test = float(result.test.get("ic", float("nan")))
    ic_design = float(result.design.get("ic", float("nan")))
    ci_low = result.test_ci[0] if result.test_ci is not None else float("nan")
    mono = float("nan")
    if not result.quintiles.empty:
        mono = _spearman(
            result.quintiles["decile"].to_numpy(dtype=float),
            result.quintiles["mean"].to_numpy(dtype=float),
        )
    min_loo = float(min((r.ic for r in result.loo.itertuples()), default=float("nan")))
    matched_ok = (
        result.matched.get("spread", float("nan")) > 0.0012
        and result.matched.get("ci_low", float("nan")) > 0
    )
    reg_ok = (
        result.regression.get("coef", float("nan")) > 0
        and result.regression.get("ci_low", float("nan")) > 0
    )
    horizon_ok = ic_test > 0 and any(
        result.horizon_ic.get(h, float("nan")) > 0 for h in ("1h", "24h")
    )

    result.criteria = {
        "P1": bool(ic_test > 0.02 and np.isfinite(ci_low) and ci_low > 0),
        "P2": bool(ic_design > 0 and ic_test > 0),
        "P3": bool(matched_ok),
        "P4": bool(reg_ok),
        "P5": bool(mono > 0.7),
        "P6": bool(min_loo > 0 and result.ex_megacap_ic > 0),
        "P7": bool(horizon_ok),
    }
    result.verdict = "EDGE" if all(result.criteria.values()) else "NO EDGE (H0)"
    result.notes.append(
        f"Exploratory: single regime (2026), in-sample; n={len(symbols)} perp symbols."
    )
    if ic_design > 0 >= ic_test:
        result.notes.append(
            "Design IC positive but flipped sign out-of-sample — selection artifact, not an edge."
        )
    if sweep:
        result.robustness = robustness_sweep(
            symbols,
            start=start,
            end=end,
            design_end=design_end,
            primary_horizon=primary_horizon,
            config=config,
            root=root,
        )
    return result


__all__ = [
    "DEFAULT_HORIZONS",
    "PRIMARY_HORIZON",
    "PRIMARY_TIMEFRAME",
    "AbsorptionResult",
    "absorption_regression",
    "absorption_signal",
    "add_outcomes",
    "block_bootstrap",
    "build_absorption_panel",
    "horizons_for_timeframe",
    "matched_spread",
    "normalized_flow",
    "robustness_sweep",
    "rolling_impact_model",
    "run_absorption_study",
]
