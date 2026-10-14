"""H3 — flow-confirmed breakout continuation.

A *breakout* is a close beyond the trailing prior-range extreme. H3 asks whether
breakouts with **persistent same-direction aggressive taker flow** continue more
than breakouts with weak or opposing flow, **after accounting for the breakout
itself**. See ``reports/cvd_breakout_flow_study.md`` for the frozen design.

All features are trailing/causal; the prior-range level excludes the current bar;
a per-symbol cooldown de-duplicates clustered breakouts; forward outcomes start one
bar after the breakout. Reuses the level detector in ``analytics/sweeps.py`` and the
estimators in ``analytics/absorption.py`` / ``analytics/divergence.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
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
from .absorption import (
    _oriented_extremes,
    _spearman,
    absorption_regression,
    block_bootstrap,
    horizons_for_timeframe,
    matched_spread,
)
from .divergence import pooled_ic
from .regimes import RegimeConfig, attach_regimes, classify_regimes
from .sweeps import add_controls, prior_range_levels

DEFAULT_HORIZONS: dict[str, int] = {"1h": 1, "4h": 4, "24h": 24, "3d": 72}
PRIMARY_HORIZON = "4h"
PRIMARY_TIMEFRAME = "1h"
PRIMARY_LOOKBACK = 48
PRIMARY_FLOW_WINDOW = 4
PRIMARY_COOLDOWN = 24
PRIMARY_BASELINES: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")


def detect_breakouts(
    frame: pl.DataFrame,
    *,
    lookback: int = PRIMARY_LOOKBACK,
    cooldown: int = PRIMARY_COOLDOWN,
    min_disp: float = 0.0,
) -> pl.DataFrame:
    """Flag close-beyond-range breakouts with direction and displacement.

    ``bdir = +1`` above the prior high, ``-1`` below the prior low. ``break_disp``
    is the signed distance beyond the level (fraction). Cooldown keeps at most one
    breakout every ``cooldown`` bars.
    """
    out = prior_range_levels(frame, lookback=lookback).with_columns(
        (pl.col("close") > pl.col("prior_high")).fill_null(False).alias("bull"),
        (pl.col("close") < pl.col("prior_low")).fill_null(False).alias("bear"),
    )
    out = out.with_columns(
        pl.when(pl.col("bull")).then(1.0).when(pl.col("bear")).then(-1.0).otherwise(0.0).alias("bdir")
    )
    out = out.with_columns(
        pl.when(pl.col("bdir") > 0)
        .then(pl.col("prior_high"))
        .when(pl.col("bdir") < 0)
        .then(pl.col("prior_low"))
        .otherwise(None)
        .alias("level")
    )
    out = out.with_columns(
        (pl.col("bdir") * (pl.col("close") - pl.col("level")) / pl.col("level")).alias("break_disp")
    )
    out = out.with_columns(
        ((pl.col("bdir") != 0) & (pl.col("break_disp") >= min_disp)).alias("breakout")
    )
    if cooldown > 0:
        flags = out["breakout"].to_numpy()
        keep = np.zeros(flags.size, dtype=bool)
        last = -(10**9)
        for i, flag in enumerate(flags):
            if flag and i - last > cooldown:
                keep[i] = True
                last = i
        out = out.with_columns(pl.Series("breakout", keep))
    return out


def flow_alignment(
    frame: pl.DataFrame,
    *,
    flow_window: int = PRIMARY_FLOW_WINDOW,
    act_window: int = 48,
    norm_window: int = 720,
) -> pl.DataFrame:
    """Aligned, activity-normalised, z-scored breakout flow and signed persistence."""
    mp = min(norm_window, max(30, norm_window // 3))
    out = frame.with_columns(
        (2.0 * pl.col("taker_buy_quote_volume") - pl.col("quote_volume")).alias("delta")
    ).with_columns(
        pl.when(pl.col("delta") > 0).then(1.0).when(pl.col("delta") < 0).then(-1.0).otherwise(0.0).alias("_sgn")
    )
    out = out.with_columns(
        pl.col("delta").rolling_sum(window_size=flow_window, min_periods=flow_window).alias("F"),
        pl.col("delta").abs().rolling_mean(window_size=act_window, min_periods=act_window).alias("act"),
        pl.col("_sgn").rolling_mean(window_size=flow_window, min_periods=flow_window).alias("_persist"),
    )
    out = out.with_columns(
        safe_div(pl.col("F"), pl.col("act") * flow_window).alias("fq"),
        (pl.col("bdir") * pl.col("_persist")).alias("persist"),
    )
    out = out.with_columns(
        safe_div(
            pl.col("fq") - pl.col("fq").rolling_mean(window_size=norm_window, min_periods=mp),
            pl.col("fq").rolling_std(window_size=norm_window, min_periods=mp),
        ).alias("fz"),
        safe_div(
            pl.col("persist") - pl.col("persist").rolling_mean(window_size=norm_window, min_periods=mp),
            pl.col("persist").rolling_std(window_size=norm_window, min_periods=mp),
        ).alias("persist_z"),
    )
    return out.with_columns((pl.col("bdir") * pl.col("fz")).alias("flow_align_z")).drop("_sgn", "_persist")


def _forward_failure(
    close: np.ndarray, level: np.ndarray, direction: np.ndarray, bars: int
) -> np.ndarray:
    """1 if price closes back through the breakout level within ``bars`` bars."""
    n = close.size
    fail = np.full(n, np.nan)
    for i in range(n):
        d = direction[i]
        if d == 0 or not np.isfinite(level[i]):
            continue
        j0, j1 = i + 1, min(n, i + 1 + bars)
        if j1 <= j0:
            continue
        seg = close[j0:j1]
        fail[i] = float(np.min(seg) < level[i]) if d > 0 else float(np.max(seg) > level[i])
    return fail


def add_outcomes(frame: pl.DataFrame, horizons: dict[str, int] | None = None) -> pl.DataFrame:
    """Oriented forward returns, continuation hits, MFE/MAE and failure flags."""
    horizons = horizons or DEFAULT_HORIZONS
    out = frame.with_columns(pl.col("bdir").alias("d"))
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
    level = frame["level"].to_numpy().astype(float)
    direction = out["d"].to_numpy().astype(float)
    for name, bars in horizons.items():
        fav, adv = _oriented_extremes(high, low, close, direction, bars)
        fail = _forward_failure(close, level, direction, bars)
        out = out.with_columns(
            pl.Series(f"mfe_{name}", fav),
            pl.Series(f"mae_{name}", adv),
            pl.Series(f"fail_{name}", fail),
        )
    return out


def build_breakout_panel(
    symbols: Sequence[str],
    *,
    start: str,
    end: str,
    timeframe: str = PRIMARY_TIMEFRAME,
    source_timeframe: str = "1m",
    lookback: int = PRIMARY_LOOKBACK,
    cooldown: int = PRIMARY_COOLDOWN,
    min_disp: float = 0.0,
    flow_window: int = PRIMARY_FLOW_WINDOW,
    act_window: int = 48,
    norm_window: int | None = None,
    horizons: dict[str, int] | None = None,
    config: Config | None = None,
    root: Path | None = None,
) -> pl.DataFrame:
    cfg = config or get_config()
    data_root = Path(root) if root is not None else cfg.paths.data_normalized
    nw = norm_window or max(30, round(30 * 86400 / timeframe_seconds(timeframe)))
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
        sig = detect_breakouts(bars, lookback=lookback, cooldown=cooldown, min_disp=min_disp)
        sig = flow_alignment(
            sig, flow_window=flow_window, act_window=act_window, norm_window=nw
        )
        sig = add_controls(sig)
        regimes = classify_regimes(
            bars, config=RegimeConfig(regime_timeframe=timeframe), source_timeframe=timeframe
        )
        sig = attach_regimes(sig, regimes)
        sig = add_outcomes(sig, horizons)
        frames.append(sig.with_columns(pl.lit(symbol).alias("symbol")))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort(["timestamp", "symbol"])


def quintile_outcomes(df: pd.DataFrame, y_col: str, fail_col: str, *, n: int = 5) -> pd.DataFrame:
    """Buckets of ``flow_align_z`` → oriented return mean/median and failure rate."""
    d = df.dropna(subset=["flow_align_z", y_col]).copy()
    if d.shape[0] < n * 5:
        return pd.DataFrame()
    d["bucket"] = pd.qcut(d["flow_align_z"].rank(method="first"), n, labels=False)
    g = d.groupby("bucket").agg(
        n=(y_col, "size"), mean=(y_col, "mean"), median=(y_col, "median"),
        fail_rate=(fail_col, "mean") if fail_col in d.columns else (y_col, "size"),
    )
    return g.reset_index()


def robustness_sweep(
    symbols: Sequence[str],
    *,
    start: str,
    end: str,
    design_end: str = "2026-05-01",
    timeframes: Sequence[str] = ("15m", "1h", "4h"),
    lookbacks: Sequence[int] = (24, 48, 96),
    flow_windows: Sequence[int] = (1, 4, 12),
    primary_horizon: str = PRIMARY_HORIZON,
    config: Config | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    cutoff = pd.Timestamp(design_end, tz="UTC")
    for tf in timeframes:
        hz = horizons_for_timeframe(tf)
        if primary_horizon not in hz:
            continue
        y_col = f"y_{primary_horizon}"
        for lb in lookbacks:
            for fw in flow_windows:
                panel = build_breakout_panel(
                    symbols, start=start, end=end, timeframe=tf, lookback=lb,
                    flow_window=fw, horizons=hz, config=config, root=root,
                )
                if panel.is_empty():
                    continue
                pdf = panel.to_pandas()
                ev = pdf[pdf["breakout"].fillna(False) & pdf[y_col].notna() & pdf["flow_align_z"].notna()]
                de = ev[ev["timestamp"] < cutoff]
                te = ev[ev["timestamp"] >= cutoff]
                rows.append(
                    {
                        "timeframe": tf, "lookback": lb, "flow_window": fw,
                        "design_ic": pooled_ic(de, "flow_align_z", y_col) if not de.empty else float("nan"),
                        "test_ic": pooled_ic(te, "flow_align_z", y_col) if not te.empty else float("nan"),
                        "n_events": float(ev.shape[0]), "n_test": float(te.shape[0]),
                    }
                )
    return pd.DataFrame(rows)


@dataclass(slots=True)
class BreakoutResult:
    symbols: list[str] = field(default_factory=list)
    timeframe: str = PRIMARY_TIMEFRAME
    primary_horizon: str = PRIMARY_HORIZON
    lookback: int = PRIMARY_LOOKBACK
    flow_window: int = PRIMARY_FLOW_WINDOW
    n_events: int = 0
    design: dict[str, float] = field(default_factory=dict)
    test: dict[str, float] = field(default_factory=dict)
    test_ci: tuple[float, float] | None = None
    quintiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    matched: dict[str, float] = field(default_factory=dict)
    regression: dict[str, float] = field(default_factory=dict)
    horizon_ic: dict[str, float] = field(default_factory=dict)
    loo: pd.DataFrame = field(default_factory=pd.DataFrame)
    ex_megacap_ic: float = float("nan")
    by_symbol: pd.DataFrame = field(default_factory=pd.DataFrame)
    robustness: pd.DataFrame = field(default_factory=pd.DataFrame)
    criteria: dict[str, bool] = field(default_factory=dict)
    verdict: str = "not evaluated"
    notes: list[str] = field(default_factory=list)
    panel: pl.DataFrame = field(default_factory=pl.DataFrame)

    def to_markdown(self) -> str:
        lines = [
            "# H3 — Flow-Confirmed Breakout Continuation — Results (exploratory, 2026)",
            "",
            f"- universe ({len(self.symbols)}): " + ", ".join(self.symbols[:12])
            + (" ..." if len(self.symbols) > 12 else ""),
            f"- timeframe {self.timeframe} · horizon {self.primary_horizon} · "
            f"lookback {self.lookback} · flow_window {self.flow_window}",
            f"- breakout events (design+test): {self.n_events}",
        ]
        if self.notes:
            lines += ["", "## Notes"] + [f"- {n}" for n in self.notes]
        lines += ["", "## Primary: event pooled rank IC(flow_align_z, oriented return)", ""]
        lines += ["| period | IC | n |", "|---|---|---|"]
        for name, block in (("design", self.design), ("test", self.test)):
            lines.append(f"| {name} | {block.get('ic', float('nan')):.4f} | {block.get('n', 0):.0f} |")
        if self.test_ci is not None:
            lines += ["", f"test IC 95% bootstrap CI: [{self.test_ci[0]:.4f}, {self.test_ci[1]:.4f}]"]
        lines += ["", "## Horizon consistency (test event IC)", "", "| horizon | IC |", "|---|---|"]
        for h, v in self.horizon_ic.items():
            lines.append(f"| {h} | {v:.4f} |")
        if not self.quintiles.empty:
            lines += ["", "## Test quintiles (flow_align_z → return / failure rate)", ""]
            lines += ["| qubit | n | mean | median | fail_rate |", "|---|---|---|---|---|"]
            for row in self.quintiles.itertuples(index=False):
                lines.append(
                    f"| {row.bucket} | {row.n} | {row.mean:.5f} | {row.median:.5f} | {row.fail_rate:.3f} |"
                )
        if self.matched:
            lines += [
                "", "## Matched control (high − low alignment)", "",
                f"- spread: {self.matched.get('spread', float('nan')) * 1e4:.2f} bps "
                f"(CI [{self.matched.get('ci_low', float('nan')) * 1e4:.2f}, "
                f"{self.matched.get('ci_high', float('nan')) * 1e4:.2f}])",
                f"- n: {self.matched.get('n', 0):.0f}",
            ]
        if self.regression:
            lines += [
                "", "## Incremental regression (flow_align_z coefficient)", "",
                f"- coef: {self.regression.get('coef', float('nan')):.5f} "
                f"(CI [{self.regression.get('ci_low', float('nan')):.5f}, "
                f"{self.regression.get('ci_high', float('nan')):.5f}])",
            ]
        if not self.by_symbol.empty:
            lines += ["", "## Per-symbol test event IC", "", "| symbol | IC | n |", "|---|---|---|"]
            for row in self.by_symbol.itertuples(index=False):
                lines.append(f"| {row.symbol} | {row.ic:.4f} | {row.n:.0f} |")
        if not self.robustness.empty:
            lines += [
                "", "## Robustness sweep (test event IC)", "",
                "| timeframe | lookback | flow_window | design_ic | test_ic | n_events |",
                "|---|---|---|---|---|---|",
            ]
            for row in self.robustness.itertuples(index=False):
                lines.append(
                    f"| {row.timeframe} | {row.lookback} | {row.flow_window} | "
                    f"{row.design_ic:.4f} | {row.test_ic:.4f} | {row.n_events:.0f} |"
                )
        lines += ["", f"## Verdict: **{self.verdict}**", "", "| # | criterion | pass |", "|---|---|---|"]
        for key, passed in self.criteria.items():
            lines.append(f"| {key} | | {'✅' if passed else '❌'} |")
        return "\n".join(lines) + "\n"


def run_breakout_study(
    *,
    symbols: Sequence[str],
    start: str,
    end: str,
    design_end: str = "2026-05-01",
    timeframe: str = PRIMARY_TIMEFRAME,
    source_timeframe: str = "1m",
    lookback: int = PRIMARY_LOOKBACK,
    cooldown: int = PRIMARY_COOLDOWN,
    flow_window: int = PRIMARY_FLOW_WINDOW,
    primary_horizon: str = PRIMARY_HORIZON,
    bootstrap: int = 1000,
    sweep: bool = True,
    config: Config | None = None,
    root: Path | None = None,
) -> BreakoutResult:
    panel = build_breakout_panel(
        symbols, start=start, end=end, timeframe=timeframe, source_timeframe=source_timeframe,
        lookback=lookback, cooldown=cooldown, flow_window=flow_window, config=config, root=root,
    )
    result = BreakoutResult(
        symbols=list(symbols), timeframe=timeframe, primary_horizon=primary_horizon,
        lookback=lookback, flow_window=flow_window,
    )
    if panel.is_empty():
        result.notes.append("panel is empty (no data for the universe)")
        return result
    result.panel = panel

    pdf = panel.to_pandas()
    y_col = f"y_{primary_horizon}"
    fail_col = f"fail_{primary_horizon}"
    events = pdf[pdf["breakout"].fillna(False) & pdf["flow_align_z"].notna() & pdf[y_col].notna()]
    result.n_events = int(events.shape[0])
    if result.n_events < 100:
        result.notes.append(f"too few events ({result.n_events}); result not meaningful")
        return result

    cutoff = pd.Timestamp(design_end, tz="UTC")
    design = events[events["timestamp"] < cutoff]
    test = events[events["timestamp"] >= cutoff]
    block = DEFAULT_HORIZONS.get(primary_horizon, 4)

    result.design = {"ic": pooled_ic(design, "flow_align_z", y_col), "n": float(design.shape[0])}
    result.test = {"ic": pooled_ic(test, "flow_align_z", y_col), "n": float(test.shape[0])}
    ci = block_bootstrap(test, lambda s: pooled_ic(s, "flow_align_z", y_col), n=bootstrap, block_timestamps=block)
    if ci.size:
        result.test_ci = (float(np.percentile(ci, 2.5)), float(np.percentile(ci, 97.5)))

    result.horizon_ic = {
        name: pooled_ic(test, "flow_align_z", f"y_{name}")
        for name in DEFAULT_HORIZONS
        if f"y_{name}" in test.columns and test[f"y_{name}"].notna().any()
    }
    result.quintiles = quintile_outcomes(test, y_col, fail_col)

    strata = ("break_disp", "range_width", "realized_vol", "past_ret")
    controls = ("break_disp", "range_width", "realized_vol", "past_ret")
    spread = matched_spread(test, y_col, signal="flow_align_z", strong="breakout", strata=strata)
    mci = block_bootstrap(
        test,
        lambda s: matched_spread(s, y_col, signal="flow_align_z", strong="breakout", strata=strata),
        n=bootstrap, block_timestamps=block,
    )
    result.matched = {
        "spread": spread, "n": float(test.shape[0]),
        "ci_low": float(np.percentile(mci, 2.5)) if mci.size else float("nan"),
        "ci_high": float(np.percentile(mci, 97.5)) if mci.size else float("nan"),
    }
    coef = absorption_regression(test, y_col, signal="flow_align_z", controls=controls)
    rci = block_bootstrap(
        test,
        lambda s: absorption_regression(s, y_col, signal="flow_align_z", controls=controls),
        n=bootstrap, block_timestamps=block,
    )
    result.regression = {
        "coef": coef,
        "ci_low": float(np.percentile(rci, 2.5)) if rci.size else float("nan"),
        "ci_high": float(np.percentile(rci, 97.5)) if rci.size else float("nan"),
    }

    result.loo = pd.DataFrame(
        [{"dropped": s, "ic": pooled_ic(test[test["symbol"] != s], "flow_align_z", y_col)} for s in symbols]
    )
    result.ex_megacap_ic = pooled_ic(test[~test["symbol"].isin(PRIMARY_BASELINES)], "flow_align_z", y_col)
    result.by_symbol = pd.DataFrame(
        [
            {"symbol": s, "ic": pooled_ic(test[test["symbol"] == s], "flow_align_z", y_col),
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
            result.quintiles["bucket"].to_numpy(dtype=float),
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
        "R1": bool(ic_test > 0.02 and np.isfinite(ci_low) and ci_low > 0),
        "R2": bool(ic_design > 0 and ic_test > 0),
        "R3": bool(matched_ok),
        "R4": bool(reg_ok),
        "R5": bool(mono > 0.7),
        "R6": bool(min_loo > 0 and result.ex_megacap_ic > 0),
        "R7": bool(horizon_ok),
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
            symbols, start=start, end=end, design_end=design_end,
            primary_horizon=primary_horizon, config=config, root=root,
        )
    return result


__all__ = [
    "DEFAULT_HORIZONS",
    "PRIMARY_FLOW_WINDOW",
    "PRIMARY_HORIZON",
    "PRIMARY_LOOKBACK",
    "PRIMARY_TIMEFRAME",
    "BreakoutResult",
    "add_outcomes",
    "build_breakout_panel",
    "detect_breakouts",
    "flow_alignment",
    "quintile_outcomes",
    "robustness_sweep",
    "run_breakout_study",
]
