"""Spot–perpetual CVD divergence analysis.

The economic claim under test: spot taker flow and perp taker flow carry
different information, so a divergence between them predicts forward price. The
signal is

    D = z(rolling spot taker flow) - z(rolling perp taker flow)

with each flow z-scored against its own trailing distribution (see
``reports/spot_cvd_divergence_study.md`` for the frozen methodology). Everything
here is trailing/causal; forward returns are measured strictly after the signal
bar with a one-bar execution delay.

This module is deliberately small and explicit: a panel builder, a handful of
estimators (pooled IC, cross-sectional IC, event study, block bootstrap), and an
orchestrator that produces design/test tables. No model is fitted.
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

# Bar horizons at a 1h analysis timeframe. The 4h primary is 4 bars.
DEFAULT_HORIZONS: dict[str, int] = {"1h": 1, "4h": 4, "24h": 24, "3d": 72}
PRIMARY_HORIZON = "4h"

PANEL_COLUMNS = (
    "timestamp",
    "symbol",
    "close",
    "D",
    "Fs",
    "Fp",
    "zs",
    "zp",
)


# --- panel construction ------------------------------------------------------


def load_market_pair(
    root: Path,
    symbol: str,
    *,
    timeframe: str = "1h",
    source_timeframe: str = "1m",
    start: str,
    end: str,
) -> pl.DataFrame | None:
    """Join resampled spot and perp bars for one symbol on the bar open time."""
    spot_key = DatasetKey("binance", "spot", DatasetType.BARS, symbol, source_timeframe)
    perp_key = DatasetKey("binance", "futures", DatasetType.BARS, symbol, source_timeframe)
    spot = read_dataset(root, spot_key, start=start, end=end)
    perp = read_dataset(root, perp_key, start=start, end=end)
    if spot.is_empty() or perp.is_empty():
        return None
    if source_timeframe != timeframe:
        spot = resample_bars(spot, timeframe=timeframe, source_timeframe=source_timeframe)
        perp = resample_bars(perp, timeframe=timeframe, source_timeframe=source_timeframe)
    if spot.is_empty() or perp.is_empty():
        return None

    spot_side = spot.select(
        "timestamp",
        pl.col("volume").alias("spot_volume"),
        pl.col("taker_buy_volume").alias("spot_taker_buy"),
        pl.col("quote_volume").alias("spot_quote_volume"),
        pl.col("taker_buy_quote_volume").alias("spot_taker_buy_quote"),
    )
    perp_side = perp.select(
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        pl.col("volume").alias("perp_volume"),
        pl.col("taker_buy_volume").alias("perp_taker_buy"),
        pl.col("quote_volume").alias("perp_quote_volume"),
        pl.col("taker_buy_quote_volume").alias("perp_taker_buy_quote"),
    )
    return perp_side.join(spot_side, on="timestamp", how="inner").sort("timestamp")


def _zscore(column: str, window: int) -> pl.Expr:
    mean = pl.col(column).rolling_mean(window_size=window, min_periods=window)
    std = pl.col(column).rolling_std(window_size=window, min_periods=window)
    return ((pl.col(column) - mean) / std).alias(f"z_{column}")


def flow_divergence(
    frame: pl.DataFrame,
    *,
    window: int = 60,
    norm_window: int = 720,
) -> pl.DataFrame:
    """Append rolling spot/perp flows, their z-scores and the divergence ``D``."""
    out = frame.with_columns(
        (2.0 * pl.col("spot_taker_buy") - pl.col("spot_volume")).alias("delta_spot"),
        (2.0 * pl.col("perp_taker_buy") - pl.col("perp_volume")).alias("delta_perp"),
        (2.0 * pl.col("spot_taker_buy_quote") - pl.col("spot_quote_volume")).alias("qdelta_spot"),
        (2.0 * pl.col("perp_taker_buy_quote") - pl.col("perp_quote_volume")).alias("qdelta_perp"),
    )
    out = out.with_columns(
        pl.col("delta_spot").rolling_sum(window_size=window, min_periods=window).alias("Fs"),
        pl.col("delta_perp").rolling_sum(window_size=window, min_periods=window).alias("Fp"),
        pl.col("qdelta_spot").rolling_sum(window_size=window, min_periods=window).alias("qFs"),
        pl.col("qdelta_perp").rolling_sum(window_size=window, min_periods=window).alias("qFp"),
    )
    out = out.with_columns(
        _zscore("Fs", norm_window).alias("zs"),
        _zscore("Fp", norm_window).alias("zp"),
        _zscore("qFs", norm_window).alias("qzs"),
        _zscore("qFp", norm_window).alias("qzp"),
    )
    out = out.with_columns(
        (pl.col("zs") - pl.col("zp")).alias("D"),
        (pl.col("qzs") - pl.col("qzp")).alias("qD"),
        (pl.col("Fs").sign() * pl.col("Fp").sign()).alias("agree"),
    )
    denom = pl.col("Fs").abs() + pl.col("Fp").abs()
    out = out.with_columns(
        pl.when(denom > 0).then((pl.col("Fs") - pl.col("Fp")) / denom).otherwise(0.0).alias("ratio")
    )
    return out


def _forward_extremes(high: np.ndarray, low: np.ndarray, close: np.ndarray, bars: int) -> tuple[np.ndarray, np.ndarray]:
    """MFE/MAE (fraction of the signal close) over ``(t, t+bars]``."""
    n = close.size
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    for i in range(n):
        j0, j1 = i + 1, min(n, i + 1 + bars)
        if j1 <= j0:
            continue
        ref = close[i]
        if ref:
            mfe[i] = float(np.max(high[j0:j1]) - ref) / ref
            mae[i] = float(np.min(low[j0:j1]) - ref) / ref
    return mfe, mae


def add_forward_returns(
    frame: pl.DataFrame,
    horizons: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Append forward returns, directional hits and MFE/MAE per horizon.

    Entry is the close of ``t+1`` (one-bar execution delay); exit is the close of
    ``t+1+h``. MFE/MAE are measured from the signal close over ``(t, t+h]``.
    """
    horizons = horizons or DEFAULT_HORIZONS
    out = frame
    for name, bars in horizons.items():
        out = out.with_columns(
            (pl.col("close").shift(-(1 + bars)) / pl.col("close").shift(-1) - 1.0).alias(
                f"fwd_{name}"
            ),
            (
                (pl.col("close").shift(-(1 + bars)) - pl.col("close").shift(-1)).sign()
                * pl.col("D").sign()
            )
            .fill_null(0)
            .alias(f"hit_{name}"),        )
    high = frame["high"].to_numpy().astype(float)
    low = frame["low"].to_numpy().astype(float)
    close = frame["close"].to_numpy().astype(float)
    for name, bars in horizons.items():
        mfe, mae = _forward_extremes(high, low, close, bars)
        out = out.with_columns(
            pl.Series(f"mfe_{name}", mfe), pl.Series(f"mae_{name}", mae)
        )
    return out


def build_divergence_panel(
    symbols: Sequence[str],
    *,
    start: str,
    end: str,
    timeframe: str = "1h",
    source_timeframe: str = "1m",
    window: int = 60,
    norm_window: int = 720,
    horizons: dict[str, int] | None = None,
    config: Config | None = None,
    root: Path | None = None,
) -> pl.DataFrame:
    """Build the cross-sectional divergence panel for a symbol list."""
    cfg = config or get_config()
    data_root = Path(root) if root is not None else cfg.paths.data_normalized
    horizons = horizons or DEFAULT_HORIZONS

    frames: list[pl.DataFrame] = []
    for symbol in symbols:
        pair = load_market_pair(
            data_root,
            symbol,
            timeframe=timeframe,
            source_timeframe=source_timeframe,
            start=start,
            end=end,
        )
        if pair is None or pair.height == 0:
            continue
        enriched = add_forward_returns(flow_divergence(pair, window=window, norm_window=norm_window), horizons)
        frames.append(enriched.with_columns(pl.lit(symbol).alias("symbol")))
    if not frames:
        return pl.DataFrame()

    panel = pl.concat(frames, how="diagonal_relaxed").sort(["timestamp", "symbol"])

    # Market-neutral forward returns (de-mean across symbols at each timestamp).
    out = panel
    cs_exprs = [
        (pl.col(f"fwd_{name}") - pl.col(f"fwd_{name}").mean().over("timestamp")).alias(f"fwd_{name}_cs")
        for name in horizons
    ]
    out = out.with_columns(cs_exprs)
    return out


# --- estimators (numpy/pandas, small panel) ---------------------------------


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xs = pd.Series(x[mask]).rank().to_numpy()
    ys = pd.Series(y[mask]).rank().to_numpy()
    if xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def pooled_ic(df: pd.DataFrame, signal: str, fwd: str) -> float:
    return _spearman(df[signal].to_numpy(dtype=float), df[fwd].to_numpy(dtype=float))


def cross_sectional_ic(df: pd.DataFrame, signal: str, fwd: str) -> float:
    """Average within-timestamp Spearman of signal vs forward return (vectorised)."""
    d = df[["timestamp", signal, fwd]].dropna()
    if d.shape[0] < 6:
        return float("nan")
    a = d.groupby("timestamp")[signal].rank().to_numpy()
    b = d.groupby("timestamp")[fwd].rank().to_numpy()
    t = d["timestamp"].to_numpy()
    tmp = pd.DataFrame({"t": t, "a": a, "b": b})
    a0 = tmp["a"] - tmp.groupby("t")["a"].transform("mean")
    b0 = tmp["b"] - tmp.groupby("t")["b"].transform("mean")
    tmp["ab"] = a0 * b0
    tmp["aa"] = a0 * a0
    tmp["bb"] = b0 * b0
    g = tmp.groupby("t")[["ab", "aa", "bb"]].sum()
    denom = np.sqrt(g["aa"] * g["bb"])
    corr = (g["ab"] / denom).replace([np.inf, -np.inf], np.nan).dropna()
    return float(corr.mean()) if not corr.empty else float("nan")


def block_bootstrap_ic(
    df: pd.DataFrame,
    signal: str,
    fwd: str,
    *,
    n: int = 1000,
    block_timestamps: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Pooled IC resampled by contiguous blocks of timestamps (overlapping-window aware)."""
    d = df[["timestamp", signal, fwd]].dropna().sort_values("timestamp")
    if d.shape[0] < 3:
        return np.array([])
    times = d["timestamp"].to_numpy()
    change = np.ones(times.size, dtype=bool)
    change[1:] = times[1:] != times[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], times.size)
    t_count = starts.size
    if t_count < 2:
        return np.array([])
    sig = d[signal].to_numpy(dtype=float)
    fw = d[fwd].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    blocks = max(1, min(block_timestamps, t_count))
    n_blocks = int(np.ceil(t_count / blocks))
    offsets = np.arange(blocks)
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
        out[i] = _spearman(sig[idx], fw[idx])
    return out[np.isfinite(out)]


def decile_table(df: pd.DataFrame, signal: str, fwd: str, *, n_deciles: int = 10) -> pd.DataFrame:
    d = df[[signal, fwd]].dropna()
    if d.shape[0] < n_deciles * 5:
        return pd.DataFrame()
    d = d.copy()
    d["decile"] = pd.qcut(d[signal].rank(method="first"), n_deciles, labels=False)
    table = d.groupby("decile")[fwd].agg(["count", "mean", "median"]).reset_index()
    return table


def event_study(df: pd.DataFrame, signal: str, fwd: str, *, n_buckets: int = 5) -> pd.DataFrame:
    d = df[[signal, fwd]].dropna().copy()
    if d.shape[0] < n_buckets * 5:
        return pd.DataFrame()
    d["bucket"] = pd.qcut(d[signal].rank(method="first"), n_buckets, labels=False)
    table = d.groupby("bucket")[fwd].agg(["count", "mean", "median"]).reset_index()
    return table


# --- orchestration -----------------------------------------------------------


@dataclass(slots=True)
class DivergenceResult:
    symbols: list[str] = field(default_factory=list)
    timeframe: str = "1h"
    primary_horizon: str = PRIMARY_HORIZON
    window: int = 60
    norm_window: int = 720
    design: dict[str, float] = field(default_factory=dict)
    test: dict[str, float] = field(default_factory=dict)
    test_ci: tuple[float, float] | None = None
    deciles: pd.DataFrame = field(default_factory=pd.DataFrame)
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    loo: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)
    criteria: dict[str, bool | None] = field(default_factory=dict)
    monotonicity: float = float("nan")
    spread: float = float("nan")
    ic_megacap_excluded: float = float("nan")
    verdict: str = "not evaluated"

    def to_markdown(self) -> str:
        lines = [
            "# Spot–Perp CVD Divergence — Results (exploratory, in-sample 2026)",
            "",
            f"- universe ({len(self.symbols)}): " + ", ".join(self.symbols),
            f"- timeframe: {self.timeframe} · primary horizon: {self.primary_horizon}",
            f"- window={self.window} · norm_window={self.norm_window}",
            "",
        ]
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        lines += ["## Primary: pooled rank IC", "", "| period | pooled IC | cross-sectional IC | n |", "|---|---|---|---|"]
        for name, block in (("design", self.design), ("test", self.test)):
            lines.append(
                f"| {name} | {block.get('ic', float('nan')):.4f} | "
                f"{block.get('ic_cs', float('nan')):.4f} | {block.get('n', 0):.0f} |"
            )
        if self.test_ci is not None:
            lines += ["", f"test IC 95% bootstrap CI: [{self.test_ci[0]:.4f}, {self.test_ci[1]:.4f}]"]
        if self.criteria:
            lines += ["", f"## Verdict: **{self.verdict}**", ""]
            lines.append("| # | criterion | value | pass |")
            lines.append("|---|---|---|---|")
            detail = {
                "C1": f"test IC {self.test.get('ic', float('nan')):.4f}"
                + (
                    f", CI [{self.test_ci[0]:.4f}, {self.test_ci[1]:.4f}]"
                    if self.test_ci is not None
                    else ""
                ),
                "C2": f"design {self.design.get('ic', float('nan')):.4f} / test {self.test.get('ic', float('nan')):.4f}",
                "C3": f"decile monotonicity {self.monotonicity:.3f}",
                "C4": "not evaluated (moot — lower-priority criteria already fail)"
                if not any(self.criteria.get(k) for k in ("C1", "C2"))
                else "control regression not run in v1",
                "C5": f"top-bottom decile spread {self.spread * 1e4:.2f} bps",
                "C6": f"min leave-one-out IC {min((r.ic for r in self.loo.itertuples()), default=float('nan')):.4f}",
                "C7": f"IC ex-BTC/ETH {self.ic_megacap_excluded:.4f}",
            }
            for key, passed in self.criteria.items():
                mark = "✅" if passed else ("—" if passed is None else "❌")
                lines.append(f"| {key} | | {detail.get(key, '')} | {mark} |")
        if not self.deciles.empty:
            lines += ["", "## Test deciles (D → forward return, market-neutral)", ""]
            lines.append("| decile | n | mean | median |")
            lines.append("|---|---|---|---|")
            for row in self.deciles.itertuples(index=False):
                lines.append(f"| {row.decile} | {row.count} | {row.mean:.5f} | {row.median:.5f} |")
        if not self.events.empty:
            lines += ["", "## Test event study (|D| quintiles)", ""]
            lines.append("| bucket | n | mean | median |")
            lines.append("|---|---|---|---|")
            for row in self.events.itertuples(index=False):
                lines.append(f"| {row.bucket} | {row.count} | {row.mean:.5f} | {row.median:.5f} |")
        if not self.loo.empty:
            lines += ["", "## Test leave-one-symbol-out pooled IC", ""]
            lines.append("| dropped | IC |")
            lines.append("|---|---|")
            for row in self.loo.itertuples(index=False):
                lines.append(f"| {row.dropped} | {row.ic:.4f} |")
        return "\n".join(lines) + "\n"


def _ic_block(df: pd.DataFrame, signal: str, fwd: str) -> dict[str, float]:
    return {"ic": pooled_ic(df, signal, fwd), "n": float(df.shape[0])}


def run_divergence_study(
    *,
    symbols: Sequence[str],
    start: str,
    end: str,
    design_end: str = "2026-05-01",
    timeframe: str = "1h",
    source_timeframe: str = "1m",
    window: int = 60,
    norm_window: int = 720,
    primary_horizon: str = PRIMARY_HORIZON,
    bootstrap: int = 1000,
    config: Config | None = None,
    root: Path | None = None,
) -> DivergenceResult:
    """Build the panel, split design/test chronologically, and tabulate the study."""
    panel = build_divergence_panel(
        symbols,
        start=start,
        end=end,
        timeframe=timeframe,
        source_timeframe=source_timeframe,
        window=window,
        norm_window=norm_window,
        config=config,
        root=root,
    )
    result = DivergenceResult(
        symbols=list(symbols),
        timeframe=timeframe,
        primary_horizon=primary_horizon,
        window=window,
        norm_window=norm_window,
    )
    if panel.is_empty():
        result.notes.append("panel is empty (no spot/perp data for the universe)")
        return result

    pdf = panel.to_pandas()
    fwd = f"fwd_{primary_horizon}"
    fwd_cs = f"fwd_{primary_horizon}_cs"
    design_end_ts = pd.Timestamp(design_end, tz="UTC")
    design = pdf[pdf["timestamp"] < design_end_ts]
    test = pdf[pdf["timestamp"] >= design_end_ts]

    result.design = _ic_block(design, "D", fwd)
    result.test = _ic_block(test, "D", fwd)
    result.design["ic_cs"] = cross_sectional_ic(design, "D", fwd)
    result.test["ic_cs"] = cross_sectional_ic(test, "D", fwd)
    ci = block_bootstrap_ic(test, "D", fwd, n=bootstrap, block_timestamps=max(1, DEFAULT_HORIZONS[primary_horizon]))
    if ci.size:
        result.test_ci = (float(np.percentile(ci, 2.5)), float(np.percentile(ci, 97.5)))
    result.deciles = decile_table(test, "D", fwd_cs)
    result.events = event_study(test, "D", fwd_cs)

    rows = []
    for symbol in symbols:
        sub = test[test["symbol"] != symbol]
        rows.append({"dropped": symbol, "ic": pooled_ic(sub, "D", fwd)})
    result.loo = pd.DataFrame(rows)

    ic_test = float(result.test.get("ic", float("nan")))
    ic_design = float(result.design.get("ic", float("nan")))
    ci_low = result.test_ci[0] if result.test_ci is not None else float("nan")
    if not result.deciles.empty:
        di = result.deciles["decile"].to_numpy(dtype=float)
        dm = result.deciles["mean"].to_numpy(dtype=float)
        result.monotonicity = _spearman(di, dm)
        result.spread = float(dm[-1] - dm[0])
    excl = test[~test["symbol"].isin(["BTCUSDT", "ETHUSDT"])]
    result.ic_megacap_excluded = pooled_ic(excl, "D", fwd)
    min_loo = float(min((r.ic for r in result.loo.itertuples()), default=float("nan")))

    result.criteria = {
        "C1": bool(ic_test > 0.02 and np.isfinite(ci_low) and ci_low > 0),
        "C2": bool(ic_design > 0 and ic_test > 0),
        "C3": bool(result.monotonicity > 0.7),
        "C5": bool(result.spread > 0.0012),
        "C6": bool(min_loo > 0),
        "C7": bool(result.ic_megacap_excluded > 0),
    }
    result.verdict = (
        "EDGE"
        if all(v is True for v in result.criteria.values())
        else "NO EDGE (H0)"
    )
    result.notes.append(
        f"Exploratory only: single regime (2026), in-sample, n={len(symbols)} cross-section."
    )
    if ic_design > 0 >= ic_test:
        result.notes.append(
            "Design IC was positive and above threshold but flipped sign out-of-sample — "
            "a selection artifact (the breakout study's failure mode), not an edge."
        )
    return result


__all__ = [
    "DEFAULT_HORIZONS",
    "PANEL_COLUMNS",
    "PRIMARY_HORIZON",
    "DivergenceResult",
    "add_forward_returns",
    "block_bootstrap_ic",
    "build_divergence_panel",
    "cross_sectional_ic",
    "decile_table",
    "event_study",
    "flow_divergence",
    "load_market_pair",
    "pooled_ic",
    "run_divergence_study",
]
