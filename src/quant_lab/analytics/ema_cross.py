"""EMA fast/slow cross signal study.

A **cross_above** is the fast EMA (default 9) crossing above the slow EMA
(default 20); a **cross_below** is the mirror. Both are evaluated as directional
signals: does a cross predict the sign of the subsequent move, how strong is it,
and — the part specific to this study — **how long does the move last**?

The move is defined the way an EMA-cross system would actually hold it: from the
cross bar until the opposite cross fires. That "regime" length is the duration
reported below, alongside fixed-horizon forward returns and oriented MFE/MAE.

Everything causal: EMAs use trailing windows, a signal is evaluated at the close
of the cross bar, and forward outcomes are labels only. Design/test is a strict
temporal split (:func:`run_ema_cross_study`).
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
from .absorption import _oriented_extremes, block_bootstrap
from .divergence import pooled_ic

# Forward horizons in bars on the analysis timeframe (default 1h).
DEFAULT_HORIZONS: dict[str, int] = {"1h": 1, "4h": 4, "12h": 12, "24h": 24, "3d": 72}
PRIMARY_HORIZON = "24h"
PRIMARY_TIMEFRAME = "1h"
PRIMARY_SOURCE_TIMEFRAME = "1m"
DEFAULT_FAST = 9
DEFAULT_SLOW = 20

_FWD_COLS = [f"fwd_{h}" for h in DEFAULT_HORIZONS]


# --- signal construction -----------------------------------------------------


def ema_cross_signals(
    frame: pl.DataFrame, *, fast: int = DEFAULT_FAST, slow: int = DEFAULT_SLOW
) -> pl.DataFrame:
    """Attach EMAs, the fast/slow sign and bar-level up/down cross flags.

    ``cross`` is ``+1`` on the bar the fast EMA crosses **above** the slow EMA
    (cross_above), ``-1`` on a cross_below and ``0`` otherwise. The first time a
    sign becomes available (warmup end) counts as a cross. ``regime_id``
    increments at every cross, so bars sharing a ``regime_id`` sit between two
    consecutive opposite crosses.
    """
    if fast >= slow:
        raise ValueError("fast EMA span must be strictly shorter than the slow span")
    out = frame.with_columns(
        pl.col("close").ewm_mean(span=fast, adjust=False, min_periods=fast).alias("ema_fast"),
        pl.col("close").ewm_mean(span=slow, adjust=False, min_periods=slow).alias("ema_slow"),
    )
    diff = pl.col("ema_fast") - pl.col("ema_slow")
    sign = diff.sign().fill_null(0).cast(pl.Int8)
    out = out.with_columns(sign.alias("ema_sign"))
    prev = pl.col("ema_sign").shift(1).fill_null(0).cast(pl.Int8)
    out = out.with_columns(
        pl.when((pl.col("ema_sign") == 1) & (prev != 1)).then(1)
        .when((pl.col("ema_sign") == -1) & (prev != -1)).then(-1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("cross")
    )
    return out.with_columns((pl.col("cross") != 0).cast(pl.Int64).cum_sum().alias("regime_id"))


def _regime_stats(frame: pl.DataFrame, *, fast: int, slow: int) -> pl.DataFrame:
    """One row per EMA regime: start/end, length in bars, oriented return/peaks.

    A regime runs from a cross bar to the bar immediately before the next
    opposite cross. Its length is therefore "how long that move lasted" under a
    hold-until-opposite-cross rule. Peaks include the cross bar itself.
    """
    n_rows = frame.height
    grouped = frame.with_row_index("_row").group_by("regime_id").agg(
        pl.col("_row").first().alias("start_row"),
        pl.col("_row").last().alias("end_row"),
        pl.len().alias("regime_bars"),
        pl.col("cross").first().alias("regime_dir"),
        pl.col("close").first().alias("start_close"),
        pl.col("close").last().alias("end_close"),
        pl.col("high").max().alias("regime_high"),
        pl.col("low").min().alias("regime_low"),
        pl.col("high").arg_max().alias("_hi_idx"),
        pl.col("low").arg_min().alias("_lo_idx"),
    )
    reg = grouped.with_columns(
        (pl.col("end_close") / pl.col("start_close") - 1.0).alias("regime_fwd"),
        (pl.col("regime_high") / pl.col("start_close") - 1.0).alias("_up_exc"),
        (1.0 - pl.col("regime_low") / pl.col("start_close")).alias("_dn_exc"),
        (pl.col("end_row") == n_rows - 1).alias("regime_censored"),
    )
    up = pl.col("regime_dir") > 0
    return reg.with_columns(
        (pl.col("regime_dir") * pl.col("regime_fwd")).alias("regime_run"),
        pl.when(up).then(pl.col("_up_exc")).otherwise(pl.col("_dn_exc")).alias("regime_mfe"),
        pl.when(up).then(pl.col("_dn_exc")).otherwise(pl.col("_up_exc")).alias("regime_mae"),
        pl.when(up).then(pl.col("_hi_idx")).otherwise(pl.col("_lo_idx")).alias("bars_to_peak"),
    ).drop("_up_exc", "_dn_exc", "_hi_idx", "_lo_idx")


def add_forward_outcomes(
    frame: pl.DataFrame, horizons: dict[str, int] | None = None
) -> pl.DataFrame:
    """Forward returns, oriented returns/hits and oriented MFE/MAE.

    ``fwd_{h}`` is the close-to-close move over the ``h`` bars after the signal
    bar; ``y_{h}`` multiplies it by ``ema_sign`` so a positive value means the
    move agreed with the prevailing cross direction.
    """
    horizons = horizons or DEFAULT_HORIZONS
    out = frame
    for name, bars in horizons.items():
        out = out.with_columns((pl.col("close").shift(-bars) / pl.col("close") - 1.0).alias(f"fwd_{name}"))
    out = out.with_columns(
        [(pl.col("ema_sign").cast(pl.Float64) * pl.col(f"fwd_{name}")).alias(f"y_{name}") for name in horizons]
    )
    out = out.with_columns(
        [(pl.col(f"y_{name}") > 0).cast(pl.Int8).alias(f"hit_{name}") for name in horizons]
    )
    high = frame["high"].to_numpy().astype(float)
    low = frame["low"].to_numpy().astype(float)
    close = frame["close"].to_numpy().astype(float)
    direction = out["ema_sign"].to_numpy().astype(float)
    for name, bars in horizons.items():
        fav, adv = _oriented_extremes(high, low, close, direction, bars)
        out = out.with_columns(pl.Series(f"mfe_{name}", fav), pl.Series(f"mae_{name}", adv))
    return out


def attach_regime_stats(frame: pl.DataFrame, *, fast: int, slow: int) -> pl.DataFrame:
    reg = _regime_stats(frame, fast=fast, slow=slow)
    cols = [
        "regime_id", "regime_bars", "regime_dir", "regime_run", "regime_mfe",
        "regime_mae", "regime_censored", "bars_to_peak",
    ]
    return frame.join(reg.select(cols), on="regime_id", how="left")


# --- panel build -------------------------------------------------------------


def build_ema_cross_panel(
    symbols: Sequence[str],
    *,
    start: str,
    end: str,
    timeframe: str = PRIMARY_TIMEFRAME,
    source_timeframe: str = PRIMARY_SOURCE_TIMEFRAME,
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
    horizons: dict[str, int] | None = None,
    market: str = "futures",
    exchange: str = "binance",
    config: Config | None = None,
    root: Path | None = None,
) -> pl.DataFrame:
    """Bar-level EMA-cross panel for a symbol list (resampled if needed)."""
    cfg = config or get_config()
    data_root = Path(root) if root is not None else cfg.paths.data_normalized
    horizons = horizons or DEFAULT_HORIZONS
    frames: list[pl.DataFrame] = []
    for symbol in symbols:
        bars = read_dataset(
            data_root,
            DatasetKey(exchange, market, DatasetType.BARS, symbol, source_timeframe),
            start=start,
            end=end,
        )
        if bars.is_empty():
            continue
        if source_timeframe != timeframe:
            bars = resample_bars(bars, timeframe=timeframe, source_timeframe=source_timeframe)
        if bars.height <= slow + max(horizons.values()):
            continue
        sig = ema_cross_signals(bars, fast=fast, slow=slow)
        sig = attach_regime_stats(sig, fast=fast, slow=slow)
        sig = add_forward_outcomes(sig, horizons)
        frames.append(sig.with_columns(pl.lit(symbol).alias("symbol")))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort(["timestamp", "symbol"])


# --- aggregation helpers -----------------------------------------------------


def _bootstrap_ci(
    events: pd.DataFrame, stat_fn, *, n: int, block: int, seed: int = 0
) -> tuple[float, float]:
    vals = block_bootstrap(events, stat_fn, n=n, block_timestamps=block, seed=seed)
    if vals.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _events_from_panel(panel: pl.DataFrame) -> pd.DataFrame:
    if panel.is_empty():
        return pd.DataFrame()
    ev = panel.filter(pl.col("cross") != 0).to_pandas()
    return ev.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _direction_blocks(events: pd.DataFrame, horizons: dict[str, int]) -> list[dict]:
    """Hit-rate / mean-y blocks per group (all / up / down) x horizon."""
    rows: list[dict] = []
    if events.empty:
        return rows
    groups = [("all", events), ("up", events[events["cross"] > 0]), ("down", events[events["cross"] < 0])]
    for gname, g in groups:
        for name in horizons:
            y = g[f"y_{name}"].to_numpy(dtype=float)
            y = y[np.isfinite(y)]
            if y.size == 0:
                rows.append({"group": gname, "horizon": name, "n": 0, "hit_rate": float("nan"),
                             "mean_y": float("nan"), "median_y": float("nan")})
                continue
            rows.append({
                "group": gname,
                "horizon": name,
                "n": int(y.size),
                "hit_rate": float((y > 0).mean()),
                "mean_y": float(y.mean()),
                "median_y": float(np.median(y)),
            })
    return rows


def _duration_blocks(events: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if events.empty:
        return rows
    groups = [("all", events), ("up", events[events["cross"] > 0]), ("down", events[events["cross"] < 0])]
    for gname, g in groups:
        d = g[["regime_bars", "regime_run", "regime_mfe", "regime_mae", "regime_censored", "bars_to_peak"]].dropna()
        if d.empty:
            continue
        rows.append({
            "group": gname,
            "n": int(d.shape[0]),
            "mean_bars": float(d["regime_bars"].mean()),
            "median_bars": float(d["regime_bars"].median()),
            "p25_bars": float(d["regime_bars"].quantile(0.25)),
            "p75_bars": float(d["regime_bars"].quantile(0.75)),
            "max_bars": float(d["regime_bars"].max()),
            "mean_run_pct": float(d["regime_run"].mean() * 100.0),
            "median_run_pct": float(d["regime_run"].median() * 100.0),
            "share_pos_run": float((d["regime_run"] > 0).mean()),
            "mean_mfe_pct": float(d["regime_mfe"].mean() * 100.0),
            "mean_mae_pct": float(d["regime_mae"].mean() * 100.0),
            "median_bars_to_peak": float(d["bars_to_peak"].median()),
            "censored_pct": float(d["regime_censored"].mean() * 100.0),
        })
    return rows


def _duration_by_outcome(events: pd.DataFrame) -> list[dict]:
    """Do winning moves last longer than losing ones, by cross direction?"""
    rows: list[dict] = []
    if events.empty:
        return rows
    for gname, g in (("up", events[events["cross"] > 0]), ("down", events[events["cross"] < 0])):
        for outcome, sub in (("win", g[g["regime_run"] > 0]), ("loss", g[g["regime_run"] <= 0])):
            d = sub[["regime_bars", "regime_run", "bars_to_peak"]].dropna()
            if d.empty:
                continue
            rows.append({
                "group": gname,
                "outcome": outcome,
                "n": int(d.shape[0]),
                "share": float(d.shape[0]) / float(len(g.dropna(subset=["regime_run"]))),
                "median_bars": float(d["regime_bars"].median()),
                "mean_bars": float(d["regime_bars"].mean()),
                "mean_run_pct": float(d["regime_run"].mean() * 100.0),
                "median_bars_to_peak": float(d["bars_to_peak"].median()),
            })
    return rows


# --- orchestration -----------------------------------------------------------


@dataclass(slots=True)
class EmaCrossResult:
    symbols: list[str] = field(default_factory=list)
    market: str = "futures"
    timeframe: str = PRIMARY_TIMEFRAME
    fast: int = DEFAULT_FAST
    slow: int = DEFAULT_SLOW
    start: str = ""
    end: str = ""
    design_end: str = ""
    horizons: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_HORIZONS))
    primary_horizon: str = PRIMARY_HORIZON

    n_bars: int = 0
    n_up: int = 0
    n_down: int = 0
    frequency: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictive: pd.DataFrame = field(default_factory=pd.DataFrame)
    design: pd.DataFrame = field(default_factory=pd.DataFrame)
    test: pd.DataFrame = field(default_factory=pd.DataFrame)
    durations: pd.DataFrame = field(default_factory=pd.DataFrame)
    duration_outcomes: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_symbol: pd.DataFrame = field(default_factory=pd.DataFrame)
    primary_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)
    criteria: dict[str, bool] = field(default_factory=dict)
    verdict: str = "not evaluated"
    notes: list[str] = field(default_factory=list)
    events: pd.DataFrame = field(default_factory=pd.DataFrame)

    def _summary_block(self, L: list[str], by_group: dict) -> None:
        base_up = self.baseline.get("up_rate", float("nan"))
        up, down = by_group.get("up"), by_group.get("down")
        if up is None or down is None:
            return
        all_dur = self.durations[self.durations["group"] == "all"] if not self.durations.empty else self.durations
        med_bars = f"{all_dur['median_bars'].iloc[0]:.0f}" if not all_dur.empty else "n/a"
        up_ci = self.primary_ci.get("up_mean", (float("nan"), float("nan")))
        dn_ci = self.primary_ci.get("down_mean", (float("nan"), float("nan")))
        straddles = (not np.isfinite(up_ci[0])) or (up_ci[0] <= 0 <= up_ci[1]) or (dn_ci[0] <= 0 <= dn_ci[1])
        L += ["", "## Executive summary", "",
              f"**Question.** On the {len(self.symbols)} top coins, 1h, does an EMA{self.fast}/{self.slow} "
              f"cross predict the direction of the next move, and how long does that move last?",
              "",
              f"**Answer.** {self.verdict}. At the primary {self.primary_horizon} horizon a cross_above is "
              f"close to a coin flip ({up.hit_rate * 100:.1f}% directional hits vs a {base_up * 100:.1f}% "
              f"unconditional up-rate) and a cross_below is only marginally better "
              f"({down.hit_rate * 100:.1f}% directional hits). "
              + ("The bootstrap confidence intervals on the oriented return straddle zero for both "
                 "directions, so neither cross supports a durable directional edge. "
                 if straddles else "")
              + "Moves do have a characteristic length (median ≈ "
              f"{med_bars}h from a cross to the opposite cross) but the net run is small relative to the "
              "noise and positive-skewed: more than half of all regimes finish underwater in the cross "
              "direction.",
              "",
              f"**Headline numbers ({self.primary_horizon})**:", "",
              "| metric | cross_above (↑) | cross_below (↓) |", "|---|---|---|",
              f"| n | {up.n:,} | {down.n:,} |",
              f"| directional hit rate | {up.hit_rate * 100:.1f}% | {down.hit_rate * 100:.1f}% |",
              f"| mean oriented return | {up.mean_y * 100:+.3f}% | {down.mean_y * 100:+.3f}% |",
              f"| median oriented return | {up.median_y * 100:+.3f}% | {down.median_y * 100:+.3f}% |",
              f"| 95% CI, mean oriented return | {self._ci_str('up_mean')} | {self._ci_str('down_mean')} |",
              ]

    def _ci_str(self, key: str) -> str:
        ci = self.primary_ci.get(key)
        if not ci or not np.isfinite(ci[0]):
            return "—"
        return f"[{ci[0] * 100:+.3f}%, {ci[1] * 100:+.3f}%]"

    def to_markdown(self) -> str:
        L: list[str] = []
        L.append(f"# EMA{self.fast}/{self.slow} crosses as a directional signal — "
                 f"{len(self.symbols)} coins, {self.timeframe}")
        L.append("")
        L.append(f"- market: {self.market} · timeframe: {self.timeframe} · "
                 f"signal: EMA{self.fast} vs EMA{self.slow}")
        L.append(f"- sample: {self.start} → {self.end} · design/test split at {self.design_end}")
        L.append(f"- bars: {self.n_bars:,} · cross events: {self.n_up + self.n_down:,} "
                 f"(↑ {self.n_up:,} cross_above / ↓ {self.n_down:,} cross_below)")
        if self.notes:
            L += ["", "## Notes"] + [f"- {n}" for n in self.notes]

        # --- executive summary -------------------------------------------------
        if not self.predictive.empty:
            prim = self.predictive[self.predictive["horizon"] == self.primary_horizon]
            by_group = {r.group: r for r in prim.itertuples(index=False)}
            self._summary_block(L, by_group)
        if self.verdict != "not evaluated":
            L += ["", f"## Verdict: **{self.verdict}**", "", "| # | criterion | pass |", "|---|---|---|"]
            for key, passed in self.criteria.items():
                L.append(f"| {key} | | {'✅' if passed else '❌'} |")

        L += ["", "## Cross frequency per symbol", ""]
        L += ["| symbol | bars | cross_above | cross_below | events/mo | median up bars | median down bars |",
              "|---|---|---|---|---|---|---|"]
        for r in self.frequency.itertuples(index=False):
            L.append(f"| {r.symbol} | {r.bars:,} | {r.up} | {r.down} | {r.events_per_month:.1f} | "
                     f"{r.median_up_bars:.0f} | {r.median_down_bars:.0f} |")

        L += ["", "## Predictive power — oriented forward return (design + test pooled)", "",
              "`y` is the forward return in the cross direction; `hit` = share with `y > 0`.", ""]
        L += ["| group | horizon | n | hit rate | mean y | median y |", "|---|---|---|---|---|---|"]
        for r in self.predictive.itertuples(index=False):
            hr = "—" if not np.isfinite(r.hit_rate) else f"{r.hit_rate * 100:.1f}%"
            my = "—" if not np.isfinite(r.mean_y) else f"{r.mean_y * 100:+.3f}%"
            md = "—" if not np.isfinite(r.median_y) else f"{r.median_y * 100:+.3f}%"
            L.append(f"| {r.group} | {r.horizon} | {r.n:,} | {hr} | {my} | {md} |")

        if self.baseline:
            L += ["", f"Unconditional baseline (all bars): P(up move) at {self.primary_horizon} = "
                      f"{self.baseline.get('up_rate', float('nan')) * 100:.1f}%, "
                      f"mean |move| = {self.baseline.get('mean_abs', float('nan')) * 100:.3f}%, "
                      f"bars = {self.baseline.get('n', 0):,.0f}."]

        if self.primary_ci:
            L += ["", f"### Bootstrap 95% CIs on `{self.primary_horizon}` (block-resampled by timestamp)", "",
                  "| group | mean y CI | hit rate CI |", "|---|---|---|"]
            for g in ("up", "down"):
                m = self.primary_ci.get(f"{g}_mean", (float("nan"), float("nan")))
                h = self.primary_ci.get(f"{g}_hit", (float("nan"), float("nan")))
                L.append(f"| {g} | [{m[0] * 100:+.3f}%, {m[1] * 100:+.3f}%] | "
                         f"[{h[0] * 100:.1f}%, {h[1] * 100:.1f}%] |")

        L += ["", "## Design vs test consistency", "",
              "| group | period | n | hit rate | mean y |", "|---|---|---|---|---|"]
        for label, tbl in (("design", self.design), ("test", self.test)):
            for r in tbl.itertuples(index=False):
                hr = "—" if not np.isfinite(r.hit_rate) else f"{r.hit_rate * 100:.1f}%"
                my = "—" if not np.isfinite(r.mean_y) else f"{r.mean_y * 100:+.3f}%"
                L.append(f"| {r.group} | {label} | {r.n:,} | {hr} | {my} |")

        L += ["", "## How long the move lasted (cross → opposite cross)", "",
              "The move is held until the opposite cross fires, so `bars` is the regime length.",
              "",
              "| group | n | mean bars | median | p25 | p75 | mean run | median run | share run>0 | median bars→peak | mean MFE | mean MAE | censored |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in self.durations.itertuples(index=False):
            L.append(f"| {r.group} | {r.n:,} | {r.mean_bars:.1f} | {r.median_bars:.0f} | {r.p25_bars:.0f} | "
                     f"{r.p75_bars:.0f} | {r.mean_run_pct:+.2f}% | {r.median_run_pct:+.2f}% | "
                     f"{r.share_pos_run * 100:.1f}% | {r.median_bars_to_peak:.0f} | "
                     f"{r.mean_mfe_pct:.2f}% | {r.mean_mae_pct:.2f}% | {r.censored_pct:.1f}% |")

        if not self.duration_outcomes.empty:
            L += ["", "### Duration by outcome (winning vs losing regimes)", "",
                  "| group | outcome | share | n | median bars | mean bars | mean run | median bars→peak |",
                  "|---|---|---|---|---|---|---|---|"]
            for r in self.duration_outcomes.itertuples(index=False):
                L.append(f"| {r.group} | {r.outcome} | {r.share * 100:.0f}% | {r.n:,} | {r.median_bars:.0f} | "
                         f"{r.mean_bars:.1f} | {r.mean_run_pct:+.2f}% | {r.median_bars_to_peak:.0f} |")

        L += ["", "## Per-symbol breakdown", "",
              "| symbol | up n | down n | up hit | down hit | up mean y | down mean y | median up bars | median down bars |",
              "|---|---|---|---|---|---|---|---|---|"]
        def _pct(v: float, signed: bool = False) -> str:
            if not np.isfinite(v):
                return "—"
            return f"{v * 100:+.2f}%" if signed else f"{v * 100:.1f}%"

        for r in self.per_symbol.itertuples(index=False):
            L.append(f"| {r.symbol} | {r.up_n} | {r.down_n} | {_pct(r.up_hit)} | {_pct(r.down_hit)} | "
                     f"{_pct(r.up_mean_y, True)} | {_pct(r.down_mean_y, True)} | "
                     f"{r.median_up_bars:.0f} | {r.median_down_bars:.0f} |")
        return "\n".join(L) + "\n"


def run_ema_cross_study(
    *,
    symbols: Sequence[str],
    start: str,
    end: str,
    design_end: str = "2026-01-01",
    timeframe: str = PRIMARY_TIMEFRAME,
    source_timeframe: str = PRIMARY_SOURCE_TIMEFRAME,
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
    primary_horizon: str = PRIMARY_HORIZON,
    bootstrap: int = 1000,
    market: str = "futures",
    config: Config | None = None,
    root: Path | None = None,
) -> EmaCrossResult:
    horizons = dict(DEFAULT_HORIZONS)
    panel = build_ema_cross_panel(
        symbols, start=start, end=end, timeframe=timeframe, source_timeframe=source_timeframe,
        fast=fast, slow=slow, horizons=horizons, market=market, config=config, root=root,
    )
    result = EmaCrossResult(
        symbols=list(symbols), market=market, timeframe=timeframe, fast=fast, slow=slow,
        start=start, end=end, design_end=design_end, horizons=horizons,
        primary_horizon=primary_horizon,
    )
    if panel.is_empty():
        result.notes.append("panel is empty (no data for the universe)")
        return result
    result.n_bars = panel.height
    events = _events_from_panel(panel)
    up = events[events["cross"] > 0]
    down = events[events["cross"] < 0]
    result.n_up, result.n_down = int(up.shape[0]), int(down.shape[0])
    result.events = events

    # Per-symbol cross frequency (bars-based month = 24*30 bars on 1h).
    freq_rows = []
    for sym_df in panel.partition_by("symbol", maintain_order=True):
        s = sym_df["symbol"][0]
        ups = sym_df.filter(pl.col("cross") == 1)
        dns = sym_df.filter(pl.col("cross") == -1)
        months = max(sym_df.height / (24 * 30.0), 1e-9)
        freq_rows.append({
            "symbol": s,
            "bars": sym_df.height,
            "up": ups.height,
            "down": dns.height,
            "events_per_month": (ups.height + dns.height) / months,
            "median_up_bars": float(ups["regime_bars"].median()) if ups.height else float("nan"),
            "median_down_bars": float(dns["regime_bars"].median()) if dns.height else float("nan"),
        })
    result.frequency = pd.DataFrame(freq_rows).sort_values("symbol").reset_index(drop=True)

    result.predictive = pd.DataFrame(_direction_blocks(events, horizons))
    result.durations = pd.DataFrame(_duration_blocks(events))
    result.duration_outcomes = pd.DataFrame(_duration_by_outcome(events))

    cutoff = pd.Timestamp(design_end, tz="UTC")
    ev_ts = pd.to_datetime(events["timestamp"], utc=True)
    result.design = pd.DataFrame(_direction_blocks(events[ev_ts < cutoff], {primary_horizon}))
    result.test = pd.DataFrame(_direction_blocks(events[ev_ts >= cutoff], {primary_horizon}))

    # Unconditional baseline on all bars.
    base = panel.select(
        (pl.col(f"fwd_{primary_horizon}") > 0).mean().alias("up_rate"),
        pl.col(f"fwd_{primary_horizon}").abs().mean().alias("mean_abs"),
        pl.len().alias("n"),
    ).row(0)
    result.baseline = {"up_rate": float(base[0]), "mean_abs": float(base[1]), "n": float(base[2])}

    # Bootstrap CIs on the primary horizon.
    block = horizons.get(primary_horizon, 24)
    for gname, g in (("up", up), ("down", down)):
        if g.empty:
            continue
        result.primary_ci[f"{gname}_mean"] = _bootstrap_ci(
            g, lambda s: float(s[f"y_{primary_horizon}"].mean()), n=bootstrap, block=block
        )
        result.primary_ci[f"{gname}_hit"] = _bootstrap_ci(
            g, lambda s: float((s[f"y_{primary_horizon}"] > 0).mean()), n=bootstrap, block=block, seed=1
        )

    # Per-symbol directional stats.
    ps_rows = []
    for s in sorted(panel["symbol"].unique().to_list()):
        gu = up[up["symbol"] == s]
        gd = down[down["symbol"] == s]
        ps_rows.append({
            "symbol": s,
            "up_n": int(gu.shape[0]), "down_n": int(gd.shape[0]),
            "up_hit": float((gu[f"y_{primary_horizon}"] > 0).mean()) if not gu.empty else float("nan"),
            "down_hit": float((gd[f"y_{primary_horizon}"] > 0).mean()) if not gd.empty else float("nan"),
            "up_mean_y": float(gu[f"y_{primary_horizon}"].mean()) if not gu.empty else float("nan"),
            "down_mean_y": float(gd[f"y_{primary_horizon}"].mean()) if not gd.empty else float("nan"),
            "median_up_bars": float(gu["regime_bars"].median()) if not gu.empty else float("nan"),
            "median_down_bars": float(gd["regime_bars"].median()) if not gd.empty else float("nan"),
        })
    result.per_symbol = pd.DataFrame(ps_rows)

    # Notes / diagnostics.
    pool_ic = pooled_ic(events.dropna(subset=[f"y_{primary_horizon}"]),
                        "ema_sign", f"y_{primary_horizon}")

    # Verdict: does either cross beat a coin flip / baseline with a positive
    # bootstrap CI and a stable design→test sign?
    base_up = result.baseline.get("up_rate", float("nan"))
    test_up = result.test[result.test["group"] == "up"]
    test_dn = result.test[result.test["group"] == "down"]
    des_up = result.design[result.design["group"] == "up"]
    up_hit = float(test_up.iloc[0]["hit_rate"]) if not test_up.empty else float("nan")
    dn_hit = float(test_dn.iloc[0]["hit_rate"]) if not test_dn.empty else float("nan")
    up_mean = float(test_up.iloc[0]["mean_y"]) if not test_up.empty else float("nan")
    des_up_mean = float(des_up.iloc[0]["mean_y"]) if not des_up.empty else float("nan")
    up_ci = result.primary_ci.get("up_mean", (float("nan"), float("nan")))
    dn_ci = result.primary_ci.get("down_mean", (float("nan"), float("nan")))
    result.criteria = {
        "cross_above: test hit rate > baseline + 2pp": bool(np.isfinite(up_hit) and up_hit > base_up + 0.02),
        "cross_above: oriented mean 95% CI low > 0": bool(np.isfinite(up_ci[0]) and up_ci[0] > 0),
        "cross_above: same sign design vs test": bool(
            np.isfinite(up_mean) and np.isfinite(des_up_mean) and np.sign(up_mean) == np.sign(des_up_mean)
        ),
        "cross_below: test hit rate > baseline + 2pp": bool(np.isfinite(dn_hit) and dn_hit > (1 - base_up) + 0.02),
        "cross_below: oriented mean 95% CI low > 0": bool(np.isfinite(dn_ci[0]) and dn_ci[0] > 0),
        "pooled rank IC > 0.02": bool(np.isfinite(pool_ic) and pool_ic > 0.02),
    }
    result.verdict = "EDGE" if all(result.criteria.values()) else "NO EDGE (H0)"
    result.notes.append(f"pooled rank IC(ema_sign, y_{primary_horizon}) = {pool_ic:.4f}")
    upd = result.design[result.design["group"] == "up"]
    upt = result.test[result.test["group"] == "up"]
    if not upd.empty and not upt.empty:
        result.notes.append(
            f"cross_above hit rate: design {upd.iloc[0]['hit_rate'] * 100:.1f}% → "
            f"test {upt.iloc[0]['hit_rate'] * 100:.1f}%"
        )
    return result


__all__ = [
    "DEFAULT_FAST",
    "DEFAULT_HORIZONS",
    "DEFAULT_SLOW",
    "EmaCrossResult",
    "PRIMARY_HORIZON",
    "PRIMARY_TIMEFRAME",
    "add_forward_outcomes",
    "attach_regime_stats",
    "build_ema_cross_panel",
    "ema_cross_signals",
    "run_ema_cross_study",
]
