"""Confluence-filter evaluation.

The raw breakout has no edge on its own; the question is whether additional
conditions select the small fraction of signals that actually run.

This module labels every signal with a "good outcome" (a meaningful favourable
excursion that does **not** immediately revert) and then measures, for each
candidate confluence filter, the probability of a good outcome versus the
unfiltered baseline. Filters are simple, explicit boolean conditions — not a
model — and are evaluated on a design period and a separate validation period
to guard against overfitting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from ..config import Config
from ..data.resample import resample_bars
from ..data.schemas import DatasetType
from ..data.storage import DatasetKey, read_dataset
from ..features.core import safe_div
from ..strategies.base import Strategy
from ..timeframes import timeframe_seconds
from .excursions import analyze_excursions
from .regimes import RegimeConfig, attach_regimes, classify_regimes, evaluate_by_regime


@dataclass(slots=True)
class LabelConfig:
    """Definition of a 'good' breakout outcome."""

    excursion_pct_threshold: float = 0.0025
    min_reversion_seconds: float = 3600.0
    horizon_bars: int | None = 10_080
    reversion_basis: str = "close"

    def to_dict(self) -> dict[str, Any]:
        return {
            "excursion_pct_threshold": self.excursion_pct_threshold,
            "min_reversion_seconds": self.min_reversion_seconds,
            "horizon_bars": self.horizon_bars,
            "reversion_basis": self.reversion_basis,
        }


@dataclass(slots=True)
class Filter:
    name: str
    expr: pl.Expr
    requires: tuple[str, ...] = ()


def default_filters() -> list[Filter]:
    """Candidate confluence conditions, all evaluated at the signal bar."""
    return [
        Filter("volume>=1.5x", pl.col("relative_volume") >= 1.5, ("relative_volume",)),
        Filter("taker_ratio>=0.55", pl.col("taker_ratio") >= 0.55, ("taker_ratio",)),
        Filter("cvd_20>0", pl.col("cvd_change_20") > 0, ("cvd_change_20",)),
        Filter("close>ema200", pl.col("close") > pl.col("ema_200"), ("ema_200",)),
        Filter("strong_close>=0.7", pl.col("close_location") >= 0.7, ("close_location",)),
        Filter("bandwidth_z<=-0.5", pl.col("bandwidth_z") <= -0.5, ("bandwidth_z",)),
        Filter("basis>0", pl.col("basis") > 0, ("basis",)),
        Filter("funding<=0", pl.col("funding_rate") <= 0, ("funding_rate",)),
        Filter("oi_change_1h>0", pl.col("oi_change_1h") > 0, ("oi_change_1h",)),
    ]


# --- context / feature assembly ---------------------------------------------


def enrich_frame(
    frame: pl.DataFrame,
    config: Config,
    *,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    market_type: str = "futures",
    source_timeframe: str = "1m",
) -> pl.DataFrame:
    """Left-join spot close, funding and open interest onto the bar frame."""
    root = config.paths.data_normalized
    out = frame

    spot_key = DatasetKey("binance", "spot", DatasetType.BARS, symbol, source_timeframe)
    spot = read_dataset(root, spot_key, start=start, end=end)
    if spot.height and source_timeframe != timeframe:
        spot = resample_bars(spot, timeframe=timeframe, source_timeframe=source_timeframe)
    if spot.height:
        out = out.join(
            spot.select(pl.col("timestamp"), pl.col("close").alias("spot_close")),
            on="timestamp",
            how="left",
        )

    funding_key = DatasetKey("binance", "futures", DatasetType.FUNDING, symbol, "na")
    funding = read_dataset(root, funding_key, start=start, end=end)
    if funding.height:
        funding = (
            funding.select(pl.col("timestamp").dt.truncate("1m").alias("timestamp"), "funding_rate")
            .unique(subset=["timestamp"], keep="last")
        )
        out = out.join(funding, on="timestamp", how="left").with_columns(
            pl.col("funding_rate").forward_fill()
        )

    oi_key = DatasetKey("binance", "futures", DatasetType.OPEN_INTEREST, symbol, "na")
    oi = read_dataset(root, oi_key, start=start, end=end)
    if oi.height:
        out = out.join(
            oi.select("timestamp", "open_interest"), on="timestamp", how="left"
        ).with_columns(pl.col("open_interest").forward_fill())

    return out


def add_confluence_features(
    frame: pl.DataFrame, *, bandwidth_window: int = 240, timeframe: str | None = None
) -> pl.DataFrame:
    """Add bar-derived confluence features (skips anything that needs missing data)."""
    bars_per_hour = 60
    if timeframe:
        try:
            bars_per_hour = max(1, int(3600 / timeframe_seconds(timeframe)))
        except ValueError:
            bars_per_hour = 1
    out = frame
    exprs = [
        pl.col("volume")
        .rolling_mean(window_size=20, min_periods=20)
        .alias("volume_mean_20"),
        safe_div(pl.col("taker_buy_volume"), pl.col("volume")).alias("taker_ratio"),
        (2.0 * pl.col("taker_buy_volume") - pl.col("volume"))
        .rolling_sum(window_size=20, min_periods=20)
        .alias("cvd_change_20"),
        pl.col("close").ewm_mean(span=200, adjust=False, min_periods=200).alias("ema_200"),
        safe_div(pl.col("close") - pl.col("low"), pl.col("high") - pl.col("low")).alias(
            "close_location"
        ),
    ]
    out = out.with_columns(exprs).with_columns(
        safe_div(pl.col("volume"), pl.col("volume_mean_20")).alias("relative_volume")
    )

    if {"bb_upper", "bb_lower", "bb_mid"}.issubset(out.columns):
        out = out.with_columns(
            safe_div(pl.col("bb_upper") - pl.col("bb_lower"), pl.col("bb_mid")).alias("bandwidth")
        ).with_columns(
            safe_div(
                pl.col("bandwidth")
                - pl.col("bandwidth").rolling_mean(window_size=bandwidth_window, min_periods=bandwidth_window),
                pl.col("bandwidth").rolling_std(window_size=bandwidth_window, min_periods=bandwidth_window),
            ).alias("bandwidth_z")
        )

    if "spot_close" in out.columns:
        out = out.with_columns(
            safe_div(pl.col("close") - pl.col("spot_close"), pl.col("spot_close")).alias("basis")
        )

    if "open_interest" in out.columns:
        out = out.with_columns(
            pl.col("open_interest").pct_change(n=bars_per_hour).alias("oi_change_1h")
        )

    return out


def label_signals(
    signals: pl.DataFrame,
    *,
    signal_col: str = "entries",
    label: LabelConfig | None = None,
) -> pl.DataFrame:
    """Attach per-signal outcome labels (joined back by row index)."""
    label = label or LabelConfig()
    result = analyze_excursions(
        signals,
        signal_col=signal_col,
        targets=(),
        horizon_bars=label.horizon_bars,
        reversion_basis=label.reversion_basis,
    )
    events = result.events
    if events.is_empty():
        return signals.with_columns(
            pl.lit(None, dtype=pl.Boolean).alias("is_good"),
            pl.lit(None, dtype=pl.Float64).alias("excursion_to_reversion_pct"),
            pl.lit(None, dtype=pl.Float64).alias("time_to_reversion_seconds"),
        )
    events = events.with_columns(
        (pl.col("excursion_to_reversion_pct") >= label.excursion_pct_threshold)
        .fill_null(False)
        .alias("good_excursion"),
        (pl.col("time_to_reversion_seconds") > label.min_reversion_seconds)
        .fill_null(True)
        .alias("slow_reversion"),
    ).with_columns(
        (pl.col("good_excursion") & pl.col("slow_reversion")).alias("is_good")
    )
    return signals.with_row_index("signal_index").join(
        events.select(
            "signal_index", "is_good", "excursion_to_reversion_pct", "time_to_reversion_seconds"
        ),
        on="signal_index",
        how="left",
    )


# --- evaluation --------------------------------------------------------------


def _summary_row(name: str, selected: pl.DataFrame, baseline_p: float, total: int) -> dict[str, Any]:
    n = selected.height
    p_good = float(selected["is_good"].mean()) if n else None
    return {
        "filter": name,
        "n_signals": n,
        "coverage": n / total if total else None,
        "p_good": p_good,
        "lift": (p_good / baseline_p) if (p_good is not None and baseline_p) else None,
        "median_excursion_pct": float(selected["excursion_to_reversion_pct"].median()) if n else None,
        "median_reversion_seconds": float(selected["time_to_reversion_seconds"].median()) if n else None,
    }


def evaluate_confluence(
    labelled: pl.DataFrame,
    filters: Sequence[Filter],
    *,
    signal_col: str = "entries",
) -> pl.DataFrame:
    """P(good | filter) and lift versus the unfiltered baseline."""
    base = labelled.filter(pl.col(signal_col).fill_null(False))
    total = base.height
    if total == 0:
        return pl.DataFrame()
    baseline_p = float(base["is_good"].mean())
    rows = [
        {
            "filter": "BASELINE (all signals)",
            "n_signals": total,
            "coverage": 1.0,
            "p_good": baseline_p,
            "lift": 1.0,
            "median_excursion_pct": float(base["excursion_to_reversion_pct"].median()),
            "median_reversion_seconds": float(base["time_to_reversion_seconds"].median()),
        }
    ]
    available = set(labelled.columns)
    for filt in filters:
        if not set(filt.requires).issubset(available):
            continue
        selected = base.filter(filt.expr)
        rows.append(_summary_row(filt.name, selected, baseline_p, total))
    return pl.DataFrame(rows)


def combine_filters(filters: Sequence[Filter]) -> Filter:
    name = " AND ".join(f.name for f in filters)
    expr = filters[0].expr
    requires: list[str] = list(filters[0].requires)
    for f in filters[1:]:
        expr = expr & f.expr
        requires.extend(f.requires)
    return Filter(name, expr, tuple(requires))


def greedy_combinations(
    labelled: pl.DataFrame,
    filters: Sequence[Filter],
    *,
    max_depth: int = 3,
    min_signals: int = 30,
    signal_col: str = "entries",
) -> list[Filter]:
    """Greedily AND filters in descending order of standalone lift."""
    table = evaluate_confluence(labelled, filters, signal_col=signal_col)
    if table.is_empty():
        return []
    ranked = (
        table.filter(pl.col("filter") != "BASELINE (all signals)")
        .filter(pl.col("n_signals") >= min_signals)
        .sort("lift", descending=True)
    )
    ordered = []
    by_name = {f.name: f for f in filters}
    for name in ranked["filter"].to_list():
        if name in by_name:
            ordered.append(by_name[name])
    combos: list[Filter] = []
    for depth in range(2, max_depth + 1):
        if len(ordered) >= depth:
            combos.append(combine_filters(ordered[:depth]))
    return combos


def prepare_labelled(
    config: Config,
    strategy: Strategy,
    *,
    symbol: str,
    timeframe: str,
    market_type: str,
    start: str,
    end: str,
    label: LabelConfig,
    signal_col: str = "entries",
    source_timeframe: str = "1m",
    regime_config: RegimeConfig | None = None,
) -> pl.DataFrame:
    """Load, enrich, regenerate signals, add features and outcome labels."""
    key = DatasetKey("binance", market_type, DatasetType.BARS, symbol, source_timeframe)
    data = read_dataset(config.paths.data_normalized, key, start=start, end=end)
    if data.height == 0:
        raise FileNotFoundError(f"no data for {key.dataset_id} in {start}..{end}")
    regimes = (
        classify_regimes(data, config=regime_config, source_timeframe=source_timeframe)
        if regime_config is not None
        else None
    )
    if source_timeframe != timeframe:
        data = resample_bars(data, timeframe=timeframe, source_timeframe=source_timeframe)
    frame = enrich_frame(
        data, config, symbol=symbol, timeframe=timeframe, start=start, end=end,
        market_type=market_type, source_timeframe=source_timeframe,
    )
    signals = strategy.prepare(frame)
    signals = add_confluence_features(signals, timeframe=timeframe)
    labelled = label_signals(signals, signal_col=signal_col, label=label)
    if regimes is not None:
        labelled = attach_regimes(labelled, regimes)
    return labelled


def run_confluence(
    config: Config,
    strategy: Strategy,
    *,
    symbol: str,
    timeframe: str = "1m",
    market_type: str = "futures",
    design_start: str,
    design_end: str,
    validate_start: str,
    validate_end: str,
    label: LabelConfig | None = None,
    filters: Sequence[Filter] | None = None,
    max_depth: int = 3,
    min_signals: int = 30,
    signal_col: str = "entries",
    source_timeframe: str = "1m",
    regime_config: RegimeConfig | None = None,
) -> dict[str, Any]:
    """Evaluate confluence filters on a design period and a validation period."""
    label = label or LabelConfig()
    filters = list(filters) if filters is not None else default_filters()

    design = prepare_labelled(
        config, strategy, symbol=symbol, timeframe=timeframe, market_type=market_type,
        start=design_start, end=design_end, label=label, signal_col=signal_col,
        source_timeframe=source_timeframe, regime_config=regime_config,
    )
    validate = prepare_labelled(
        config, strategy, symbol=symbol, timeframe=timeframe, market_type=market_type,
        start=validate_start, end=validate_end, label=label, signal_col=signal_col,
        source_timeframe=source_timeframe, regime_config=regime_config,
    )
    combos = greedy_combinations(
        design, filters, max_depth=max_depth, min_signals=min_signals, signal_col=signal_col
    )
    regime_design = (
        evaluate_by_regime(design, filters, signal_col=signal_col)
        if regime_config is not None
        else pl.DataFrame()
    )
    regime_validate = (
        evaluate_by_regime(validate, filters, signal_col=signal_col)
        if regime_config is not None
        else pl.DataFrame()
    )
    return {
        "label": label.to_dict(),
        "regime": regime_config.to_dict() if regime_config is not None else None,
        "design_table": evaluate_confluence(design, filters, signal_col=signal_col),
        "validate_table": evaluate_confluence(validate, filters, signal_col=signal_col),
        "combos": combos,
        "combo_design": evaluate_confluence(design, combos, signal_col=signal_col) if combos else pl.DataFrame(),
        "combo_validate": evaluate_confluence(validate, combos, signal_col=signal_col) if combos else pl.DataFrame(),
        "regime_design": regime_design,
        "regime_validate": regime_validate,
    }
