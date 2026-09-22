"""Causal market-regime classification.

Regimes are computed on a coarser *regime timeframe* (default 1h) from trailing
statistics only, then attached to finer-grained signals as-of the signal time.
Two orthogonal dimensions are labelled:

* **trend** — ``up`` / ``down`` / ``range``, from a volatility-normalised
  trailing move (a z-score of the cumulative log return, so it is comparable
  across price levels and regimes).
* **volatility** — ``highvol`` / ``lowvol``, from whether realised volatility
  sits above its trailing median.

The combined label is ``{trend}_{vol}`` (e.g. ``up_lowvol``).

Causality: every regime column is shifted by one regime bar, so the label
attached to a signal only uses regime bars that had already *closed* before the
signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import polars as pl

from ..data.resample import resample_bars
from ..features.core import safe_div


@dataclass(slots=True)
class RegimeConfig:
    regime_timeframe: str = "1h"
    trend_window: int = 24
    vol_window: int = 24
    vol_rank_window: int = 720
    trend_threshold: float = 1.0
    vol_high_quantile: float = 0.5
    causal_shift: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime_timeframe": self.regime_timeframe,
            "trend_window": self.trend_window,
            "vol_window": self.vol_window,
            "vol_rank_window": self.vol_rank_window,
            "trend_threshold": self.trend_threshold,
            "vol_high_quantile": self.vol_high_quantile,
            "causal_shift": self.causal_shift,
        }


def classify_regimes(
    bars: pl.DataFrame,
    *,
    config: RegimeConfig | None = None,
    source_timeframe: str = "1m",
) -> pl.DataFrame:
    """Return per-regime-bar labels (``timestamp``, ``regime*``, features)."""
    config = config or RegimeConfig()
    frame = bars
    if source_timeframe != config.regime_timeframe:
        frame = resample_bars(
            frame, timeframe=config.regime_timeframe, source_timeframe=source_timeframe
        )
    frame = frame.sort("timestamp")

    logp = pl.col("close").log()
    frame = frame.with_columns(
        (logp - logp.shift(config.trend_window)).alias("_trend_ret"),
        (logp - logp.shift(1)).alias("_logret1"),
    ).with_columns(
        pl.col("_logret1")
        .rolling_std(window_size=config.vol_window, min_periods=config.vol_window)
        .alias("realized_vol")
    ).with_columns(
        safe_div(
            pl.col("_trend_ret"),
            pl.col("realized_vol") * math.sqrt(config.trend_window),
        ).alias("trend_strength"),
        pl.col("realized_vol")
        .rolling_quantile(
            config.vol_high_quantile,
            interpolation="linear",
            window_size=config.vol_rank_window,
            min_periods=max(2, config.vol_rank_window // 4),
        )
        .alias("vol_threshold"),
    ).with_columns(
        pl.when(pl.col("trend_strength") > config.trend_threshold)
        .then(pl.lit("up"))
        .when(pl.col("trend_strength") < -config.trend_threshold)
        .then(pl.lit("down"))
        .otherwise(pl.lit("range"))
        .alias("regime_trend"),
        pl.when(pl.col("realized_vol") > pl.col("vol_threshold"))
        .then(pl.lit("highvol"))
        .otherwise(pl.lit("lowvol"))
        .alias("regime_vol"),
    ).with_columns(
        (pl.col("regime_trend") + "_" + pl.col("regime_vol")).alias("regime")
    )

    if config.causal_shift:
        shift = config.causal_shift
        frame = frame.with_columns(
            pl.col("regime").shift(shift).alias("regime"),
            pl.col("regime_trend").shift(shift).alias("regime_trend"),
            pl.col("regime_vol").shift(shift).alias("regime_vol"),
        )

    keep = [
        "timestamp", "regime", "regime_trend", "regime_vol",
        "trend_strength", "realized_vol",
    ]
    # Drop warm-up rows where the regime inputs are not yet available.
    valid = frame.filter(
        pl.col("trend_strength").is_not_null()
        & pl.col("realized_vol").is_not_null()
        & pl.col("vol_threshold").is_not_null()
        & pl.col("regime").is_not_null()
    )
    return valid.select([c for c in keep if c in valid.columns]).sort("timestamp")


def attach_regimes(signals: pl.DataFrame, regimes: pl.DataFrame) -> pl.DataFrame:
    """As-of join regime labels onto signals (backward, causal)."""
    if regimes.is_empty():
        return signals.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("regime"),
            pl.lit(None, dtype=pl.Utf8).alias("regime_trend"),
            pl.lit(None, dtype=pl.Utf8).alias("regime_vol"),
        )
    labels = regimes.select("timestamp", "regime", "regime_trend", "regime_vol").sort(
        "timestamp"
    )
    return (
        signals.sort("timestamp")
        .join_asof(labels, on="timestamp", strategy="backward")
        .sort("timestamp")
    )


def gate_signals(
    signals: pl.DataFrame,
    allowed_regimes,
    *,
    regime_col: str = "regime",
    columns: tuple[str, ...] = ("entries", "short_entries"),
) -> pl.DataFrame:
    """Zero out entry signals that occur outside ``allowed_regimes``.

    Exits are left untouched so open positions can always be closed.
    """
    allowed = list(allowed_regimes)
    exprs = []
    for col in columns:
        if col in signals.columns:
            exprs.append(
                pl.when(pl.col(regime_col).is_in(allowed))
                .then(pl.col(col))
                .otherwise(pl.lit(False))
                .cast(pl.Boolean)
                .alias(col)
            )
    return signals.with_columns(exprs) if exprs else signals


def evaluate_by_regime(
    labelled: pl.DataFrame,
    filters,
    *,
    regime_col: str = "regime",
    signal_col: str = "entries",
) -> pl.DataFrame:
    """P(good | filter, regime) with lift measured against that regime's baseline."""
    if regime_col not in labelled.columns:
        raise KeyError(f"{regime_col!r} not in labelled frame")
    base = labelled.filter(pl.col(signal_col).fill_null(False)).filter(
        pl.col(regime_col).is_not_null()
    )
    if base.height == 0:
        return pl.DataFrame()
    rows: list[dict[str, Any]] = []
    available = set(labelled.columns)
    for regime in sorted(base[regime_col].unique().to_list()):
        sub = base.filter(pl.col(regime_col) == regime)
        n_reg = sub.height
        p_reg = float(sub["is_good"].mean())
        rows.append(
            {
                "regime": regime,
                "filter": "BASELINE (all signals)",
                "n_signals": n_reg,
                "p_good": p_reg,
                "lift": 1.0,
            }
        )
        for filt in filters:
            if not set(filt.requires).issubset(available):
                continue
            selected = sub.filter(filt.expr)
            n = selected.height
            p = float(selected["is_good"].mean()) if n else None
            rows.append(
                {
                    "regime": regime,
                    "filter": filt.name,
                    "n_signals": n,
                    "p_good": p,
                    "lift": (p / p_reg) if (p is not None and p_reg) else None,
                }
            )
    return pl.DataFrame(rows)
