"""Post-signal excursion and reversion analysis.

For every breakout signal we look forward from the signal bar and measure how
long it takes price to reach configurable milestones, and how long until it
reverts to the signal price.

Three milestone definitions are supported because "50% / 90% above the signal"
is ambiguous:

``pct_signal``
    Level is a percentage above the *signal price*:
    ``ref * (1 + fraction)`` (for longs). A +90% level is rarely reached.

``pct_excursion``
    Level is a percentage of the *eventual maximum favourable excursion*:
    ``ref + fraction * max_favourable_move``. Measures how quickly the move is
    captured.

``atr``
    Level is a multiple of ATR at the signal: ``ref + fraction * ATR``.

Time is reported both in bars and in wall-clock seconds (which accounts for
data gaps). Signals whose milestone is not reached within the horizon are
censored and counted in the hit-rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl


@dataclass(slots=True)
class ExcursionResult:
    events: pl.DataFrame
    summary: pl.DataFrame
    config: dict[str, Any] = field(default_factory=dict)
    reversion_cdf: pl.DataFrame = field(default_factory=lambda: pl.DataFrame())
    excursion_by_bucket: pl.DataFrame = field(default_factory=lambda: pl.DataFrame())

    def format(self) -> str:
        lines = ["Post-signal excursion analysis"]
        for k, v in self.config.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
        if self.summary.is_empty():
            lines.append("  no signals found")
            return "\n".join(lines)
        lines.append(self.summary.to_pandas().to_string(index=False))
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = ["# Post-signal excursion analysis", ""]
        for k, v in self.config.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
        if self.summary.is_empty():
            lines.append("_No signals found._")
            return "\n".join(lines)
        cols = self.summary.columns
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for row in self.summary.iter_rows(named=True):
            lines.append("| " + " | ".join(_md_cell(row[c]) for c in cols) + " |")
        if not self.reversion_cdf.is_empty():
            lines.append("")
            lines.append("## Time to reversion (cumulative)")
            lines.append("")
            lines.append("| within | fraction reverted |")
            lines.append("| --- | --- |")
            for row in self.reversion_cdf.iter_rows(named=True):
                lines.append(f"| {row['label']} | {row['fraction']:.3f} |")
        if not self.excursion_by_bucket.is_empty():
            lines.append("")
            lines.append("## Excursion before reversion, by reversion-speed bucket")
            lines.append("")
            lines.append(
                "`*_value` is the favourable price move in quote currency before "
                "price reverted to the signal price; `*_pct` is that move as a "
                "fraction of the signal price. Buckets are disjoint."
            )
            lines.append("")
            bcols = self.excursion_by_bucket.columns
            lines.append("| " + " | ".join(bcols) + " |")
            lines.append("| " + " | ".join("---" for _ in bcols) + " |")
            for row in self.excursion_by_bucket.iter_rows(named=True):
                lines.append("| " + " | ".join(_md_cell(row[c]) for c in bcols) + " |")
        lines.append("")
        lines.append(
            "Times are measured from the **close of the signal bar**. "
            "Bars are 1 bar = one timeframe interval; seconds are wall-clock "
            "(they account for data gaps)."
        )
        return "\n".join(lines)


def _label(fraction: float) -> str:
    return f"{fraction * 100:g}pct".replace(".", "_")


def _md_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.6g}"
    return str(value)


DEFAULT_REVERSION_BUCKETS: list[tuple[float, float, str]] = [
    (0, 60, "<=1m"),
    (60, 300, "1m-5m"),
    (300, 900, "5m-15m"),
    (900, 3_600, "15m-1h"),
    (3_600, 14_400, "1h-4h"),
    (14_400, 86_400, "4h-1d"),
    (86_400, 604_800, "1d-7d"),
]


def _bucket_stats(
    label: str, mask: np.ndarray, excursion: np.ndarray, pct: np.ndarray
) -> dict[str, Any]:
    n = int(mask.sum())
    if n == 0:
        return {"bucket": label, "n": 0}
    e = excursion[mask]
    p = pct[mask]
    finite_p = p[np.isfinite(p)]
    return {
        "bucket": label,
        "n": n,
        "mean_value": float(np.mean(e)),
        "median_value": float(np.median(e)),
        "min_value": float(np.min(e)),
        "max_value": float(np.max(e)),
        "p25_value": float(np.percentile(e, 25)),
        "p75_value": float(np.percentile(e, 75)),
        "mean_pct": float(np.mean(finite_p)) if finite_p.size else None,
        "median_pct": float(np.median(finite_p)) if finite_p.size else None,
        "min_pct": float(np.min(finite_p)) if finite_p.size else None,
        "max_pct": float(np.max(finite_p)) if finite_p.size else None,
    }


def excursion_by_reversion_bucket(
    events: pl.DataFrame,
    buckets: list[tuple[float, float, str]] | None = None,
) -> pl.DataFrame:
    """Excursion realised before reversion, grouped by reversion-speed bucket.

    Buckets are disjoint intervals on ``time_to_reversion_seconds`` so that the
    excursion statistics describe distinct groups (a cumulative view would make
    every later bucket nearly identical to the first).
    """
    if events.is_empty() or "time_to_reversion_seconds" not in events.columns:
        return pl.DataFrame()
    t = (
        events["time_to_reversion_seconds"]
        .cast(pl.Float64)
        .fill_null(float("inf"))
        .to_numpy()
    )
    excursion = (
        events["excursion_to_reversion"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    )
    pct = (
        events["excursion_to_reversion_pct"]
        .cast(pl.Float64)
        .fill_null(float("nan"))
        .to_numpy()
    )
    rows = [
        _bucket_stats(label, (t > lo) & (t <= hi), excursion, pct)
        for lo, hi, label in (buckets or DEFAULT_REVERSION_BUCKETS)
    ]
    rows.append(_bucket_stats("not_reverted_7d", ~np.isfinite(t), excursion, pct))
    return pl.DataFrame(rows)


def _stats(values: np.ndarray) -> dict[str, float | None]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"mean": None, "median": None, "p25": None, "p75": None}
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
    }


def analyze_excursions(
    data: pl.DataFrame,
    *,
    signal_col: str = "entries",
    mode: str = "pct_excursion",
    targets: Sequence[float] = (0.5, 0.9),
    horizon_bars: int | None = 10_080,
    side: str = "long",
    atr_col: str = "atr",
    reversion_basis: str = "close",
) -> ExcursionResult:
    """Measure time-to-milestone and time-to-reversion for each signal.

    The reference price is the **close of the signal bar**. Execution timing is
    the backtest engine's concern; this analysis is about the raw path.

    ``reversion_basis`` selects how "reverted to the signal price" is detected:
    ``"close"`` requires a bar to *close* back at/beyond the signal price,
    ``"touch"`` counts an intrabar wick to that level.
    """
    if mode not in {"pct_signal", "pct_excursion", "atr"}:
        raise ValueError(f"unknown mode {mode!r}")
    if side not in {"long", "short"}:
        raise ValueError(f"unknown side {side!r}")
    if reversion_basis not in {"close", "touch"}:
        raise ValueError(f"unknown reversion_basis {reversion_basis!r}")
    if signal_col not in data.columns:
        raise KeyError(f"{signal_col!r} not in data")

    ts = data["timestamp"].to_numpy().astype("datetime64[us]").astype(np.int64)
    close = data["close"].to_numpy().astype(float)
    high = data["high"].to_numpy().astype(float)
    low = data["low"].to_numpy().astype(float)
    atr = data[atr_col].to_numpy().astype(float) if atr_col in data.columns else None
    signals = data[signal_col].fill_null(False).to_numpy().astype(bool)
    n = len(close)

    signal_indices = np.flatnonzero(signals)
    rows: list[dict[str, Any]] = []
    labels = [_label(t) for t in targets]

    for i in signal_indices:
        remaining = n - i - 1
        h = remaining if horizon_bars is None else min(horizon_bars, remaining)
        row: dict[str, Any] = {
            "signal_index": int(i),
            "signal_time": data["timestamp"][int(i)],
            "ref_price": float(close[i]),
            "side": side,
        }
        if h <= 0:
            row["max_favourable_move"] = None
            row["time_to_max_seconds"] = None
            row["time_to_max_bars"] = None
            row["reverted"] = None
            row["time_to_reversion_seconds"] = None
            row["time_to_reversion_bars"] = None
            row["reverted_touch"] = None
            row["time_to_reversion_touch_seconds"] = None
            row["time_to_reversion_touch_bars"] = None
            row["excursion_to_reversion"] = None
            row["excursion_to_reversion_pct"] = None
            for lab in labels:
                row[f"hit_{lab}"] = None
                row[f"time_to_{lab}_seconds"] = None
                row[f"time_to_{lab}_bars"] = None
            rows.append(row)
            continue

        seg_high = high[i + 1 : i + 1 + h]
        seg_low = low[i + 1 : i + 1 + h]
        seg_close = close[i + 1 : i + 1 + h]
        seg_ts = ts[i + 1 : i + 1 + h]
        ref = close[i]

        if side == "long":
            fav = seg_high - ref
            excursion = float(max(0.0, np.max(fav)))
            adv = seg_high
            rev = seg_low
        else:
            fav = ref - seg_low
            excursion = float(max(0.0, np.max(fav)))
            adv = -seg_low  # favourable direction increases
            rev = seg_high
        rev_level = ref
        atr_unit = atr[i] if atr is not None else float("nan")

        row["max_favourable_move"] = excursion
        if excursion > 0:
            j = int(np.argmax(fav))
            row["time_to_max_bars"] = j + 1
            row["time_to_max_seconds"] = float((seg_ts[j] - ts[i]) / 1_000_000)
        else:
            row["time_to_max_bars"] = None
            row["time_to_max_seconds"] = None

        # Reversion to the signal price.
        touch_mask = rev <= rev_level if side == "long" else rev >= rev_level
        close_mask = seg_close <= rev_level if side == "long" else seg_close >= rev_level
        chosen = close_mask if reversion_basis == "close" else touch_mask

        row["reverted_touch"] = bool(touch_mask.any())
        if touch_mask.any():
            jt = int(np.argmax(touch_mask))
            row["time_to_reversion_touch_bars"] = jt + 1
            row["time_to_reversion_touch_seconds"] = float((seg_ts[jt] - ts[i]) / 1_000_000)
        else:
            row["time_to_reversion_touch_bars"] = None
            row["time_to_reversion_touch_seconds"] = None

        if chosen.any():
            j = int(np.argmax(chosen))
            row["reverted"] = True
            row["time_to_reversion_bars"] = j + 1
            row["time_to_reversion_seconds"] = float((seg_ts[j] - ts[i]) / 1_000_000)
            reversion_bars = j + 1
        else:
            row["reverted"] = False
            row["time_to_reversion_bars"] = None
            row["time_to_reversion_seconds"] = None
            reversion_bars = None

        # Excursion actually realised before price reverted to the signal price.
        if reversion_bars is not None:
            k = reversion_bars
            if side == "long":
                exc_rev = float(max(0.0, np.max(seg_high[:k]) - ref))
            else:
                exc_rev = float(max(0.0, ref - np.min(seg_low[:k])))
        else:
            exc_rev = excursion
        row["excursion_to_reversion"] = exc_rev
        row["excursion_to_reversion_pct"] = (exc_rev / ref) if ref else None

        for t, lab in zip(targets, labels, strict=True):
            if mode == "pct_excursion" and excursion <= 0:
                hit, bars = False, None
            else:
                if mode == "pct_signal":
                    level = ref * (1.0 + t) if side == "long" else ref * (1.0 - t)
                elif mode == "pct_excursion":
                    level = ref + t * excursion if side == "long" else ref - t * excursion
                else:  # atr
                    level = ref + t * atr_unit if side == "long" else ref - t * atr_unit
                mask = adv >= level if side == "long" else adv <= level
                if mask.any():
                    j = int(np.argmax(mask))
                    hit, bars = True, j + 1
                else:
                    hit, bars = False, None
            row[f"hit_{lab}"] = hit
            row[f"time_to_{lab}_bars"] = bars
            row[f"time_to_{lab}_seconds"] = (
                float((seg_ts[bars - 1] - ts[i]) / 1_000_000) if bars is not None else None
            )
            row[f"reverted_before_{lab}"] = (
                bool(reversion_bars is not None and (bars is None or reversion_bars < bars))
                if reversion_bars is not None
                else False
            )
        rows.append(row)

    events = pl.DataFrame(rows) if rows else pl.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    if rows:
        for lab in labels:
            hits = np.array([r[f"hit_{lab}"] for r in rows], dtype=object)
            hit_bool = np.array([bool(x) for x in hits])
            bars = np.array(
                [r[f"time_to_{lab}_bars"] if r[f"time_to_{lab}_bars"] is not None else np.nan for r in rows],
                dtype=float,
            )
            secs = np.array(
                [r[f"time_to_{lab}_seconds"] if r[f"time_to_{lab}_seconds"] is not None else np.nan for r in rows],
                dtype=float,
            )
            bstats = _stats(bars)
            sstats = _stats(secs)
            summary_rows.append(
                {
                    "milestone": lab,
                    "n_signals": len(rows),
                    "hit_rate": float(hit_bool.mean()),
                    "median_bars": bstats["median"],
                    "mean_bars": bstats["mean"],
                    "p25_bars": bstats["p25"],
                    "p75_bars": bstats["p75"],
                    "median_seconds": sstats["median"],
                    "mean_seconds": sstats["mean"],
                    "p25_seconds": sstats["p25"],
                    "p75_seconds": sstats["p75"],
                }
            )
        rev_secs = np.array(
            [r["time_to_reversion_seconds"] if r["time_to_reversion_seconds"] is not None else np.nan for r in rows],
            dtype=float,
        )
        rev_bars = np.array(
            [r["time_to_reversion_bars"] if r["time_to_reversion_bars"] is not None else np.nan for r in rows],
            dtype=float,
        )
        bstats = _stats(rev_bars)
        sstats = _stats(rev_secs)
        summary_rows.append(
            {
                "milestone": "reversion_to_signal",
                "n_signals": len(rows),
                "hit_rate": float(np.mean([bool(r["reverted"]) for r in rows])),
                "median_bars": bstats["median"],
                "mean_bars": bstats["mean"],
                "p25_bars": bstats["p25"],
                "p75_bars": bstats["p75"],
                "median_seconds": sstats["median"],
                "mean_seconds": sstats["mean"],
                "p25_seconds": sstats["p25"],
                "p75_seconds": sstats["p75"],
            }
        )

    summary = pl.DataFrame(summary_rows) if summary_rows else pl.DataFrame()

    reversion_cdf = pl.DataFrame()
    if rows and "time_to_reversion_seconds" in events.columns:
        secs = events["time_to_reversion_seconds"].fill_null(float("inf")).to_numpy()
        thresholds = [
            (60, "1m"), (300, "5m"), (900, "15m"), (3600, "1h"),
            (14400, "4h"), (86400, "1d"), (604800, "7d"),
        ]
        reversion_cdf = pl.DataFrame(
            [
                {
                    "within_seconds": t,
                    "label": lab,
                    "fraction": float((secs <= t).mean()),
                }
                for t, lab in thresholds
            ]
        )

    config = {
        "signal_col": signal_col,
        "mode": mode if targets else "reversion_only",
        "horizon_bars": horizon_bars,
        "side": side,
        "reversion_basis": reversion_basis,
        "reference_price": "close of signal bar",
        "n_signals": len(rows),
    }
    if targets:
        config["targets"] = list(targets)
    return ExcursionResult(
        events=events,
        summary=summary,
        config=config,
        reversion_cdf=reversion_cdf,
        excursion_by_bucket=excursion_by_reversion_bucket(events) if rows else pl.DataFrame(),
    )
