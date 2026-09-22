"""Market-regime classification tests."""

from __future__ import annotations

import polars as pl

from quant_lab.analytics.regimes import (
    RegimeConfig,
    attach_regimes,
    classify_regimes,
    evaluate_by_regime,
)
from quant_lab.synthetic import synthetic_bars

CFG = RegimeConfig(
    regime_timeframe="15m",
    trend_window=8,
    vol_window=8,
    vol_rank_window=32,
    trend_threshold=1.0,
)


def test_classify_uptrend():
    bars = synthetic_bars(6000, seed=61, drift=0.005, volatility=0.002)
    regimes = classify_regimes(bars, config=CFG, source_timeframe="1m")
    assert regimes.height > 300
    trends = set(regimes["regime_trend"].unique().to_list())
    assert trends == {"up"}


def test_classify_downtrend():
    bars = synthetic_bars(6000, seed=62, drift=-0.005, volatility=0.002)
    regimes = classify_regimes(bars, config=CFG, source_timeframe="1m")
    assert set(regimes["regime_trend"].unique().to_list()) == {"down"}


def test_classify_range():
    bars = synthetic_bars(6000, seed=63, drift=0.0, volatility=0.002)
    regimes = classify_regimes(bars, config=CFG, source_timeframe="1m")
    assert "range" in set(regimes["regime_trend"].unique().to_list())


def test_regime_labels_are_causal():
    bars = synthetic_bars(6000, seed=64, drift=0.003, volatility=0.004)
    cutoff = 3000  # multiple of the 15m regime bucket
    full = classify_regimes(bars, config=CFG, source_timeframe="1m")
    part = classify_regimes(bars.head(cutoff), config=CFG, source_timeframe="1m")
    joined = part.select("timestamp", pl.col("regime").alias("regime_part")).join(
        full.select("timestamp", "regime"), on="timestamp", how="inner"
    )
    assert joined.height > 0
    assert (joined["regime"] == joined["regime_part"]).all()


def test_attach_regimes_asof():
    bars = synthetic_bars(6000, seed=65, drift=0.004, volatility=0.003)
    regimes = classify_regimes(bars, config=CFG, source_timeframe="1m")
    # signals on a couple of arbitrary timestamps
    signals = pl.DataFrame(
        {"timestamp": [regimes["timestamp"][100], regimes["timestamp"][200]]}
    )
    out = attach_regimes(signals, regimes)
    assert out["regime"].null_count() == 0
    assert out["regime"][0] == regimes["regime"][100]


def test_evaluate_by_regime():
    labelled = pl.DataFrame(
        {
            "entries": [True] * 4,
            "is_good": [True, False, True, True],
            "regime": ["up_lowvol", "up_lowvol", "range_highvol", "range_highvol"],
            "x": [1, 0, 1, 1],
        }
    )
    from quant_lab.analytics.confluence import Filter

    table = evaluate_by_regime(labelled, [Filter("x=1", pl.col("x") == 1, ("x",))])
    by = {(r["regime"], r["filter"]): r for r in table.iter_rows(named=True)}
    up = by[("up_lowvol", "BASELINE (all signals)")]
    assert up["n_signals"] == 2 and up["p_good"] == 0.5
    x1 = by[("range_highvol", "x=1")]
    assert x1["n_signals"] == 2 and x1["lift"] == 1.0
