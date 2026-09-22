"""Feature correctness and causality tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.features import FeatureSpec, compute_features
from quant_lab.features.core import atr, ema, rsi, sma


def test_sma_matches_pandas(bars):
    out = bars.with_columns(sma(bars, window=20).alias("sma20"))
    expected = pd.Series(bars["close"].to_numpy()).rolling(20).mean().to_numpy()
    got = out["sma20"].to_numpy()
    mask = ~np.isnan(expected)
    assert np.allclose(got[mask], expected[mask])


def test_ema_matches_pandas(bars):
    out = bars.with_columns(ema(bars, span=50).alias("ema50"))
    expected = pd.Series(bars["close"].to_numpy()).ewm(span=50, adjust=False).mean().to_numpy()
    got = out["ema50"].to_numpy()
    assert np.allclose(got[49:], expected[49:])


def test_rsi_bounds(bars):
    out = bars.with_columns(rsi(bars, period=14).alias("rsi"))
    values = out["rsi"].drop_nulls()
    assert values.min() >= 0
    assert values.max() <= 100


def test_atr_positive(bars):
    out = bars.with_columns(atr(bars, period=14).alias("atr"))
    values = out["atr"].drop_nulls()
    assert (values > 0).all()


def test_features_are_causal(bars):
    """Appending future bars must not change past feature values."""
    cutoff = 400
    truncated = bars.head(cutoff)
    specs = [
        FeatureSpec("sma", {"window": 20}),
        FeatureSpec("ema", {"span": 50}),
        FeatureSpec("atr", {"period": 14}),
        FeatureSpec("rsi", {"period": 14}),
        FeatureSpec("zscore", {"window": 20}),
        FeatureSpec("realized_volatility", {"window": 20}),
    ]
    full = compute_features(bars, specs)
    part = compute_features(truncated, specs)
    for spec in specs:
        col = spec.output
        a = full[col].head(cutoff).to_numpy()
        b = part[col].to_numpy()
        mask = ~(np.isnan(a) | np.isnan(b))
        assert np.allclose(a[mask], b[mask]), f"{col} leaks future information"


def test_compute_features_aliases(bars):
    out = compute_features(bars, [FeatureSpec("ema", {"span": 10}, alias="my_ema")])
    assert "my_ema" in out.columns
