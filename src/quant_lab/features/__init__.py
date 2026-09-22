"""Reusable feature API.

Importing this package registers all built-in features.

    from quant_lab.features import FeatureSpec, compute_features

    df = compute_features(df, [FeatureSpec("ema", {"span": 100}), "rsi"])
"""

# Importing these modules registers their features as a side effect.
from . import core as _core  # noqa: F401,E402
from . import market as _market  # noqa: F401,E402
from .base import (
    FeatureSpec,
    available_features,
    compute_features,
    feature,
    get_feature,
    parse_specs,
)
from .core import (  # noqa: E402
    atr,
    bollinger_bandwidth,
    bollinger_lower,
    bollinger_mid,
    bollinger_percent_b,
    bollinger_upper,
    ema,
    momentum,
    realized_volatility,
    returns,
    roc,
    rolling_max,
    rolling_min,
    rolling_volume,
    rsi,
    safe_div,
    sma,
    stdev,
    true_range,
    vwap,
    zscore,
)

__all__ = [
    "FeatureSpec",
    "compute_features",
    "parse_specs",
    "available_features",
    "get_feature",
    "feature",
    "sma",
    "ema",
    "atr",
    "rsi",
    "vwap",
    "returns",
    "rolling_volume",
    "realized_volatility",
    "zscore",
    "roc",
    "stdev",
    "rolling_max",
    "rolling_min",
    "momentum",
    "true_range",
    "safe_div",
    "bollinger_mid",
    "bollinger_upper",
    "bollinger_lower",
    "bollinger_bandwidth",
    "bollinger_percent_b",
]
