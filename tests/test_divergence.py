"""Spot–perp CVD divergence: definition, causality, alignment and estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from quant_lab.analytics.divergence import (
    add_forward_returns,
    build_divergence_panel,
    cross_sectional_ic,
    flow_divergence,
    pooled_ic,
)
from quant_lab.data.schemas import DatasetType
from quant_lab.data.storage import DatasetKey, write_partition
from quant_lab.synthetic import synthetic_bars


def _pair_frame(n: int = 900) -> pl.DataFrame:
    """A spot+perp style frame with the columns ``flow_divergence`` expects."""
    spot = synthetic_bars(n, seed=1)
    perp = synthetic_bars(n, seed=2)
    return perp.select(
        "timestamp", "open", "high", "low", "close",
        pl.col("volume").alias("perp_volume"),
        pl.col("taker_buy_volume").alias("perp_taker_buy"),
        pl.col("quote_volume").alias("perp_quote_volume"),
        pl.col("taker_buy_quote_volume").alias("perp_taker_buy_quote"),
    ).with_columns(
        spot["volume"].alias("spot_volume"),
        spot["taker_buy_volume"].alias("spot_taker_buy"),
        spot["quote_volume"].alias("spot_quote_volume"),
        spot["taker_buy_quote_volume"].alias("spot_taker_buy_quote"),
    )


def test_flow_divergence_definition():
    frame = _pair_frame()
    out = flow_divergence(frame, window=10, norm_window=50)
    expected_fs = (2.0 * frame["spot_taker_buy"] - frame["spot_volume"]).rolling_sum(
        window_size=10, min_periods=10
    )
    assert out["Fs"].to_list()[10:] == expected_fs.to_list()[10:]
    # D is exactly zs - zp wherever both are defined.
    both = out.drop_nulls(["zs", "zp"])
    assert (both["D"] - (both["zs"] - both["zp"])).abs().max() < 1e-12


def test_divergence_is_causal():
    frame = _pair_frame(1200)
    full = flow_divergence(frame, window=10, norm_window=50)
    part = flow_divergence(frame.head(800), window=10, norm_window=50)
    a = full["D"].head(800).to_numpy()
    b = part["D"].to_numpy()
    mask = np.isfinite(a) & np.isfinite(b)
    assert mask.sum() > 0
    assert np.allclose(a[mask], b[mask])


def test_forward_return_alignment_and_delay(tmp_path):
    frame = _pair_frame(400)
    out = add_forward_returns(flow_divergence(frame, window=5, norm_window=20), {"4h": 4})
    close = frame["close"].to_numpy()
    # Entry at t+1, exit at t+1+4; the last five rows have no forward return.
    for i in range(len(close) - 5):
        assert abs(out["fwd_4h"][i] - (close[i + 5] / close[i + 1] - 1.0)) < 1e-12
    assert out["fwd_4h"].to_list()[-5:] == [None] * 5


def test_build_divergence_panel_end_to_end(tmp_path):
    root = tmp_path / "normalized"
    symbol = "TESTUSDT"
    bars = synthetic_bars(5000, seed=3, start="2026-01-01T00:00:00")
    write_partition(bars, DatasetKey("binance", "spot", DatasetType.BARS, symbol, "1m"), root, 2026, 1)
    write_partition(bars, DatasetKey("binance", "futures", DatasetType.BARS, symbol, "1m"), root, 2026, 1)
    panel = build_divergence_panel(
        [symbol],
        start="2026-01-01",
        end="2026-02-01",
        timeframe="1h",
        source_timeframe="1m",
        window=5,
        norm_window=20,
        horizons={"1h": 1, "4h": 4},
        root=root,
    )
    assert panel.height > 0
    assert "D" in panel.columns and panel["D"].null_count() < panel.height
    assert "fwd_4h_cs" in panel.columns


def test_pooled_ic_detects_synthetic_edge():
    rng = np.random.default_rng(0)
    d = rng.normal(size=4000)
    df = pd.DataFrame({"D": d, "fwd": d * 1e-3 + rng.normal(scale=1e-3, size=4000)})
    assert pooled_ic(df, "D", "fwd") > 0.5
    df["fwd"] = -df["fwd"]
    assert pooled_ic(df, "D", "fwd") < -0.5


def test_cross_sectional_ic_averages_within_timestamp():
    rows = []
    for t in range(200):
        for s in range(10):
            d = float(s)
            rows.append({"timestamp": t, "D": d, "fwd": d + 0.001 * (t % 3)})
    df = pd.DataFrame(rows)
    assert cross_sectional_ic(df, "D", "fwd") > 0.9
