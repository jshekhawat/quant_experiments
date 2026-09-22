"""Explicit execution assumptions.

Every backtest states, in machine- and human-readable form:

initial capital, fees, maker/taker fees, slippage, position sizing, leverage,
maximum position size, funding, execution delay and intrabar policy.

Defaults are non-zero for costs (fees + slippage) so a backtest never silently
assumes free, frictionless trading.
"""

from __future__ import annotations

from typing import Any

from ..config import ExecutionConfig

# Human descriptions, surfaced in every report.
FIELD_HELP: dict[str, str] = {
    "initial_capital": "Starting equity in quote currency.",
    "fees": "Flat per-side fee applied to market orders (fraction).",
    "maker_fee": "Maker fee (fraction); recorded, used when explicitly requested.",
    "taker_fee": "Taker fee (fraction); market orders pay this when configured.",
    "slippage": "Per-side price slippage (fraction).",
    "position_size": "Fraction of equity allocated per entry.",
    "leverage": "Notional exposure multiplier.",
    "max_position_size": "Hard cap on position fraction.",
    "funding": "Per-interval funding drag applied to open notional.",
    "execution_delay_bars": "Bars between signal and execution (>=1 for realism).",
    "allow_short": "Whether short signals are simulated.",
    "intrabar_policy": "Policy when a bar touches both stop and target.",
}

VALID_INTRABAR = {"worst_case", "best_case", "stop_first", "target_first"}


def validate_assumptions(cfg: ExecutionConfig) -> None:
    if cfg.initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    for name in ("fees", "maker_fee", "taker_fee", "slippage"):
        value = getattr(cfg, name)
        if value < 0 or value >= 1:
            raise ValueError(f"{name} must be in [0, 1), got {value}")
    if cfg.leverage < 1:
        raise ValueError("leverage must be >= 1")
    if cfg.position_size <= 0:
        raise ValueError("position_size must be positive")
    if cfg.execution_delay_bars < 0:
        raise ValueError("execution_delay_bars must be >= 0")
    if cfg.intrabar_policy not in VALID_INTRABAR:
        raise ValueError(f"intrabar_policy must be one of {sorted(VALID_INTRABAR)}")


def effective_size_fraction(cfg: ExecutionConfig, overrides: dict[str, Any] | None = None) -> float:
    value = cfg.position_size
    if overrides and overrides.get("position_size") is not None:
        value = float(overrides["position_size"])
    levered = value * cfg.leverage
    return max(0.0, min(levered, cfg.max_position_size))


def describe(cfg: ExecutionConfig) -> dict[str, Any]:
    data = cfg.to_dict()
    return {k: {"value": v, "help": FIELD_HELP.get(k, "")} for k, v in data.items()}


def format_assumptions(cfg: ExecutionConfig) -> str:
    width = max(len(k) for k in FIELD_HELP)
    lines = ["Execution assumptions:"]
    for key, value in cfg.to_dict().items():
        lines.append(f"  {key:<{width}} : {value}    # {FIELD_HELP.get(key, '')}")
    return "\n".join(lines)
