"""Timeframe parsing and annualization helpers."""

from __future__ import annotations

import re

_UNITS = {
    "s": 1,
    "m": 60,
    "min": 60,
    "h": 3600,
    "d": 86400,
    "w": 7 * 86400,
}

_PANDAS_UNITS = {
    "s": "s",
    "m": "min",
    "h": "h",
    "d": "D",
    "w": "W",
}

_SECONDS_PER_YEAR = 365.0 * 24 * 3600


def timeframe_seconds(timeframe: str) -> int:
    match = re.fullmatch(r"(\d+)\s*([a-zA-Z]+)", timeframe.strip())
    if not match:
        raise ValueError(f"cannot parse timeframe {timeframe!r}")
    qty, unit = int(match.group(1)), match.group(2).lower()
    unit = {"min": "m", "mins": "m", "hour": "h", "hours": "h", "day": "d", "days": "d"}.get(unit, unit)
    if unit not in _UNITS:
        raise ValueError(f"unknown timeframe unit {unit!r}")
    return qty * _UNITS[unit]


def periods_per_year(timeframe: str) -> float:
    return _SECONDS_PER_YEAR / timeframe_seconds(timeframe)


def pandas_freq(timeframe: str) -> str:
    match = re.fullmatch(r"(\d+)\s*([a-zA-Z]+)", timeframe.strip())
    if not match:
        raise ValueError(f"cannot parse timeframe {timeframe!r}")
    qty, unit = match.group(1), match.group(2).lower()
    unit = {"min": "m", "mins": "m", "hour": "h", "hours": "h", "day": "d", "days": "d"}.get(unit, unit)
    if unit not in _PANDAS_UNITS:
        raise ValueError(f"unknown timeframe unit {unit!r}")
    return f"{qty}{_PANDAS_UNITS[unit]}"


def infer_timeframe(index) -> str | None:
    """Infer a timeframe from a pandas DatetimeIndex of bar timestamps."""
    import pandas as pd

    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return None
    delta = index.to_series().diff().dropna().median()
    seconds = int(delta.total_seconds())
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"
