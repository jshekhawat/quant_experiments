"""Event-driven execution validation interface (Phase 4 scaffold).

VectorBT is the fast hypothesis-testing engine. NautilusTrader is the intended
secondary engine for realistic, event-driven execution simulation (bar/quote/
trade data, order book, multiple instruments/venues, fees, slippage, latency).

This module defines a *clean interface* between the two and translates the
lab's signal frames into an explicit, inspectable order schedule. The
NautilusTrader adapter is optional and import-guarded: the research lab works
perfectly well without it (``Polars + DuckDB + VectorBT``).

Nothing here is a live-trading connector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import polars as pl

from ..config import ExecutionConfig


@dataclass(slots=True)
class OrderIntent:
    timestamp: str
    symbol: str
    side: str  # "buy" | "sell"
    action: str  # "open" | "close"
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "side": self.side,
            "action": self.action,
            "reason": self.reason,
        }


def signal_frame_to_intents(
    signal_frame: pl.DataFrame,
    *,
    symbol: str,
    execution_delay_bars: int = 1,
) -> list[OrderIntent]:
    """Translate boolean signals into explicit order intents.

    The order timestamp is ``signal bar + execution_delay_bars`` and the
    fill happens at that bar (open for delay >= 1, close for delay 0). This is
    the same timing contract as the VectorBT engine, expressed explicitly.
    """
    sf = signal_frame
    n = sf.height
    ts = sf["timestamp"].to_list()
    intents: list[OrderIntent] = []
    columns = {
        "entries": ("buy", "open"),
        "exits": ("sell", "close"),
        "short_entries": ("sell", "open"),
        "short_exits": ("buy", "close"),
    }
    for col, (side, action) in columns.items():
        if col not in sf.columns:
            continue
        flags = sf[col].to_list()
        for i, flag in enumerate(flags):
            if not flag:
                continue
            j = i + execution_delay_bars
            if j >= n:
                continue
            intents.append(
                OrderIntent(
                    timestamp=str(ts[j]),
                    symbol=symbol,
                    side=side,
                    action=action,
                    reason=col,
                )
            )
    intents.sort(key=lambda x: (x.timestamp, x.reason))
    return intents


def nautilus_available() -> bool:
    try:
        import nautilus_trader  # noqa: F401

        return True
    except ImportError:
        return False


class EventDrivenEngine(ABC):
    """Interface for a realistic execution-simulation engine."""

    name: str = "event"

    @abstractmethod
    def run(
        self,
        data: pl.DataFrame,
        intents: list[OrderIntent],
        *,
        execution: ExecutionConfig,
        symbol: str,
    ) -> dict[str, Any]:
        """Run the simulation and return a summary dictionary."""


class NautilusEngine(EventDrivenEngine):
    """Optional NautilusTrader adapter.

    Install with ``uv sync --extra event``. The adapter is intentionally thin:
    the same :class:`~quant_lab.strategies.base.Strategy` concept should be
    expressible in both engines, but they do not need to share internals.
    """

    name = "nautilus"

    def run(
        self,
        data: pl.DataFrame,
        intents: list[OrderIntent],
        *,
        execution: ExecutionConfig,
        symbol: str,
    ) -> dict[str, Any]:
        if not nautilus_available():
            raise ImportError(
                "NautilusTrader is not installed. Install the optional extra with "
                "`uv sync --extra event`, or use the VectorBT engine for research. "
                "The basic lab does not require NautilusTrader."
            )
        raise NotImplementedError(
            "The NautilusTrader execution adapter is a Phase 4 scaffold. The order "
            "intents are ready (see signal_frame_to_intents); wire them into a "
            "BacktestEngine for your installed Nautilus version. This is "
            "deliberately not a live-trading connector."
        )


def get_event_engine(name: str = "nautilus") -> EventDrivenEngine:
    if name == "nautilus":
        return NautilusEngine()
    raise KeyError(f"unknown event engine {name!r}")
