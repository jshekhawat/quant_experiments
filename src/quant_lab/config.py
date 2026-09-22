"""Configuration loading.

The lab keeps configuration in a single TOML file (``configs/default.toml`` by
default). Nothing here is required to run: every value has a sane default so
that the package works even without a config file present.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Best-effort repository root.

    ``QUANTLAB_ROOT`` wins if set, otherwise we walk up from this file looking
    for ``pyproject.toml`` and fall back to the current working directory.
    """
    env = os.environ.get("QUANTLAB_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


@dataclass(slots=True)
class Paths:
    data_raw: Path
    data_normalized: Path
    data_catalogs: Path
    experiments: Path

    @classmethod
    def from_dict(cls, d: dict[str, Any], root: Path) -> Paths:
        def p(key: str, default: str) -> Path:
            value = Path(d.get(key, default)).expanduser()
            return value if value.is_absolute() else (root / value)

        return cls(
            data_raw=p("data_raw", "data/raw"),
            data_normalized=p("data_normalized", "data/normalized"),
            data_catalogs=p("data_catalogs", "data/catalogs"),
            experiments=p("experiments", "experiments"),
        )

    def ensure(self) -> None:
        for path in (self.data_raw, self.data_normalized, self.data_catalogs, self.experiments):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class DataConfig:
    exchange: str = "binance"
    market: str = "futures"
    base_url: str = "https://data.binance.vision"
    request_timeout: int = 120
    verify_checksums: bool = True
    prefer_monthly: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataConfig:
        return cls(**{k: d[k] for k in cls.__slots__ if k in d})  # type: ignore[arg-type]


@dataclass(slots=True)
class ExecutionConfig:
    """Explicit execution assumptions recorded with every backtest."""

    initial_capital: float = 100_000.0
    fees: float = 0.0004
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    slippage: float = 0.0002
    position_size: float = 1.0
    leverage: float = 1.0
    max_position_size: float = 1.0
    funding: float = 0.0
    execution_delay_bars: int = 1
    allow_short: bool = True
    intrabar_policy: str = "worst_case"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecutionConfig:
        return cls(**{k: d[k] for k in cls.__slots__ if k in d})  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}

    @property
    def total_cost_per_side(self) -> float:
        """Convenience: fee + slippage applied per side."""
        return self.fees + self.slippage


@dataclass(slots=True)
class BacktestConfig:
    annualization: float = 365.0
    initial_capital: float = 100_000.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BacktestConfig:
        return cls(**{k: d[k] for k in cls.__slots__ if k in d})  # type: ignore[arg-type]


@dataclass(slots=True)
class Config:
    root: Path
    paths: Paths
    data: DataConfig = field(default_factory=DataConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> Config:
        root = project_root()
        if path is None:
            path = os.environ.get("QUANTLAB_CONFIG")
        if path is None:
            path = root / "configs" / "default.toml"
        path = Path(path)
        raw: dict[str, Any] = {}
        if path.exists():
            with path.open("rb") as fh:
                raw = tomllib.load(fh)
        return cls(
            root=root,
            paths=Paths.from_dict(raw.get("paths", {}), root),
            data=DataConfig.from_dict(raw.get("data", {})),
            execution=ExecutionConfig.from_dict(raw.get("execution", {})),
            backtest=BacktestConfig.from_dict(raw.get("backtest", {})),
        )


_DEFAULT: Config | None = None


def get_config(path: Path | str | None = None, *, reload: bool = False) -> Config:
    """Return a cached default config (or load an explicit one)."""
    global _DEFAULT
    if path is not None:
        return Config.load(path)
    if _DEFAULT is None or reload:
        _DEFAULT = Config.load()
    return _DEFAULT
