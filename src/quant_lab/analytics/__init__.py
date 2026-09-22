"""Analytics: chronological splits and statistical robustness."""

from .confluence import (
    Filter,
    LabelConfig,
    add_confluence_features,
    combine_filters,
    default_filters,
    enrich_frame,
    evaluate_confluence,
    greedy_combinations,
    label_signals,
    prepare_labelled,
    run_confluence,
)
from .excursions import ExcursionResult, analyze_excursions, excursion_by_reversion_bucket
from .gating import RegimeBacktest, run_regime_backtest
from .regimes import (
    RegimeConfig,
    attach_regimes,
    classify_regimes,
    evaluate_by_regime,
    gate_signals,
)
from .robustness import (
    Distribution,
    RobustnessReport,
    bootstrap_returns,
    drawdown_distribution,
    metrics_from_returns,
    monte_carlo_trades,
    parameter_sensitivity,
    return_distribution,
)
from .splits import (
    Split,
    chronological_split,
    split_by_dates,
    split_summary,
    walk_forward_splits,
)

__all__ = [
    "Split",
    "chronological_split",
    "split_by_dates",
    "walk_forward_splits",
    "split_summary",
    "ExcursionResult",
    "analyze_excursions",
    "excursion_by_reversion_bucket",
    "Filter",
    "LabelConfig",
    "enrich_frame",
    "add_confluence_features",
    "label_signals",
    "evaluate_confluence",
    "combine_filters",
    "greedy_combinations",
    "default_filters",
    "prepare_labelled",
    "run_confluence",
    "RegimeConfig",
    "classify_regimes",
    "attach_regimes",
    "evaluate_by_regime",
    "gate_signals",
    "RegimeBacktest",
    "run_regime_backtest",
    "RobustnessReport",
    "Distribution",
    "bootstrap_returns",
    "monte_carlo_trades",
    "parameter_sensitivity",
    "drawdown_distribution",
    "return_distribution",
    "metrics_from_returns",
]
