"""Command line interface: ``quantlab``.

Examples::

    quantlab data download binance --market futures --symbol BTCUSDT \
        --interval 1m --start 2024-01-01 --end 2024-03-01

    quantlab backtest --strategy volatility_breakout --symbol BTCUSDT \
        --timeframe 1m --start 2024-01-01 --end 2024-03-01
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from ..config import Config, get_config
from ..data.build import build_dataset, make_source
from ..data.catalog import Catalog, catalog_path
from ..data.schemas import DatasetType
from ..data.storage import DatasetKey, read_dataset
from ..data.validate import validate_dataset
from ..features import available_features
from ..reporting import (
    format_summary,
    list_experiments,
    load_result,
    save_experiment,
    save_report,
)
from ..strategies import available_strategies, build_strategy, get_strategy

app = typer.Typer(
    name="quantlab",
    help="Personal crypto quantitative research lab (local, no live trading).",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Download, build, inspect and validate datasets.", no_args_is_help=True)
experiments_app = typer.Typer(help="Inspect saved experiments.", no_args_is_help=True)
features_app = typer.Typer(help="Inspect the feature library.", no_args_is_help=True)
strategies_app = typer.Typer(help="Inspect available strategies.", no_args_is_help=True)
analyze_app = typer.Typer(help="Post-signal path analysis.", no_args_is_help=True)

app.add_typer(data_app, name="data")
app.add_typer(experiments_app, name="experiments")
app.add_typer(features_app, name="features")
app.add_typer(strategies_app, name="strategies")
app.add_typer(analyze_app, name="analyze")

console = Console()
CONFIG_OPTION = typer.Option(None, "--config", help="Path to a TOML config file.")


def _cfg(config: Path | None) -> Config:
    return get_config(config)


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_sets(items: list[str] | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise typer.BadParameter(f"expected key=value, got {item!r}")
        key, value = item.split("=", 1)
        params[key.strip()] = _coerce(value.strip())
    return params


def _load_bars(
    cfg: Config,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    market: str,
) -> tuple[Any, DatasetKey]:
    key = DatasetKey(
        exchange="binance",
        market_type=market,
        dataset=DatasetType.BARS,
        symbol=symbol,
        timeframe=timeframe,
    )
    df = read_dataset(cfg.paths.data_normalized, key, start=start, end=end)
    if df.height == 0:
        console.print(
            f"[red]No normalized data for {key.dataset_id}.[/red]\n"
            f"Run: quantlab data build binance --market {market} --symbol {symbol} "
            f"--interval {timeframe} --start {start} --end {end}"
        )
        raise typer.Exit(code=1)
    return df, key


# --- data --------------------------------------------------------------------
@data_app.command("download")
def data_download(
    exchange: str = typer.Argument("binance"),
    market: str = typer.Option("futures", "--market", help="futures | futures_cm | spot"),
    symbol: str = typer.Option(..., "--symbol"),
    dataset: str = typer.Option("bars", "--dataset", help="bars|trades|agg_trades|funding|metrics|open_interest|mark_price_bars|index_price_bars|premium_index_bars"),
    interval: str = typer.Option(None, "--interval", help="Required for bar-like datasets, e.g. 1m."),
    start: str = typer.Option(..., "--start", help="YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="YYYY-MM-DD"),
    granularity: str = typer.Option(None, "--granularity", help="monthly | daily"),
    config: Path = CONFIG_OPTION,
) -> None:
    """Download raw bulk archives (resumable, checksum-verified)."""
    cfg = _cfg(config)
    if exchange != "binance":
        console.print(f"[red]Only the Binance adapter is implemented so far (got {exchange}).[/red]")
        raise typer.Exit(code=1)
    try:
        ds = DatasetType(dataset)
    except ValueError:
        console.print(f"[red]Unknown dataset {dataset!r}[/red]")
        raise typer.Exit(code=1) from None
    source = make_source(cfg, market)
    result = source.download(
        market_type=market,
        symbol=symbol,
        dataset=ds,
        interval=interval,
        start=start,
        end=end,
        granularity=granularity,
    )
    console.print(f"[green]{result.summary()}[/green]")


@data_app.command("build")
def data_build(
    exchange: str = typer.Argument("binance"),
    market: str = typer.Option("futures", "--market"),
    symbol: str = typer.Option(..., "--symbol"),
    dataset: str = typer.Option("bars", "--dataset"),
    interval: str = typer.Option(None, "--interval"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    granularity: str = typer.Option(None, "--granularity"),
    no_download: bool = typer.Option(False, "--no-download", help="Normalize existing raw archives only."),
    config: Path = CONFIG_OPTION,
) -> None:
    """Download (unless --no-download) and normalize a dataset to Parquet."""
    cfg = _cfg(config)
    try:
        ds = DatasetType(dataset)
    except ValueError:
        console.print(f"[red]Unknown dataset {dataset!r}[/red]")
        raise typer.Exit(code=1) from None
    result = build_dataset(
        dataset=ds,
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        market_type=market,
        exchange=exchange,
        download=not no_download,
        granularity=granularity,
        config=cfg,
    )
    console.print(f"[green]{result.summary()}[/green]")


@data_app.command("inspect")
def data_inspect(
    exchange: str = typer.Option(None, "--exchange"),
    market: str = typer.Option(None, "--market"),
    symbol: str = typer.Option(None, "--symbol"),
    dataset: str = typer.Option(None, "--dataset"),
    timeframe: str = typer.Option(None, "--timeframe"),
    as_json: bool = typer.Option(False, "--json"),
    config: Path = CONFIG_OPTION,
) -> None:
    """Show what datasets are in the catalog."""
    cfg = _cfg(config)
    catalog = Catalog.load(catalog_path(cfg.paths.data_catalogs))
    entries = catalog.filter(
        exchange=exchange, market_type=market, symbol=symbol, dataset_type=dataset, timeframe=timeframe
    )
    if as_json:
        console.print_json(json.dumps([e.to_dict() for e in entries], default=str))
        return
    if not entries:
        console.print("[yellow]Catalog is empty. Run `quantlab data build ...` first.[/yellow]")
        return
    table = Table(title="Dataset catalog")
    for col in ("exchange", "market_type", "dataset_type", "symbol", "timeframe", "start", "end", "row_count", "source"):
        table.add_column(col)
    for e in entries:
        table.add_row(
            e.exchange, e.market_type, e.dataset_type, e.symbol, e.timeframe,
            e.start[:19], e.end[:19], str(e.row_count), e.source,
        )
    console.print(table)


@data_app.command("validate")
def data_validate(
    symbol: str = typer.Option(..., "--symbol"),
    dataset: str = typer.Option("bars", "--dataset"),
    interval: str = typer.Option(None, "--interval"),
    market: str = typer.Option("futures", "--market"),
    start: str = typer.Option(None, "--start"),
    end: str = typer.Option(None, "--end"),
    out: Path = typer.Option(None, "--out", help="Write the JSON report here."),
    config: Path = CONFIG_OPTION,
) -> None:
    """Validate a normalized dataset and print a data-quality report."""
    cfg = _cfg(config)
    try:
        ds = DatasetType(dataset)
    except ValueError:
        console.print(f"[red]Unknown dataset {dataset!r}[/red]")
        raise typer.Exit(code=1) from None
    key = DatasetKey("binance", market, ds, symbol, interval or "na")
    report = validate_dataset(cfg.paths.data_normalized, key, start=start, end=end)
    color = "green" if report.passed else "red"
    console.print(f"[{color}]{report.format()}[/{color}]")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
        console.print(f"wrote {out}")


# --- backtest ----------------------------------------------------------------
@app.command("backtest")
def backtest(
    strategy: str = typer.Option(..., "--strategy"),
    symbol: str = typer.Option(..., "--symbol"),
    timeframe: str = typer.Option("1m", "--timeframe"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    market: str = typer.Option("futures", "--market"),
    params: list[str] = typer.Option(None, "--set", help="Parameter override key=value (repeatable)."),
    no_save: bool = typer.Option(False, "--no-save", help="Do not persist the experiment."),
    no_report: bool = typer.Option(False, "--no-report", help="Do not write plots/tearsheet."),
    config: Path = CONFIG_OPTION,
) -> None:
    """Run a VectorBT backtest and print a full summary."""
    from ..backtest.engine import run_backtest

    cfg = _cfg(config)
    df, key = _load_bars(cfg, symbol, timeframe, start, end, market)
    try:
        strategy_obj = build_strategy(strategy, **_parse_sets(params))
    except (KeyError, TypeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    result = run_backtest(
        df,
        strategy_obj,
        symbol=symbol,
        timeframe=timeframe,
        dataset_id=key.dataset_id,
        start=start,
        end=end,
    )

    experiment_id = None
    if not no_save:
        record = save_experiment(result, cfg.paths.experiments, root=cfg.root, command="quantlab backtest")
        experiment_id = record.experiment_id
        if not no_report:
            report_dir = cfg.paths.experiments / "reports" / experiment_id
            save_report(result, report_dir, experiment_id=experiment_id)

    console.print(format_summary(result, experiment_id))
    if experiment_id:
        console.print(f"\n[green]experiment id: {experiment_id}[/green]")


# --- optimize ----------------------------------------------------------------
@app.command("optimize")
def optimize(
    strategy: str = typer.Option(..., "--strategy"),
    symbol: str = typer.Option(..., "--symbol"),
    timeframe: str = typer.Option("1m", "--timeframe"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    market: str = typer.Option("futures", "--market"),
    metric: str = typer.Option("sharpe", "--metric"),
    method: str = typer.Option("grid", "--method", help="grid | optuna"),
    grid: list[str] = typer.Option(None, "--grid", help="name=v1,v2,v3 (repeatable)."),
    trials: int = typer.Option(50, "--trials", help="Optuna trials."),
    out: Path = typer.Option(None, "--out", help="Write results parquet here."),
    config: Path = CONFIG_OPTION,
) -> None:
    """Run a parameter sweep (grid) or Optuna search."""
    from ..optimization import best_by, parameter_grid, run_sweep

    cfg = _cfg(config)
    df, key = _load_bars(cfg, symbol, timeframe, start, end, market)
    strategy_cls = get_strategy(strategy)

    if method == "grid":
        overrides = {k: [_coerce(v) for v in vals.split(",")] for k, vals in (i.split("=", 1) for i in grid or [])}
        sweep_grid = parameter_grid(strategy_cls, overrides)
        if not sweep_grid:
            console.print("[red]No grid given and strategy declares no param_grid.[/red]")
            raise typer.Exit(code=1)
        results = run_sweep(
            df, strategy_cls, sweep_grid, symbol=symbol, timeframe=timeframe,
            dataset_id=key.dataset_id, execution=cfg.execution, progress=True,
        )
        dest = out or (cfg.paths.experiments / "results" / f"sweep_{strategy}_{symbol}.parquet")
        dest.parent.mkdir(parents=True, exist_ok=True)
        results.write_parquet(dest)
        console.print(f"[green]{results.height} combinations -> {dest}[/green]")
        if metric in results.columns:
            console.print(f"best by {metric}: {json.dumps(best_by(results, metric), default=str)}")
        return

    if method == "optuna":
        from ..optimization.optuna_opt import optimize as optuna_optimize

        space: dict[str, Any] = {}
        for item in grid or []:
            name, spec = item.split("=", 1)
            if spec.startswith(("int:", "float:")):
                kind, lo, hi = spec.split(":")
                space[name] = (kind, float(lo), float(hi))
            else:
                space[name] = [_coerce(v) for v in spec.split(",")]
        if not space:
            console.print("[red]--method optuna requires at least one --grid search space.[/red]")
            raise typer.Exit(code=1)
        try:
            opt = optuna_optimize(
                df, strategy_cls, space, metric=metric, n_trials=trials,
                symbol=symbol, timeframe=timeframe, dataset_id=key.dataset_id,
                execution=cfg.execution,
            )
        except ImportError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        console.print(f"[green]best {metric}={opt.best_value:.4f} params={opt.best_params}[/green]")
        if out:
            opt.to_polars().write_parquet(out)
        return

    console.print(f"[red]Unknown method {method!r}[/red]")
    raise typer.Exit(code=1)


# --- report ------------------------------------------------------------------
@app.command("report")
def report(
    experiment_id: str = typer.Argument(..., help="Experiment id, or 'latest'."),
    no_plots: bool = typer.Option(False, "--no-plots"),
    no_tearsheet: bool = typer.Option(False, "--no-tearsheet"),
    config: Path = CONFIG_OPTION,
) -> None:
    """Regenerate the report for a saved experiment."""
    cfg = _cfg(config)
    if experiment_id == "latest":
        records = list_experiments(cfg.paths.experiments)
        if not records:
            console.print("[yellow]No experiments saved yet.[/yellow]")
            raise typer.Exit(code=1)
        experiment_id = sorted(records, key=lambda r: r.created_at)[-1].experiment_id

    try:
        result = load_result(cfg.paths.experiments, experiment_id)
    except FileNotFoundError:
        console.print(f"[red]Unknown experiment {experiment_id!r}[/red]")
        raise typer.Exit(code=1) from None

    out_dir = cfg.paths.experiments / "reports" / experiment_id
    paths = save_report(
        result, out_dir, experiment_id=experiment_id,
        make_plots=not no_plots, make_tearsheet=not no_tearsheet,
    )
    console.print(format_summary(result, experiment_id))
    console.print(f"\n[green]report: {paths.summary_md}[/green]")
    for name, path in paths.plots.items():
        console.print(f"  plot {name}: {path}")
    if paths.tearsheet:
        console.print(f"  tearsheet: {paths.tearsheet}")


# --- dashboard ---------------------------------------------------------------
@app.command("dashboard")
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (localhost only by default)."),
    port: int = typer.Option(8501, "--port"),
    config: Path = CONFIG_OPTION,
) -> None:
    """Launch the optional read-only Streamlit experiment viewer.

    Requires the optional extra: ``uv sync --extra dashboard``. The viewer is
    local-only, performs no writes, and never talks to an exchange.
    """
    import subprocess
    import sys

    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        console.print(
            "[red]Streamlit is not installed. Install it with "
            "`uv sync --extra dashboard`.[/red]"
        )
        raise typer.Exit(code=1) from exc

    app_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    env = dict(os.environ)
    if config is not None:
        env["QUANTLAB_CONFIG"] = str(config)
    console.print(f"[green]Launching read-only viewer at http://{host}:{port}[/green]")
    code = subprocess.call(
        [
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.address", host, "--server.port", str(port),
            "--server.headless", "true",
        ],
        env=env,
    )
    raise typer.Exit(code=code)


# --- experiments -------------------------------------------------------------
@experiments_app.command("list")
def experiments_list(
    as_json: bool = typer.Option(False, "--json"),
    config: Path = CONFIG_OPTION,
) -> None:
    """List saved experiments."""
    cfg = _cfg(config)
    records = list_experiments(cfg.paths.experiments)
    if as_json:
        console.print_json(json.dumps([r.to_dict() for r in records], default=str))
        return
    if not records:
        console.print("[yellow]No experiments saved yet.[/yellow]")
        return
    table = Table(title="Experiments")
    for col in ("experiment_id", "created_at", "strategy", "symbol", "timeframe", "total_return", "sharpe", "max_drawdown"):
        table.add_column(col)
    for r in sorted(records, key=lambda x: x.created_at):
        m = r.metrics or {}
        table.add_row(
            r.experiment_id, r.created_at[:19], r.strategy, r.symbol, r.timeframe,
            _fmt(m.get("total_return")), _fmt(m.get("sharpe")), _fmt(m.get("max_drawdown")),
        )
    console.print(table)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


# --- analyze -----------------------------------------------------------------
@analyze_app.command("excursions")
def analyze_excursions_cmd(
    strategy: str = typer.Option("bollinger_breakout", "--strategy"),
    symbol: str = typer.Option(..., "--symbol"),
    timeframe: str = typer.Option("1m", "--timeframe"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    market: str = typer.Option("futures", "--market"),
    mode: str = typer.Option("pct_excursion", "--mode", help="pct_signal | pct_excursion | atr"),
    targets: str = typer.Option("0.5,0.9", "--targets", help="Comma-separated fractions."),
    reversion_only: bool = typer.Option(False, "--reversion-only", help="Only measure time to reversion."),
    horizon_days: float = typer.Option(7.0, "--horizon-days", help="<=0 means to end of data."),
    side: str = typer.Option("long", "--side", help="long | short"),
    reversion_basis: str = typer.Option("close", "--reversion-basis", help="close | touch"),
    params: list[str] = typer.Option(None, "--set", help="Strategy parameter key=value (repeatable)."),
    out: Path = typer.Option(None, "--out", help="Directory for events.parquet + summary.parquet."),
    config: Path = CONFIG_OPTION,
) -> None:
    """Measure time-to-milestone and time-to-reversion after each signal."""
    from ..analytics import analyze_excursions
    from ..features import FeatureSpec, compute_features
    from ..timeframes import timeframe_seconds

    cfg = _cfg(config)
    df, key = _load_bars(cfg, symbol, timeframe, start, end, market)
    strategy_obj = build_strategy(strategy, **_parse_sets(params))
    signals = strategy_obj.prepare(df)

    if mode == "atr" and "atr" not in signals.columns:
        signals = compute_features(signals, [FeatureSpec("atr", {"period": 14}, alias="atr")])

    target_list = tuple(_coerce(t) for t in targets.split(",") if t.strip()) if not reversion_only else ()
    horizon = None
    if horizon_days and horizon_days > 0:
        horizon = int(horizon_days * 86400 / timeframe_seconds(timeframe))

    result = analyze_excursions(
        signals,
        signal_col="entries" if side == "long" else "short_entries",
        mode=mode,
        targets=target_list,
        horizon_bars=horizon,
        side=side,
        reversion_basis=reversion_basis,
    )
    console.print(result.format())
    if not result.excursion_by_bucket.is_empty():
        console.print("\nExcursion before reversion, by reversion-speed bucket:")
        console.print(result.excursion_by_bucket.to_pandas().to_string(index=False))

    if out:
        out.mkdir(parents=True, exist_ok=True)
        if not result.events.is_empty():
            result.events.write_parquet(out / "events.parquet")
            result.events.write_csv(out / "events.csv")
        if not result.summary.is_empty():
            result.summary.write_parquet(out / "summary.parquet")
        if not result.reversion_cdf.is_empty():
            result.reversion_cdf.write_parquet(out / "reversion_cdf.parquet")
        if not result.excursion_by_bucket.is_empty():
            result.excursion_by_bucket.write_parquet(out / "excursion_by_bucket.parquet")
        (out / "summary.md").write_text(result.to_markdown(), encoding="utf-8")
        console.print(f"[green]wrote {out}/ (events.parquet, summary.parquet, summary.md)[/green]")


# --- confluence evaluation ---------------------------------------------------
def _print_table(df, title: str) -> None:
    console.print(f"\n[bold]{title}[/bold]")
    if df.is_empty():
        console.print("  (empty)")
        return
    console.print(df.to_pandas().to_string(index=False))


@analyze_app.command("confluence")
def analyze_confluence_cmd(
    strategy: str = typer.Option("bollinger_breakout", "--strategy"),
    symbol: str = typer.Option(..., "--symbol"),
    timeframe: str = typer.Option("1m", "--timeframe"),
    source_timeframe: str = typer.Option("1m", "--source-timeframe", help="Native bar timeframe to resample from."),
    market: str = typer.Option("futures", "--market"),
    start: str = typer.Option(..., "--start", help="Design period start."),
    end: str = typer.Option(..., "--end", help="Design period end."),
    validate_start: str = typer.Option(..., "--validate-start"),
    validate_end: str = typer.Option(..., "--validate-end"),
    excursion_pct: float = typer.Option(0.0025, "--excursion-pct"),
    min_reversion_hours: float = typer.Option(1.0, "--min-reversion-hours"),
    horizon_days: float = typer.Option(7.0, "--horizon-days"),
    reversion_basis: str = typer.Option("close", "--reversion-basis", help="close | touch"),
    depth: int = typer.Option(3, "--depth", help="Max filters in a greedy combination."),
    min_signals: int = typer.Option(30, "--min-signals"),
    by_regime: bool = typer.Option(False, "--by-regime", help="Also evaluate per market regime."),
    regime_timeframe: str = typer.Option("1h", "--regime-timeframe"),
    regime_trend_window: int = typer.Option(24, "--regime-trend-window"),
    regime_vol_window: int = typer.Option(24, "--regime-vol-window"),
    regime_trend_threshold: float = typer.Option(1.0, "--regime-trend-threshold"),
    params: list[str] = typer.Option(None, "--set", help="Strategy parameter key=value (repeatable)."),
    out: Path = typer.Option(None, "--out"),
    config: Path = CONFIG_OPTION,
) -> None:
    """Evaluate confluence filters on a design period and validate out-of-sample."""
    from ..analytics import LabelConfig, RegimeConfig, run_confluence
    from ..timeframes import timeframe_seconds

    cfg = _cfg(config)
    strategy_obj = build_strategy(strategy, **_parse_sets(params))
    horizon = int(horizon_days * 86400 / timeframe_seconds(timeframe)) if horizon_days > 0 else None
    label = LabelConfig(
        excursion_pct_threshold=excursion_pct,
        min_reversion_seconds=min_reversion_hours * 3600.0,
        horizon_bars=horizon,
        reversion_basis=reversion_basis,
    )
    regime_config = (
        RegimeConfig(
            regime_timeframe=regime_timeframe,
            trend_window=regime_trend_window,
            vol_window=regime_vol_window,
            trend_threshold=regime_trend_threshold,
        )
        if by_regime
        else None
    )

    result = run_confluence(
        cfg,
        strategy_obj,
        symbol=symbol,
        timeframe=timeframe,
        market_type=market,
        design_start=start,
        design_end=end,
        validate_start=validate_start,
        validate_end=validate_end,
        label=label,
        max_depth=depth,
        min_signals=min_signals,
        source_timeframe=source_timeframe,
        regime_config=regime_config,
    )

    console.print(
        f"[bold]label[/bold]: good = excursion >= {excursion_pct:.4%} AND "
        f"reversion > {min_reversion_hours:g}h (horizon {horizon_days:g}d, "
        f"{reversion_basis} basis)"
    )
    _print_table(result["design_table"], f"Design {start} -> {end}")
    _print_table(result["validate_table"], f"Validation {validate_start} -> {validate_end}")
    if result["combos"]:
        _print_table(result["combo_design"], "Greedy combos (design)")
        _print_table(result["combo_validate"], "Greedy combos (validation)")
    if by_regime:
        _print_table(result["regime_design"], "By regime (design)")
        _print_table(result["regime_validate"], "By regime (validation)")

    if out:
        out.mkdir(parents=True, exist_ok=True)
        for name in (
            "design_table", "validate_table", "combo_design", "combo_validate",
            "regime_design", "regime_validate",
        ):
            df = result[name]
            if not df.is_empty():
                df.write_parquet(out / f"{name}.parquet")
        lines = [
            "# Confluence filter evaluation",
            "",
            f"- design: {start} -> {end}",
            f"- validation: {validate_start} -> {validate_end}",
            f"- label: excursion >= {excursion_pct} and reversion > {min_reversion_hours}h "
            f"({reversion_basis} basis, horizon {horizon_days}d)",
            "",
            "## Design",
            "",
            result["design_table"].to_pandas().to_markdown(index=False)
            if not result["design_table"].is_empty() else "_empty_",
            "",
            "## Validation",
            "",
            result["validate_table"].to_pandas().to_markdown(index=False)
            if not result["validate_table"].is_empty() else "_empty_",
            "",
        ]
        if result["combos"]:
            lines += [
                "## Greedy combinations (validation)",
                "",
                result["combo_validate"].to_pandas().to_markdown(index=False),
                "",
            ]
        if by_regime and not result["regime_validate"].is_empty():
            lines += [
                "## By regime (validation)",
                "",
                result["regime_validate"].to_pandas().to_markdown(index=False),
                "",
            ]
        (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]wrote {out}/[/green]")


# --- regime-gated backtest ---------------------------------------------------
@analyze_app.command("regime-backtest")
def analyze_regime_backtest_cmd(
    strategy: str = typer.Option("bollinger_breakout", "--strategy"),
    symbol: str = typer.Option(..., "--symbol"),
    timeframe: str = typer.Option("15m", "--timeframe"),
    source_timeframe: str = typer.Option("1m", "--source-timeframe"),
    market: str = typer.Option("futures", "--market"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    allow_regimes: str = typer.Option(
        "up_highvol,up_lowvol", "--allow-regimes", help="Comma-separated regimes to trade in."
    ),
    regime_timeframe: str = typer.Option("1h", "--regime-timeframe"),
    regime_trend_window: int = typer.Option(24, "--regime-trend-window"),
    regime_vol_window: int = typer.Option(24, "--regime-vol-window"),
    regime_trend_threshold: float = typer.Option(1.0, "--regime-trend-threshold"),
    params: list[str] = typer.Option(None, "--set", help="Strategy parameter key=value (repeatable)."),
    out: Path = typer.Option(None, "--out"),
    config: Path = CONFIG_OPTION,
) -> None:
    """Backtest a strategy gated to selected market regimes vs ungated."""
    from ..analytics import RegimeConfig, run_regime_backtest
    from ..reporting import format_summary

    cfg = _cfg(config)
    strategy_obj = build_strategy(strategy, **_parse_sets(params))
    allowed = [r.strip() for r in allow_regimes.split(",") if r.strip()]
    regime_config = RegimeConfig(
        regime_timeframe=regime_timeframe,
        trend_window=regime_trend_window,
        vol_window=regime_vol_window,
        trend_threshold=regime_trend_threshold,
    )
    result = run_regime_backtest(
        cfg, strategy_obj, symbol=symbol, timeframe=timeframe, market_type=market,
        start=start, end=end, allowed_regimes=allowed, regime_config=regime_config,
    )
    console.print("\n[bold]Signals by regime[/bold]")
    console.print(result.regime_counts.to_pandas().to_string(index=False))
    console.print(f"\nallowed regimes: {allowed}")
    console.print("\n[bold]UNGATED[/bold]")
    console.print(format_summary(result.ungated))
    console.print("\n[bold]GATED[/bold]")
    console.print(format_summary(result.gated))

    if out:
        out.mkdir(parents=True, exist_ok=True)
        result.regime_counts.write_parquet(out / "regime_counts.parquet")
        import json

        (out / "metrics.json").write_text(
            json.dumps(
                {"ungated": result.ungated.metrics, "gated": result.gated.metrics,
                 "allowed_regimes": allowed, "config": result.config},
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
        console.print(f"[green]wrote {out}/[/green]")


# --- features / strategies ---------------------------------------------------
@features_app.command("list")
def features_list() -> None:
    """List registered features."""
    for name in available_features():
        console.print(name)


@strategies_app.command("list")
def strategies_list() -> None:
    """List registered strategies and their parameters."""
    for name in available_strategies():
        cls = get_strategy(name)
        console.print(f"[bold]{name}[/bold]  params={cls.default_params}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
