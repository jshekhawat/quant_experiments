"""Streamlit app: a read-only viewer over saved experiments.

Run via ``quantlab dashboard`` (preferred) or ``streamlit run`` directly.
Nothing here writes files or places orders.
"""

from __future__ import annotations

import streamlit as st

from quant_lab.config import get_config
from quant_lab.dashboard.data import (
    assumptions_table,
    catalog_table,
    list_summaries,
    load_bundle,
    metrics_table,
)

METRIC_ROWS = [
    ("total_return", "Total return"),
    ("cagr", "CAGR"),
    ("sharpe", "Sharpe"),
    ("sortino", "Sortino"),
    ("max_drawdown", "Max drawdown"),
    ("win_rate", "Win rate"),
    ("profit_factor", "Profit factor"),
    ("trade_count", "Trades"),
]


def _fmt_pct(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_num(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def render() -> None:
    st.set_page_config(page_title="quant-lab", layout="wide")
    cfg = get_config()

    st.title("quant-lab")
    st.caption(
        "Read-only local viewer for saved experiments. No live data, no writes, "
        "no trading. Reports live in `experiments/reports/<id>/`."
    )

    summaries = list_summaries(cfg.paths.experiments)

    with st.sidebar:
        st.header("Experiments")
        if summaries.is_empty():
            st.info(
                "No experiments yet. Run:\n\n"
                "```\nquantlab backtest --strategy volatility_breakout \\\n"
                "    --symbol BTCUSDT --timeframe 1m \\\n"
                "    --start 2024-01-01 --end 2024-03-01\n```"
            )
            st.stop()
        ordered = summaries.sort("created_at", descending=True)
        options = ordered["experiment_id"].to_list()
        selected = st.selectbox("Select an experiment", options)
        st.divider()
        st.subheader("Dataset catalog")
        catalog = catalog_table(cfg)
        if catalog.is_empty():
            st.write("empty")
        else:
            st.dataframe(catalog.to_pandas(), use_container_width=True, hide_index=True)

    bundle = load_bundle(cfg.paths.experiments, selected)
    record = bundle.record

    st.subheader(f"{record.strategy} — {record.symbol} {record.timeframe}")
    st.caption(
        f"experiment `{record.experiment_id}` · created {record.created_at[:19]} · "
        f"dataset `{record.dataset_id}` · {record.data_range.get('start', '?')} → "
        f"{record.data_range.get('end', '?')}"
    )

    cols = st.columns(4)
    for i, (key, label) in enumerate(METRIC_ROWS[:4]):
        cols[i].metric(label, _fmt_pct(record.metrics.get(key)) if "return" in key or "drawdown" in key or "rate" in key else _fmt_num(record.metrics.get(key)))
    cols2 = st.columns(4)
    for i, (key, label) in enumerate(METRIC_ROWS[4:]):
        value = record.metrics.get(key)
        if key == "trade_count":
            cols2[i].metric(label, int(value) if value is not None else "n/a")
        elif key in {"win_rate", "max_drawdown"}:
            cols2[i].metric(label, _fmt_pct(value))
        else:
            cols2[i].metric(label, _fmt_num(value))

    tab_metrics, tab_equity, tab_trades, tab_assumptions, tab_report = st.tabs(
        ["Metrics", "Equity & drawdown", "Trades", "Execution assumptions", "Report"]
    )

    with tab_metrics:
        st.dataframe(metrics_table(record).to_pandas(), use_container_width=True, hide_index=True)
        if record.params:
            st.markdown("**Parameters**")
            st.json(record.params)

    with tab_equity:
        if bundle.equity.empty:
            st.info("No equity curve stored for this experiment.")
        else:
            eq = bundle.equity.set_index("timestamp")
            st.markdown("**Equity**")
            st.line_chart(eq[["equity"]], height=280)
            if "drawdown" in eq.columns:
                st.markdown("**Drawdown**")
                st.area_chart(eq[["drawdown"]], height=220)

    with tab_trades:
        if bundle.trades.empty:
            st.info("No trades recorded.")
        else:
            st.caption(f"{len(bundle.trades)} trades (showing up to 1000)")
            st.dataframe(bundle.trades.head(1000), use_container_width=True, hide_index=True)

    with tab_assumptions:
        st.dataframe(
            assumptions_table(record).to_pandas(), use_container_width=True, hide_index=True
        )
        if record.git_commit:
            st.caption(f"git commit: {record.git_commit}")
        st.caption(f"random seed: {record.random_seed}")

    with tab_report:
        if bundle.tearsheet is not None:
            st.download_button(
                "Download QuantStats tearsheet (HTML)",
                data=bundle.tearsheet.read_bytes(),
                file_name=f"{record.experiment_id}-tearsheet.html",
                mime="text/html",
            )
            with st.expander("Preview tearsheet"):
                st.components.v1.html(bundle.tearsheet.read_text(encoding="utf-8"), height=900, scrolling=True)
        else:
            st.info("No tearsheet available (QuantStats may have been unavailable or too few bars).")
        summary_md = bundle.report_dir / "summary.md"
        if summary_md.exists():
            with st.expander("summary.md"):
                st.markdown(summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    render()
