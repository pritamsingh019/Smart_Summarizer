from pathlib import Path

import pandas as pd
import streamlit as st

from utils.helpers import format_metric, humanize_label, infer_file_type
from utils.logger import get_logger

LOGGER = get_logger("smart_summariser.app")
PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = PROJECT_ROOT / "sample_data" / "example_sales.csv"
STRUCTURED_FILE_TYPES = {"csv", "excel"}
TEXT_FILE_TYPES = {"pdf", "txt"}
DEFAULT_SPACY_MODEL = "en_core_web_sm"

st.set_page_config(page_title="Smart Summariser", layout="centered", page_icon="📊")


# ─────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────

def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', 'Segoe UI', sans-serif;
            }
            .stApp {
                background: #0E1117;
                color: #E6EDF3;
            }

            /* Scrollbar */
            ::-webkit-scrollbar { width: 6px; height: 6px; }
            ::-webkit-scrollbar-track { background: #0E1117; }
            ::-webkit-scrollbar-thumb { background: rgba(0,194,255,0.25); border-radius: 999px; }
            ::-webkit-scrollbar-thumb:hover { background: rgba(0,194,255,0.45); }

            /* Sidebar */
            [data-testid="stSidebar"] {
                background: #0D1119;
                border-right: 1px solid rgba(255,255,255,0.06);
            }

            /* Section header */
            .section-label {
                color: #00C2FF;
                font-size: 0.70rem;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                font-weight: 700;
                margin-bottom: 0.3rem;
            }
            .section-title {
                color: #E6EDF3;
                font-size: 1.15rem;
                font-weight: 700;
                margin-bottom: 0.7rem;
                border-bottom: 1px solid rgba(255,255,255,0.07);
                padding-bottom: 0.45rem;
            }

            /* Summary card row */
            .summary-row {
                display: flex;
                gap: 1rem;
                margin-bottom: 1.5rem;
                flex-wrap: wrap;
            }
            .summary-card {
                flex: 1;
                min-width: 120px;
                background: linear-gradient(160deg, rgba(30,36,50,0.98), rgba(19,24,35,0.97));
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 16px;
                padding: 1.0rem 1.1rem;
                text-align: center;
            }
            .summary-card-label {
                color: #7B8899;
                font-size: 0.72rem;
                letter-spacing: 0.09em;
                text-transform: uppercase;
                font-weight: 600;
                margin-bottom: 0.4rem;
            }
            .summary-card-value {
                color: #E6EDF3;
                font-size: 1.8rem;
                font-weight: 800;
                line-height: 1.1;
                letter-spacing: -0.02em;
            }
            .summary-card-sub {
                color: #8B9AAA;
                font-size: 0.78rem;
                margin-top: 0.3rem;
            }

            /* Structured data summary table */
            .col-summary-table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 1.2rem;
                font-size: 0.88rem;
            }
            .col-summary-table th {
                background: rgba(0,194,255,0.10);
                color: #00C2FF;
                font-size: 0.72rem;
                letter-spacing: 0.10em;
                text-transform: uppercase;
                font-weight: 700;
                padding: 0.55rem 0.9rem;
                text-align: left;
                border-bottom: 1px solid rgba(0,194,255,0.15);
            }
            .col-summary-table td {
                padding: 0.50rem 0.9rem;
                color: #CBD5E1;
                border-bottom: 1px solid rgba(255,255,255,0.05);
                vertical-align: top;
                line-height: 1.55;
            }
            .col-summary-table tr:hover td {
                background: rgba(255,255,255,0.025);
            }
            .col-name {
                color: #E6EDF3;
                font-weight: 600;
            }
            .col-type-badge {
                display: inline-block;
                padding: 0.10rem 0.50rem;
                border-radius: 999px;
                font-size: 0.68rem;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }
            .badge-numeric  { background: rgba(52,211,153,0.12); color: #34D399; border: 1px solid rgba(52,211,153,0.25); }
            .badge-category { background: rgba(0,194,255,0.10);  color: #00C2FF; border: 1px solid rgba(0,194,255,0.20); }
            .badge-datetime { background: rgba(245,158,11,0.10); color: #F59E0B; border: 1px solid rgba(245,158,11,0.25); }

            /* Chart wrapper */
            .chart-wrap {
                background: rgba(26,31,43,0.7);
                border: 1px solid rgba(255,255,255,0.065);
                border-radius: 16px;
                padding: 0.6rem 0.8rem 0.2rem;
                margin-bottom: 1.1rem;
            }

            /* Insight item (legacy) */
            .insight-item {
                background: rgba(0,194,255,0.05);
                border-left: 3px solid #00C2FF;
                border-radius: 0 8px 8px 0;
                padding: 0.65rem 0.95rem;
                margin-bottom: 0.65rem;
                color: #CBD5E1;
                font-size: 0.91rem;
                line-height: 1.68;
            }
            .insight-title {
                color: #E6EDF3;
                font-weight: 700;
                margin-bottom: 0.15rem;
                font-size: 0.82rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }

            /* ── Ranked Insight Cards ── */
            .ri-card {
                background: linear-gradient(160deg, rgba(20,26,38,0.98), rgba(14,19,28,0.97));
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 14px;
                padding: 1.0rem 1.15rem 0.85rem;
                margin-bottom: 0.95rem;
                position: relative;
                transition: border-color 0.2s;
            }
            .ri-card:hover { border-color: rgba(0,194,255,0.25); }
            .ri-header {
                display: flex;
                align-items: flex-start;
                gap: 0.7rem;
                margin-bottom: 0.55rem;
            }
            .ri-rank {
                font-size: 1.35rem;
                line-height: 1;
                flex-shrink: 0;
                margin-top: 0.05rem;
            }
            .ri-title-block { flex: 1; min-width: 0; }
            .ri-title {
                color: #E6EDF3;
                font-size: 0.93rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                margin-bottom: 0.25rem;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .ri-badges { display: flex; gap: 0.4rem; flex-wrap: wrap; }
            .ri-badge {
                display: inline-block;
                padding: 0.08rem 0.55rem;
                border-radius: 999px;
                font-size: 0.64rem;
                font-weight: 800;
                letter-spacing: 0.09em;
                text-transform: uppercase;
            }
            .badge-data    { background: rgba(52,211,153,0.12); color: #34D399; border: 1px solid rgba(52,211,153,0.25); }
            .badge-text    { background: rgba(0,194,255,0.10);  color: #00C2FF; border: 1px solid rgba(0,194,255,0.22); }
            .badge-fusion  { background: rgba(167,139,250,0.12); color: #A78BFA; border: 1px solid rgba(167,139,250,0.25); }
            .badge-anomaly { background: rgba(251,113,133,0.12); color: #FB7185; border: 1px solid rgba(251,113,133,0.25); }
            .ri-score-wrap {
                display: flex;
                align-items: center;
                gap: 0.6rem;
                margin-bottom: 0.6rem;
            }
            .ri-score-bar-bg {
                flex: 1;
                height: 4px;
                background: rgba(255,255,255,0.07);
                border-radius: 999px;
                overflow: hidden;
            }
            .ri-score-bar-fill {
                height: 100%;
                border-radius: 999px;
                background: linear-gradient(90deg, #0099CC, #00C2FF);
                transition: width 0.5s;
            }
            .ri-score-label {
                color: #7B8899;
                font-size: 0.68rem;
                font-weight: 700;
                white-space: nowrap;
            }
            .ri-description {
                color: #B8C4D0;
                font-size: 0.89rem;
                line-height: 1.72;
                margin-bottom: 0.5rem;
            }
            .ri-evidence {
                margin-top: 0.45rem;
                display: flex;
                flex-wrap: wrap;
                gap: 0.35rem;
            }
            .ri-ev-chip {
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.09);
                border-radius: 8px;
                padding: 0.15rem 0.6rem;
                font-size: 0.74rem;
                color: #8B9AAA;
                font-family: 'Courier New', monospace;
            }
            .ri-fusion-matches {
                background: rgba(167,139,250,0.06);
                border: 1px solid rgba(167,139,250,0.15);
                border-radius: 10px;
                padding: 0.65rem 0.85rem;
                margin-top: 0.5rem;
            }
            .ri-match-title {
                color: #A78BFA;
                font-size: 0.71rem;
                font-weight: 700;
                letter-spacing: 0.10em;
                text-transform: uppercase;
                margin-bottom: 0.4rem;
            }
            .ri-match-pill {
                display: inline-block;
                background: rgba(167,139,250,0.10);
                border: 1px solid rgba(167,139,250,0.20);
                border-radius: 999px;
                padding: 0.12rem 0.55rem;
                font-size: 0.73rem;
                color: #C4B5FD;
                margin: 2px;
            }

            /* Text fallback box */
            .text-summary-box {
                background: linear-gradient(160deg, rgba(26,31,43,0.98), rgba(19,24,35,0.97));
                border: 1px solid rgba(0,194,255,0.15);
                border-radius: 16px;
                padding: 1.2rem 1.3rem;
                margin-bottom: 1.2rem;
            }
            .text-summary-title {
                color: #00C2FF;
                font-size: 0.78rem;
                letter-spacing: 0.10em;
                text-transform: uppercase;
                font-weight: 700;
                margin-bottom: 0.55rem;
            }
            .text-summary-body {
                color: #CED8E2;
                font-size: 0.92rem;
                line-height: 1.72;
            }

            /* Empty state */
            .empty-state {
                background: rgba(26,31,43,0.6);
                border: 1px dashed rgba(255,255,255,0.10);
                border-radius: 16px;
                padding: 2.5rem 1.5rem;
                color: #6B7785;
                text-align: center;
                font-size: 0.92rem;
                line-height: 1.65;
            }
            .empty-state-icon {
                font-size: 2.5rem;
                margin-bottom: 0.7rem;
                opacity: 0.55;
            }

            /* Overrides */
            div[data-testid="stMetric"] {
                background: rgba(26,31,43,0.7);
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 14px;
                padding: 0.75rem 1rem;
            }
            .stDataFrame { border-radius: 12px; overflow: hidden; }
            .stExpander { border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 12px !important; }
            button[kind="primary"], .stButton>button {
                background: linear-gradient(90deg, #0099CC, #00C2FF);
                border: none;
                color: #fff;
                font-weight: 600;
                border-radius: 10px;
            }
            button[kind="primary"]:hover, .stButton>button:hover {
                opacity: 0.88;
                transform: translateY(-1px);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _section(label: str, title: str) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def _empty(icon: str, message: str) -> None:
    st.markdown(
        f'<div class="empty-state"><div class="empty-state-icon">{icon}</div>{message}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# NLP / CACHE
# ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def prepare_nlp_resources():
    try:
        from utils.bootstrap import bootstrap_nlp, load_spacy_model
        status = bootstrap_nlp(logger=LOGGER, download_missing=True)
        model_name = status.get("spacy", {}).get("model", DEFAULT_SPACY_MODEL)
        nlp = load_spacy_model(model_name, logger=LOGGER) if status.get("spacy", {}).get("available") else None
        return status, nlp
    except Exception as exc:
        LOGGER.warning("NLP bootstrap failed: %s", exc)
        return {}, None


@st.cache_data(show_spinner=False)
def run_cached_analysis(file_payloads: tuple, insight_depth: int, selected_columns: tuple, enable_nlp: bool):
    try:
        from core.orchestrator import SmartSummariserOrchestrator
        nlp = None
        if enable_nlp:
            _status, nlp = prepare_nlp_resources()
        orchestrator = SmartSummariserOrchestrator(nlp=nlp, logger=LOGGER)
        return orchestrator.process_file_payloads(
            file_payloads,
            insight_depth=insight_depth,
            selected_columns=list(selected_columns),
        )
    except Exception as exc:
        LOGGER.warning("Analysis failed: %s", exc)
        return {"data": None, "text": None, "fusion": None, "warnings": [], "errors": [str(exc)], "debug": {}}


@st.cache_data(show_spinner=False)
def preview_structured_columns(file_payloads: tuple) -> list[str]:
    try:
        from modules.data_loader import load_file_payload
        for file_name, _mime, raw_bytes in file_payloads:
            loaded = load_file_payload(file_name, raw_bytes, logger=LOGGER)
            if loaded.get("dataframe") is not None and not loaded.get("error"):
                return loaded["dataframe"].columns.tolist()
    except Exception as exc:
        LOGGER.warning("Column preview failed: %s", exc)
    return []


def serialise_uploads(uploaded_files) -> tuple:
    return tuple(
        (f.name, f.type or "", f.getvalue()) for f in uploaded_files
    )


def contains_text_payloads(file_payloads: tuple) -> bool:
    """True only if there are PDF/TXT payloads that yielded TEXT (not tables)."""
    return any(infer_file_type(fn) in TEXT_FILE_TYPES for fn, _, _ in file_payloads)


# ─────────────────────────────────────────────────────────────────
# SECTION 1 — SUMMARY (KPI cards)
# ─────────────────────────────────────────────────────────────────

def render_summary_section(results: dict) -> None:
    _section("Section 1", "Summary")
    try:
        data_results = results.get("data") or {}
        text_results = results.get("text") or {}

        cards = []

        if data_results.get("success"):
            kpis   = data_results.get("analysis", {}).get("kpis", {})
            stats  = data_results.get("analysis", {}).get("statistics", {})
            source = data_results.get("source", {})

            rows     = kpis.get("rows", "—")
            cols     = kpis.get("columns", "—")
            num_cols = kpis.get("numerical_columns", 0)
            cat_cols = kpis.get("categorical_columns", 0)

            cards.append({"icon": "🗄️", "label": "Total Rows",    "value": format_metric(rows),     "sub": f"{cols} columns"})
            cards.append({"icon": "🔢", "label": "Numeric Cols",  "value": format_metric(num_cols), "sub": "numeric features"})
            cards.append({"icon": "🏷️", "label": "Category Cols", "value": format_metric(cat_cols), "sub": "categorical features"})

            # Show first numeric mean / max
            if stats:
                key_col  = next(iter(stats))
                mean_val = f"{stats[key_col].get('mean', 0):,.2f}"
                max_val  = f"{stats[key_col].get('max',  0):,.2f}"
                cards.append({"icon": "⊘", "label": f"Avg · {humanize_label(key_col)}", "value": mean_val, "sub": f"Max: {max_val}"})

        if text_results.get("success"):
            text_analysis = text_results.get("analysis", {})
            word_count = text_analysis.get("word_count", "—")
            sent_count = text_analysis.get("sentence_count", "—")
            cards.append({"icon": "📝", "label": "Words", "value": format_metric(word_count), "sub": f"{format_metric(sent_count)} sentences"})

        if not cards:
            _empty("📂", "Upload a file to see the summary.")
            return

        card_htmls = []
        for c in cards:
            card_htmls.append(f"""
            <div class="summary-card">
                <div class="summary-card-label">{c['icon']} {c['label']}</div>
                <div class="summary-card-value">{c['value']}</div>
                <div class="summary-card-sub">{c['sub']}</div>
            </div>""")

        st.markdown(
            f'<div class="summary-row">{"".join(card_htmls)}</div>',
            unsafe_allow_html=True,
        )

    except Exception as exc:
        st.error(f"Summary section error: {exc}")


# ─────────────────────────────────────────────────────────────────
# SECTION 2 — STRUCTURED DATA SUMMARY
# ─────────────────────────────────────────────────────────────────

def render_structured_summary_section(results: dict) -> None:
    _section("Section 2", "Structured Data Summary")

    try:
        data_results = results.get("data") or {}

        if not data_results.get("success"):
            text_results = results.get("text") or {}
            if text_results.get("success"):
                _empty("📄", "PDF text mode — no tabular data to summarise column-wise.")
            else:
                _empty("📋", "Upload a CSV, Excel, or PDF with a table to see column-wise summary.")
            return

        df: pd.DataFrame = data_results.get("dataframe", pd.DataFrame())
        schema = data_results.get("schema", {})
        stats  = data_results.get("analysis", {}).get("statistics", {})

        if df.empty:
            _empty("📋", "No data available.")
            return

        numerical  = set(schema.get("numerical",  []))
        categorical = set(schema.get("categorical", []))
        datetime   = set(schema.get("datetime",   []))

        rows_html = ""
        for col in df.columns:
            if col in numerical:
                badge = '<span class="col-type-badge badge-numeric">numeric</span>'
                col_stats = stats.get(col, {})
                mean_v = col_stats.get("mean", None)
                min_v  = col_stats.get("min",  None)
                max_v  = col_stats.get("max",  None)
                samples = df[col].dropna().head(3).tolist()
                sample_txt = ", ".join(f"{v:,.2f}" if isinstance(v, float) else str(v) for v in samples)
                detail = f"avg: {mean_v:,.2f} &nbsp;|&nbsp; min: {min_v:,.2f} &nbsp;|&nbsp; max: {max_v:,.2f}" if mean_v is not None else sample_txt
            elif col in datetime:
                badge = '<span class="col-type-badge badge-datetime">datetime</span>'
                samples = df[col].dropna().head(3).astype(str).tolist()
                sample_txt = ", ".join(samples)
                detail = sample_txt
            else:
                badge = '<span class="col-type-badge badge-category">categorical</span>'
                samples = df[col].dropna().head(3).astype(str).tolist()
                sample_txt = ", ".join(samples)
                unique_n = int(df[col].nunique(dropna=True))
                detail = f"{sample_txt} &nbsp;({unique_n} unique)"

            rows_html += f"""
            <tr>
                <td><span class="col-name">{col}</span></td>
                <td>{badge}</td>
                <td>{detail}</td>
                <td style="color:#8B9AAA;">{int(df[col].notna().sum())}</td>
            </tr>"""

        table_html = f"""
        <table class="col-summary-table">
            <thead>
                <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Sample / Summary</th>
                    <th>Non-null</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>"""

        st.markdown(table_html, unsafe_allow_html=True)

        with st.expander("📋 Full Statistics (describe)", expanded=False):
            try:
                st.dataframe(df.describe(include="all").round(2), use_container_width=True)
            except Exception:
                pass

    except Exception as exc:
        st.error(f"Structured summary section error: {exc}")


# ─────────────────────────────────────────────────────────────────
# SECTION 3 — CHARTS (max 4)
# ─────────────────────────────────────────────────────────────────

def render_charts_section(results: dict) -> None:
    _section("Section 3", "Charts")

    try:
        data_results = results.get("data") or {}
        if not data_results.get("success"):
            _empty("📊", "Upload a CSV, Excel, or PDF with a table to generate charts.")
            return

        import plotly.express as px

        df: pd.DataFrame = data_results.get("dataframe", pd.DataFrame())
        schema = data_results.get("schema", {})

        if df.empty:
            _empty("📊", "No data available to chart.")
            return

        numeric_cols    = [c for c in schema.get("numerical",  []) if c in df.columns]
        categorical_cols = [c for c in schema.get("categorical", []) if c in df.columns]
        datetime_cols   = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        CHART_LAYOUT = dict(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="#E6EDF3",
            font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#E6EDF3"),
            margin=dict(l=0, r=0, t=40, b=0),
            hoverlabel=dict(bgcolor="#111827", bordercolor="#1F2937", font_color="#E6EDF3"),
        )

        charts_rendered = 0

        # Chart 1 — Line chart (datetime × numeric)
        if charts_rendered < 4 and datetime_cols and numeric_cols:
            try:
                dt_col  = datetime_cols[0]
                val_col = numeric_cols[0]
                frame   = df[[dt_col, val_col]].dropna().sort_values(dt_col)
                if not frame.empty:
                    fig = px.line(frame, x=dt_col, y=val_col, markers=True,
                                  color_discrete_sequence=["#34D399"])
                    fig.update_traces(line_width=2.5, marker_size=6,
                                      fill="tozeroy", fillcolor="rgba(52,211,153,0.07)")
                    fig.update_layout(
                        title=dict(text=f"📈 {humanize_label(val_col)} Over Time",
                                   font=dict(size=13, color="#E6EDF3"), x=0),
                        showlegend=False,
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                        **CHART_LAYOUT,
                    )
                    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True, key="chart_1_line")
                    st.markdown('</div>', unsafe_allow_html=True)
                    charts_rendered += 1
            except Exception as exc:
                LOGGER.warning("Line chart failed: %s", exc)

        # Chart 2 — Histogram (first numeric col)
        if charts_rendered < 4 and numeric_cols:
            try:
                col = numeric_cols[0]
                fig = px.histogram(df, x=col, nbins=20,
                                   color_discrete_sequence=["#00C2FF"], opacity=0.88)
                fig.update_traces(marker_line_width=0)
                fig.update_layout(
                    title=dict(text=f"📊 {humanize_label(col)} Distribution",
                               font=dict(size=13, color="#E6EDF3"), x=0),
                    showlegend=False, bargap=0.06,
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                    **CHART_LAYOUT,
                )
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, key="chart_2_histogram")
                st.markdown('</div>', unsafe_allow_html=True)
                charts_rendered += 1
            except Exception as exc:
                LOGGER.warning("Histogram failed: %s", exc)

        # Chart 3 — Bar chart (top categorical col)
        if charts_rendered < 4 and categorical_cols:
            try:
                col    = categorical_cols[0]
                counts = df[col].fillna("Missing").astype(str).value_counts().head(10).reset_index()
                counts.columns = [col, "count"]
                counts = counts.sort_values("count", ascending=True)
                fig = px.bar(
                    counts, x="count", y=col, orientation="h",
                    color=col,
                    color_discrete_sequence=["#00C2FF","#34D399","#F59E0B","#FB7185","#A78BFA",
                                             "#38BDF8","#F97316","#2DD4BF"],
                    text="count",
                )
                fig.update_traces(marker_line_width=0, textposition="outside", textfont_size=11)
                fig.update_layout(
                    title=dict(text=f"📊 {humanize_label(col)} Breakdown",
                               font=dict(size=13, color="#E6EDF3"), x=0),
                    showlegend=False, bargap=0.18,
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False),
                    **CHART_LAYOUT,
                )
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, key="chart_3_bar")
                st.markdown('</div>', unsafe_allow_html=True)
                charts_rendered += 1
            except Exception as exc:
                LOGGER.warning("Bar chart failed: %s", exc)

        # Chart 4 — Pie / donut (second categorical or same col)
        if charts_rendered < 4 and categorical_cols:
            try:
                # Pick second categorical if available for diversity
                pie_col = categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[0]
                counts  = df[pie_col].fillna("Missing").astype(str).value_counts().head(6).reset_index()
                counts.columns = [pie_col, "count"]
                fig = px.pie(
                    counts, names=pie_col, values="count", hole=0.50,
                    color=pie_col,
                    color_discrete_sequence=["#00C2FF","#34D399","#F59E0B","#FB7185","#A78BFA","#38BDF8"],
                )
                fig.update_traces(
                    textposition="inside", textinfo="percent+label",
                    marker=dict(line=dict(color="#0E1117", width=2)),
                )
                fig.update_layout(
                    title=dict(text=f"🥧 {humanize_label(pie_col)} Share",
                               font=dict(size=13, color="#E6EDF3"), x=0),
                    showlegend=False,
                    **CHART_LAYOUT,
                )
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, key="chart_4_pie")
                st.markdown('</div>', unsafe_allow_html=True)
                charts_rendered += 1
            except Exception as exc:
                LOGGER.warning("Pie chart failed: %s", exc)

        if charts_rendered == 0:
            _empty("📊", "No charts could be generated. Ensure your file has numeric or categorical columns.")

    except Exception as exc:
        st.error(f"Charts section error: {exc}")


# ─────────────────────────────────────────────────────────────────
# SECTION 4 — RANKED INSIGHTS (Intelligence Engine)
# ─────────────────────────────────────────────────────────────────

_RANK_MEDALS = ["🥇", "🥈", "🥉"]
_SOURCE_BADGE = {
    "data":    ('badge-data',    'DATA'),
    "text":    ('badge-text',    'TEXT'),
    "fusion":  ('badge-fusion',  'FUSION'),
    "anomaly": ('badge-anomaly', 'ANOMALY'),
}


def _render_ranked_insight_card(rank: int, ins: dict) -> str:
    """Build the HTML for a single ranked insight card."""
    title   = ins.get("title", "Insight")
    desc    = ins.get("description", ins.get("text", ""))
    score   = float(ins.get("score", ins.get("priority", 50) / 100))
    evidence = ins.get("evidence", [])
    source  = ins.get("source", "data")

    rank_icon = _RANK_MEDALS[rank] if rank < 3 else f"#{rank + 1}"
    badge_cls, badge_label = _SOURCE_BADGE.get(source, ('badge-data', source.upper()))
    score_pct = int(min(100, max(0, score * 100)))
    score_bar_color = (
        "linear-gradient(90deg,#34D399,#10B981)" if score >= 0.85
        else "linear-gradient(90deg,#F59E0B,#FBBF24)" if score >= 0.65
        else "linear-gradient(90deg,#0099CC,#00C2FF)"
    )

    # Evidence chips
    ev_html = ""
    if evidence:
        chips = "".join(
            f'<span class="ri-ev-chip">{ev}</span>' for ev in evidence[:6]
        )
        ev_html = f'<div class="ri-evidence">{chips}</div>'

    return f"""
    <div class="ri-card">
        <div class="ri-header">
            <div class="ri-rank">{rank_icon}</div>
            <div class="ri-title-block">
                <div class="ri-title">{title}</div>
                <div class="ri-badges">
                    <span class="ri-badge {badge_cls}">{badge_label}</span>
                </div>
            </div>
        </div>
        <div class="ri-score-wrap">
            <div class="ri-score-bar-bg">
                <div class="ri-score-bar-fill" style="width:{score_pct}%;background:{score_bar_color};"></div>
            </div>
            <div class="ri-score-label">Score {score_pct}%</div>
        </div>
        <div class="ri-description">{desc}</div>
        {ev_html}
    </div>"""


def render_insights_section(results: dict) -> None:
    _section("Section 4", "Ranked Insights")

    try:
        # ── PRIMARY PATH: use unified ranked_insights from intelligence engine ──
        ranked = results.get("ranked_insights") or []

        # ranked_insights may be Insight objects or dicts; normalise to dicts
        ranked_dicts: list[dict] = []
        for item in ranked:
            if isinstance(item, dict):
                ranked_dicts.append(item)
            elif hasattr(item, "to_dict"):
                ranked_dicts.append(item.to_dict())

        if ranked_dicts:
            # Show top-line score summary
            top_score = float(ranked_dicts[0].get("score", 1.0))
            sources_present = list(dict.fromkeys(
                item.get("source", "data") for item in ranked_dicts
            ))
            source_labels = " · ".join(
                f'<span class="ri-badge {_SOURCE_BADGE.get(s, ("badge-data",""))[0]}"'
                f' style="margin-right:4px">{_SOURCE_BADGE.get(s,("badge-data", s.upper()))[1]}</span>'
                for s in sources_present
            )
            st.markdown(
                f'<div style="margin-bottom:0.9rem;font-size:0.81rem;color:#7B8899;">'
                f'<span style="color:#E6EDF3;font-weight:700;">{len(ranked_dicts)}</span> insights ranked · '
                f'Top score: <span style="color:#34D399;font-weight:700;">{int(top_score*100)}%</span> · '
                f'Sources: {source_labels}</div>',
                unsafe_allow_html=True,
            )

            # Render each card
            for rank, ins in enumerate(ranked_dicts):
                st.markdown(
                    _render_ranked_insight_card(rank, ins),
                    unsafe_allow_html=True,
                )

            # ── Fusion keyword matches (if present) ──────────────
            fusion_results = results.get("fusion") or {}
            matches = fusion_results.get("matches", [])
            if matches:
                pills = "".join(
                    f'<span class="ri-match-pill">'
                    f'{m.get("term", "")} → {humanize_label(m.get("column", ""))}'
                    f' <span style="opacity:0.6">({int(m.get("score",0)*100)}%)</span>'
                    f'</span>'
                    for m in matches[:12]
                )
                method = matches[0].get("match_method", "difflib") if matches else "difflib"
                st.markdown(
                    f'<div class="ri-fusion-matches">'
                    f'<div class="ri-match-title">🔗 NLP ↔ Data Keyword Links '
                    f'<span style="opacity:0.5;font-weight:400;text-transform:none;font-size:0.68rem;">'
                    f'via {method}</span></div>'
                    f'{pills}</div>',
                    unsafe_allow_html=True,
                )

            # ── Keyword tags from text pipeline ─────────────────
            text_results = results.get("text") or {}
            if text_results.get("success"):
                keywords: list[dict] = text_results.get("analysis", {}).get("keywords", [])
                if keywords:
                    kw_items = keywords[:12]
                    badges = " ".join(
                        f'<span style="background:rgba(0,194,255,0.08);color:#7FDBFF;'
                        f'padding:0.15rem 0.6rem;border-radius:999px;font-size:0.76rem;'
                        f'margin:2px;display:inline-block;border:1px solid rgba(0,194,255,0.16);'
                        f'opacity:{0.5 + 0.5 * kw.get("importance", kw.get("score", 0)):.2f};">'
                        f'{kw["term"]}</span>'
                        for kw in kw_items
                    )
                    st.markdown(
                        f'<div class="text-summary-box" style="margin-top:0.8rem;">'
                        f'<div class="text-summary-title">🔑 Top Keywords · opacity = importance</div>'
                        f'<div style="padding-top:0.4rem;">{badges}</div></div>',
                        unsafe_allow_html=True,
                    )
            return

        # ── FALLBACK PATH: no ranked_insights — legacy renderer ───────────────
        data_results  = results.get("data")  or {}
        text_results  = results.get("text")  or {}
        rendered_anything = False

        if data_results.get("success"):
            rendered_anything = True
            legacy_insights: list[dict] = data_results.get("insights", [])
            if legacy_insights:
                for ins in legacy_insights:
                    title = ins.get("title", "")
                    text  = ins.get("text", ins.get("description", ""))
                    if text:
                        st.markdown(
                            f'<div class="insight-item">'
                            f'<div class="insight-title">{title}</div>{text}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        if text_results.get("success"):
            rendered_anything = True
            pipeline_insights: list[dict] = text_results.get("insights", [])
            text_analysis = text_results.get("analysis", {})
            summary = text_analysis.get("summary", "").strip()
            keywords: list[dict] = text_analysis.get("keywords", [])

            if summary:
                st.markdown(
                    f'<div class="text-summary-box">'
                    f'<div class="text-summary-title">📄 Document Summary</div>'
                    f'<div class="text-summary-body">{summary}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            for ins in pipeline_insights[:6]:
                title = ins.get("title", "")
                text  = ins.get("text", "")
                if text:
                    st.markdown(
                        f'<div class="insight-item">'
                        f'<div class="insight-title">{title}</div>{text}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            if keywords:
                top_kw = [kw["term"] for kw in keywords[:12]]
                kwbadges = " ".join(
                    f'<span style="background:rgba(0,194,255,0.10);color:#7FDBFF;'
                    f'padding:0.15rem 0.6rem;border-radius:999px;font-size:0.78rem;'
                    f'margin:2px;display:inline-block;border:1px solid rgba(0,194,255,0.18);">'
                    f'{kw}</span>'
                    for kw in top_kw
                )
                st.markdown(
                    f'<div class="text-summary-box"><div class="text-summary-title">🔑 Top Keywords</div>'
                    f'<div style="padding-top:0.3rem;">{kwbadges}</div></div>',
                    unsafe_allow_html=True,
                )

        if not rendered_anything:
            _empty("💡", "Upload a CSV, Excel, or PDF to see data insights.")

    except Exception as exc:
        st.error(f"Insights section error: {exc}")


# ─────────────────────────────────────────────────────────────────
# WELCOME STATE
# ─────────────────────────────────────────────────────────────────

def render_welcome() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">📂</div>
            <strong style="color:#E6EDF3;font-size:1.0rem;">No files uploaded yet</strong><br><br>
            Upload a <strong>CSV or Excel</strong> for structured analytics and charts.<br>
            Upload a <strong>PDF</strong> — if it contains a table it becomes a full data dashboard.<br>
            Upload a <strong>TXT or text-only PDF</strong> for extractive document summary.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    inject_styles()

    # ── Header ──
    st.markdown(
        """
        <div style="padding:1.2rem 0 0.5rem;">
            <div style="color:#00C2FF;font-size:0.72rem;letter-spacing:0.16em;text-transform:uppercase;font-weight:700;margin-bottom:0.3rem;">Analytics Dashboard</div>
            <div style="font-size:2.0rem;font-weight:800;color:#E6EDF3;line-height:1.1;margin-bottom:0.4rem;">Smart Summariser</div>
            <div style="color:#8B98A5;font-size:0.93rem;line-height:1.65;">Upload CSV, Excel, or PDF — get instant structured summaries, charts, and insights.</div>
            <hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:1.1rem 0 0.2rem;">
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 📂 Upload Files")
        uploaded_files = st.file_uploader(
            "CSV, Excel, PDF, or TXT",
            type=["csv", "xlsx", "xls", "pdf", "txt"],
            accept_multiple_files=True,
            help="PDF files with embedded tables will be treated as structured data.",
        )

        st.markdown("---")
        insight_depth = st.slider(
            "⚙️ Insight depth",
            min_value=1, max_value=5, value=3,
            help="Higher = more detailed insights.",
        )

        payloads     = serialise_uploads(uploaded_files) if uploaded_files else tuple()
        preview_cols = preview_structured_columns(payloads) if payloads else []
        default_cols = preview_cols[: min(6, len(preview_cols))] if len(preview_cols) > 6 else preview_cols

        if preview_cols:
            st.markdown("---")
            selected_columns = st.multiselect(
                "🔬 Columns to analyse",
                options=preview_cols,
                default=default_cols,
            )
        else:
            selected_columns = []

        if SAMPLE_DATA_PATH.exists():
            st.markdown("---")
            st.caption(f"💡 Sample: `{SAMPLE_DATA_PATH.name}`")

    # ── No files ──
    if not uploaded_files:
        render_welcome()
        return

    text_detected = contains_text_payloads(payloads)

    if text_detected:
        with st.spinner("Preparing NLP resources…"):
            try:
                prepare_nlp_resources()
            except Exception:
                pass

    active_columns = tuple(selected_columns or preview_cols)

    with st.spinner("Analysing your files…"):
        results = run_cached_analysis(payloads, insight_depth, active_columns, text_detected)

    # Surface errors / warnings
    for err in results.get("errors", []):
        st.error(err)
    for warn in results.get("warnings", []):
        st.warning(warn)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══ SECTION 1: SUMMARY KPIs ══
    render_summary_section(results)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══ SECTION 2: STRUCTURED DATA SUMMARY ══
    render_structured_summary_section(results)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══ SECTION 3: CHARTS ══
    render_charts_section(results)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══ SECTION 4: INSIGHTS ══
    render_insights_section(results)


if __name__ == "__main__":
    main()
