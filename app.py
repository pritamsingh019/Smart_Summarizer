"""
app.py
──────
Smart Summariser — Streamlit dashboard.

Presentation-layer complete rewrite.  All backend pipeline / orchestrator
calls are preserved verbatim.  Only the rendering layer changes.

Architecture
────────────
main()
  ├── inject_global_css(theme)     — injects CSS vars + static stylesheet
  ├── [no file] → render_landing(theme)
  │       ├── render_top_bar(theme)
  │       ├── render_hero()
  │       ├── render_upload_card()
  │       └── render_feature_cards()
  └── [file uploaded] → render_dashboard(theme, results, …)
          ├── render_top_bar(theme)
          ├── render_kpi_row(results)
          ├── render_charts_grid(results, theme)
          └── render_insights_list(results, has_nlp_text_upload)

Sidebar is wired inside main() via st.sidebar.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.helpers import format_metric, humanize_label, infer_file_type
from utils.logger import get_logger
from utils.theme import get_theme

LOGGER = get_logger("smart_summariser.app")
PROJECT_ROOT      = Path(__file__).resolve().parent
CSS_PATH          = PROJECT_ROOT / "assets" / "styles.css"
SAMPLE_DATA_PATH  = PROJECT_ROOT / "sample_data" / "example_sales.csv"
STRUCTURED_FILE_TYPES = {"csv", "excel"}
TEXT_FILE_TYPES       = {"pdf", "txt"}
DEFAULT_SPACY_MODEL   = "en_core_web_sm"

st.set_page_config(
    page_title="Smart Summariser",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CSS INJECTION
# ─────────────────────────────────────────────────────────────────

_STATIC_CSS: str | None = None   # module-level cache


def _load_static_css() -> str:
    """Load assets/styles.css once and cache the result in memory.

    Returns
    -------
    str
        Raw CSS content, or empty string if the file cannot be read.
    """
    global _STATIC_CSS
    if _STATIC_CSS is None:
        try:
            _STATIC_CSS = CSS_PATH.read_text(encoding="utf-8")
        except Exception as exc:
            LOGGER.warning("Could not load assets/styles.css: %s", exc)
            _STATIC_CSS = ""
    return _STATIC_CSS


def inject_global_css(theme: dict) -> None:
    """Inject the static stylesheet and the theme's CSS custom properties.

    Calls ``st.markdown`` twice:

    1. The full contents of ``assets/styles.css`` wrapped in ``<style>``.
    2. A ``<style>`` block that sets CSS custom properties on ``html``
       derived from *theme* (bg, text, border, accent).

    Parameters
    ----------
    theme : dict
        Design-token dict returned by :func:`utils.theme.get_theme`.
    """
    static = _load_static_css()
    if static:
        st.markdown(f"<style>{static}</style>", unsafe_allow_html=True)

    # Dynamic colour custom properties
    st.markdown(f"""
<style>
html {{
    --bg-primary:    {theme['bg_primary']};
    --bg-secondary:  {theme['bg_secondary']};
    --text-primary:  {theme['text_primary']};
    --text-secondary:{theme['text_secondary']};
    --border:        {theme['border']};
    --accent:        {theme['accent']};
    --accent-text:   {theme['accent_text']};
}}
body, .stApp, .stApp > .main {{
    background: var(--bg-primary) !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SVG ICONS  (inline, no CDN)
# ─────────────────────────────────────────────────────────────────

_SUN_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" '
    'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="4"/>'
    '<line x1="12" y1="2"  x2="12" y2="6"/>'
    '<line x1="12" y1="18" x2="12" y2="22"/>'
    '<line x1="4.93"  y1="4.93"  x2="7.76"  y2="7.76"/>'
    '<line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/>'
    '<line x1="2"  y1="12" x2="6"  y2="12"/>'
    '<line x1="18" y1="12" x2="22" y2="12"/>'
    '<line x1="4.93"  y1="19.07" x2="7.76"  y2="16.24"/>'
    '<line x1="16.24" y1="7.76"  x2="19.07" y2="4.93"/>'
    '</svg>'
)

_MOON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" '
    'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>'
    '</svg>'
)


# ─────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────

_TAIL_SENTINEL = "This sustained"


def strip_templated_prose(description: str) -> str:
    """Remove the boilerplate tail from trend insight descriptions.

    Splits on the sentinel string ``"This sustained"``, takes the text
    before it, and strips trailing whitespace / punctuation.  Does not
    modify ``insight_generator.py``.

    Parameters
    ----------
    description : str
        Raw description string from the insight generator.

    Returns
    -------
    str
        Cleaned description, or the original string if the sentinel is
        absent.
    """
    if _TAIL_SENTINEL in description:
        description = description.split(_TAIL_SENTINEL)[0].rstrip(" .,;")
    return description


def _empty_state(icon: str, title: str, sub: str = "") -> None:
    """Render a centred, bordered empty-state card.

    Parameters
    ----------
    icon  : str
        Emoji or short text used as the visual anchor.
    title : str
        Short bold headline (serif italic in CSS).
    sub   : str, optional
        Secondary body text (smaller, text-secondary colour).
    """
    st.markdown(
        f'<div class="ss-empty-state">'
        f'<div class="ss-empty-icon">{icon}</div>'
        f'<div class="ss-empty-title">{title}</div>'
        f'<div class="ss-empty-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _hr() -> None:
    """Render a themed horizontal rule."""
    st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# CACHE / PIPELINE WRAPPERS  (verbatim — do not modify)
# ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def prepare_nlp_resources():
    """Bootstrap spaCy NLP resources (cached across reruns).

    Returns
    -------
    tuple[dict, object | None]
        (status_dict, spacy_nlp_model_or_None)
    """
    try:
        from utils.bootstrap import bootstrap_nlp, load_spacy_model
        status = bootstrap_nlp(logger=LOGGER, download_missing=True)
        model_name = status.get("spacy", {}).get("model", DEFAULT_SPACY_MODEL)
        nlp = (
            load_spacy_model(model_name, logger=LOGGER)
            if status.get("spacy", {}).get("available")
            else None
        )
        return status, nlp
    except Exception as exc:
        LOGGER.warning("NLP bootstrap failed: %s", exc)
        return {}, None


@st.cache_data(show_spinner=False)
def run_cached_analysis(
    file_payloads: tuple,
    insight_depth: int,
    selected_columns: tuple,
    enable_nlp: bool,
) -> dict:
    """Run the full orchestrator pipeline, cached by inputs.

    Parameters
    ----------
    file_payloads    : tuple
        Serialised uploaded files — ``(name, mime, bytes)`` triples.
    insight_depth    : int
        Slider value 1–5 passed to the pipeline.
    selected_columns : tuple
        Column filter; empty tuple means all columns.
    enable_nlp       : bool
        True when a text-bearing file is present.

    Returns
    -------
    dict
        Orchestrator result dict with keys: data, text, fusion,
        ranked_insights, warnings, errors, debug.
    """
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
        return {
            "data": None, "text": None, "fusion": None,
            "warnings": [], "errors": [str(exc)], "debug": {},
        }


@st.cache_data(show_spinner=False)
def preview_structured_columns(file_payloads: tuple) -> list[str]:
    """Return column names from the first structured file in *file_payloads*.

    Parameters
    ----------
    file_payloads : tuple
        Serialised uploaded files.

    Returns
    -------
    list[str]
        Column names, or empty list on failure.
    """
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
    """Convert Streamlit UploadedFile objects to a hashable tuple.

    Parameters
    ----------
    uploaded_files : list[UploadedFile]
        Files from ``st.file_uploader``.

    Returns
    -------
    tuple
        Each element is ``(name: str, mime: str, content: bytes)``.
    """
    return tuple((f.name, f.type or "", f.getvalue()) for f in uploaded_files)


def contains_text_payloads(file_payloads: tuple) -> bool:
    """Return True only if a PDF/TXT file is present in *file_payloads*.

    Parameters
    ----------
    file_payloads : tuple
        Serialised uploads from :func:`serialise_uploads`.

    Returns
    -------
    bool
    """
    return any(infer_file_type(fn) in TEXT_FILE_TYPES for fn, _, _ in file_payloads)


# ─────────────────────────────────────────────────────────────────
# THEME TOGGLE  (shared between top bar and sidebar)
# ─────────────────────────────────────────────────────────────────

def render_theme_toggle(key_suffix: str = "") -> None:
    """Render the sun/moon SVG theme-toggle button.

    Writes ``st.session_state["theme"]`` and calls ``st.rerun()`` on
    click.  Uses inline SVG icons (no emoji, no CDN).

    Parameters
    ----------
    key_suffix : str, optional
        Appended to the widget key to allow placement in multiple
        locations without key collisions.
    """
    is_dark = st.session_state.get("theme", "dark") == "dark"
    # Plain Unicode glyphs — st.button does not render HTML labels
    label = "\u2600" if is_dark else "\u263D"   # ☀ sun / ☽ crescent moon

    # Inject button-specific style to strip the default padding / background
    st.markdown("""
<style>
div[data-testid="stButton"].ss-toggle > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    padding: 5px 8px !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    min-width: 0 !important;
    width: auto !important;
}
div[data-testid="stButton"].ss-toggle > button:hover {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

    # Wrap in a div with class ss-toggle
    st.markdown('<div class="ss-toggle" style="display:inline-block">', unsafe_allow_html=True)
    clicked = st.button(label, key=f"theme_toggle_{key_suffix}", help="Toggle light / dark mode")
    st.markdown('</div>', unsafe_allow_html=True)

    if clicked:
        st.session_state["theme"] = "dark" if not is_dark else "light"
        st.rerun()


# ─────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────

def render_top_bar(theme: dict) -> None:
    """Render the top bar: wordmark (left) and theme toggle (right).

    Renders in normal document flow — not sticky — to avoid z-index and
    scroll surprises inside Streamlit's iframe layout.

    Parameters
    ----------
    theme : dict
        Active design-token dict from :func:`utils.theme.get_theme`.
    """
    col_wm, col_spacer, col_toggle = st.columns([6, 3, 1])
    with col_wm:
        st.markdown(
            '<div class="ss-top-bar">'
            '<span class="ss-wordmark">Smart Summariser</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_toggle:
        st.markdown('<div style="padding-top:18px;">', unsafe_allow_html=True)
        render_theme_toggle(key_suffix="topbar")
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# LANDING SCREEN
# ─────────────────────────────────────────────────────────────────

def render_hero() -> None:
    """Render the large Times New Roman bold italic headline on the landing screen."""
    st.markdown(
        '<div class="ss-hero">'
        '<h1 class="ss-hero-headline">What&#8217;s on your mind\u2009?</h1>'
        '</div>',
        unsafe_allow_html=True,
    )
    # Ensure headline is dark regardless of theme
    st.markdown("""
<style>
.ss-hero-headline {
    font-family: "Times New Roman", Times, serif !important;
    font-style: italic !important;
    font-weight: 700 !important;
    color: #1A1A1A !important;
    font-size: clamp(32px, 5vw, 52px) !important;
    letter-spacing: -1px;
    line-height: 1.1;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)


def render_upload_card() -> None:
    """Render the centred dashed upload card on the landing screen.

    The file uploader widget inside this card writes to
    ``st.session_state["file_uploader"]`` so the main function can
    read the result without the card needing to return it.
    """
    # Cloud-upload SVG (inline, no CDN)
    upload_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" '
        'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" '
        'style="opacity:0.4;color:var(--text-primary)">'
        '<polyline points="16 16 12 12 8 16"/>'
        '<line x1="12" y1="12" x2="12" y2="21"/>'
        '<path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>'
        '</svg>'
    )

    st.markdown(
        '<div class="ss-upload-card">'
        f'<div class="ss-upload-card-icon">{upload_svg}</div>'
        '<div class="ss-upload-card-title">Upload a file to get started</div>'
        '<div class="ss-upload-helper">'
        'CSV or Excel &rarr; structured analytics, charts &amp; insights.<br>'
        'PDF with table &rarr; full data dashboard.<br>'
        'TXT or prose PDF &rarr; document summary &amp; keyword extraction.'
        '</div>'
        '<div class="ss-upload-helper" style="margin-top:8px;font-style:italic;">'
        '&uarr; Use the sidebar uploader on the left to choose your file.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    # NOTE: no st.file_uploader here — the sidebar owns the single
    # key="file_uploader" widget to avoid StreamlitDuplicateElementKey.


def render_feature_cards() -> None:
    """Render the four-card feature grid below the upload card."""
    st.markdown(
        '<div class="ss-eyebrow">GET STARTED</div>',
        unsafe_allow_html=True,
    )

    cards = [
        ("📊", "Structured Analytics",
         "Upload CSV or Excel to generate charts, KPIs and trends."),
        ("📈", "Full Data Dashboard",
         "Turn your data into interactive dashboards and reports."),
        ("📄", "Document Summary",
         "Get concise summaries and extract key insights."),
        ("🔍", "Keyword Extraction",
         "Extract important keywords and themes from documents."),
    ]

    cols = st.columns(4, gap="small")
    for col, (icon, title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="ss-feature-card">'
                f'<div class="ss-feature-icon-badge">{icon}</div>'
                f'<div class="ss-feature-title">{title}</div>'
                f'<div class="ss-feature-body">{body}</div>'
                f'<div class="ss-feature-arrow">→</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_landing(theme: dict) -> None:
    """Render the complete landing screen (no file uploaded).

    Parameters
    ----------
    theme : dict
        Active design-token dict.
    """
    render_hero()
    render_upload_card()
    render_feature_cards()


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────

def render_sidebar(theme: dict) -> tuple[int, list[str], list]:
    """Render the sidebar and return the user's selections.

    Contains: wordmark, theme toggle, nav, depth slider, file uploader,
    column selector, and file chips.

    Parameters
    ----------
    theme : dict
        Active design-token dict (used for toggle icon).

    Returns
    -------
    tuple[int, list[str], list]
        ``(insight_depth, selected_columns, uploaded_files)``
    """
    with st.sidebar:
        # ── Wordmark + toggle row ──────────────────────────────────
        c_wm, c_btn = st.columns([5, 2])
        with c_wm:
            st.markdown(
                '<div class="sb-wordmark">Smart Summariser</div>',
                unsafe_allow_html=True,
            )
        with c_btn:
            st.markdown('<div style="padding-top:4px;">', unsafe_allow_html=True)
            render_theme_toggle(key_suffix="sidebar")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Nav (cosmetic — sections are in-page) ─────────────────
        st.markdown("""
<div class="sb-nav">
  <div class="sb-nav-item active">
    <span class="sb-nav-icon">◧</span>Overview
  </div>
  <div class="sb-nav-item">
    <span class="sb-nav-icon">◫</span>Charts
  </div>
  <div class="sb-nav-item">
    <span class="sb-nav-icon">◎</span>Insights
  </div>
</div>""", unsafe_allow_html=True)

        _hr()

        # ── Insight depth ──────────────────────────────────────────
        st.markdown(
            '<div class="sb-section-label">Insight depth</div>',
            unsafe_allow_html=True,
        )
        insight_depth = st.slider(
            "Insight depth",
            min_value=1, max_value=5, value=3,
            key="depth_slider",
            label_visibility="collapsed",
        )
        st.markdown(
            f'<div class="sb-depth-meta">Level {insight_depth} of 5</div>',
            unsafe_allow_html=True,
        )

        _hr()

        # ── File uploader ──────────────────────────────────────────
        st.markdown(
            '<div class="sb-section-label">Upload files</div>',
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["csv", "xlsx", "xls", "pdf", "txt"],
            accept_multiple_files=True,
            key="file_uploader",
            label_visibility="collapsed",
        )

        payloads     = serialise_uploads(uploaded_files) if uploaded_files else tuple()
        preview_cols = preview_structured_columns(payloads) if payloads else []
        default_cols = (
            preview_cols[: min(6, len(preview_cols))]
            if len(preview_cols) > 6
            else preview_cols
        )

        # ── Column selector ────────────────────────────────────────
        if preview_cols:
            _hr()
            st.markdown(
                '<div class="sb-section-label">Columns</div>',
                unsafe_allow_html=True,
            )
            selected_columns = st.multiselect(
                "Active columns",
                options=preview_cols,
                default=default_cols,
                key="col_select",
                label_visibility="collapsed",
            )
            n_active = len(selected_columns or preview_cols)
            st.markdown(
                f'<div class="sb-col-meta">'
                f'{n_active} of {len(preview_cols)} active</div>',
                unsafe_allow_html=True,
            )
        else:
            selected_columns = []

        # ── File chips ─────────────────────────────────────────────
        if uploaded_files:
            _hr()
            for f in uploaded_files:
                kb       = len(f.getvalue()) / 1024
                size_str = f"{kb:.1f} KB" if kb < 1000 else f"{kb / 1024:.1f} MB"
                st.markdown(
                    f'<div class="sb-file-chip">'
                    f'<span class="sb-chip-name">{f.name}</span>'
                    f'<span class="sb-chip-size">{size_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    return insight_depth, selected_columns, uploaded_files


# ─────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────

def render_kpi_row(results: dict) -> None:
    """Render the four KPI cards in the Overview section.

    Reads ``results["data"]["analysis"]["kpis"]`` and ``["statistics"]``.
    Silently skips if the data pipeline did not succeed.

    Parameters
    ----------
    results : dict
        Full orchestrator output dict.
    """
    data_results = results.get("data") or {}
    if not data_results.get("success"):
        return

    kpis  = data_results.get("analysis", {}).get("kpis", {})
    stats = data_results.get("analysis", {}).get("statistics", {})
    df    = data_results.get("dataframe", pd.DataFrame())

    rows      = kpis.get("rows", 0)
    total_cols = kpis.get("columns", 0)
    num_cols  = kpis.get("numerical_columns", 0)
    cat_cols  = kpis.get("categorical_columns", 0)

    # Missing % across numeric columns
    missing_pct = 0
    if not df.empty and num_cols > 0:
        numeric_names = data_results.get("schema", {}).get("numerical", [])
        numeric_in_df = [c for c in numeric_names if c in df.columns]
        if numeric_in_df:
            total_cells = rows * len(numeric_in_df)
            missing_cells = df[numeric_in_df].isna().sum().sum()
            missing_pct = round(missing_cells / total_cells * 100, 1) if total_cells > 0 else 0

    # Unique values count across categorical columns
    cat_unique = 0
    cat_names  = data_results.get("schema", {}).get("categorical", [])
    cat_in_df  = [c for c in cat_names if c in df.columns]
    if cat_in_df:
        cat_unique = int(df[cat_in_df[0]].nunique(dropna=True))

    # Average of the largest numeric column
    avg_label = "Avg metric"
    avg_value = "—"
    avg_hint  = ""
    if stats:
        key_col = next(iter(stats))
        mean_v  = stats[key_col].get("mean", None)
        if mean_v is not None:
            avg_label = f"Avg {humanize_label(key_col)}"
            avg_value = (
                f"{mean_v:,.0f}" if abs(mean_v) >= 100 else f"{mean_v:,.2f}"
            )
            max_v = stats[key_col].get("max", 0)
            avg_hint = f"max {format_metric(max_v)}"

    st.markdown(
        f'<div class="ss-kpi-grid">'

        # Card 1 — Total rows
        f'<div class="ss-kpi-card">'
        f'<div class="ss-kpi-label">Total rows</div>'
        f'<div class="ss-kpi-value">{format_metric(rows)}</div>'
        f'<div class="ss-kpi-sub">{format_metric(total_cols)} columns</div>'
        f'</div>'

        # Card 2 — Numeric columns
        f'<div class="ss-kpi-card">'
        f'<div class="ss-kpi-label">Numeric</div>'
        f'<div class="ss-kpi-value">{format_metric(num_cols)}</div>'
        f'<div class="ss-kpi-sub">{missing_pct}% missing</div>'
        f'</div>'

        # Card 3 — Categorical columns
        f'<div class="ss-kpi-card">'
        f'<div class="ss-kpi-label">Categories</div>'
        f'<div class="ss-kpi-value">{format_metric(cat_cols)}</div>'
        f'<div class="ss-kpi-sub">{format_metric(cat_unique)} unique values</div>'
        f'</div>'

        # Card 4 — Avg metric
        f'<div class="ss-kpi-card">'
        f'<div class="ss-kpi-label">{avg_label}</div>'
        f'<div class="ss-kpi-value">{avg_value}</div>'
        f'<div class="ss-kpi-sub">{avg_hint}</div>'
        f'</div>'

        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# CHARTS GRID
# ─────────────────────────────────────────────────────────────────

def _is_year_col(col: str, series: pd.Series) -> bool:
    """Return True if *col* is a year-like ordinal column.

    A column is year-like if its name contains "year" (case-insensitive),
    it has ≤20 unique values, and all non-null integer values fall in
    [1900, 2100].

    Parameters
    ----------
    col    : str            Column name.
    series : pd.Series      The column data.

    Returns
    -------
    bool
    """
    name_match = "year" in col.lower()
    if not name_match:
        return False
    unique_vals = series.dropna().unique()
    if len(unique_vals) > 20:
        return False
    try:
        int_vals = [int(v) for v in unique_vals]
        return all(1900 <= v <= 2100 for v in int_vals)
    except (ValueError, TypeError):
        return False


def _plotly_base_layout(theme: dict) -> dict:
    """Return a Plotly layout dict with theme-aware colours."""
    return dict(
        paper_bgcolor=theme["plotly_bg"],
        plot_bgcolor=theme["plotly_bg"],
        font=dict(
            family="-apple-system, Segoe UI, system-ui, sans-serif",
            size=12,
            color=theme["text_primary"],
        ),
        margin=dict(l=40, r=20, t=30, b=40),
        hoverlabel=dict(
            bgcolor=theme["plotly_hover_bg"],
            bordercolor=theme["border"],
            font_color=theme["text_primary"],
            font_size=12,
        ),
        showlegend=False,
    )


def _axis_x(theme: dict, **extra) -> dict:
    """Return x-axis kwargs for Plotly update_xaxes."""
    base = dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont=dict(size=11, color=theme["plotly_tick"]),
    )
    base.update(extra)
    return base


def _axis_y(theme: dict, **extra) -> dict:
    """Return y-axis kwargs for Plotly update_yaxes."""
    base = dict(
        showgrid=True,
        gridcolor=theme["plotly_grid"],
        zeroline=False,
        showline=False,
        tickfont=dict(size=11, color=theme["plotly_tick"]),
    )
    base.update(extra)
    return base


def render_charts_grid(results: dict, theme: dict) -> None:
    """Render the 2-column chart grid in the Charts section.

    Chart selection logic per column:

    * Year-like columns (name contains "year", ≤20 uniq, values 1900–2100):
      → vertical bar of largest numeric metric summed per year.
    * Other numeric columns → histogram (15–20 bins).
    * Categorical columns with ≤20 unique values → multi-colour donut pie
    * Categorical columns with >20 unique values → skip; show caption.

    Parameters
    ----------
    results : dict
        Full orchestrator output dict.
    theme   : dict
        Active design-token dict.
    """
    import plotly.express as px

    st.markdown(
        '<div class="ss-section-heading">Charts</div>',
        unsafe_allow_html=True,
    )

    data_results = results.get("data") or {}
    if not data_results.get("success"):
        _empty_state("📊", "No chart data",
                     "Upload a CSV or Excel file to generate charts.")
        return

    df: pd.DataFrame = data_results.get("dataframe", pd.DataFrame())
    schema = data_results.get("schema", {})

    if df.empty:
        _empty_state("📊", "Empty dataset", "No rows to visualise.")
        return

    palette       = theme["palette"]
    numeric_cols   = [c for c in schema.get("numerical",   []) if c in df.columns]
    categorical_cols = [c for c in schema.get("categorical", []) if c in df.columns]
    layout        = _plotly_base_layout(theme)

    # ── Accumulate (title, fig | None, skip_msg | "") tuples ──────
    figs: list[tuple[str, object | None, str]] = []

    # Separate year-like columns from regular numeric columns
    year_cols        = [c for c in numeric_cols if _is_year_col(c, df[c])]
    non_year_numeric = [c for c in numeric_cols if c not in year_cols]

    # ── Chart A: Year bar (only when a year col exists) ───────────
    for yr_col in year_cols[:1]:
        metric = next((c for c in non_year_numeric), None)
        if metric is None:
            continue
        try:
            agg = (
                df.groupby(yr_col)[metric]
                .sum().reset_index().sort_values(yr_col)
            )
            fig = px.bar(
                agg, x=yr_col, y=metric,
                color_discrete_sequence=[palette[0]],
                text=metric,
            )
            fig.update_traces(
                marker_line_width=0,
                texttemplate="%{text:,.0f}",
                textposition="outside",
                textfont=dict(size=10, color=theme["text_secondary"]),
            )
            fig.update_layout(
                xaxis_type="category",
                yaxis_range=[0, agg[metric].max() * 1.25],
                **layout,
            )
            fig.update_xaxes(**_axis_x(theme))
            fig.update_yaxes(**_axis_y(theme))
            figs.append((
                f"{humanize_label(metric)} by {humanize_label(yr_col)}",
                fig, "",
            ))
        except Exception as exc:
            LOGGER.warning("Year-bar chart failed: %s", exc)

    # ── Chart 1: Histogram — distribution of first numeric col ────
    if len(non_year_numeric) >= 1:
        col = non_year_numeric[0]
        try:
            fig = px.histogram(
                df, x=col, nbins=20,
                color_discrete_sequence=[palette[0]],
                opacity=0.88,
            )
            fig.update_traces(marker_line_width=0)
            fig.update_layout(bargap=0.05, **layout)
            fig.update_xaxes(**_axis_x(theme))
            fig.update_yaxes(**_axis_y(theme))
            figs.append((f"{humanize_label(col)} distribution", fig, ""))
        except Exception as exc:
            LOGGER.warning("Histogram failed (%s): %s", col, exc)

    # ── Chart 2: Line chart — sorted trend for second numeric col ─
    if len(non_year_numeric) >= 2:
        col = non_year_numeric[1]
        try:
            trend_df = df[[col]].dropna().reset_index(drop=True)
            trend_df = trend_df.sort_values(col).reset_index(drop=True)
            trend_df["Record"] = range(len(trend_df))
            # Build fill colour: palette[1] at 12% opacity
            hex_c = palette[1].lstrip("#")
            r2, g2, b2 = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
            fill_rgba = f"rgba({r2},{g2},{b2},0.12)"
            fig = px.line(
                trend_df, x="Record", y=col,
                color_discrete_sequence=[palette[1]],
            )
            fig.update_traces(
                line=dict(width=2),
                fill="tozeroy",
                fillcolor=fill_rgba,
            )
            fig.update_layout(**layout)
            fig.update_xaxes(title_text="Record (sorted)", **_axis_x(theme))
            fig.update_yaxes(**_axis_y(theme))
            figs.append((f"{humanize_label(col)} trend", fig, ""))
        except Exception as exc:
            LOGGER.warning("Line chart failed (%s): %s", col, exc)

    # ── Chart 3: Candlestick — quartile-based OHLC per category ───
    #    Open=Q1, Close=Q3, Low=5th pct, High=95th pct of numeric col
    if non_year_numeric and categorical_cols:
        candle_metric = (non_year_numeric[2] if len(non_year_numeric) >= 3
                         else non_year_numeric[0])
        cat_col_c = categorical_cols[0]
        n_cats_c  = df[cat_col_c].nunique(dropna=True)
        if n_cats_c <= 20:
            try:
                import plotly.graph_objects as go
                grp = df.groupby(cat_col_c)[candle_metric]
                ohlc = pd.DataFrame({
                    "x":    grp.apply(lambda s: s.dropna().quantile(0.5)),
                    "open": grp.apply(lambda s: s.dropna().quantile(0.25)),
                    "high": grp.apply(lambda s: s.dropna().quantile(0.95)),
                    "low":  grp.apply(lambda s: s.dropna().quantile(0.05)),
                    "close":grp.apply(lambda s: s.dropna().quantile(0.75)),
                }).reset_index()
                # Sort by median descending
                ohlc = ohlc.sort_values("x", ascending=False)
                candle_fig = go.Figure(data=[go.Candlestick(
                    x=ohlc[cat_col_c],
                    open=ohlc["open"],
                    high=ohlc["high"],
                    low=ohlc["low"],
                    close=ohlc["close"],
                    increasing_line_color=palette[1],   # green
                    decreasing_line_color=palette[3],   # coral/red
                    increasing_fillcolor=palette[1],
                    decreasing_fillcolor=palette[3],
                    line=dict(width=1),
                    whiskerwidth=0.5,
                )])
                candle_layout = {k: v for k, v in layout.items()
                                 if k != "showlegend"}
                candle_fig.update_layout(
                    showlegend=False,
                    xaxis_rangeslider_visible=False,
                    **candle_layout,
                )
                candle_fig.update_xaxes(**_axis_x(theme))
                candle_fig.update_yaxes(**_axis_y(theme))
                figs.append((
                    f"{humanize_label(candle_metric)} range"
                    f" by {humanize_label(cat_col_c)}",
                    candle_fig, "",
                ))
            except Exception as exc:
                LOGGER.warning("Candlestick chart failed: %s", exc)

    # 3. Market share — first categorical × first numeric metric (grouped sum → donut)
    if categorical_cols and non_year_numeric:
        cat_col    = categorical_cols[0]
        metric_col = non_year_numeric[0]
        n_unique   = df[cat_col].nunique(dropna=True)
        if n_unique <= 25:
            try:
                agg = (
                    df.groupby(cat_col)[metric_col]
                    .sum().reset_index()
                    .sort_values(metric_col, ascending=False)
                    .head(10)
                )
                n_slices     = len(agg)
                slice_colors = [palette[i % len(palette)] for i in range(n_slices)]
                total        = agg[metric_col].sum()
                # Add percentage label so slices show market share
                agg["pct"] = (agg[metric_col] / total * 100).round(1)
                fig = px.pie(
                    agg,
                    names=cat_col,
                    values=metric_col,
                    hole=0.55,
                    color_discrete_sequence=slice_colors,
                )
                fig.update_traces(
                    textposition="outside",
                    textinfo="label+percent",
                    marker=dict(line=dict(
                        color=theme["bg_secondary"], width=2,
                    )),
                    textfont=dict(
                        size=11,
                        color=theme["text_primary"],
                        family="-apple-system, Segoe UI, system-ui, sans-serif",
                    ),
                    pull=[0.03] * n_slices,
                )
                # Remove showlegend from local call — it already lives in layout
                mkt_layout = {k: v for k, v in layout.items() if k != "showlegend"}
                fig.update_layout(showlegend=False, **mkt_layout)
                figs.append((
                    f"{humanize_label(cat_col)} market share"
                    f" by {humanize_label(metric_col)}",
                    fig, "",
                ))
            except Exception as exc:
                LOGGER.warning("Market-share chart failed: %s", exc)

    if not figs:
        _empty_state(
            "📊", "No charts available",
            "Ensure your file has numeric or categorical columns.",
        )
        return

    # ── Render as a 2-column CSS grid ─────────────────────────────
    figs = figs[:4]   # cap at 4 charts
    chart_idx = 0     # global counter — guarantees unique keys across all pairs
    i = 0
    while i < len(figs):
        pair = figs[i : i + 2]
        cols = st.columns(2, gap="medium")
        for col_widget, (title, fig, skip_msg) in zip(cols, pair):
            with col_widget:
                if skip_msg:
                    st.markdown(
                        f'<div class="ss-chart-card">'
                        f'<div class="ss-chart-title">{title}</div>'
                        f'<div class="ss-chart-skip-msg">{skip_msg}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="ss-chart-card">',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="ss-chart-title">{title}</div>',
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        fig,
                        width="stretch",
                        config={"displayModeBar": False},
                        key=f"chart_{chart_idx}",
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                chart_idx += 1
        i += 2


# ─────────────────────────────────────────────────────────────────
# STATISTICS TABLE  (collapsible)
# ─────────────────────────────────────────────────────────────────

def _render_stats_table(results: dict) -> None:
    """Render a collapsible descriptive-statistics table for numeric columns.

    Uses ``st.expander`` so it is hidden by default and only expands when
    the user explicitly clicks it.  Shows count, mean, std, min, median,
    max, and missing-% for every numeric column in the dataset.

    Parameters
    ----------
    results : dict
        Full orchestrator output dict.
    """
    data_results = results.get("data") or {}
    if not data_results.get("success"):
        return

    df: pd.DataFrame = data_results.get("dataframe", pd.DataFrame())
    schema = data_results.get("schema", {})
    numeric_cols = [c for c in schema.get("numerical", []) if c in df.columns]

    if not numeric_cols or df.empty:
        return

    with st.expander("📊 Statistics table", expanded=False):
        num_df = df[numeric_cols]

        # Build a clean stats frame
        desc = num_df.describe(percentiles=[0.5]).T   # mean, std, min, 50%, max, count
        desc = desc.rename(columns={
            "count": "Count",
            "mean":  "Mean",
            "std":   "Std dev",
            "min":   "Min",
            "50%":   "Median",
            "max":   "Max",
        })
        # Add missing %
        desc["Missing %"] = (num_df.isna().sum() / len(df) * 100).round(1)

        # Round numbers for readability
        for col in ["Mean", "Std dev", "Min", "Median", "Max"]:
            if col in desc.columns:
                desc[col] = desc[col].apply(
                    lambda v: f"{v:,.2f}" if abs(v) < 1_000_000 else f"{v:,.0f}"
                    if pd.notna(v) else "—"
                )
        desc["Count"]     = desc["Count"].apply(lambda v: f"{int(v):,}")
        desc["Missing %"] = desc["Missing %"].apply(lambda v: f"{v:.1f}%")

        # Rename index to "Column"
        desc.index.name = "Column"
        desc = desc.reset_index()

        st.dataframe(
            desc,
            width="stretch",
            hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────
# INSIGHTS LIST
# ─────────────────────────────────────────────────────────────────

_SOURCE_PILL_KEY = {
    "data":    "pill_data",
    "anomaly": "pill_anomaly",
    "text":    "pill_nlp",
    "nlp":     "pill_nlp",
    "fusion":  "pill_fusion",
}

_SOURCE_DISPLAY = {
    "data":    "DATA",
    "anomaly": "ANOMALY",
    "text":    "NLP",
    "nlp":     "NLP",
    "fusion":  "FUSION",
}


def _pill_html(source: str, theme: dict) -> str:
    """Build the HTML for a source pill badge.

    Parameters
    ----------
    source : str
        Insight source key (``"data"``, ``"anomaly"``, ``"text"``,
        ``"fusion"``).
    theme  : dict
        Active design-token dict.

    Returns
    -------
    str
        HTML ``<span>`` string.
    """
    key     = _SOURCE_PILL_KEY.get(source, "pill_default")
    colours = theme.get(key, theme["pill_default"])
    label   = _SOURCE_DISPLAY.get(source, source.upper())
    return (
        f'<span class="ss-ins-pill" '
        f'style="background:{colours["bg"]};color:{colours["text"]}">'
        f'{label}</span>'
    )


def _insight_card_html(ins: dict, theme: dict) -> str:
    """Build the HTML string for a single insight card.

    No rank badge is rendered.  Order in the list implies rank.

    Parameters
    ----------
    ins   : dict   Insight dict (title, description, score, source, evidence).
    theme : dict   Active design-token dict.

    Returns
    -------
    str
        Complete ``<div class="ss-ins-card">`` HTML string.
    """
    title    = ins.get("title", "Insight")
    desc     = strip_templated_prose(
        ins.get("description", ins.get("text", ""))
    )
    source   = ins.get("source", "data")
    evidence = ins.get("evidence", [])

    pill     = _pill_html(source, theme)

    ev_html  = ""
    if evidence:
        chips = "".join(
            f'<span class="ss-evidence-chip">{e}</span>'
            for e in evidence[:5]
        )
        ev_html = f'<div class="ss-evidence-row">{chips}</div>'

    return (
        f'<div class="ss-ins-card">'
        f'  <div class="ss-ins-top-row">'
        f'    <div class="ss-ins-title">{title}</div>'
        f'    {pill}'
        f'  </div>'
        f'  <div class="ss-ins-body">{desc}</div>'
        f'  {ev_html}'
        f'</div>'
    )


def render_insights_list(results: dict, has_nlp_text_upload: bool) -> None:
    """Render the stacked insight cards in the Insights section.

    Walks ``results["ranked_insights"]``, converts Insight objects to
    dicts, strips templated prose, and renders each card with a source
    pill.  Appends a dashed NLP placeholder card when no text file was
    uploaded.

    Parameters
    ----------
    results              : dict   Full orchestrator output dict.
    has_nlp_text_upload  : bool   True if a TXT/PDF text file is present.
    """
    st.markdown(
        '<div class="ss-section-heading">Insights</div>',
        unsafe_allow_html=True,
    )

    theme = get_theme(st.session_state.get("theme", "dark"))

    ranked = results.get("ranked_insights") or []
    ranked_dicts: list[dict] = []
    for item in ranked:
        if isinstance(item, dict):
            ranked_dicts.append(item)
        elif hasattr(item, "to_dict"):
            ranked_dicts.append(item.to_dict())

    if ranked_dicts:
        cards_html = "".join(
            _insight_card_html(ins, theme) for ins in ranked_dicts
        )
        st.markdown(cards_html, unsafe_allow_html=True)
    else:
        # Fallback: try data insights or text summary
        data_results = results.get("data") or {}
        text_results = results.get("text") or {}
        rendered = False

        if data_results.get("success"):
            for ins in data_results.get("insights", []):
                text = ins.get("text", ins.get("description", ""))
                if text:
                    rendered = True
                    d = {"title": ins.get("title", ""), "description": text, "source": "data"}
                    st.markdown(_insight_card_html(d, theme), unsafe_allow_html=True)

        if text_results.get("success"):
            summary = text_results.get("analysis", {}).get("summary", "").strip()
            if summary:
                rendered = True
                d = {"title": "Document Summary", "description": summary, "source": "text"}
                st.markdown(_insight_card_html(d, theme), unsafe_allow_html=True)

        if not rendered:
            _empty_state("💡", "No insights yet",
                         "Upload a CSV, Excel, or PDF to see insights.")
            return

    # ── NLP placeholder card (CSV-only upload) ─────────────────────
    if not has_nlp_text_upload:
        st.markdown(
            '<div class="ss-nlp-placeholder">'
            '<div class="ss-nlp-placeholder-title">More insights available</div>'
            '<div class="ss-nlp-placeholder-body">'
            'Upload a TXT or text-bearing PDF to see NLP keywords, '
            'summaries, and cross-modal fusion insights.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────
# DASHBOARD  (file uploaded)
# ─────────────────────────────────────────────────────────────────

def render_dashboard(
    theme: dict,
    results: dict,
    has_nlp_text_upload: bool,
) -> None:
    """Render the full dashboard after file upload.

    Parameters
    ----------
    theme               : dict   Active design-token dict.
    results             : dict   Orchestrator output dict.
    has_nlp_text_upload : bool   True if a text file was uploaded.
    """
    # top bar removed — sidebar wordmark + toggle is sufficient
    # Surface pipeline errors / warnings
    for err in results.get("errors", []):
        st.error(err)
    for warn in results.get("warnings", []):
        st.warning(warn)

    # ── Section 1: Overview ────────────────────────────────────────
    st.markdown(
        '<div class="ss-section-heading">Overview</div>',
        unsafe_allow_html=True,
    )
    render_kpi_row(results)

    # ── Section 2: Charts ──────────────────────────────────────────
    render_charts_grid(results, theme)

    # ── Section 2b: Statistics table (collapsible) ──────────────────────
    _render_stats_table(results)

    # ── Section 3: Insights ────────────────────────────────────────
    render_insights_list(results, has_nlp_text_upload)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Application entry point.

    Initialises session state, injects CSS, renders sidebar, then
    branches to the landing screen (no upload) or the dashboard.
    """
    # ── Session state defaults ─────────────────────────────────────
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"

    theme = get_theme(st.session_state["theme"])
    inject_global_css(theme)

    # ── Sidebar ────────────────────────────────────────────────────
    insight_depth, selected_columns, uploaded_files = render_sidebar(theme)

    # ── Branch ────────────────────────────────────────────────────
    if not uploaded_files:
        render_landing(theme)
        return

    # ── Analysis ──────────────────────────────────────────────────
    payloads          = serialise_uploads(uploaded_files)
    preview_cols      = preview_structured_columns(payloads)
    has_nlp           = contains_text_payloads(payloads)
    active_columns    = tuple(selected_columns or preview_cols)

    if has_nlp:
        with st.spinner("Preparing NLP resources…"):
            try:
                prepare_nlp_resources()
            except Exception:
                pass

    with st.spinner("Analysing your files…"):
        results = run_cached_analysis(payloads, insight_depth, active_columns, has_nlp)

    render_dashboard(theme, results, has_nlp_text_upload=has_nlp)


if __name__ == "__main__":
    main()
