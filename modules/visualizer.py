import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.helpers import humanize_label
from utils.logger import get_logger

# ── Design tokens ──────────────────────────────────────────────────────────────
PAPER_COLOR = "#0E1117"
PLOT_COLOR  = "#151B27"
FONT_COLOR  = "#E6EDF3"
ACCENT_COLOR = "#00C2FF"
MUTED_GRID  = "rgba(255,255,255,0.05)"
HOVER_BG    = "#111827"
HOVER_BORDER = "#1F2937"

# Curated sequential palette — vivid but harmonious
COLOR_SEQUENCE = [
    "#00C2FF",  # cyan
    "#34D399",  # emerald
    "#F59E0B",  # amber
    "#FB7185",  # rose
    "#A78BFA",  # violet
    "#38BDF8",  # sky
    "#F97316",  # orange
    "#2DD4BF",  # teal
]


# ── Shared theme application ───────────────────────────────────────────────────

def _apply_theme(figure, title: str, hovermode: str = "closest"):
    figure.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, color=FONT_COLOR, family="Inter, Segoe UI, sans-serif"),
            x=0,
            xanchor="left",
            pad=dict(l=4),
        ),
        template="plotly_dark",
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=PLOT_COLOR,
        font=dict(color=FONT_COLOR, family="Inter, Segoe UI, sans-serif", size=12),
        margin=dict(l=28, r=28, t=54, b=32),
        hovermode=hovermode,
        hoverlabel=dict(
            bgcolor=HOVER_BG,
            bordercolor=HOVER_BORDER,
            font_color=FONT_COLOR,
            font_size=12,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        legend_title_text="",
    )
    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont=dict(size=11),
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor=MUTED_GRID,
        zeroline=False,
        showline=False,
        tickfont=dict(size=11),
    )
    return figure


# ── Individual chart builders ──────────────────────────────────────────────────

def _histogram(dataframe: pd.DataFrame, column: str):
    figure = px.histogram(
        dataframe,
        x=column,
        nbins=30,
        color_discrete_sequence=[ACCENT_COLOR],
        opacity=0.88,
    )
    figure.update_traces(
        marker_line_width=0,
        marker_color=ACCENT_COLOR,
        # Gradient effect via a second invisible bar layer is not natively supported,
        # so we use a fixed accent and add a subtle border glow via opacity.
    )
    figure.update_layout(bargap=0.06)
    return _apply_theme(figure, f"{humanize_label(column)} Distribution")


def _category_counts(dataframe: pd.DataFrame, column: str, top_n: int = 8) -> pd.DataFrame:
    counts = dataframe[column].fillna("Missing").astype(str).value_counts().head(top_n).reset_index()
    counts.columns = [column, "count"]
    return counts


def _bar_chart(dataframe: pd.DataFrame, column: str):
    counts = _category_counts(dataframe, column, top_n=10)
    # Sort descending for visual clarity
    counts = counts.sort_values("count", ascending=True)
    figure = px.bar(
        counts,
        x="count",
        y=column,
        orientation="h",
        color=column,
        color_discrete_sequence=COLOR_SEQUENCE,
        text="count",
    )
    figure.update_traces(
        marker_line_width=0,
        textposition="outside",
        textfont_size=11,
    )
    figure.update_layout(showlegend=False, bargap=0.18)
    return _apply_theme(figure, f"{humanize_label(column)} Breakdown")


def _pie_chart(dataframe: pd.DataFrame, column: str):
    counts = _category_counts(dataframe, column, top_n=6)
    # Pull the largest slice slightly
    max_idx = counts["count"].idxmax() if not counts.empty else 0
    pull = [0.07 if i == max_idx else 0 for i in range(len(counts))]
    figure = px.pie(
        counts,
        names=column,
        values="count",
        hole=0.50,
        color=column,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    figure.update_traces(
        textposition="inside",
        textinfo="percent+label",
        pull=pull,
        marker=dict(line=dict(color=PAPER_COLOR, width=2)),
    )
    return _apply_theme(figure, f"{humanize_label(column)} Share")


def _line_chart(trend: dict):
    frame = trend["timeseries"].copy()
    frame.columns = ["x", "y"]
    figure = px.line(
        frame,
        x="x",
        y="y",
        markers=True,
        color_discrete_sequence=["#34D399"],
    )
    figure.update_traces(
        line_width=2.5,
        marker_size=7,
        marker_color="#34D399",
        # Area fill under line
        fill="tozeroy",
        fillcolor="rgba(52,211,153,0.08)",
    )
    return _apply_theme(figure, f"{humanize_label(trend['metric'])} Trend", hovermode="x unified")


def _box_plot(dataframe: pd.DataFrame, column: str):
    figure = px.box(
        dataframe,
        y=column,
        color_discrete_sequence=["#F59E0B"],
        points="outliers",
    )
    figure.update_traces(
        marker=dict(color="#FB7185", size=5, opacity=0.75),
        line_color="#F59E0B",
    )
    return _apply_theme(figure, f"{humanize_label(column)} Spread")


def _heatmap(correlation_matrix: pd.DataFrame):
    figure = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns.tolist(),
            y=correlation_matrix.index.tolist(),
            colorscale=[
                [0.0,  "#0E1117"],
                [0.25, "#164E63"],
                [0.5,  "#0891B2"],
                [0.75, "#22D3EE"],
                [1.0,  "#A5F3FC"],
            ],
            zmin=-1.0,
            zmax=1.0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=11),
            hoverongaps=False,
        )
    )
    figure.update_layout(xaxis_tickangle=-35)
    return _apply_theme(figure, "Correlation Heatmap")


def _anomaly_scatter(top_rows: pd.DataFrame, numeric_columns: list[str]):
    if len(numeric_columns) < 2 or top_rows.empty:
        return None
    x_column, y_column = numeric_columns[:2]
    plot_frame = top_rows.copy()
    color_column = "_isolation_anomaly" if "_isolation_anomaly" in plot_frame.columns else None
    size_series = np.abs(
        plot_frame.get("_anomaly_score", pd.Series([1] * len(plot_frame), index=plot_frame.index))
    )
    figure = px.scatter(
        plot_frame,
        x=x_column,
        y=y_column,
        color=color_column,
        size=size_series,
        hover_data=[column for column in plot_frame.columns if not column.startswith("_")][:6],
        color_discrete_sequence=["#FB7185", "#00C2FF"],
        opacity=0.85,
    )
    figure.update_traces(marker=dict(line=dict(width=0)))
    return _apply_theme(figure, "Anomaly Scatter")


def _numeric_scatter(dataframe: pd.DataFrame, numeric_columns: list[str]):
    if len(numeric_columns) < 2:
        return None
    x_column, y_column = numeric_columns[:2]
    plot_frame = dataframe[[x_column, y_column]].dropna().head(2_000)
    if plot_frame.empty:
        return None
    figure = px.scatter(
        plot_frame,
        x=x_column,
        y=y_column,
        opacity=0.70,
        color_discrete_sequence=["#A78BFA"],
        trendline="ols",
        trendline_color_override="#FBBF24",
    )
    figure.update_traces(marker=dict(size=5, line=dict(width=0)))
    return _apply_theme(figure, f"{humanize_label(x_column)} vs {humanize_label(y_column)}")


# ── Public entry point ─────────────────────────────────────────────────────────

def build_visualizations(
    dataframe: pd.DataFrame,
    schema: dict,
    analysis: dict,
    anomalies: dict,
    selected_columns: list[str] | tuple[str, ...] | None = None,
    logger=None,
) -> dict:
    logger = logger or get_logger(__name__)
    charts: dict[str, list] = {"primary": [], "advanced": [], "category": [], "error": None}

    try:
        selected = set(selected_columns or dataframe.columns.tolist())
        numeric_columns    = [c for c in schema.get("numerical",   []) if c in selected]
        categorical_columns = [c for c in schema.get("categorical", []) if c in selected]

        # ── Primary: trend lines, then histograms ──
        for trend in analysis.get("trends", []):
            if trend["metric"] in numeric_columns:
                charts["primary"].append(
                    {"title": f"{humanize_label(trend['metric'])} Trend", "figure": _line_chart(trend)}
                )
            if len(charts["primary"]) >= 2:
                break

        for column in numeric_columns[:2]:
            charts["primary"].append(
                {"title": f"{humanize_label(column)} Distribution", "figure": _histogram(dataframe, column)}
            )

        # ── Category: pie + bar ──
        for column in categorical_columns[:2]:
            charts["category"].append({"title": f"{humanize_label(column)} Share",      "figure": _pie_chart(dataframe, column)})
            charts["category"].append({"title": f"{humanize_label(column)} Breakdown",  "figure": _bar_chart(dataframe, column)})

        # ── Advanced: box, heatmap, anomaly scatter, numeric scatter ──
        for column in numeric_columns[:2]:
            charts["advanced"].append(
                {"title": f"{humanize_label(column)} Spread", "figure": _box_plot(dataframe, column)}
            )

        correlation_matrix = analysis.get("correlation_matrix", pd.DataFrame())
        if not correlation_matrix.empty:
            charts["advanced"].append({"title": "Correlation Heatmap", "figure": _heatmap(correlation_matrix)})

        anomaly_chart = _anomaly_scatter(anomalies.get("top_rows", pd.DataFrame()), numeric_columns)
        if anomaly_chart is not None:
            charts["advanced"].append({"title": "Anomaly Scatter", "figure": anomaly_chart})

        scatter_chart = _numeric_scatter(dataframe, numeric_columns)
        if scatter_chart is not None:
            charts["advanced"].append({"title": "Metric Relationship", "figure": scatter_chart})

        # Cap each group
        charts["primary"]  = charts["primary"][:4]
        charts["advanced"] = charts["advanced"][:4]
        charts["category"] = charts["category"][:4]
        return charts

    except Exception as exc:
        logger.warning("Visualization build failed: %s", exc)
        charts["error"] = f"Visualization build failed: {exc}"
        return charts
