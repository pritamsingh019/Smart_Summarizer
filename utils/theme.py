"""
utils/theme.py
──────────────
Single source of truth for Smart Summariser design tokens.

Usage::

    from utils.theme import get_theme

    t = get_theme(st.session_state.theme)   # "light" | "dark"
    palette = t["palette"]
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────
# TOKEN DICTS
# ─────────────────────────────────────────────────────────────────

LIGHT: dict[str, object] = {
    # Surface colours
    "bg_primary":    "#F5F1E8",
    "bg_secondary":  "#FFFFFF",
    # Text
    "text_primary":  "#1A1A1A",
    "text_secondary":"#666666",
    # Borders
    "border":        "rgba(0,0,0,0.08)",
    # Accent (buttons, highlights)
    "accent":        "#1A1A1A",
    "accent_text":   "#FFFFFF",
    # Plotly chart palette — vivid, readable on cream
    "palette": [
        "#4A90E2",  # cobalt blue
        "#50C878",  # emerald green
        "#F5A623",  # amber
        "#E94B3C",  # coral red
        "#9B59B6",  # medium purple
        "#1ABC9C",  # turquoise
        "#E67E22",  # carrot orange
        "#34495E",  # wet asphalt
    ],
    # Source-pill semantic colours
    "pill_data":    {"bg": "#EBF5FB", "text": "#1A5276"},
    "pill_anomaly": {"bg": "#FDEDEC", "text": "#922B21"},
    "pill_nlp":     {"bg": "#EAF2FF", "text": "#1F618D"},
    "pill_fusion":  {"bg": "#F3E5F5", "text": "#6C3483"},
    "pill_default": {"bg": "#F2F3F4", "text": "#424949"},
    # Plotly common
    "plotly_bg":      "rgba(0,0,0,0)",
    "plotly_grid":    "rgba(0,0,0,0.06)",
    "plotly_tick":    "#666666",
    "plotly_hover_bg":"#FFFFFF",
}

DARK: dict[str, object] = {
    # Surface colours
    "bg_primary":    "#0F0F0F",
    "bg_secondary":  "#1A1A1A",
    # Text
    "text_primary":  "#F5F5F5",
    "text_secondary":"#A0A0A0",
    # Borders
    "border":        "rgba(255,255,255,0.08)",
    # Accent
    "accent":        "#F5F5F5",
    "accent_text":   "#0F0F0F",
    # Plotly chart palette — same colours, read well on dark too
    "palette": [
        "#4A90E2",
        "#50C878",
        "#F5A623",
        "#E94B3C",
        "#9B59B6",
        "#1ABC9C",
        "#E67E22",
        "#34495E",
    ],
    # Source-pill semantic colours (desaturated for dark bg)
    "pill_data":    {"bg": "#1A2B3C", "text": "#85C1E9"},
    "pill_anomaly": {"bg": "#2C1011", "text": "#F1948A"},
    "pill_nlp":     {"bg": "#1A2340", "text": "#7FB3D3"},
    "pill_fusion":  {"bg": "#2A1A2E", "text": "#C39BD3"},
    "pill_default": {"bg": "#2C2C2C", "text": "#AAAAAA"},
    # Plotly common
    "plotly_bg":      "rgba(0,0,0,0)",
    "plotly_grid":    "rgba(255,255,255,0.06)",
    "plotly_tick":    "#A0A0A0",
    "plotly_hover_bg":"#1A1A1A",
}


# ─────────────────────────────────────────────────────────────────
# PUBLIC HELPER
# ─────────────────────────────────────────────────────────────────

def get_theme(name: str) -> dict[str, object]:
    """Return the design-token dict for *name*.

    Parameters
    ----------
    name : str
        ``"light"`` or ``"dark"``.  Any other value is treated as
        ``"light"``.

    Returns
    -------
    dict[str, object]
        Flat token dict.  Colour values are CSS strings.  ``palette``
        is a ``list[str]`` of hex colour codes.
    """
    return DARK if name == "dark" else LIGHT
