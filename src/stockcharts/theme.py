"""Shared dark theme for the Dash dashboards.

Extracted from the screener app so the macro dashboard renders identically
without depending on it.  Colours match the TradingView-style charts produced
by :mod:`stockcharts.charts.interactive`.
"""

from __future__ import annotations

from typing import Any

from dash import html

# ---------------------------------------------------------------------------
# Dark theme colours (matching TradingView charts)
# ---------------------------------------------------------------------------
BG = "#131722"
CARD_BG = "#1e222d"
TEXT = "#d1d4dc"
MUTED = "#9598a1"
ACCENT = "#2962ff"
GREEN = "#26a69a"
RED = "#ef5350"

# Status colours for the macro watchlist.
AMBER = "#ff9800"
PANEL_BORDER = "#363a45"
INPUT_BG = "#2a2e39"

STATUS_COLORS: dict[str, str] = {
    "ok": GREEN,
    "watch": AMBER,
    "alert": RED,
    "unknown": MUTED,
}


def _label(text: str) -> html.Label:
    """Return a muted form label.

    Args:
        text: Label text.

    Returns:
        A styled Dash label.
    """
    return html.Label(text, style={"color": MUTED, "fontSize": "12px", "marginBottom": "2px"})


def _card(children: list, **kwargs: object) -> html.Div:
    """Wrap children in a rounded card.

    Args:
        children: Child components.
        **kwargs: Extra props; a ``style`` entry is merged over the default.

    Returns:
        A styled Dash div.
    """
    style: dict[str, Any] = {
        "backgroundColor": CARD_BG,
        "borderRadius": "6px",
        "padding": "16px",
        "marginBottom": "12px",
    }
    extra = kwargs.get("style", {})
    if isinstance(extra, dict):
        style.update(extra)
    return html.Div(children, style=style)


INPUT_STYLE = {
    "backgroundColor": INPUT_BG,
    "color": TEXT,
    "border": f"1px solid {PANEL_BORDER}",
    "borderRadius": "4px",
    "width": "100%",
    "padding": "6px 10px",
    "marginBottom": "10px",
}

DROPDOWN_STYLE = {
    "backgroundColor": INPUT_BG,
    "color": TEXT,
}


__all__ = [
    "ACCENT",
    "AMBER",
    "BG",
    "CARD_BG",
    "DROPDOWN_STYLE",
    "GREEN",
    "INPUT_BG",
    "INPUT_STYLE",
    "MUTED",
    "PANEL_BORDER",
    "RED",
    "STATUS_COLORS",
    "TEXT",
    "_card",
    "_label",
]
