"""Dash dashboard for tracking Treasury yields and macro stress.

Runs separately from the screener app in :mod:`stockcharts.app`: the two share
a theme but nothing else, since the screener's run-a-screen state machine has
no bearing on a macro panel.  Start it with ``stockcharts-yields-app``.

Fetching happens in a background callback so a cold refresh -- which paces
several FRED requests to stay inside the endpoint's throttle -- never freezes
the page.  The refreshed values travel through a ``dcc.Store`` as plain
scalars; the underlying series stay in the on-disk cache and are re-read by
the chart callbacks, which keeps the browser payload small.
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from datetime import date
from threading import Timer
from typing import Any

import diskcache
from dash import Dash, DiskcacheManager, Input, Output, State, dash_table, dcc, html

from stockcharts.charts.yields import build_curve_snapshot_figure, build_yield_figure
from stockcharts.macro.config import DISCLAIMER, KEY_DATES, SCENARIOS, Scenario
from stockcharts.macro.manual import load_manual_inputs, set_checklist, set_manual_value
from stockcharts.macro.snapshot import (
    MacroSnapshot,
    auto_scenario_signs,
    build_snapshot,
    evaluate_watchlist,
)
from stockcharts.theme import (
    ACCENT,
    AMBER,
    BG,
    CARD_BG,
    GREEN,
    INPUT_STYLE,
    MUTED,
    PANEL_BORDER,
    RED,
    TEXT,
    _card,
    _label,
)

_bg_cache = diskcache.Cache(os.path.join(os.path.dirname(__file__), ".bg_cache_macro"))
background_manager = DiskcacheManager(_bg_cache)

# History accumulates permanently in the local parquet cache, so a long
# lookback is slow only on its first use.
LOOKBACK_OPTIONS = [
    {"label": "3M", "value": 90},
    {"label": "6M", "value": 180},
    {"label": "1Y", "value": 365},
    {"label": "2Y", "value": 730},
    {"label": "5Y", "value": 1825},
    {"label": "10Y", "value": 3650},
    {"label": "20Y", "value": 7300},
]

TILE_SPECS: list[tuple[str, str]] = [
    ("ten_year", "10y nominal"),
    ("thirty_year", "30y nominal"),
    ("real_ten", "10y real"),
    ("breakeven", "10y breakeven"),
    ("term_premium", "Term premium"),
    ("spread_2s10s", "2s10s"),
]

STALE_DAYS = 30


# ---------------------------------------------------------------------------
# Snapshot -> plain payload
# ---------------------------------------------------------------------------


def snapshot_payload(snapshot: MacroSnapshot) -> dict[str, Any]:
    """Reduce a snapshot to JSON-safe scalars for the browser store.

    Args:
        snapshot: Assembled macro data.

    Returns:
        A mapping of tiles, rows, automatic signs and errors.
    """
    ten = snapshot.tenor(10.0)
    thirty = snapshot.tenor(30.0)
    tiles = {
        "ten_year": (ten, snapshot.tenor_change_bp(10.0), "%"),
        "thirty_year": (thirty, snapshot.tenor_change_bp(30.0), "%"),
        "real_ten": (snapshot.tenor(10.0, real=True), None, "%"),
        "breakeven": (snapshot.latest("breakeven_10y"), snapshot.change_bp("breakeven_10y"), "%"),
        "term_premium": (
            snapshot.latest("term_premium_10y"),
            snapshot.change_bp("term_premium_10y"),
            "%",
        ),
        "spread_2s10s": (snapshot.curve_spread_bp(2.0, 10.0), None, "bp"),
    }
    return {
        "fetched_at": snapshot.fetched_at,
        "expected_short_rate": snapshot.expected_short_rate(),
        "tiles": {k: {"value": v[0], "change_bp": v[1], "unit": v[2]} for k, v in tiles.items()},
        "rows": [
            {
                "key": r.key,
                "indicator": r.label,
                "now": r.now,
                "concerning": r.concerning,
                "status": r.status,
                "reason": r.reason,
                "as_of": r.as_of.isoformat() if r.as_of else "",
                "source": r.source,
            }
            for r in evaluate_watchlist(snapshot)
        ],
        "auto_signs": auto_scenario_signs(snapshot),
        "errors": snapshot.errors,
        "as_of": {k: v.isoformat() for k, v in snapshot.as_of_by_source.items()},
    }


# ---------------------------------------------------------------------------
# Layout pieces
# ---------------------------------------------------------------------------


def _tile(key: str, label: str) -> html.Div:
    """Build one headline statistic tile.

    Args:
        key: Tile identifier, used for the callback output id.
        label: Display label.

    Returns:
        A styled tile.
    """
    return html.Div(
        [
            html.Div(label, style={"color": MUTED, "fontSize": "11px", "marginBottom": "4px"}),
            html.Div(
                "--",
                id=f"tile-{key}-value",
                style={"color": TEXT, "fontSize": "24px", "fontWeight": "600"},
            ),
            html.Div(
                "",
                id=f"tile-{key}-change",
                style={"fontSize": "12px", "marginTop": "2px", "minHeight": "16px"},
            ),
        ],
        style={
            "backgroundColor": CARD_BG,
            "borderRadius": "6px",
            "padding": "14px 18px",
            "flex": "1 1 150px",
            "minWidth": "150px",
        },
    )


def _scenario_card(scenario: Scenario) -> html.Div:
    """Build one scenario card with its checklist.

    Args:
        scenario: Scenario definition.

    Returns:
        A styled card.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        scenario.key,
                        style={
                            "backgroundColor": ACCENT,
                            "color": "#fff",
                            "borderRadius": "4px",
                            "padding": "1px 8px",
                            "fontWeight": "700",
                            "marginRight": "8px",
                        },
                    ),
                    html.Span(scenario.name, style={"color": TEXT, "fontWeight": "600"}),
                ]
            ),
            html.Div(
                scenario.likelihood,
                style={"color": AMBER, "fontSize": "11px", "margin": "4px 0"},
            ),
            html.Div(
                scenario.description,
                style={"color": MUTED, "fontSize": "12px", "marginBottom": "8px"},
            ),
            html.Div(
                "signs met: --",
                id=f"scenario-{scenario.key}-count",
                style={"color": TEXT, "fontSize": "12px", "marginBottom": "6px"},
            ),
            dcc.Checklist(
                id=f"scenario-{scenario.key}-checklist",
                options=[
                    {
                        "label": f" {s.label}" + (" (auto)" if s.auto else ""),
                        "value": s.key,
                        "disabled": s.auto,
                    }
                    for s in scenario.signs
                ],
                value=[],
                style={"color": MUTED, "fontSize": "12px"},
                labelStyle={"display": "block", "marginBottom": "3px"},
            ),
        ],
        style={
            "backgroundColor": CARD_BG,
            "borderRadius": "6px",
            "padding": "14px",
            "flex": "1 1 300px",
            "minWidth": "280px",
        },
    )


def _key_dates_rows() -> list[html.Tr]:
    """Build the key-dates table rows.

    Returns:
        One row per configured date, greying out stale entries.
    """
    today = date.today()
    rows: list[html.Tr] = []
    for entry in KEY_DATES:
        delta = (entry.day - today).days
        stale = delta < -STALE_DAYS
        when = f"in {delta}d" if delta >= 0 else f"{-delta}d ago"
        colour = MUTED if stale or delta < 0 else TEXT
        rows.append(
            html.Tr(
                [
                    html.Td(entry.day.isoformat(), style={"padding": "4px 12px", "color": MUTED}),
                    html.Td(when, style={"padding": "4px 12px", "color": colour}),
                    html.Td(entry.label, style={"padding": "4px 12px", "color": colour}),
                    html.Td(
                        entry.note,
                        style={"padding": "4px 12px", "color": MUTED, "fontSize": "12px"},
                    ),
                ]
            )
        )
    return rows


def build_macro_layout() -> html.Div:
    """Return the dashboard layout.

    Returns:
        The root component.
    """
    manual = load_manual_inputs()
    header = html.Div(
        [
            html.Div(
                [
                    html.H2(
                        "Treasury Yield Tracker",
                        style={"color": GREEN, "margin": "0", "fontSize": "22px"},
                    ),
                    html.Div(
                        "Not yet loaded",
                        id="fetched-at",
                        style={"color": MUTED, "fontSize": "12px", "marginTop": "2px"},
                    ),
                ],
                style={"flex": "1"},
            ),
            html.Button(
                "Refresh data",
                id="refresh-btn",
                n_clicks=0,
                style={
                    "backgroundColor": ACCENT,
                    "color": "#fff",
                    "border": "none",
                    "borderRadius": "4px",
                    "padding": "8px 18px",
                    "cursor": "pointer",
                    "fontWeight": "600",
                },
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "6px"},
    )

    decomposition = html.Div(
        "",
        id="decomposition-line",
        style={"color": TEXT, "fontSize": "13px", "marginBottom": "12px"},
    )

    tiles = html.Div(
        [_tile(key, label) for key, label in TILE_SPECS],
        style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "14px"},
    )

    charts = _card(
        [
            html.Div(
                [
                    _label("Lookback"),
                    dcc.RadioItems(
                        id="lookback-radio",
                        options=LOOKBACK_OPTIONS,
                        value=365,
                        inline=True,
                        style={"color": MUTED, "fontSize": "12px"},
                        labelStyle={"marginRight": "14px"},
                    ),
                ],
                style={"marginBottom": "8px"},
            ),
            dcc.Loading(dcc.Graph(id="decomposition-chart"), type="dot", color=ACCENT),
        ]
    )

    curve_card = _card([dcc.Loading(dcc.Graph(id="curve-chart"), type="dot", color=ACCENT)])

    watchlist = _card(
        [
            html.H3("Watchlist", style={"color": TEXT, "fontSize": "16px", "marginTop": "0"}),
            dash_table.DataTable(
                id="watchlist-table",
                columns=[
                    {"name": "Indicator", "id": "indicator"},
                    {"name": "Now", "id": "now"},
                    {"name": "Getting concerning", "id": "concerning"},
                    {"name": "Status", "id": "status"},
                    {"name": "Why", "id": "reason"},
                    {"name": "As of", "id": "as_of"},
                    {"name": "Source", "id": "source"},
                ],
                data=[],
                style_as_list_view=True,
                style_header={
                    "backgroundColor": BG,
                    "color": MUTED,
                    "fontWeight": "600",
                    "border": f"1px solid {PANEL_BORDER}",
                },
                style_cell={
                    "backgroundColor": CARD_BG,
                    "color": TEXT,
                    "textAlign": "left",
                    "padding": "8px",
                    "fontSize": "13px",
                    "border": f"1px solid {PANEL_BORDER}",
                    "whiteSpace": "normal",
                    "height": "auto",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": '{status} = "alert"'},
                        "backgroundColor": "rgba(239,83,80,0.16)",
                    },
                    {
                        "if": {"filter_query": '{status} = "watch"'},
                        "backgroundColor": "rgba(255,152,0,0.13)",
                    },
                    {
                        "if": {"filter_query": '{status} = "unknown"'},
                        "color": MUTED,
                    },
                ],
            ),
            html.Div(
                "",
                id="watchlist-summary",
                style={"color": MUTED, "fontSize": "12px", "marginTop": "8px"},
            ),
        ]
    )

    manual_inputs = _card(
        [
            html.H3(
                "Manual inputs",
                style={"color": TEXT, "fontSize": "16px", "marginTop": "0"},
            ),
            html.Div(
                "These two have no free API, so they are entered by hand.",
                style={"color": MUTED, "fontSize": "12px", "marginBottom": "10px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            _label("Hyperscaler order coverage (x)"),
                            dcc.Input(
                                id="manual-hyperscaler",
                                type="number",
                                value=manual.value("hyperscaler_coverage"),
                                step=0.05,
                                style=INPUT_STYLE,
                            ),
                        ],
                        style={"flex": "1 1 220px"},
                    ),
                    html.Div(
                        [
                            _label("S&P 500 trailing EPS ($)"),
                            dcc.Input(
                                id="manual-eps",
                                type="number",
                                value=manual.value("sp500_trailing_eps"),
                                step=0.5,
                                style=INPUT_STYLE,
                            ),
                        ],
                        style={"flex": "1 1 220px"},
                    ),
                    html.Div(
                        [
                            _label("As of"),
                            dcc.DatePickerSingle(
                                id="manual-date",
                                date=date.today().isoformat(),
                                display_format="YYYY-MM-DD",
                            ),
                        ],
                        style={"flex": "0 0 auto"},
                    ),
                ],
                style={"display": "flex", "gap": "14px", "flexWrap": "wrap"},
            ),
            html.Button(
                "Save",
                id="manual-save-btn",
                n_clicks=0,
                style={
                    "backgroundColor": GREEN,
                    "color": "#fff",
                    "border": "none",
                    "borderRadius": "4px",
                    "padding": "7px 20px",
                    "cursor": "pointer",
                    "marginTop": "6px",
                },
            ),
            html.Span(
                "",
                id="manual-save-status",
                style={"color": MUTED, "fontSize": "12px", "marginLeft": "12px"},
            ),
        ]
    )

    scenarios = _card(
        [
            html.H3(
                "Scenarios that would get the Fed buying Treasuries",
                style={"color": TEXT, "fontSize": "16px", "marginTop": "0"},
            ),
            html.Div(
                [_scenario_card(s) for s in SCENARIOS],
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            ),
        ]
    )

    key_dates = _card(
        [
            html.H3("Key dates", style={"color": TEXT, "fontSize": "16px", "marginTop": "0"}),
            html.Table(_key_dates_rows(), style={"width": "100%", "borderCollapse": "collapse"}),
        ]
    )

    return html.Div(
        [
            dcc.Store(id="macro-store"),
            dcc.Interval(id="auto-refresh", interval=15 * 60 * 1000, n_intervals=0),
            header,
            html.Div(
                "",
                id="data-status",
                style={"color": AMBER, "fontSize": "12px", "marginBottom": "8px"},
            ),
            decomposition,
            tiles,
            charts,
            curve_card,
            watchlist,
            manual_inputs,
            scenarios,
            key_dates,
            html.Div(
                DISCLAIMER,
                style={
                    "color": MUTED,
                    "fontSize": "11px",
                    "marginTop": "16px",
                    "paddingTop": "10px",
                    "borderTop": f"1px solid {PANEL_BORDER}",
                },
            ),
        ],
        style={
            "backgroundColor": BG,
            "minHeight": "100vh",
            "padding": "20px 26px",
            "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        },
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_app() -> Dash:
    """Build the macro dashboard and register its callbacks.

    Returns:
        The configured Dash app.
    """
    app = Dash(__name__, title="Treasury Yield Tracker")
    app.layout = build_macro_layout()

    @app.callback(
        Output("macro-store", "data"),
        Input("refresh-btn", "n_clicks"),
        Input("auto-refresh", "n_intervals"),
        background=True,
        manager=background_manager,
        running=[(Output("refresh-btn", "disabled"), True, False)],
        prevent_initial_call=False,
    )
    def refresh(_clicks: int | None, _intervals: int | None) -> dict[str, object]:
        snapshot = build_snapshot()
        return snapshot_payload(snapshot)

    @app.callback(
        Output("fetched-at", "children"),
        Output("data-status", "children"),
        Output("decomposition-line", "children"),
        Output("watchlist-table", "data"),
        Output("watchlist-summary", "children"),
        *[Output(f"tile-{key}-value", "children") for key, _ in TILE_SPECS],
        *[Output(f"tile-{key}-change", "children") for key, _ in TILE_SPECS],
        *[Output(f"tile-{key}-change", "style") for key, _ in TILE_SPECS],
        Input("macro-store", "data"),
    )
    def render(payload: dict | None) -> tuple:  # noqa: ANN201
        if not payload:
            blanks = ["--"] * len(TILE_SPECS)
            blank_styles = [{"fontSize": "12px"}] * len(TILE_SPECS)
            return ("Loading...", "", "", [], "", *blanks, *[""] * len(TILE_SPECS), *blank_styles)

        values: list[str] = []
        changes: list[str] = []
        styles: list[dict[str, Any]] = []
        for key, _ in TILE_SPECS:
            tile = payload["tiles"].get(key, {})
            value, change, unit = tile.get("value"), tile.get("change_bp"), tile.get("unit", "%")
            values.append(
                "--" if value is None else (f"{value:+.0f}bp" if unit == "bp" else f"{value:.2f}%")
            )
            if change is None:
                changes.append("")
                styles.append({"fontSize": "12px", "color": MUTED})
            else:
                changes.append(f"{change:+.0f}bp / 5d")
                styles.append({"fontSize": "12px", "color": RED if change > 0 else GREEN})

        rows = payload["rows"]
        counts = {
            level: sum(1 for r in rows if r["status"] == level)
            for level in ("alert", "watch", "ok", "unknown")
        }
        summary = (
            f"{counts['alert']} alert, {counts['watch']} watch, "
            f"{counts['ok']} ok, {counts['unknown']} no data"
        )

        expected = payload.get("expected_short_rate")
        ten = payload["tiles"]["ten_year"].get("value")
        premium = payload["tiles"]["term_premium"].get("value")
        line = ""
        if ten is not None and expected is not None and premium is not None:
            line = (
                f"10y {ten:.2f}%  =  expected short rate {expected:.2f}%"
                f"  +  term premium {premium:.2f}%"
            )

        errors = payload.get("errors") or []
        status = f"{len(errors)} data issue(s): " + "; ".join(errors[:3]) if errors else ""

        return (
            f"fetched {payload['fetched_at']}",
            status,
            line,
            rows,
            summary,
            *values,
            *changes,
            *styles,
        )

    @app.callback(
        Output("decomposition-chart", "figure"),
        Output("curve-chart", "figure"),
        Input("macro-store", "data"),
        Input("lookback-radio", "value"),
    )
    def draw(_payload: dict | None, lookback: int | None) -> tuple:  # noqa: ANN201
        # Series are re-read from the on-disk cache rather than shipped
        # through the browser store, which keeps the payload small.
        window = lookback or 365
        snapshot = build_snapshot(include_equities=False, lookback_days=window)
        expected = None
        zero = snapshot.series.get("zero_coupon_10y")
        premium = snapshot.series.get("term_premium_10y")
        if zero is not None and premium is not None:
            expected = (zero - premium).dropna()
        decomposition = build_yield_figure(
            snapshot.curve,
            snapshot.real_curve,
            snapshot.series,
            expected_short=expected,
            lookback_days=window,
        )
        return decomposition, build_curve_snapshot_figure(snapshot.curve)

    @app.callback(
        Output("manual-save-status", "children"),
        Input("manual-save-btn", "n_clicks"),
        State("manual-hyperscaler", "value"),
        State("manual-eps", "value"),
        State("manual-date", "date"),
        prevent_initial_call=True,
    )
    def save_manual(
        _clicks: int | None,
        coverage: float | None,
        eps: float | None,
        as_of_text: str | None,
    ) -> str:
        try:
            as_of = date.fromisoformat(str(as_of_text)[:10]) if as_of_text else date.today()
        except ValueError:
            as_of = date.today()
        saved = []
        for key, value in (
            ("hyperscaler_coverage", coverage),
            ("sp500_trailing_eps", eps),
        ):
            if value is None:
                continue
            try:
                set_manual_value(key, float(value), as_of=as_of, note="entered in dashboard")
                saved.append(key)
            except (KeyError, ValueError) as error:
                return f"Could not save {key}: {error}"
        if not saved:
            return "Nothing to save"
        return f"Saved {len(saved)} value(s) at {as_of.isoformat()} -- refresh to re-evaluate"

    for scenario in SCENARIOS:
        _register_checklist(app, scenario.key)

    return app


def _register_checklist(app: Dash, scenario_key: str) -> None:
    """Wire one scenario checklist to the snapshot and the manual store.

    Automatic signs are driven by the snapshot and never written to disk;
    manual ticks are persisted.  The two key spaces are disjoint, so neither
    can overwrite the other.

    Args:
        app: Dash app to register on.
        scenario_key: Scenario identifier.
    """

    @app.callback(
        Output(f"scenario-{scenario_key}-checklist", "value"),
        Output(f"scenario-{scenario_key}-count", "children"),
        Input("macro-store", "data"),
        Input(f"scenario-{scenario_key}-checklist", "value"),
        prevent_initial_call=False,
    )
    def sync(
        payload: dict | None,
        ticked: list | None,
        scenario_key: str = scenario_key,
    ) -> tuple:  # noqa: ANN201
        scenario = next(s for s in SCENARIOS if s.key == scenario_key)
        auto = (payload or {}).get("auto_signs", {})
        manual_keys = [s.key for s in scenario.signs if not s.auto]
        manual_ticked = {k: k in (ticked or []) for k in manual_keys}
        try:
            set_checklist(scenario_key, manual_ticked)
        except OSError:
            pass  # a read-only store must not break the page
        selected = [k for k, on in manual_ticked.items() if on]
        selected += [s.key for s in scenario.signs if s.auto and auto.get(s.key)]
        return selected, f"signs met: {len(selected)}/{len(scenario.signs)}"


def main_macro_app() -> int:
    """Run the Treasury yield dashboard.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Treasury yield and macro stress dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start on the default port
  stockcharts-yields-app

  # Pick a port and skip opening a browser
  stockcharts-yields-app --port 8055 --no-browser
        """,
    )
    parser.add_argument("--port", type=int, default=8051, help="Port (default: 8051)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    parser.add_argument("--debug", action="store_true", help="Run with the Dash reloader")
    args = parser.parse_args()

    app = create_app()
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Treasury yield dashboard on {url}")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


__all__ = [
    "build_macro_layout",
    "create_app",
    "main_macro_app",
    "snapshot_payload",
]


if __name__ == "__main__":
    sys.exit(main_macro_app())
