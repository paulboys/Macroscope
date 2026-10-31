from unittest.mock import patch

import pandas as pd
import pytest
from dash import Dash

from stockcharts.macro.config import SCENARIOS
from stockcharts.macro.manual import ManualInputs
from stockcharts.macro.snapshot import MacroSnapshot
from stockcharts.macro_app import (
    TILE_SPECS,
    build_macro_layout,
    create_app,
    snapshot_payload,
)


def _snapshot() -> MacroSnapshot:
    index = pd.date_range("2026-09-01", periods=20, freq="D")
    snap = MacroSnapshot(fetched_at="2026-09-26T00:00:00Z")
    snap.curve = pd.DataFrame(
        {2.0: [4.80] * 19 + [4.81], 10.0: [5.00] * 19 + [5.17], 30.0: [5.30] * 19 + [5.49]},
        index=index,
    )
    snap.real_curve = pd.DataFrame({10.0: [2.70] * 19 + [2.83]}, index=index)
    snap.series = {
        "breakeven_10y": pd.Series([2.30] * 19 + [2.34], index=index),
        "term_premium_10y": pd.Series([0.85] * 19 + [0.96], index=index),
        "zero_coupon_10y": pd.Series([4.90] * 19 + [5.01], index=index),
        "iorb": pd.Series([3.90] * 20, index=index),
    }
    snap.manual = ManualInputs()
    return snap


def _walk(node, acc):
    node_id = getattr(node, "id", None)
    if isinstance(node_id, str):
        acc.add(node_id)
    children = getattr(node, "children", None)
    if children is None:
        return acc
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            _walk(child, acc)
    return acc


# --- layout -----------------------------------------------------------------


def test_layout_contains_every_component_the_callbacks_target():
    found = _walk(build_macro_layout(), set())

    required = {
        "macro-store",
        "auto-refresh",
        "refresh-btn",
        "fetched-at",
        "data-status",
        "decomposition-line",
        "watchlist-table",
        "watchlist-summary",
        "decomposition-chart",
        "curve-chart",
        "lookback-radio",
        "manual-hyperscaler",
        "manual-eps",
        "manual-date",
        "manual-save-btn",
        "manual-save-status",
    }
    assert required <= found


def test_layout_has_a_tile_for_every_spec():
    found = _walk(build_macro_layout(), set())

    for key, _ in TILE_SPECS:
        assert f"tile-{key}-value" in found
        assert f"tile-{key}-change" in found


def test_layout_has_a_checklist_per_scenario():
    found = _walk(build_macro_layout(), set())

    for scenario in SCENARIOS:
        assert f"scenario-{scenario.key}-checklist" in found
        assert f"scenario-{scenario.key}-count" in found


def test_create_app_registers_all_callbacks():
    with patch("stockcharts.macro_app.build_snapshot", return_value=_snapshot()):
        app = create_app()

    assert isinstance(app, Dash)
    # refresh, render, draw, manual save, plus one sync per scenario
    assert len(app.callback_map) == 4 + len(SCENARIOS)


# --- payload ----------------------------------------------------------------


def test_payload_is_json_safe_scalars():
    import json

    payload = snapshot_payload(_snapshot())
    json.dumps(payload, allow_nan=False)  # would raise on a Series or NaN

    assert payload["tiles"]["ten_year"]["value"] == 5.17
    assert payload["tiles"]["spread_2s10s"]["unit"] == "bp"
    assert len(payload["rows"]) == 9


def test_payload_carries_every_field_the_table_columns_need():
    row = snapshot_payload(_snapshot())["rows"][0]

    for column in ("indicator", "now", "concerning", "status", "reason", "as_of", "source"):
        assert column in row


def test_payload_of_empty_snapshot_does_not_raise():
    payload = snapshot_payload(MacroSnapshot())

    assert len(payload["rows"]) == 9
    assert all(r["status"] == "unknown" for r in payload["rows"])


def test_payload_expected_short_rate_is_derived():
    payload = snapshot_payload(_snapshot())

    assert round(payload["expected_short_rate"], 2) == round(5.01 - 0.96, 2)


# --- checklist namespacing --------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.key)
def test_auto_signs_are_disabled_in_the_checklist_options(scenario):
    found = build_macro_layout()
    checklists = []

    def collect(node):
        if getattr(node, "id", None) == f"scenario-{scenario.key}-checklist":
            checklists.append(node)
        children = getattr(node, "children", None)
        if children is None:
            return
        if not isinstance(children, (list, tuple)):
            children = [children]
        for child in children:
            if hasattr(child, "children") or hasattr(child, "id"):
                collect(child)

    collect(found)
    assert checklists, f"no checklist for scenario {scenario.key}"
    options = checklists[0].options
    for option, sign in zip(options, scenario.signs, strict=False):
        # Automatic signs are computed, so the user must not toggle them.
        assert option["disabled"] is sign.auto
