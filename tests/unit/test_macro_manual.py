import json
from datetime import date

import pytest

from stockcharts.macro.config import MANUAL_DEFAULTS
from stockcharts.macro.manual import (
    ManualInputs,
    default_manual_path,
    load_manual_inputs,
    save_manual_inputs,
    set_checklist,
    set_manual_value,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "manual_inputs.json"
    monkeypatch.setenv("STOCKCHARTS_MACRO_DATA", str(path))
    return path


def test_missing_file_returns_defaults(store):
    inputs = load_manual_inputs()

    assert not store.exists()
    assert inputs.value("hyperscaler_coverage") == MANUAL_DEFAULTS["hyperscaler_coverage"].value


def test_save_and_load_round_trip(store):
    inputs = load_manual_inputs()
    set_manual_value("hyperscaler_coverage", 1.35, as_of=date(2026, 9, 20), note="July books")

    reloaded = load_manual_inputs()
    assert reloaded.value("hyperscaler_coverage") == 1.35
    assert reloaded.as_of("hyperscaler_coverage") == date(2026, 9, 20)
    assert reloaded.entries["hyperscaler_coverage"].note == "July books"
    # The untouched default survives the partial update.
    assert reloaded.value("sp500_trailing_eps") == inputs.value("sp500_trailing_eps")


def test_env_var_controls_location(store):
    set_manual_value("hyperscaler_coverage", 1.1)

    assert store.exists()
    assert default_manual_path() == store


def test_unknown_key_is_rejected(store):
    with pytest.raises(KeyError):
        set_manual_value("not_a_real_key", 1.0)


def test_corrupt_json_falls_back_to_defaults_with_a_warning(store):
    store.write_text("{not json at all", encoding="utf-8")

    inputs = load_manual_inputs()

    assert inputs.value("hyperscaler_coverage") is not None
    assert inputs.warnings
    assert "using defaults" in inputs.warnings[0]


def test_unknown_stored_key_is_ignored_not_fatal(store):
    store.write_text(
        json.dumps({"schema": 1, "entries": {"bogus": {"value": 1, "as_of": "2026-01-01"}}}),
        encoding="utf-8",
    )

    inputs = load_manual_inputs()

    assert "bogus" not in inputs.entries
    assert any("bogus" in w for w in inputs.warnings)


def test_malformed_entry_is_ignored_not_fatal(store):
    store.write_text(
        json.dumps({"schema": 1, "entries": {"hyperscaler_coverage": {"value": "abc"}}}),
        encoding="utf-8",
    )

    inputs = load_manual_inputs()

    # Falls back to the default rather than raising.
    assert inputs.value("hyperscaler_coverage") == MANUAL_DEFAULTS["hyperscaler_coverage"].value
    assert inputs.warnings


def test_atomic_write_leaves_no_temp_file(store):
    set_manual_value("hyperscaler_coverage", 1.4)

    leftovers = list(store.parent.glob(".manual_*"))
    assert leftovers == []


def test_checklist_round_trip_and_unknown_signs_dropped(store):
    set_checklist("A", {"a_wide_bid_ask": True, "not_a_sign": True})

    inputs = load_manual_inputs()

    assert inputs.is_ticked("A", "a_wide_bid_ask") is True
    assert inputs.is_ticked("A", "not_a_sign") is False
    assert "not_a_sign" not in inputs.checklists["A"]


def test_is_ticked_defaults_false_for_untouched_scenario(store):
    assert load_manual_inputs().is_ticked("D", "d_ycc_talk") is False


def test_saved_file_is_valid_json_with_schema(store):
    save_manual_inputs(ManualInputs(entries=dict(MANUAL_DEFAULTS)))

    payload = json.loads(store.read_text(encoding="utf-8"))
    assert payload["schema"] == 1
    assert "updated_at" in payload
    assert "hyperscaler_coverage" in payload["entries"]
