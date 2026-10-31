from datetime import date

from stockcharts.macro.config import (
    KEY_DATES,
    MANUAL_DEFAULTS,
    SCENARIOS,
    WATCHLIST,
    scenario_sign_keys,
    watchlist_keys,
    worst_status,
)
from stockcharts.macro.series import SERIES, TENOR_FIELDS, fred_series_id


def test_watchlist_has_nine_rows():
    assert len(WATCHLIST) == 9


def test_watchlist_keys_are_unique():
    keys = watchlist_keys()
    assert len(keys) == len(set(keys))


def test_every_watchlist_series_key_exists_in_catalog():
    for row in WATCHLIST:
        if row.series_key is not None:
            assert row.series_key in SERIES, f"{row.key} -> unknown series {row.series_key}"


def test_alert_is_more_severe_than_watch_in_threshold_direction():
    for row in WATCHLIST:
        if row.watch is None or row.alert is None:
            continue
        if row.rule in {"term_premium", "yields"}:
            continue  # these compare changes, not levels
        if row.direction == "above":
            assert row.alert >= row.watch, row.key
        else:
            assert row.alert <= row.watch, row.key


def test_scenarios_cover_all_four_paths():
    assert [s.key for s in SCENARIOS] == ["A", "B", "C", "D"]


def test_every_scenario_has_at_least_four_signs():
    for scenario in SCENARIOS:
        assert len(scenario.signs) >= 4, scenario.key


def test_scenario_sign_keys_are_globally_unique():
    keys = scenario_sign_keys()
    assert len(keys) == len(set(keys))


def test_auto_and_manual_sign_namespaces_are_disjoint():
    auto = {s.key for sc in SCENARIOS for s in sc.signs if s.auto}
    manual = {s.key for sc in SCENARIOS for s in sc.signs if not s.auto}
    assert auto.isdisjoint(manual)


def test_key_dates_are_sorted():
    days = [d.day for d in KEY_DATES]
    assert days == sorted(days)


def test_key_dates_are_real_dates():
    for entry in KEY_DATES:
        assert isinstance(entry.day, date)


def test_manual_defaults_cover_every_manual_series():
    manual_keys = {k for k, spec in SERIES.items() if spec.source == "manual"}
    assert manual_keys <= set(MANUAL_DEFAULTS)


def test_fred_series_id_resolves():
    assert fred_series_id("breakeven_10y") == "T10YIE"
    assert fred_series_id("term_premium_10y") == "THREEFYTP10"


def test_fred_series_id_rejects_non_fred_series():
    try:
        fred_series_id("brent")
    except ValueError as error:
        assert "not a FRED series" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_tenor_fields_are_ascending():
    values = list(TENOR_FIELDS.values())
    assert values == sorted(values)


def test_worst_status_picks_most_severe():
    assert worst_status(["ok", "watch", "alert"]) == "alert"
    assert worst_status(["ok", "watch"]) == "watch"
    assert worst_status(["ok", "unknown"]) == "ok"
    assert worst_status([]) == "unknown"
