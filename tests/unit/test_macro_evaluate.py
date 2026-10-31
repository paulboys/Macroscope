import pytest

from stockcharts.macro.config import WATCHLIST, Threshold
from stockcharts.macro.evaluate import (
    evaluate_auction,
    evaluate_erp,
    evaluate_level,
    evaluate_move,
    evaluate_repo_stress,
    evaluate_risk_off_yield_jump,
    evaluate_term_premium,
    evaluate_yields,
)

ROWS = {row.key: row for row in WATCHLIST}


def _spec(key):
    return ROWS[key]


# --- evaluate_level ---------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(2.80, "ok"), (2.90, "watch"), (2.95, "watch"), (3.00, "alert"), (3.20, "alert")],
)
def test_evaluate_level_above_direction_boundaries(value, expected):
    status, _ = evaluate_level(value, _spec("real_10y"))
    assert status == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(2.00, "ok"), (1.50, "watch"), (1.30, "watch"), (1.20, "alert"), (1.00, "alert")],
)
def test_evaluate_level_below_direction_boundaries(value, expected):
    status, _ = evaluate_level(value, _spec("hyperscaler_coverage"))
    assert status == expected


def test_evaluate_level_without_data():
    assert evaluate_level(None, _spec("real_10y")) == ("unknown", "no data")


def test_evaluate_level_tolerates_missing_thresholds():
    spec = Threshold(
        key="x",
        label="x",
        rule="level",
        direction="above",
        watch=None,
        alert=None,
        units="pct",
        concerning="",
        source="fred",
    )
    status, reason = evaluate_level(1.23, spec)
    assert status == "ok"
    assert isinstance(reason, str)


# --- evaluate_move ----------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "expected"),
    [(14.0, "ok"), (15.0, "watch"), (24.0, "watch"), (25.0, "alert"), (-26.0, "alert")],
)
def test_evaluate_move_boundaries(change, expected):
    status, _ = evaluate_move(change)
    assert status == expected


def test_evaluate_move_is_direction_agnostic():
    assert evaluate_move(30.0)[0] == evaluate_move(-30.0)[0] == "alert"


# --- evaluate_yields --------------------------------------------------------


def test_evaluate_yields_flags_fast_move_even_at_safe_levels():
    status, reason = evaluate_yields(4.50, 4.70, 40.0, _spec("yields_10y_30y"))
    assert status == "alert"
    assert "40bp" in reason


def test_evaluate_yields_flags_high_level_without_a_move():
    status, reason = evaluate_yields(5.60, 5.80, 2.0, _spec("yields_10y_30y"))
    assert status == "alert"
    assert "10y 5.60" in reason


def test_evaluate_yields_quiet_market_is_ok():
    status, _ = evaluate_yields(4.50, 4.70, 3.0, _spec("yields_10y_30y"))
    assert status == "ok"


def test_evaluate_yields_without_any_data():
    assert evaluate_yields(None, None, None, _spec("yields_10y_30y"))[0] == "unknown"


# --- evaluate_term_premium --------------------------------------------------


def test_term_premium_rising_while_expected_short_falls_is_alert():
    status, reason = evaluate_term_premium(1.20, 15.0, -10.0)
    assert status == "alert"
    assert "expected short rates fell" in reason


def test_term_premium_rising_with_short_rates_is_only_watch():
    status, _ = evaluate_term_premium(1.20, 15.0, 20.0)
    assert status == "watch"


def test_term_premium_flat_is_ok():
    assert evaluate_term_premium(0.96, 2.0, 5.0)[0] == "ok"


def test_term_premium_without_change_is_unknown():
    assert evaluate_term_premium(0.96, None, -10.0)[0] == "unknown"


def test_term_premium_rising_with_unknown_short_rate_does_not_escalate():
    assert evaluate_term_premium(1.20, 15.0, None)[0] == "watch"


# --- evaluate_auction -------------------------------------------------------


def test_auction_weak_cover_and_indirect_is_alert():
    # The 2026-09-17 10y reopening: 2.24x cover, 59.1% indirect.
    status, reason = evaluate_auction(2.24, 59.1, _spec("auction"))
    assert status == "alert"
    assert "cover" in reason and "indirect" in reason


def test_auction_strong_result_is_ok():
    # The 2026-09-09 10y reopening: 2.71x cover, 79.2% indirect.
    assert evaluate_auction(2.71, 79.2, _spec("auction"))[0] == "ok"


def test_auction_takes_worst_of_two_signals():
    # Healthy cover but collapsing indirect share still alerts.
    assert evaluate_auction(2.90, 65.0, _spec("auction"))[0] == "alert"


def test_auction_without_data():
    assert evaluate_auction(None, None, _spec("auction"))[0] == "unknown"


# --- evaluate_repo_stress ---------------------------------------------------


@pytest.mark.parametrize(
    ("sofr", "iorb", "srf", "expected"),
    [
        (3.88, 3.90, 0.0, "ok"),  # below ceiling, no facility use
        (3.90, 3.90, 0.0, "watch"),  # exactly at the ceiling
        (3.96, 3.90, 0.0, "alert"),  # +6bp through the ceiling
        (3.85, 3.90, 30.0, "alert"),  # calm spread but heavy facility use
        (3.85, 3.90, 10.0, "watch"),
    ],
)
def test_repo_stress_matrix(sofr, iorb, srf, expected):
    status, _ = evaluate_repo_stress(sofr, iorb, srf)
    assert status == expected


def test_repo_stress_without_data():
    assert evaluate_repo_stress(None, None, None)[0] == "unknown"


# --- evaluate_erp -----------------------------------------------------------


def test_erp_negative_with_selloff_and_rising_yields_is_alert():
    status, reason = evaluate_erp(-1.21, -4.0, 20.0)
    assert status == "alert"
    assert "no hedge" in reason


def test_erp_negative_without_selloff_is_only_watch():
    assert evaluate_erp(-1.21, 1.0, 20.0)[0] == "watch"


def test_erp_negative_with_selloff_but_falling_yields_is_watch():
    # Bonds are still hedging, so this is not the dysfunction case.
    assert evaluate_erp(-1.21, -4.0, -15.0)[0] == "watch"


def test_erp_healthy_cushion_is_ok():
    assert evaluate_erp(1.50, -4.0, 20.0)[0] == "ok"


def test_erp_without_data():
    assert evaluate_erp(None, -4.0, 20.0)[0] == "unknown"


# --- evaluate_risk_off_yield_jump -------------------------------------------


def test_risk_off_jump_alerts_on_big_selloff_with_rising_yields():
    assert evaluate_risk_off_yield_jump(20.0, -4.0)[0] == "alert"


def test_risk_off_jump_watches_small_divergence():
    assert evaluate_risk_off_yield_jump(5.0, -1.0)[0] == "watch"


def test_risk_off_jump_ok_when_bonds_rally_on_a_selloff():
    status, reason = evaluate_risk_off_yield_jump(-20.0, -4.0)
    assert status == "ok"
    assert "hedging" in reason


def test_risk_off_jump_without_data():
    assert evaluate_risk_off_yield_jump(None, -4.0)[0] == "unknown"
