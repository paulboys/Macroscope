from datetime import date
from unittest.mock import patch

import pandas as pd

from stockcharts.macro.config import WATCHLIST
from stockcharts.macro.manual import ManualInputs
from stockcharts.macro.snapshot import (
    MacroSnapshot,
    auto_scenario_signs,
    build_snapshot,
    evaluate_watchlist,
)


def _curve() -> pd.DataFrame:
    index = pd.date_range("2026-09-01", periods=20, freq="D")
    return pd.DataFrame(
        {
            2.0: [4.80] * 19 + [4.81],
            10.0: [5.00] * 19 + [5.17],
            30.0: [5.30] * 19 + [5.49],
        },
        index=index,
    )


def _snapshot() -> MacroSnapshot:
    index = pd.date_range("2026-09-01", periods=20, freq="D")
    snap = MacroSnapshot(fetched_at="2026-09-26T00:00:00Z", curve=_curve())
    snap.real_curve = pd.DataFrame({10.0: [2.70] * 19 + [2.83]}, index=index)
    snap.series = {
        "breakeven_10y": pd.Series([2.30] * 19 + [2.34], index=index),
        "term_premium_10y": pd.Series([0.85] * 19 + [0.96], index=index),
        "zero_coupon_10y": pd.Series([4.90] * 19 + [5.01], index=index),
        "iorb": pd.Series([3.90] * 20, index=index),
    }
    snap.auctions = pd.DataFrame(
        [
            {
                "auction_date": pd.Timestamp("2026-09-17"),
                "bid_to_cover": 2.24,
                "indirect_pct": 59.1,
                "security_term": "9-Year 10-Month",
            }
        ]
    )
    snap.sofr = pd.DataFrame({"rate": [3.88]}, index=[pd.Timestamp("2026-09-24")])
    snap.repo_ops = pd.DataFrame({"accepted_bn": [0.0]}, index=[pd.Timestamp("2026-09-25")])
    snap.brent = pd.Series([104.32], index=[pd.Timestamp("2026-09-25")])
    snap.equity = pd.Series([6100.0] * 19 + [6180.0], index=index)
    snap.manual = ManualInputs()
    return snap


# --- shape ------------------------------------------------------------------


def test_watchlist_always_returns_every_configured_row():
    rows = evaluate_watchlist(_snapshot())

    assert len(rows) == len(WATCHLIST)
    assert [r.key for r in rows] == [w.key for w in WATCHLIST]


def test_empty_snapshot_does_not_raise_and_reports_unknown():
    rows = evaluate_watchlist(MacroSnapshot())

    assert len(rows) == len(WATCHLIST)
    assert all(r.status == "unknown" for r in rows)
    assert all(r.now == "--" or "--" in r.now for r in rows)


# --- derived quantities -----------------------------------------------------


def test_expected_short_rate_is_zero_coupon_minus_term_premium():
    snap = _snapshot()

    assert round(snap.expected_short_rate(), 3) == round(5.01 - 0.96, 3)


def test_expected_short_rate_is_none_without_inputs():
    assert MacroSnapshot().expected_short_rate() is None


def test_curve_spread_is_in_basis_points():
    assert round(_snapshot().curve_spread_bp(2.0, 10.0)) == 36


def test_tenor_change_is_in_basis_points():
    assert round(_snapshot().tenor_change_bp(10.0)) == 17


def test_equity_risk_premium_uses_manual_eps():
    snap = _snapshot()
    snap.manual.entries = {}
    assert snap.equity_risk_premium() is None

    from stockcharts.macro.config import ManualEntry

    snap.manual.entries["sp500_trailing_eps"] = ManualEntry(245.50, date(2026, 6, 30))
    # 245.50 / 6180 * 100 - 5.17 = -1.20
    assert round(snap.equity_risk_premium(), 2) == -1.20


# --- row wiring -------------------------------------------------------------


def test_auction_row_reflects_the_newest_auction():
    rows = {r.key: r for r in evaluate_watchlist(_snapshot())}

    assert rows["auction"].status == "alert"
    assert "2.24" in rows["auction"].now


def test_repo_row_shows_spread_below_ceiling():
    rows = {r.key: r for r in evaluate_watchlist(_snapshot())}

    assert rows["repo_stress"].status == "ok"
    assert rows["repo_stress"].now == "-2bp"


def test_yields_row_combines_both_tenors():
    rows = {r.key: r for r in evaluate_watchlist(_snapshot())}

    assert "5.17%" in rows["yields_10y_30y"].now
    assert "5.49%" in rows["yields_10y_30y"].now


# --- auto scenario signs ----------------------------------------------------


def test_every_auto_sign_is_computed():
    """An auto sign renders as a disabled checkbox, so it must be computable.

    If a sign marked auto is missing from this mapping, the dashboard shows a
    box the user cannot tick and the dashboard never ticks either.
    """
    from stockcharts.macro.config import SCENARIOS

    signs = auto_scenario_signs(_snapshot())
    declared = {s.key for sc in SCENARIOS for s in sc.signs if s.auto}

    assert declared == set(signs)


def test_auto_signs_report_false_when_a_series_is_missing():
    # ig_oas is absent from this snapshot: report False, never omit the key.
    signs = auto_scenario_signs(_snapshot())

    assert signs["b_ig_spreads"] is False
    assert signs["a_repo_above_admin"] is False


def test_auto_sign_detects_repo_above_ceiling():
    snap = _snapshot()
    snap.sofr = pd.DataFrame({"rate": [3.96]}, index=[pd.Timestamp("2026-09-24")])

    assert auto_scenario_signs(snap)["a_repo_above_admin"] is True


# --- graceful degradation ---------------------------------------------------


def test_build_snapshot_records_errors_instead_of_raising():
    class Boom:
        def fetch_treasury_curve(self, real=False):
            raise RuntimeError("treasury down")

        def fetch_many(self, ids, start=None):
            return {}, ["DGS10: throttled"]

        def fetch_auctions(self, term, limit):
            raise RuntimeError("fiscaldata down")

        def fetch_sofr(self, limit):
            raise RuntimeError("nyfed down")

        def fetch_repo_ops(self, limit):
            raise RuntimeError("nyfed down")

    with patch("stockcharts.macro.snapshot.load_manual_inputs", return_value=ManualInputs()):
        snap = build_snapshot(client=Boom(), include_equities=False)

    assert len(snap.errors) >= 4
    assert any("treasury" in e for e in snap.errors)
    assert any("throttled" in e for e in snap.errors)
    # The snapshot is still usable and the table still renders.
    assert len(evaluate_watchlist(snap)) == len(WATCHLIST)


def test_manual_warnings_propagate_into_snapshot_errors():
    class Quiet:
        def fetch_treasury_curve(self, real=False):
            return pd.DataFrame()

        def fetch_many(self, ids, start=None):
            return {}, []

        def fetch_auctions(self, term, limit):
            return pd.DataFrame()

        def fetch_sofr(self, limit):
            return pd.DataFrame()

        def fetch_repo_ops(self, limit):
            return pd.DataFrame()

    noisy = ManualInputs(warnings=["Could not read manual_inputs.json, using defaults"])
    snap = build_snapshot(client=Quiet(), manual=noisy, include_equities=False)

    assert any("using defaults" in e for e in snap.errors)
