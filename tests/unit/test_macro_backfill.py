import pandas as pd

from stockcharts.macro.backfill import EARLIEST_TREASURY_YEAR, backfill_history, cache_summary
from stockcharts.macro.client import MacroClient
from stockcharts.macro.series import CORE_FRED_KEYS
from tests.unit.test_macro_client import TREASURY_XML, Response


def _client(tmp_path, fail_on=None):
    requested: list[str] = []

    def opener(request, timeout):
        url = request.full_url
        requested.append(url)
        if fail_on and fail_on in url:
            raise OSError("simulated outage")
        if "treasury" in url or "field_tdr_date_value" in url:
            return Response(TREASURY_XML)
        index = pd.bdate_range("2026-09-01", "2026-09-25")
        rows = [f"{d.date()},{4.0 + i * 0.01:.3f}" for i, d in enumerate(index)]
        return Response(("\n".join(["observation_date,V", *rows])).encode())

    client = MacroClient(
        user_agent="Test App",
        opener=opener,
        request_delay=0.0,
        max_retries=1,
        cache_dir=tmp_path,
        use_cache=True,
    )
    return client, requested


def test_backfill_covers_the_whole_lookback_window(tmp_path):
    """The window, not the calendar-year count, sets the span.

    A 3y window reaches into the calendar year 3 years ago, so backfilling
    only 3 calendar years leaves the earliest one uncached and the first deep
    dashboard load still pays to fetch it.
    """
    from datetime import date, timedelta

    client, requested = _client(tmp_path)

    code = backfill_history(years=3, client=client, verbose=False)

    treasury = [u for u in requested if "field_tdr_date_value" in u]
    years = {int(u.split("value=")[1]) for u in treasury}
    assert code == 0
    assert min(years) <= (date.today() - timedelta(days=3 * 365)).year
    assert max(years) == date.today().year
    # Nominal and real are separate datasets, so two requests per year.
    assert len(treasury) == len(years) * 2


def test_backfill_covers_every_core_fred_series(tmp_path):
    client, requested = _client(tmp_path)

    backfill_history(years=2, client=client, verbose=False)

    fred = [u for u in requested if "fredgraph.csv" in u]
    assert len(fred) == len(CORE_FRED_KEYS)


def test_backfill_reports_failures_without_aborting(tmp_path):
    client, requested = _client(tmp_path, fail_on="field_tdr_date_value=2026")

    code = backfill_history(years=2, client=client, verbose=False)

    assert code == 1  # signals that something failed
    # The FRED half still ran despite the Treasury failure.
    assert any("fredgraph.csv" in u for u in requested)


def test_backfill_does_not_reach_before_the_feed_exists(tmp_path):
    client, requested = _client(tmp_path)

    backfill_history(years=200, client=client, verbose=False)

    years = {int(u.split("value=")[1]) for u in requested if "field_tdr_date_value" in u}
    assert min(years) >= EARLIEST_TREASURY_YEAR


def test_backfill_is_cheap_the_second_time(tmp_path):
    client, requested = _client(tmp_path)
    backfill_history(years=3, client=client, verbose=False)
    first = len(requested)
    client._memo.clear()

    backfill_history(years=3, client=client, verbose=False)

    # Past years and covered FRED history come from disk; only the current
    # Treasury year and the FRED tails are re-requested.
    assert len(requested) - first < first


def test_cache_summary_describes_what_is_stored(tmp_path):
    client, _ = _client(tmp_path)
    backfill_history(years=2, client=client, verbose=False)

    summary = cache_summary(client=client)

    assert summary["files"] > 0
    assert summary["size_mb"] >= 0
    assert len(summary["treasury_years"]) >= 2
    assert "manual_keys" in summary


def test_cache_summary_on_an_empty_cache(tmp_path):
    client = MacroClient(user_agent="Test App", cache_dir=tmp_path / "nothing")

    summary = cache_summary(client=client)

    assert summary["files"] == 0
    assert summary["treasury_years"] == []
