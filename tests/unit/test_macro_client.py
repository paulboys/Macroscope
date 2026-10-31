import os
import urllib.error
from datetime import date
from unittest.mock import patch

import pandas as pd

from stockcharts.macro.client import (
    MacroClient,
    parse_auctions_json,
    parse_fred_csv,
    parse_repo_ops_json,
    parse_sofr_json,
    parse_treasury_xml,
)

FRED_CSV = b"""observation_date,DGS10
2026-09-21,4.96
2026-09-22,.
2026-09-23,5.11
2026-09-24,5.18
"""

# FRED emitted this header for years; positional parsing must survive it.
FRED_CSV_LEGACY = b"""DATE,DGS10
2026-09-23,5.11
2026-09-24,5.18
"""

TREASURY_XML = b"""<?xml version="1.0" encoding="utf-8" standalone="yes" ?>
<feed xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
      xmlns="http://www.w3.org/2005/Atom">
<entry><content type="application/xml">
<m:properties>
<d:NEW_DATE m:type="Edm.DateTime">2026-01-02T00:00:00</d:NEW_DATE>
<d:BC_2YEAR m:type="Edm.Double">3.47</d:BC_2YEAR>
<d:BC_4MONTH m:type="Edm.Double"></d:BC_4MONTH>
<d:BC_10YEAR m:type="Edm.Double">4.19</d:BC_10YEAR>
<d:BC_30YEAR m:type="Edm.Double">4.86</d:BC_30YEAR>
</m:properties>
</content></entry>
<entry><content type="application/xml">
<m:properties>
<d:NEW_DATE m:type="Edm.DateTime">2026-09-24T00:00:00</d:NEW_DATE>
<d:BC_2YEAR m:type="Edm.Double">4.10</d:BC_2YEAR>
<d:BC_10YEAR m:type="Edm.Double">5.18</d:BC_10YEAR>
<d:BC_30YEAR m:type="Edm.Double">5.45</d:BC_30YEAR>
</m:properties>
</content></entry>
</feed>
"""

# The Sept 9 10-year reopening, filed under a 9-Year 11-Month term.
AUCTIONS_JSON = b"""{"data":[
 {"auction_date":"2026-09-09","security_term":"9-Year 11-Month",
  "original_security_term":"10-Year","bid_to_cover_ratio":"2.710000",
  "comp_accepted":"36000000000","indirect_bidder_accepted":"28512000000",
  "high_yield":"4.0330","offering_amt":"39000000000"}
]}"""

SOFR_JSON = b"""{"refRates":[
 {"effectiveDate":"2026-09-24","type":"SOFR","percentRate":3.88,"volumeInBillions":2990},
 {"effectiveDate":"2026-09-23","type":"SOFR","percentRate":3.87,"volumeInBillions":2950}
]}"""

REPO_JSON = b"""{"repo":{"operations":[
 {"operationId":"RP 092526 25","operationDate":"2026-09-25","operationType":"Repo",
  "term":"Overnight","totalAmtSubmitted":0,"totalAmtAccepted":0},
 {"operationId":"RRP 092526 1","operationDate":"2026-09-25","operationType":"Reverse Repo",
  "term":"Overnight","totalAmtSubmitted":0,"totalAmtAccepted":123000000000}
]}}"""


class Response:
    """Minimal urlopen-compatible response for the injected opener."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        """Return the canned body."""
        return self.payload

    def close(self) -> None:
        """Match the urlopen contract."""
        return None


# --- FRED CSV ---------------------------------------------------------------


def test_parse_fred_csv_reads_values_and_drops_missing():
    series = parse_fred_csv(FRED_CSV, "DGS10")

    assert len(series) == 3  # the "." row is dropped
    assert series.iloc[-1] == 5.18
    assert series.index[-1] == pd.Timestamp("2026-09-24")
    assert series.name == "DGS10"


def test_parse_fred_csv_handles_legacy_date_header():
    series = parse_fred_csv(FRED_CSV_LEGACY, "DGS10")

    assert len(series) == 2
    assert series.iloc[-1] == 5.18


def test_parse_fred_csv_handles_empty_body():
    assert parse_fred_csv(b"", "DGS10").empty
    assert parse_fred_csv(b"observation_date,DGS10\n", "DGS10").empty


# --- Treasury XML -----------------------------------------------------------


def test_parse_treasury_xml_builds_curve():
    curve = parse_treasury_xml(TREASURY_XML)

    assert list(curve.index) == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-09-24")]
    assert curve.loc[pd.Timestamp("2026-09-24"), 10.0] == 5.18
    assert curve.loc[pd.Timestamp("2026-09-24"), 30.0] == 5.45


def test_parse_treasury_xml_blank_tenor_is_missing_not_zero():
    curve = parse_treasury_xml(TREASURY_XML)

    # BC_4MONTH was blank on the first row: absent or NaN, never 0.0.
    if 4 / 12 in curve.columns:
        assert pd.isna(curve.loc[pd.Timestamp("2026-01-02"), 4 / 12])


def test_parse_treasury_xml_handles_garbage():
    assert parse_treasury_xml(b"").empty
    assert parse_treasury_xml(b"<not-xml").empty


# --- Auctions ---------------------------------------------------------------


def test_parse_auctions_uses_competitive_accepted_for_indirect_share():
    frame = parse_auctions_json(AUCTIONS_JSON)
    row = frame.iloc[0]

    assert row["bid_to_cover"] == 2.71
    # 28.512bn / 36.0bn = 79.2%, the convention quoted in market commentary.
    assert round(row["indirect_pct"], 1) == 79.2
    assert row["original_security_term"] == "10-Year"


def test_parse_auctions_handles_empty():
    assert parse_auctions_json(b"").empty
    assert parse_auctions_json(b'{"data":[]}').empty


# --- New York Fed -----------------------------------------------------------


def test_parse_sofr_json():
    frame = parse_sofr_json(SOFR_JSON)

    assert frame.loc[pd.Timestamp("2026-09-24"), "rate"] == 3.88
    assert frame.index.is_monotonic_increasing


def test_parse_repo_ops_excludes_reverse_repo():
    frame = parse_repo_ops_json(REPO_JSON)

    assert len(frame) == 1  # the reverse repo row is excluded
    assert frame.iloc[0]["accepted_bn"] == 0.0


# --- Client transport -------------------------------------------------------


def _client(opener, **kwargs):
    kwargs.setdefault("request_delay", 0.0)
    kwargs.setdefault("use_cache", False)
    return MacroClient(user_agent="Test App test@example.com", opener=opener, **kwargs)


def test_client_sends_user_agent_and_builds_fred_url():
    calls: list[tuple[str, str]] = []

    def opener(request, timeout):
        calls.append((request.full_url, request.get_header("User-agent")))
        return Response(FRED_CSV)

    series = _client(opener).fetch_series("DGS10", start=date(2026, 9, 1))

    assert series.iloc[-1] == 5.18
    assert "id=DGS10" in calls[0][0]
    assert "cosd=2026-09-01" in calls[0][0]
    assert calls[0][1] == "Test App test@example.com"


def test_client_sends_no_accept_header():
    """FRED's CSV endpoint hangs when one is present.

    It accepts the connection and then never responds -- for any value,
    including the ``*/*`` that used to be sent here and a correct
    ``text/csv``. That cost 63 seconds of retries per series and made the
    macro snapshot unusable, while the same URL fetched in 0.2 seconds with
    the header omitted. Omitting it is semantically identical under HTTP, and
    the Treasury, FiscalData and NY Fed endpoints return byte-identical
    responses either way.
    """
    seen: list[str | None] = []

    def opener(request, timeout):
        seen.append(request.get_header("Accept"))
        return Response(FRED_CSV)

    _client(opener).fetch_series("DGS10", start=date(2026, 9, 1))

    assert seen == [None]


def test_client_retries_on_throttling_then_succeeds():
    attempts = {"n": 0}

    def opener(request, timeout):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)
        return Response(FRED_CSV)

    with patch("stockcharts.macro.client.time.sleep") as sleep:
        series = _client(opener).fetch_series("DGS10")

    assert attempts["n"] == 3
    assert series.iloc[-1] == 5.18
    assert sleep.called  # backed off between attempts


def test_client_does_not_retry_on_404():
    attempts = {"n": 0}

    def opener(request, timeout):
        attempts["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    try:
        _client(opener).fetch_series("NOPE")
    except urllib.error.HTTPError as error:
        assert error.code == 404
    else:  # pragma: no cover
        raise AssertionError("expected HTTPError")
    assert attempts["n"] == 1


def test_client_paces_requests():
    def opener(request, timeout):
        return Response(FRED_CSV)

    client = _client(opener, request_delay=5.0)
    with patch("stockcharts.macro.client.time.sleep") as sleep:
        client.fetch_series("DGS10")
        client._memo.clear()  # force a second real request
        client.fetch_series("DGS2")

    assert sleep.called


def test_fetch_many_collects_errors_without_raising():
    def opener(request, timeout):
        if "BAD" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
        return Response(FRED_CSV)

    results, errors = _client(opener).fetch_many(["DGS10", "BAD", "DGS2"])

    assert set(results) == {"DGS10", "DGS2"}
    assert len(errors) == 1
    assert "BAD" in errors[0]


def test_cache_round_trip_avoids_second_request(tmp_path):
    calls = {"n": 0}

    def opener(request, timeout):
        calls["n"] += 1
        return Response(FRED_CSV)

    client = MacroClient(
        user_agent="Test App",
        opener=opener,
        request_delay=0.0,
        cache_dir=tmp_path,
        use_cache=True,
    )
    first = client.fetch_series("DGS10")
    client._memo.clear()  # defeat the in-memory memo, leaving only the disk cache
    second = client.fetch_series("DGS10")

    assert calls["n"] == 1
    assert first.iloc[-1] == second.iloc[-1] == 5.18


def test_auction_query_filters_on_original_security_term(tmp_path):
    calls: list[str] = []

    def opener(request, timeout):
        calls.append(request.full_url)
        return Response(AUCTIONS_JSON)

    _client(opener).fetch_auctions(term="10-Year")

    # Reopenings carry terms like "9-Year 11-Month", so filtering on
    # security_term would miss them entirely.
    assert "original_security_term" in calls[0]
    assert "security_term:eq:10-Year" in calls[0].replace("%20", " ")


TREASURY_XML_2025 = TREASURY_XML.replace(b"2026-", b"2025-")


def test_treasury_history_spans_multiple_calendar_years(tmp_path):
    """The feed is paginated by year, so a 2y window needs several requests."""
    requested: list[str] = []

    def opener(request, timeout):
        requested.append(request.full_url)
        payload = TREASURY_XML_2025 if "value=2025" in request.full_url else TREASURY_XML
        return Response(payload)

    client = MacroClient(
        user_agent="Test App",
        opener=opener,
        request_delay=0.0,
        cache_dir=tmp_path,
        use_cache=False,
    )
    frame = client.fetch_treasury_history(lookback_days=730, end=date(2026, 9, 25))

    years = {int(u.split("value=")[1]) for u in requested}
    assert years == {2024, 2025, 2026}
    # Rows from more than one year end up in one frame, oldest first.
    assert frame.index.min().year < 2026
    assert frame.index.is_monotonic_increasing


def test_treasury_history_survives_one_missing_year(tmp_path):
    def opener(request, timeout):
        if "value=2025" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 500, "boom", {}, None)
        return Response(TREASURY_XML)

    client = MacroClient(
        user_agent="Test App",
        opener=opener,
        request_delay=0.0,
        max_retries=1,
        cache_dir=tmp_path,
        use_cache=False,
    )
    frame = client.fetch_treasury_history(lookback_days=730, end=date(2026, 9, 25))

    assert not frame.empty  # the other years still come through


def test_completed_years_are_cached_indefinitely(tmp_path):
    calls = {"n": 0}

    def opener(request, timeout):
        calls["n"] += 1
        return Response(TREASURY_XML)

    client = MacroClient(
        user_agent="Test App",
        opener=opener,
        request_delay=0.0,
        cache_dir=tmp_path,
        use_cache=True,
    )
    client.fetch_treasury_curve(year=2024)
    client._memo.clear()
    # Backdate the cache well past the six-hour window.
    cached = tmp_path / "treasury_nominal_2024.parquet"
    os.utime(cached, (0, 0))
    client.fetch_treasury_curve(year=2024)

    assert calls["n"] == 1  # a finished year never needs re-fetching


def _csv_for(start: str, end: str = "2026-09-25", begins: str | None = None) -> bytes:
    """Build a FRED-shaped CSV, optionally starting later than requested."""
    first = max(pd.Timestamp(start), pd.Timestamp(begins)) if begins else pd.Timestamp(start)
    index = pd.bdate_range(first, end)
    rows = [f"{d.date()},{4.0 + i * 0.001:.3f}" for i, d in enumerate(index)]
    return ("\n".join(["observation_date,X", *rows])).encode()


def _counting_client(tmp_path, begins=None):
    requested: list[str] = []

    def opener(request, timeout):
        cosd = request.full_url.split("cosd=")[1][:10]
        requested.append(cosd)
        return Response(_csv_for(cosd, begins=begins))

    client = MacroClient(
        user_agent="Test App",
        opener=opener,
        request_delay=0.0,
        cache_dir=tmp_path,
        use_cache=True,
    )
    return client, requested


def test_cache_accumulates_history_when_a_longer_window_is_asked_for(tmp_path):
    client, requested = _counting_client(tmp_path)

    short = client.fetch_series("X", start=date(2025, 9, 25))
    client._memo.clear()
    long = client.fetch_series("X", start=date(2016, 9, 25))

    # A deeper request must not be answered from the shallower cache.
    assert len(long) > len(short) * 5
    assert long.index.min() < pd.Timestamp("2017-01-01")
    assert requested == ["2025-09-25", "2016-09-25"]


def test_shorter_window_after_a_deep_fetch_hits_the_cache(tmp_path):
    client, requested = _counting_client(tmp_path)

    client.fetch_series("X", start=date(2016, 9, 25))
    client._memo.clear()
    trimmed = client.fetch_series("X", start=date(2025, 9, 25))

    assert len(requested) == 1  # no second download
    assert trimmed.index.min() >= pd.Timestamp("2025-09-25")


def test_stale_cache_tops_up_only_the_tail(tmp_path):
    client, requested = _counting_client(tmp_path)
    client.fetch_series("X", start=date(2016, 9, 25))
    client._memo.clear()
    os.utime(tmp_path / "fred_X.parquet", (0, 0))  # force staleness

    refreshed = client.fetch_series("X", start=date(2016, 9, 25))

    # The second request starts near today, not back at 2016.
    assert pd.Timestamp(requested[-1]) > pd.Timestamp("2026-01-01")
    assert refreshed.index.min() < pd.Timestamp("2017-01-01")  # history retained


def test_coverage_is_tracked_by_request_not_by_first_observation(tmp_path):
    # A weekend start never matches an observation; a series like IORB simply
    # begins later. Neither may force a permanent full re-download.
    client, requested = _counting_client(tmp_path, begins="2021-07-29")
    client.fetch_series("X", start=date(2016, 9, 25))
    client._memo.clear()
    os.utime(tmp_path / "fred_X.parquet", (0, 0))

    client.fetch_series("X", start=date(2016, 9, 25))

    assert pd.Timestamp(requested[-1]) > pd.Timestamp("2026-01-01")


def test_covered_start_never_narrows(tmp_path):
    import json

    client, _ = _counting_client(tmp_path)
    client.fetch_series("X", start=date(2016, 9, 25))
    client._memo.clear()
    client.fetch_series("X", start=date(2025, 9, 25))

    meta = json.loads((tmp_path / "fred_X.meta.json").read_text(encoding="utf-8"))
    assert meta["covered_start"] == "2016-09-25"


def test_timeouts_are_retried(tmp_path):
    """A read timeout raises TimeoutError, not a URLError subclass.

    Catching only URLError meant a read timeout aborted with no retry, which
    lost roughly a quarter of the years in a Treasury backfill.
    """
    attempts = {"n": 0}

    def opener(request, timeout):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise TimeoutError("read timed out")
        return Response(FRED_CSV)

    client = MacroClient(
        user_agent="Test App",
        opener=opener,
        request_delay=0.0,
        cache_dir=tmp_path,
        use_cache=False,
    )
    with patch("stockcharts.macro.client.time.sleep"):
        series = client.fetch_series("DGS10")

    assert attempts["n"] == 3
    assert series.iloc[-1] == 5.18


def test_connection_errors_are_retried(tmp_path):
    attempts = {"n": 0}

    def opener(request, timeout):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionResetError("reset by peer")
        return Response(FRED_CSV)

    client = MacroClient(
        user_agent="Test App",
        opener=opener,
        request_delay=0.0,
        cache_dir=tmp_path,
        use_cache=False,
    )
    with patch("stockcharts.macro.client.time.sleep"):
        client.fetch_series("DGS10")

    assert attempts["n"] == 2


def test_treasury_requests_get_a_longer_timeout(tmp_path):
    from stockcharts.macro.client import TREASURY_TIMEOUT

    seen: list[float] = []

    def opener(request, timeout):
        seen.append(timeout)
        return Response(TREASURY_XML)

    client = MacroClient(
        user_agent="Test App",
        opener=opener,
        request_delay=0.0,
        timeout=20.0,
        cache_dir=tmp_path,
        use_cache=False,
    )
    client.fetch_treasury_curve(year=2026)

    assert seen == [TREASURY_TIMEOUT]
    assert TREASURY_TIMEOUT > 20.0
