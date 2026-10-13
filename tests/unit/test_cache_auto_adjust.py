"""Adjusted and raw OHLC must never share a cache entry.

The cache keyed on ticker + interval only, so a series fetched with
auto_adjust=False and one fetched with auto_adjust=True wrote to the same
Parquet file. Whichever landed first was then returned for both, with no
error -- and a stale-cache incremental update would splice adjusted bars onto
raw history, giving a frame that changed basis part-way through.
"""

from pathlib import Path

import pandas as pd

from stockcharts.data.cache import (
    _cache_path,
    _is_cache_fresh,
    _parse_stem,
    cache_stats,
    clear_cache,
    load_cached,
    save_cache,
)


def _frame(close=100.0, periods=10):
    idx = pd.bdate_range("2024-01-01", periods=periods)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )


# --- path construction ------------------------------------------------------


def test_adjusted_and_raw_get_different_paths(tmp_path):
    raw = _cache_path("SPY", "1d", tmp_path, auto_adjust=False)
    adj = _cache_path("SPY", "1d", tmp_path, auto_adjust=True)

    assert raw != adj


def test_raw_path_is_unchanged(tmp_path):
    # Existing caches stay valid: the default name must not move.
    assert _cache_path("SPY", "1d", tmp_path).name == "SPY_1d.parquet"


def test_parse_stem_roundtrips():
    for ticker, interval, adjusted in (
        ("SPY", "1d", False),
        ("SPY", "1d", True),
        ("BRK_B", "1wk", True),
    ):
        stem = _cache_path(ticker, interval, Path("."), adjusted).stem
        assert _parse_stem(stem) == (ticker, interval, adjusted)


# --- the bug --------------------------------------------------------------


def test_raw_cache_is_not_served_as_adjusted(tmp_path):
    save_cache("SPY", _frame(close=100.0), "1d", tmp_path, auto_adjust=False)

    assert load_cached("SPY", "1d", tmp_path, auto_adjust=True) is None


def test_adjusted_cache_is_not_served_as_raw(tmp_path):
    save_cache("SPY", _frame(close=95.0), "1d", tmp_path, auto_adjust=True)

    assert load_cached("SPY", "1d", tmp_path, auto_adjust=False) is None


def test_both_series_coexist(tmp_path):
    save_cache("SPY", _frame(close=100.0), "1d", tmp_path, auto_adjust=False)
    save_cache("SPY", _frame(close=95.0), "1d", tmp_path, auto_adjust=True)

    raw = load_cached("SPY", "1d", tmp_path, auto_adjust=False)
    adj = load_cached("SPY", "1d", tmp_path, auto_adjust=True)

    assert raw is not None and adj is not None
    assert raw["Close"].iloc[0] == 100.0
    assert adj["Close"].iloc[0] == 95.0


def test_freshness_is_per_adjustment(tmp_path):
    save_cache("SPY", _frame(), "1d", tmp_path, auto_adjust=False)

    # The raw file exists and was just written; the adjusted one does not.
    assert _is_cache_fresh("SPY", "1d", tmp_path, auto_adjust=False) is True
    assert _is_cache_fresh("SPY", "1d", tmp_path, auto_adjust=True) is False


# --- housekeeping over two files per ticker --------------------------------


def test_clear_cache_removes_adjusted_files(tmp_path):
    save_cache("SPY", _frame(), "1d", tmp_path, auto_adjust=False)
    save_cache("SPY", _frame(), "1d", tmp_path, auto_adjust=True)

    # Interval filtering used stem.endswith("_1d"), which an "_adj" suffix
    # defeated, so adjusted files survived a clear.
    assert clear_cache("SPY", "1d", tmp_path) == 2
    assert list(tmp_path.glob("*.parquet")) == []


def test_clear_cache_can_target_one_basis(tmp_path):
    save_cache("SPY", _frame(), "1d", tmp_path, auto_adjust=False)
    save_cache("SPY", _frame(), "1d", tmp_path, auto_adjust=True)

    assert clear_cache("SPY", "1d", tmp_path, auto_adjust=True) == 1
    assert load_cached("SPY", "1d", tmp_path, auto_adjust=False) is not None
    assert load_cached("SPY", "1d", tmp_path, auto_adjust=True) is None


def test_cache_stats_counts_each_ticker_once(tmp_path):
    save_cache("SPY", _frame(), "1d", tmp_path, auto_adjust=False)
    save_cache("SPY", _frame(), "1d", tmp_path, auto_adjust=True)

    stats = cache_stats(tmp_path)

    assert stats["files"] == 2
    assert stats["tickers"] == 1


# --- the flag actually reaches the cache ------------------------------------


def test_fetch_ohlc_keys_the_fresh_cache_path_on_auto_adjust(monkeypatch):
    """The fast path must look up the adjustment it was asked for.

    This is the exact route the bug took: a fresh cache entry was returned
    before auto_adjust was ever consulted, so a request for adjusted prices
    got raw ones and the caller had no way to tell.
    """
    seen = {}

    def fake_load_cached(ticker, interval="1d", cache_dir=None, auto_adjust=False):
        seen["load"] = auto_adjust
        return _frame()

    def fake_is_fresh(ticker, interval="1d", cache_dir=None, auto_adjust=False):
        seen["fresh"] = auto_adjust
        return True

    monkeypatch.setattr("stockcharts.data.fetch.load_cached", fake_load_cached)
    monkeypatch.setattr("stockcharts.data.fetch._is_cache_fresh", fake_is_fresh)

    from stockcharts.data.fetch import fetch_ohlc

    out = fetch_ohlc("SPY", interval="1d", lookback="1y", auto_adjust=True, use_cache=True)

    assert seen["load"] is True
    assert seen["fresh"] is True
    assert not out.empty
