"""Data fetching utilities using yfinance.

Function:
    fetch_ohlc(
        ticker,
        interval="1d",
        lookback: str | None = None,
        start: str | None = None,
        end: str | None = None,
        auto_adjust: bool = False,
    )

Parameter semantics:
    - interval: Aggregation interval for candles ('1d', '1wk', '1mo').
    - lookback: Relative period for history breadth ('5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max').
    - start/end: Explicit date range (YYYY-MM-DD). If both provided they override lookback.
      Passing values like '3mo' to start/end is invalid and will be ignored.

We guard against accidentally sending a lookback string where a date is required by validating format.

Returns a pandas DataFrame with columns: Open, High, Low, Close, Volume
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import sys
from collections.abc import Iterator
from datetime import timedelta
from typing import cast

import pandas as pd

from stockcharts.data.cache import _is_cache_fresh, cache_stats, load_cached, save_cache


@contextlib.contextmanager
def _suppress_stderr() -> Iterator[None]:
    """Suppress stderr output at the OS file-descriptor level (thread-safe).

    Unlike ``sys.stderr = io.StringIO()``, this operates on the underlying
    file descriptor so it also catches output from C libraries and threads
    spawned by yfinance.  Falls back to a no-op if the file descriptor is
    unavailable (e.g. in some WSGI / notebook environments).
    """
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, io.UnsupportedOperation, OSError):
        yield  # can't redirect – just pass through
        return

    saved_fd = os.dup(stderr_fd)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


try:
    import yfinance as yf
    from yfinance.exceptions import YFInvalidPeriodError
except ImportError as e:  # pragma: no cover - guidance only
    raise ImportError(
        "yfinance must be installed to use fetch_ohlc. Install with `pip install yfinance`."
    ) from e

logger = logging.getLogger(__name__)

VALID_INTERVALS = {"1d", "1wk", "1mo"}  # Aggregation intervals: daily, weekly, monthly
VALID_LOOKBACK = {"5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
# Extended lookback periods that map to 'max' (yfinance doesn't support >10y directly)
EXTENDED_LOOKBACK_TO_MAX = {"20y", "30y", "40y", "50y", "100y"}

# Ordered fallback chain: when yfinance rejects a period we try the next shorter one.
_PERIOD_FALLBACK_ORDER = ["max", "10y", "5y", "2y", "1y", "6mo", "3mo", "1mo", "5d", "1d"]

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_date(s: str | None) -> bool:
    return bool(s and DATE_RE.match(s))


def _cache_covers_lookback(cached: pd.DataFrame, lookback: str | None) -> bool:
    """Return True if *cached* data extends far enough back for *lookback*.

    If the user asked for a specific long period and we only have a
    small window cached, this returns False so the caller can do a full
    re-download instead of serving truncated data.

    For ``"max"`` we always return True — the cache already holds all
    data that was available when it was last downloaded, and an
    incremental update will bring it up to date.  A full re-download
    would be wasteful since yfinance can only return what exists.
    """
    if lookback is None or lookback in ("5d", "1mo", "max"):
        return True  # short lookbacks or "max" — anything cached is fine

    # Map lookback strings to approximate timedelta thresholds.
    # We allow a ~10% tolerance so minor gaps don't trigger re-downloads.
    _thresholds = {
        "3mo": timedelta(days=80),
        "6mo": timedelta(days=160),
        "1y": timedelta(days=330),
        "2y": timedelta(days=660),
        "5y": timedelta(days=1700),
        "10y": timedelta(days=3400),
        "ytd": timedelta(days=0),  # handled below
    }
    threshold = _thresholds.get(lookback)
    if threshold is None:
        return True  # unknown lookback — don't block

    if lookback == "ytd":
        # Just need data from Jan 1 of this year
        import datetime as _dt

        jan1 = pd.Timestamp(_dt.date(pd.Timestamp.now().year, 1, 1))
        return cached.index[0] <= jan1

    span = pd.Timestamp.now() - cached.index[0]
    return span >= threshold


def _lookback_start(lookback: str | None, last: pd.Timestamp) -> pd.Timestamp | None:
    """Return the earliest bar a lookback should include.

    Args:
        lookback: Relative period such as ``"5y"``.
        last: Timestamp of the most recent bar available.

    Returns:
        The cutoff timestamp, or None when the lookback implies no cutoff.
    """
    if not lookback or lookback == "max":
        return None
    if lookback == "ytd":
        return pd.Timestamp(year=last.year, month=1, day=1)
    days = {
        "5d": 5,
        "1mo": 31,
        "3mo": 92,
        "6mo": 183,
        "1y": 366,
        "2y": 731,
        "5y": 1827,
        "10y": 3653,
    }.get(lookback)
    return None if days is None else last - timedelta(days=days)


def _window_slice(
    df: pd.DataFrame,
    lookback: str | None,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Trim a frame to the window the caller actually asked for.

    The cache stores one frame per (ticker, interval) regardless of the
    window it was fetched for, so a cached frame must be sliced before it is
    returned or a caller asking for a historical window receives later bars
    too.

    Args:
        df: Cached frame, indexed by date.
        lookback: Relative period, used only when no explicit dates are given.
        start: Inclusive start date, if any.
        end: Inclusive end date, if any.

    Returns:
        The trimmed frame.
    """
    if df.empty:
        return df
    sliced = df
    if start:
        sliced = sliced[sliced.index >= pd.Timestamp(start)]
    if end:
        sliced = sliced[sliced.index <= pd.Timestamp(end)]
    if not start and not end:
        cutoff = _lookback_start(lookback, pd.Timestamp(df.index[-1]))
        if cutoff is not None:
            sliced = sliced[sliced.index >= cutoff]
    return sliced


def _normalize_date(s: str | None) -> str | None:
    """Return date string if valid YYYY-MM-DD else None."""
    if not _is_date(s):
        return None
    return s


def _validate_and_build_download_kwargs(
    interval: str,
    lookback: str | None,
    start: str | None,
    end: str | None,
    auto_adjust: bool,
) -> dict:
    """Validate parameters and build kwargs for yf.download.

    Args:
        interval: Aggregation interval ('1d', '1wk', '1mo').
        lookback: Relative period for history.
        start: Start date YYYY-MM-DD.
        end: End date YYYY-MM-DD.
        auto_adjust: Whether to adjust OHLC for splits/dividends.

    Returns:
        Dictionary with validated download kwargs for yfinance.

    Raises:
        ValueError: If interval or lookback is not in valid set.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Allowed: {sorted(VALID_INTERVALS)}")

    start = _normalize_date(start)
    end = _normalize_date(end)

    if lookback and (start or end):
        # Explicit date range takes precedence; ignore lookback
        lookback = None

    if not lookback and not start and not end:
        lookback = "1y"

    # Convert extended lookback periods to 'max' (yfinance only supports up to 10y)
    if lookback and lookback in EXTENDED_LOOKBACK_TO_MAX:
        lookback = "max"

    if lookback and lookback not in VALID_LOOKBACK:
        raise ValueError(f"Unsupported lookback '{lookback}'. Allowed: {sorted(VALID_LOOKBACK)}")

    download_kwargs = {
        "interval": interval,
        "progress": False,
        "auto_adjust": auto_adjust,
    }
    if start or end:
        # Forward whichever bound was given. Requiring both meant an
        # end-only "as of" request fell through to period=None.
        if start:
            download_kwargs["start"] = start
        if end:
            download_kwargs["end"] = end
    else:
        download_kwargs["period"] = lookback

    return download_kwargs


def _normalize_single_ticker_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a single-ticker DataFrame from yfinance.

    Args:
        df: Raw DataFrame from yfinance download.
        ticker: Stock symbol (used for error messages).

    Returns:
        DataFrame with standardized columns [Open, High, Low, Close, Volume].

    Raises:
        ValueError: If data is empty or missing required columns.
    """
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize columns
    df = df.reset_index().set_index(df.index.names[0])

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in data: {missing}")
    return df[needed].copy()


def fetch_ohlc(
    ticker: str,
    interval: str = "1d",
    lookback: str | None = None,
    start: str | None = None,
    end: str | None = None,
    auto_adjust: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLC data for a single ticker.

    Args:
        ticker: Stock symbol to download.
        interval: Aggregation interval ('1d', '1wk', '1mo').
        lookback: Relative period for history ('1y', '5y', 'max', etc.).
        start: Start date YYYY-MM-DD (overrides lookback if end also provided).
        end: End date YYYY-MM-DD.
        auto_adjust: Whether to adjust OHLC for splits/dividends.
        use_cache: Whether to read/write from the local Parquet cache.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume].

    Raises:
        ValueError: If data is empty or missing required columns.

    Note:
        - If both start and end are valid dates they override lookback.
        - If either start/end is invalid (e.g. '3mo'), it is ignored.
        - If nothing specified, default lookback = '1y'.
        - When *use_cache* is True the function checks for a local
          Parquet file first and only downloads data that is newer than
          the most recent cached bar.
    """
    # Validate up front: the cache must not be able to mask a bad parameter.
    download_kwargs = _validate_and_build_download_kwargs(
        interval, lookback, start, end, auto_adjust
    )
    # Slice with the *normalised* dates: the validator discards unparseable
    # ones (start="3mo" becomes None), and the cache slice must agree with it
    # or it would try to parse "3mo" as a timestamp.
    slice_start = _normalize_date(start)
    slice_end = _normalize_date(end)
    effective_lookback = download_kwargs.get("period") if not (slice_start or slice_end) else None

    # --- Try cache first (incremental update) ---
    if use_cache:
        cached = load_cached(ticker, interval, auto_adjust=auto_adjust)
        if cached is not None and not cached.empty:
            # 1) FRESHNESS FIRST.  If the file was written after the
            #    last market close, no new bars can possibly exist —
            #    return immediately regardless of lookback.  This
            #    prevents re-downloading for "max" when the cache
            #    already holds all available data.
            if _is_cache_fresh(ticker, interval, auto_adjust=auto_adjust):
                logger.debug("%s: cache is fresh, skipping network", ticker)
                return _window_slice(cached, effective_lookback, slice_start, slice_end)

            # 2) Cache is STALE (older than last market close).
            #    Check whether it covers the requested lookback.
            #    If not, we need a full re-download.
            if not _cache_covers_lookback(cached, lookback):
                logger.debug(
                    "%s: cache starts %s, too short for lookback=%s — re-downloading",
                    ticker,
                    cached.index[0].date(),
                    lookback,
                )
                # Fall through to full download below

            else:
                # 3) Cache covers the lookback but is stale — do an
                #    incremental download from near the last bar.
                last_date = cached.index[-1]
                if interval == "1wk":
                    delta = timedelta(days=14)
                elif interval == "1mo":
                    delta = timedelta(days=60)
                else:
                    delta = timedelta(days=5)

                inc_start = (last_date - delta).strftime("%Y-%m-%d")
                try:
                    with _suppress_stderr():
                        new_df = yf.download(
                            ticker,
                            start=inc_start,
                            interval=interval,
                            progress=False,
                            auto_adjust=auto_adjust,
                            threads=False,
                        )
                except Exception:
                    # Stale cache is better than nothing, but still only the
                    # window that was asked for.
                    return _window_slice(cached, effective_lookback, slice_start, slice_end)

                if new_df is not None and not new_df.empty:
                    if isinstance(new_df.columns, pd.MultiIndex):
                        new_df.columns = new_df.columns.get_level_values(0)
                    needed = ["Open", "High", "Low", "Close", "Volume"]
                    if all(c in new_df.columns for c in needed):
                        new_df = new_df[needed]
                        combined = pd.concat([cached, new_df])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        combined = combined.sort_index()
                        save_cache(ticker, combined, interval, auto_adjust=auto_adjust)
                        return _window_slice(combined, effective_lookback, slice_start, slice_end)
                return _window_slice(cached, effective_lookback, slice_start, slice_end)

    # Suppress noisy yfinance error output during download attempts.
    yf_logger = logging.getLogger("yfinance")
    prev_yf_level = yf_logger.level

    df = pd.DataFrame()

    # First attempt with the requested period.
    yf_logger.setLevel(logging.CRITICAL)
    try:
        with _suppress_stderr():
            df = yf.download(ticker, **download_kwargs)
    except YFInvalidPeriodError:
        pass  # will be handled by fallback below
    finally:
        yf_logger.setLevel(prev_yf_level)

    # If the download returned empty and we used a period (not date range),
    # walk down the fallback chain to progressively shorter periods.
    if df.empty and "period" in download_kwargs:
        requested = download_kwargs["period"]
        try:
            idx = _PERIOD_FALLBACK_ORDER.index(requested)
        except ValueError:
            idx = 0
        for fallback in _PERIOD_FALLBACK_ORDER[idx + 1 :]:
            yf_logger.setLevel(logging.CRITICAL)
            try:
                with _suppress_stderr():
                    download_kwargs["period"] = fallback
                    df = yf.download(ticker, **download_kwargs)
            except YFInvalidPeriodError:
                continue
            finally:
                yf_logger.setLevel(prev_yf_level)
            if not df.empty:
                logger.debug(
                    "%s: period '%s' returned no data, succeeded with '%s'",
                    ticker,
                    requested,
                    fallback,
                )
                break

    df = _normalize_single_ticker_df(df, ticker)
    if use_cache and not df.empty:
        save_cache(ticker, df, interval, auto_adjust=auto_adjust)
    return df


def fetch_ohlc_batch(
    tickers: list[str],
    interval: str = "1d",
    lookback: str | None = None,
    start: str | None = None,
    end: str | None = None,
    auto_adjust: bool = False,
    threads: bool = True,
    progress: bool = False,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLC data for multiple tickers in a single batch request.

    Uses yfinance's built-in threading for parallel downloads, which is
    significantly faster than sequential single-ticker downloads.

    When *use_cache* is True (default), tickers that already have cached
    data receive an incremental update (only new bars since the last
    cached date), while completely uncached tickers are downloaded in
    full via the normal batch path.  After downloading, every result
    is saved to the cache so the next run is near-instant.

    Args:
        tickers: List of stock symbols to download.
        interval: Aggregation interval ('1d', '1wk', '1mo').
        lookback: Relative period for history ('1y', '5y', 'max', etc.).
        start: Start date YYYY-MM-DD (overrides lookback if end also provided).
        end: End date YYYY-MM-DD.
        auto_adjust: Whether to adjust OHLC for splits/dividends.
        threads: Use multi-threading for faster downloads (default: True).
        progress: Show download progress bar (default: False).
        use_cache: Read/write from the local Parquet cache.

    Returns:
        Dictionary mapping ticker symbols to their OHLC DataFrames.
        Failed downloads are silently omitted from results.
    """
    if not tickers:
        return {}

    # --- Cache-aware pre-processing ---
    results: dict[str, pd.DataFrame] = {}
    tickers_to_download: list[str] = []

    if use_cache:
        stats = cache_stats()
        if stats["files"] > 0:
            logger.info(
                "Cache: %d tickers cached, %.1f MB on disk",
                stats["tickers"],
                stats["size_mb"],
            )

        # Separate into: fresh (skip network), stale (incremental), uncached (full)
        cached_map: dict[str, pd.DataFrame] = {}  # stale cached tickers
        fresh_count = 0
        for t in tickers:
            cached = load_cached(t, interval, auto_adjust=auto_adjust)
            if cached is not None and not cached.empty:
                # FRESHNESS FIRST.  If written after last market close,
                # no new bars can exist — use directly.  This prevents
                # re-downloading for lookback="max" when the cache
                # already holds all available data.
                if _is_cache_fresh(t, interval, auto_adjust=auto_adjust):
                    results[t] = cached
                    fresh_count += 1
                elif not _cache_covers_lookback(cached, lookback):
                    # Stale AND too short — full download needed.
                    tickers_to_download.append(t)
                else:
                    # Stale but covers the lookback — incremental.
                    cached_map[t] = cached
            else:
                tickers_to_download.append(t)

        if fresh_count:
            logger.info(
                "%d tickers served from fresh cache (no network)",
                fresh_count,
            )

        # --- Batch-incremental update for stale cached tickers ---
        if cached_map:
            # Find the earliest "last cached date" across all cached tickers
            # so we can do a single batch download covering all of them.
            if interval == "1wk":
                delta = timedelta(days=14)
            elif interval == "1mo":
                delta = timedelta(days=60)
            else:
                delta = timedelta(days=5)

            earliest_start = min((df.index[-1] - delta) for df in cached_map.values())
            inc_start = earliest_start.strftime("%Y-%m-%d")

            cached_tickers = list(cached_map.keys())
            inc_df = pd.DataFrame()
            yf_logger = logging.getLogger("yfinance")
            prev_lvl = yf_logger.level
            yf_logger.setLevel(logging.CRITICAL)
            try:
                with _suppress_stderr():
                    inc_df = yf.download(
                        cached_tickers,
                        start=inc_start,
                        interval=interval,
                        progress=False,
                        auto_adjust=auto_adjust,
                        group_by="ticker",
                        threads=True,
                    )
            except Exception:
                pass
            finally:
                yf_logger.setLevel(prev_lvl)

            needed = ["Open", "High", "Low", "Close", "Volume"]

            for t in cached_tickers:
                old = cached_map[t]
                new_data: pd.DataFrame | None = None

                if not inc_df.empty:
                    try:
                        if len(cached_tickers) == 1:
                            # Single ticker: columns are flat OHLCV
                            candidate = inc_df.copy()
                            if isinstance(candidate.columns, pd.MultiIndex):
                                candidate.columns = candidate.columns.get_level_values(0)
                            new_data = candidate
                        elif t in inc_df.columns.get_level_values(0):
                            # Selecting a top level of a MultiIndex yields a
                            # frame, which the stubs type as a Series.
                            candidate = cast("pd.DataFrame", inc_df[t].copy())
                            candidate = candidate.dropna(how="all")
                            new_data = candidate
                    except Exception:
                        pass

                if new_data is not None and not new_data.empty:
                    if isinstance(new_data.columns, pd.MultiIndex):
                        new_data.columns = new_data.columns.get_level_values(0)
                    if all(c in new_data.columns for c in needed):
                        new_data = new_data[needed]
                        combined = pd.concat([old, new_data])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        combined = combined.sort_index()
                        save_cache(t, combined, interval, auto_adjust=auto_adjust)
                        results[t] = combined
                        continue
                # Fallback: serve stale cache
                results[t] = old

        if not tickers_to_download:
            logger.info("All %d tickers served from cache (batch-incremental)", len(results))
            return results

        logger.info(
            "%d tickers from cache, %d tickers need full download",
            len(results),
            len(tickers_to_download),
        )
    else:
        tickers_to_download = list(tickers)

    download_kwargs = _validate_and_build_download_kwargs(
        interval, lookback, start, end, auto_adjust
    )
    download_kwargs["progress"] = progress
    download_kwargs["threads"] = threads
    download_kwargs["group_by"] = "ticker"

    # Temporarily suppress noisy yfinance error messages for tickers
    # that reject certain period values (e.g. warrants with only '1d','5d').
    # yfinance prints "N Failed downloads:" directly to stderr via print(),
    # so we redirect at the OS file-descriptor level (thread-safe, unlike
    # replacing sys.stderr which breaks in multi-threaded / Dash contexts).
    yf_logger = logging.getLogger("yfinance")
    prev_level = yf_logger.level
    yf_logger.setLevel(logging.CRITICAL)
    try:
        with _suppress_stderr():
            batch_df = yf.download(tickers_to_download, **download_kwargs)
    finally:
        yf_logger.setLevel(prev_level)

    if batch_df.empty:
        return results

    # Handle single vs multiple ticker return format
    if len(tickers_to_download) == 1:
        # Single ticker: columns are just OHLCV
        ticker = tickers_to_download[0]
        try:
            df = _normalize_single_ticker_df(batch_df.copy(), ticker)
            if use_cache:
                save_cache(ticker, df, interval, auto_adjust=auto_adjust)
            results[ticker] = df
        except ValueError:
            pass  # Skip failed ticker
    else:
        # Multiple tickers: MultiIndex columns (ticker, OHLCV)
        for ticker in tickers_to_download:
            try:
                if ticker not in batch_df.columns.get_level_values(0):
                    continue

                ticker_df = batch_df[ticker].copy()

                # Drop rows where all values are NaN (no data for this date)
                ticker_df = ticker_df.dropna(how="all")

                if ticker_df.empty:
                    continue

                # Standardize columns
                ticker_df = ticker_df.reset_index().set_index(ticker_df.index.names[0])

                needed = ["Open", "High", "Low", "Close", "Volume"]
                missing = [c for c in needed if c not in ticker_df.columns]
                if missing:
                    continue

                ticker_df = ticker_df[needed].copy()
                if use_cache:
                    save_cache(ticker, ticker_df, interval, auto_adjust=auto_adjust)
                results[ticker] = ticker_df
            except Exception:
                # Skip any ticker that fails processing
                continue

    return results


__all__ = ["fetch_ohlc", "fetch_ohlc_batch"]
