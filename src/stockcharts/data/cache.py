"""Local Parquet cache for OHLC data.

First download stores full history.  Subsequent calls fetch only the
delta (new bars since the last cached date) and append, so screens that
run repeatedly only download the newest data per ticker.

Cache location defaults to ``<project>/cache/ohlc/`` and can be
overridden with the ``STOCKCHARTS_CACHE`` environment variable.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(
    os.environ.get("STOCKCHARTS_CACHE", Path(__file__).resolve().parents[3] / "cache" / "ohlc")
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


_ADJ_TAG = "adj"


def _cache_path(
    ticker: str,
    interval: str,
    cache_dir: Path,
    auto_adjust: bool = False,
) -> Path:
    """Return the parquet file path for a *ticker* + *interval* + adjustment.

    Adjusted and unadjusted history are different data for the same ticker,
    because ``auto_adjust`` rewrites OHLC for splits and dividends. They must
    not share a file: when they did, whichever was fetched first was returned
    for both, with no error and no warning.

    Args:
        ticker: Stock symbol.
        interval: Aggregation interval ('1d', '1wk', '1mo').
        cache_dir: Cache directory.
        auto_adjust: Whether the cached OHLC is split/dividend adjusted.
    """
    safe = ticker.replace("/", "_").replace("\\", "_").upper()
    tag = f"_{_ADJ_TAG}" if auto_adjust else ""
    return cache_dir / f"{safe}_{interval}{tag}.parquet"


def _parse_stem(stem: str) -> tuple[str, str, bool]:
    """Split a cache filename stem into ``(ticker, interval, auto_adjust)``.

    Tickers may themselves contain underscores, since '/' is replaced on the
    way in (``BRK/B`` becomes ``BRK_B``), so the ticker is everything before
    the interval rather than the first field.

    Args:
        stem: Filename stem, e.g. ``AAPL_1d`` or ``AAPL_1d_adj``.

    Returns:
        Tuple of (ticker, interval, auto_adjust).
    """
    parts = stem.split("_")
    adjusted = len(parts) > 2 and parts[-1] == _ADJ_TAG
    if adjusted:
        parts = parts[:-1]
    if len(parts) < 2:
        return stem, "", adjusted
    return "_".join(parts[:-1]), parts[-1], adjusted


def _last_market_close() -> datetime:
    """Return the most recent US equity market close (4:00 PM ET).

    Accounts for weekends — Saturday and Sunday map back to the
    preceding Friday's close.  US market holidays are **not** handled
    yet; worst case is one extra download after a holiday.
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    now_et = datetime.now(tz=et)
    today_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    weekday = now_et.weekday()  # Mon=0 … Sun=6

    if weekday == 5:  # Saturday → Friday close
        last_close = today_close - timedelta(days=1)
    elif weekday == 6:  # Sunday → Friday close
        last_close = today_close - timedelta(days=2)
    elif now_et < today_close:  # Weekday, before today's close
        # Previous trading day's close (skip weekends)
        if weekday == 0:  # Monday before close → Friday
            last_close = today_close - timedelta(days=3)
        else:
            last_close = today_close - timedelta(days=1)
    else:  # Weekday, after today's close
        last_close = today_close

    return last_close


def _is_cache_fresh(
    ticker: str,
    interval: str = "1d",
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    auto_adjust: bool = False,
) -> bool:
    """Check if a cache file is recent enough to skip a network update.

    Uses US equity market hours to determine freshness:

    - **Daily (``1d``)**: fresh if the file was written *after* the most
      recent market close (4:00 PM ET).  On weekends the reference is
      Friday's close.
    - **Weekly (``1wk``)**: fresh if written after the most recent
      Friday close.
    - **Monthly (``1mo``)**: fresh if written within the last 15 days.
    """
    from zoneinfo import ZoneInfo

    path = _cache_path(ticker, interval, cache_dir, auto_adjust)
    if not path.exists():
        return False

    et = ZoneInfo("America/New_York")
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=et)

    if interval == "1mo":
        return (datetime.now(tz=et) - mtime) < timedelta(days=15)

    last_close = _last_market_close()

    if interval == "1wk":
        # Roll last_close back to the most recent Friday close
        days_since_friday = (last_close.weekday() - 4) % 7
        last_close = last_close - timedelta(days=days_since_friday)

    return mtime > last_close


def load_cached(
    ticker: str,
    interval: str = "1d",
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    auto_adjust: bool = False,
) -> pd.DataFrame | None:
    """Load cached OHLC data from disk, or ``None`` if not cached.

    Args:
        ticker: Stock symbol.
        interval: Aggregation interval ('1d', '1wk', '1mo').
        cache_dir: Cache directory.
        auto_adjust: Load the split/dividend-adjusted series rather than the
            raw one. The two are cached separately and never interchanged.
    """
    path = _cache_path(ticker, interval, cache_dir, auto_adjust)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        return df
    except Exception as e:
        logger.debug("Cache read error for %s: %s", ticker, e)
        return None


def save_cache(
    ticker: str,
    df: pd.DataFrame,
    interval: str = "1d",
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    auto_adjust: bool = False,
) -> None:
    """Persist OHLC data to disk as Parquet.

    Args:
        ticker: Stock symbol.
        df: OHLC frame to persist.
        interval: Aggregation interval ('1d', '1wk', '1mo').
        cache_dir: Cache directory.
        auto_adjust: Whether *df* holds split/dividend-adjusted OHLC. Passing
            this wrongly would overwrite the other series.
    """
    if df is None or df.empty:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, interval, cache_dir, auto_adjust)
    try:
        df.to_parquet(path)
    except Exception as e:
        logger.debug("Cache write error for %s: %s", ticker, e)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def clear_cache(
    ticker: str | None = None,
    interval: str | None = None,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    auto_adjust: bool | None = None,
) -> int:
    """Delete cached files.

    Args:
        ticker: Restrict to this ticker (``None`` = all).
        interval: Restrict to this interval (``None`` = all).
        cache_dir: Cache directory.
        auto_adjust: Restrict to adjusted (``True``) or raw (``False``) files.
            ``None``, the default, clears both so that a plain
            ``clear_cache()`` still empties the cache completely.

    Returns:
        Number of files deleted.
    """
    if not cache_dir.exists():
        return 0

    count = 0
    for f in cache_dir.glob("*.parquet"):
        # Parse rather than match on substrings: an "_adj" suffix made
        # stem.endswith("_1d") false, so adjusted files survived a clear.
        file_ticker, file_interval, file_adjusted = _parse_stem(f.stem)
        if ticker:
            safe = ticker.replace("/", "_").replace("\\", "_").upper()
            if file_ticker != safe:
                continue
        if interval and file_interval != interval:
            continue
        if auto_adjust is not None and file_adjusted != auto_adjust:
            continue
        f.unlink()
        count += 1
    return count


def cache_stats(cache_dir: Path = _DEFAULT_CACHE_DIR) -> dict:
    """Return cache statistics.

    Returns:
        Dict with keys ``files``, ``size_mb``, ``tickers``.
    """
    if not cache_dir.exists():
        return {"files": 0, "size_mb": 0.0, "tickers": 0}

    files = list(cache_dir.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in files)
    # Count distinct tickers, not files: one ticker may hold both an adjusted
    # and a raw series, which a plain rsplit would report as two tickers.
    tickers = {_parse_stem(f.stem)[0] for f in files}

    return {
        "files": len(files),
        "size_mb": round(total_size / (1024 * 1024), 2),
        "tickers": len(tickers),
    }


__all__ = ["load_cached", "save_cache", "clear_cache", "cache_stats", "_is_cache_fresh"]
