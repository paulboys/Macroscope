"""Download earnings-announcement dates for the sp900 universe from SEC EDGAR.

Source: the submissions API (data.sec.gov/submissions/CIK##########.json), one
JSON per company plus paged history files. Every 8-K carries an `items` field;
Item 2.02 ("Results of Operations and Financial Condition") is the earnings
release and exists from August 2004. 10-Q and 10-K filing dates are kept too,
for companies whose release is not filed as an 8-K and for Phase C.

Only filings from FROM_DATE on are kept, and history pages that end before it are
not fetched (big banks have dozens of pages of structured-note filings).

Requests carry SEC_USER_AGENT from the environment (SECClient reads it) and are
spaced REQUEST_DELAY apart, under the SEC's 10 per second limit.

Resumable: each ticker is written to sec_earnings/<ticker>.parquet as it
finishes, and finished tickers are skipped on a rerun. A ticker the SEC does not
know is recorded in sec_earnings/_missing.json.

usage (stockcharts env):
  python fetch_sec_earnings.py frame_sp900_universe.csv [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
from pathlib import Path

import pandas as pd

from stockcharts.screener.sec_insider import SEC_SUBMISSIONS_URL, SECClient

FROM_DATE = "2008-01-01"
FORMS = {"8-K", "8-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A"}
REQUEST_DELAY = 0.15
OUT = Path("sec_earnings")
FIELDS = ("form", "filingDate", "acceptanceDateTime", "reportDate", "items", "accessionNumber")


def get_json(client: SECClient, url: str, tries: int = 5) -> dict:
    """GET with backoff on rate limiting and server errors."""
    for attempt in range(tries):
        try:
            return client._get_json(url)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
        time.sleep(2.0 * 2**attempt)
    raise RuntimeError("unreachable")


def rows_from(block: dict) -> pd.DataFrame:
    """Column-oriented filings block to a frame of the kept forms and dates."""
    df = pd.DataFrame({k: block.get(k, [""] * len(block["form"])) for k in FIELDS})
    return df[df["form"].isin(FORMS) & (df["filingDate"] >= FROM_DATE)]


def fetch_ticker(client: SECClient, ticker: str) -> pd.DataFrame:
    """All kept filings for one ticker, newest first."""
    cik = client.resolve_cik(ticker)
    sub = get_json(client, SEC_SUBMISSIONS_URL.format(cik=cik))
    parts = [rows_from(sub["filings"]["recent"])]
    for page in sub["filings"].get("files", []):
        if page["filingTo"] < FROM_DATE:
            continue
        parts.append(
            rows_from(get_json(client, "https://data.sec.gov/submissions/" + page["name"]))
        )
    out = pd.concat(parts, ignore_index=True).drop_duplicates("accessionNumber")
    return out.assign(ticker=ticker, cik=cik)


def main() -> None:
    """Fetch every ticker in the universe that is not already on disk."""
    parser = argparse.ArgumentParser()
    parser.add_argument("universe")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not os.environ.get("SEC_USER_AGENT"):
        sys.exit("SEC_USER_AGENT is not set in this environment")

    tickers = pd.read_csv(args.universe)["ticker"].tolist()[: args.limit]
    OUT.mkdir(exist_ok=True)
    missing_path = OUT / "_missing.json"
    missing = json.loads(missing_path.read_text()) if missing_path.exists() else {}
    client = SECClient(request_delay=REQUEST_DELAY)
    t0 = time.perf_counter()
    for n, ticker in enumerate(tickers, 1):
        path = OUT / f"{ticker}.parquet"
        if path.exists() or ticker in missing:
            continue
        try:
            df = fetch_ticker(client, ticker)
        except ValueError as exc:  # not in the SEC ticker map
            missing[ticker] = str(exc)
            missing_path.write_text(json.dumps(missing, indent=1))
            continue
        df.to_parquet(path)
        if n % 25 == 0 or n == len(tickers):
            print(
                f"  {n}/{len(tickers)}  {ticker}: {len(df)} filings, "
                f"{(df['items'].str.contains('2.02', regex=False)).sum()} item 2.02  "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    print(f"done; not found at the SEC: {sorted(missing)}")


if __name__ == "__main__":
    main()
