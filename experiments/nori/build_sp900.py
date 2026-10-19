"""Build the S&P 500 + S&P MidCap 400 feature frame for the frozen GBM.

Why this universe: established, liquid, sector-diverse companies, so a model
trained once should need refitting rarely. Features, target and the shared
non-overlapping date grid are exactly those of build_names.py, so
gbm_tune.py / nori_fair.py read the output unchanged.

Two files are written next to each other:

  <out>.parquet        32 features + target + naive_momentum + ticker + bar
  <out>_meta.parquet   same rows, same order, NOT features:
                         entry_open   open of the NEXT bar (the first fill possible
                                      after a close-time signal)
                         target_open  close[t+15] / open[t+1] - 1
                         close, dollar_vol_60, hist_bars

target_open keeps the exit date of `target` (close of t+15) and moves only the
entry from close[t] to open[t+1], so target_open minus target isolates the
overnight gap that a live strategy cannot avoid.

"Established" is applied PER ROW using only information up to that date: at
least MIN_HIST_BARS of history, median 60-day dollar volume of at least
MIN_DOLLAR_VOL, and a close of at least MIN_PRICE. A per-ticker "15 years of
history" rule would drop good newer names and says nothing about a given date.
Prices are split/dividend adjusted, which understates old dollar volume and
price, so these floors bite slightly harder in the distant past.

SURVIVORSHIP BIAS REMAINS. The lists are today's index members. A 2005 row for
a company that joined the index in 2018 is there because it did well. The
per-row filters reduce this and do not remove it; cross-sectional ranking
cancels the shared drift but not all of it. Read backtests on this frame as
optimistic.

usage (stockcharts env):
  python build_sp900.py frame_sp900.parquet [--offset 0..14] [--limit N]

  --offset  which of the 15 interleaved grids to sample (default 0)
  --limit   only the first N tickers, for a smoke test

Resumable. Raw downloads are cached by fetch_ohlc_batch (cache/ohlc), and each
chunk of CHUNK tickers is written to <out>_parts/ as it finishes, so an
interrupted run restarts at the first unfinished chunk. Delete the parts folder
to rebuild from scratch (e.g. after changing the filters or the grid offset).
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from build_frame_long import HORIZON, features_for  # noqa: E402

from stockcharts.data.fetch import fetch_ohlc, fetch_ohlc_batch  # noqa: E402

MIN_HIST_BARS = 3 * 252
MIN_DOLLAR_VOL = 5e6
MIN_PRICE = 5.0
MIN_ROWS = 20
CHUNK = 20  # small on purpose: the build has to fit beside whatever else is running
USER_AGENT = "Mozilla/5.0 (compatible; stockcharts-research)"
SOURCES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
}
META_COLS = ["ticker", "entry_open", "target_open", "close", "dollar_vol_60", "hist_bars"]


def rss_mb() -> float:
    """This process's resident memory in MB, or NaN if psutil is not installed."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return float("nan")


def wiki_symbols(url: str) -> list[str]:
    """Ticker symbols from the first table on a Wikipedia index-constituents page.

    Dots become dashes (BRK.B -> BRK-B), which is how yfinance spells them.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8")
    table = pd.read_html(io.StringIO(html))[0]
    return [str(s).strip().upper().replace(".", "-") for s in table["Symbol"]]


def load_universe() -> pd.DataFrame:
    """The S&P 500 plus MidCap 400 as (ticker, source), S&P 500 winning any overlap."""
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source, url in SOURCES.items():
        for ticker in wiki_symbols(url):
            if ticker not in seen:
                seen.add(ticker)
                rows.append((ticker, source))
    return pd.DataFrame(rows, columns=["ticker", "source"])


def meta_for(df: pd.DataFrame) -> pd.DataFrame:
    """Columns that describe a row but must never be fed to the model."""
    entry = df["Open"].shift(-1)
    return pd.DataFrame(
        {
            "entry_open": entry,
            "target_open": df["Close"].shift(-HORIZON) / entry - 1.0,
            "close": df["Close"],
            "dollar_vol_60": (df["Close"] * df["Volume"]).rolling(60).median(),
            "hist_bars": np.arange(1, len(df) + 1),
        },
        index=df.index,
    )


def build_rows(
    df: pd.DataFrame, bench: pd.DataFrame, vix: pd.Series, grid: set, counts: dict[str, int]
) -> pd.DataFrame:
    """One ticker's grid rows after the per-row filters; counts tallies what was cut."""
    f = features_for(df, bench, vix).dropna()
    f = f[f.index.isin(grid)]
    full = f.join(meta_for(df))
    counts["grid_rows"] += len(full)
    keep = (
        (full["hist_bars"] >= MIN_HIST_BARS)
        & (full["dollar_vol_60"] >= MIN_DOLLAR_VOL)
        & (full["close"] >= MIN_PRICE)
    )
    counts["cut_history"] += int((full["hist_bars"] < MIN_HIST_BARS).sum())
    counts["cut_dollar_vol"] += int((full["dollar_vol_60"] < MIN_DOLLAR_VOL).sum())
    counts["cut_price"] += int((full["close"] < MIN_PRICE).sum())
    return full[keep]


def main() -> None:
    """Fetch, featurise, filter and write the frame, its meta file and an audit report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not 0 <= args.offset < HORIZON:
        sys.exit(f"--offset must be in 0..{HORIZON - 1}")
    out = Path(args.out)

    bench = fetch_ohlc("SPY", interval="1d", lookback="max", auto_adjust=True)
    vix = fetch_ohlc("^VIX", interval="1d", lookback="max", auto_adjust=True)["Close"]
    grid = set(bench.index[args.offset :: HORIZON])
    print(
        f"benchmark SPY {len(bench)} bars from {bench.index[0].date()}; "
        f"grid offset {args.offset}: {len(grid)} dates",
        flush=True,
    )

    universe = load_universe()
    if args.limit:
        universe = universe.head(args.limit)
    universe.assign(asof=date.today().isoformat()).to_csv(
        out.with_name(out.stem + "_universe.csv"), index=False
    )
    print(
        f"universe {len(universe)} tickers "
        f"({(universe.source == 'sp500').sum()} S&P 500, "
        f"{(universe.source == 'sp400').sum()} MidCap 400)",
        flush=True,
    )

    counts = {"grid_rows": 0, "cut_history": 0, "cut_dollar_vol": 0, "cut_price": 0}
    report: list[dict] = []
    tickers = universe["ticker"].tolist()
    source_of = dict(zip(universe["ticker"], universe["source"], strict=True))
    parts = out.with_name(out.stem + "_parts")
    parts.mkdir(exist_ok=True)
    for n, start in enumerate(range(0, len(tickers), CHUNK)):
        chunk = tickers[start : start + CHUNK]
        stem = parts / f"part_{n:04d}"
        marker = stem.with_suffix(".json")
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["tickers"] != chunk:
                sys.exit(f"{marker} was built from a different universe; delete {parts} to restart")
            report += saved["report"]
            for k in counts:
                counts[k] += saved["counts"][k]
            print(f"  chunk {n:>3} resumed from disk", flush=True)
            continue
        chunk_counts = dict.fromkeys(counts, 0)
        chunk_report: list[dict] = []
        chunk_frames: list[pd.DataFrame] = []
        try:
            data = fetch_ohlc_batch(chunk, interval="1d", lookback="max", auto_adjust=True)
        except Exception as exc:
            print(f"  chunk {n}: batch failed ({exc})", flush=True)
            data = {}
        for ticker in chunk:
            df = data.get(ticker)
            row = {"ticker": ticker, "source": source_of[ticker], "bars": 0, "rows": 0}
            if df is None or len(df) == 0:
                chunk_report.append({**row, "status": "no data"})
                continue
            row["bars"] = len(df)
            try:
                rows = build_rows(df, bench, vix, grid, chunk_counts)
            except Exception as exc:
                chunk_report.append({**row, "status": f"feature error: {exc}"})
                continue
            if len(rows) < MIN_ROWS:
                chunk_report.append({**row, "rows": len(rows), "status": "too few rows"})
                continue
            chunk_frames.append(rows.assign(ticker=ticker))
            chunk_report.append({**row, "rows": len(rows), "status": "ok"})
        del data
        if chunk_frames:
            part = pd.concat(chunk_frames)
            for c in part.columns:
                if part[c].dtype == np.float64:
                    part[c] = part[c].astype(np.float32)
            part.to_parquet(stem.with_suffix(".parquet"))
        del chunk_frames
        gc.collect()
        # The marker is written last: a chunk counts as done only if its rows are on disk.
        marker.write_text(
            json.dumps({"tickers": chunk, "counts": chunk_counts, "report": chunk_report})
        )
        report += chunk_report
        for k in counts:
            counts[k] += chunk_counts[k]
        ok = sum(1 for r in report if r["status"] == "ok")
        print(
            f"  chunk {n:>3} done, tickers {start + len(chunk):>4}/{len(tickers)}  kept {ok}  "
            f"rss {rss_mb():.0f} MB",
            flush=True,
        )

    pd.DataFrame(report).to_csv(out.with_name(out.stem + "_report.csv"), index=False)
    files = sorted(parts.glob("part_*.parquet"))
    if not files:
        sys.exit("no ticker produced usable rows")

    full = pd.concat([pd.read_parquet(f) for f in files])
    full = full.reset_index().sort_values([full.index.name or "index", "ticker"], kind="stable")
    full = full.set_index(full.columns[0])
    for c in full.columns:
        if full[c].dtype == np.float64:
            full[c] = full[c].astype(np.float32)

    main_frame = full.drop(columns=[c for c in META_COLS if c != "ticker"])
    main_frame["bar"] = 0  # unused downstream; dates carry the ordering
    meta = full[META_COLS]
    assert (main_frame.index == meta.index).all() and (main_frame.ticker == meta.ticker).all()

    status = pd.DataFrame(report)["status"].value_counts().to_dict()
    print(f"\nticker outcomes: {status}")
    print(
        f"grid rows before filters {counts['grid_rows']:,}; rows failing each floor "
        f"(overlapping): history {counts['cut_history']:,}, "
        f"dollar volume {counts['cut_dollar_vol']:,}, price {counts['cut_price']:,}"
    )
    per_date = main_frame.groupby(level=0).size()
    print(
        f"names {main_frame.ticker.nunique()}, rows {len(main_frame):,}, "
        f"dates {per_date.size}, {main_frame.index.min().date()} to {main_frame.index.max().date()}"
    )
    print(
        f"names per date: min {per_date.min()}, median {int(per_date.median())}, "
        f"max {per_date.max()}"
    )
    gap = (meta.entry_open / meta.close - 1.0).dropna()
    diff = (meta.target_open - main_frame.target).dropna()
    print(f"overnight gap open[t+1]/close[t]-1: mean {gap.mean():+.5f}, std {gap.std():.4f}")
    print(
        f"target_open - target: mean {diff.mean():+.5f}, std {diff.std():.4f}; "
        f"corr(target, target_open) {main_frame.target.corr(meta.target_open):.4f}"
    )

    main_frame.to_parquet(out)
    meta.to_parquet(out.with_name(out.stem + "_meta.parquet"))
    print(f"wrote {out} and {out.stem}_meta.parquet")


if __name__ == "__main__":
    main()
