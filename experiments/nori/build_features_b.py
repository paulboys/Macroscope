"""Phase B features: earnings-announcement timing and reaction, from SEC 8-K Item 2.02.

Input: sec_earnings/<ticker>.parquet from fetch_sec_earnings.py, the Phase A
frame, and the cached prices. Output: the frame with ern_ columns appended, row
for row.

Event dating, which is what keeps this point in time. An announcement accepted by
the SEC before 16:00 New York time on a trading day is reacted to on that day; one
accepted at or after 16:00 (or on a non-trading day) is reacted to on the next
trading day. That reaction day R is the event date. Events within 10 trading days
of an earlier one are dropped (8-K re-filings, second 2.02 items).

Features at a date t use only events with R <= t (timing) or R + 1 <= t (reaction
over R-1 -> R+1, so the whole window is observed):

  ern_days_since      trading days since the last event's reaction day
  ern_days_to_next    calendar days to the estimated next announcement: the event
                      four back plus 364 days, else last + 91, floored at 0
  ern_due_15d         1 if that estimate falls within the next 15 calendar days
  ern_hour            timing of the last release: 0 before open, 1 in hours, 2 after
  ern_ar_3d           sum of abnormal daily returns (stock minus sector median)
                      over R..R+1, i.e. the reaction
  ern_gap_ar          abnormal overnight gap into R (open[R]/close[R-1] - 1, minus
                      the sector median gap)
  ern_post_ret        abnormal return since the reaction window, R+2 .. t
                      (post-earnings drift so far)

Sector medians use the Wikipedia GICS sectors, as in build_features_a.py.

usage (stockcharts env):
  python build_features_b.py frame_sp900_a.parquet frame_sp900_ab.parquet [--leak-check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from build_features_a import LEAK_CUTOFFS, load_panel, load_sectors  # noqa: E402
from build_sp900 import rss_mb  # noqa: E402

from stockcharts.data.cache import load_cached  # noqa: E402

EVENTS_DIR = Path("sec_earnings")
MIN_GAP_BARS = 10
CLOSE_HOUR = 16.0
NEW_COLS = [
    "ern_days_since",
    "ern_days_to_next",
    "ern_due_15d",
    "ern_hour",
    "ern_ar_3d",
    "ern_gap_ar",
    "ern_post_ret",
]


def load_events(ticker: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reaction-day positions (into `index`) and release timing for one ticker."""
    path = EVENTS_DIR / f"{ticker}.parquet"
    if not path.exists():
        return pd.DataFrame({"pos": [], "hour": []}, dtype=int)
    d = pd.read_parquet(path)
    d = d[(d["form"] == "8-K") & d["items"].str.contains("2.02", regex=False)]
    acc = pd.to_datetime(d["acceptanceDateTime"], utc=True, format="ISO8601").dt.tz_convert(
        "America/New_York"
    )
    clock = (acc.dt.hour + acc.dt.minute / 60).to_numpy()
    day = acc.dt.tz_localize(None).dt.normalize()
    # First trading day on or after the filing day; after-close filings move one more.
    pos = index.searchsorted(day.to_numpy())
    on_trading_day = (pos < len(index)) & (index[np.minimum(pos, len(index) - 1)] == day.to_numpy())
    after_close = on_trading_day & (clock >= CLOSE_HOUR)
    pos = pos + after_close
    hour = np.where(~on_trading_day | after_close, 2, np.where(clock < 9.5, 0, 1))
    ev = pd.DataFrame({"pos": pos, "hour": hour}).sort_values("pos", kind="stable")
    ev = ev[ev["pos"] < len(index)]
    keep, last = [], -(10**9)
    for p in ev["pos"]:
        keep.append(p - last >= MIN_GAP_BARS)
        if keep[-1]:
            last = p
    return ev[np.array(keep, dtype=bool)].reset_index(drop=True)  # a bare [] would pick columns


def ticker_features(
    ev: pd.DataFrame,
    index: pd.DatetimeIndex,
    tpos: np.ndarray,
    cum_abn: np.ndarray,
    abn_gap: np.ndarray,
    gap_ok: np.ndarray,
) -> np.ndarray:
    """The ern_ columns (len(tpos) x len(NEW_COLS)) for one ticker at positions tpos."""
    out = np.full((len(tpos), len(NEW_COLS)), np.nan)
    if ev.empty:
        return out
    rpos, hour = ev["pos"].to_numpy(), ev["hour"].to_numpy()
    known = np.searchsorted(rpos, tpos, side="right") - 1  # last event with R <= t
    seen = np.searchsorted(rpos + 1, tpos, side="right") - 1  # last with R + 1 <= t
    dates = index.to_numpy()
    for i, (t, k, s) in enumerate(zip(tpos, known, seen, strict=True)):
        if k >= 0:
            out[i, 3] = hour[k]
            est = dates[rpos[k - 3]] + np.timedelta64(364, "D") if k >= 3 else None
            if est is None or est <= dates[t]:
                est = dates[rpos[k]] + np.timedelta64(91, "D")
            days = max(float((est - dates[t]) / np.timedelta64(1, "D")), 0.0)
            out[i, 1], out[i, 2] = days, float(days <= 15)
        if s >= 0:
            r = rpos[s]
            out[i, 0] = t - r
            if r >= 1:
                out[i, 4] = cum_abn[r + 1] - cum_abn[r - 1]
                out[i, 5] = abn_gap[r] if gap_ok[r] else np.nan
                out[i, 6] = cum_abn[t] - cum_abn[r + 1]
    return out


def all_features(
    panel: dict[str, pd.DataFrame],
    spy: pd.Series,
    sectors: pd.Series,
    rows: pd.DataFrame,
    cut: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """ern_ columns for every (Date, ticker) in `rows`; `cut` drops events after it."""
    index = panel["Close"].index
    pieces = []
    for sector, members in sectors.groupby(sectors):
        cols = [t for t in members.index if t in panel["Close"].columns]
        close, open_ = panel["Close"][cols], panel["Open"][cols]
        r = close.pct_change(fill_method=None)
        gap = open_ / close.shift(1) - 1.0
        abn = r.sub(r.median(axis=1), axis=0)
        abn_gap = gap.sub(gap.median(axis=1), axis=0)
        cum = abn.fillna(0.0).cumsum()
        for t in cols:
            sel = rows[rows["ticker"] == t]
            if sel.empty:
                continue
            ev = load_events(t, index)
            if cut is not None:
                ev = ev[index[ev["pos"]] <= cut].reset_index(drop=True)
            tpos = index.get_indexer(sel.index)
            vals = ticker_features(
                ev, index, tpos, cum[t].to_numpy(), abn_gap[t].to_numpy(), gap[t].notna().to_numpy()
            )
            pieces.append(
                pd.DataFrame(
                    vals,
                    index=pd.MultiIndex.from_arrays([sel.index, sel["ticker"]]),
                    columns=NEW_COLS,
                )
            )
        print(f"  {sector:<24} rss {rss_mb():.0f} MB", flush=True)
    return pd.concat(pieces).astype(np.float32)


def leak_check(
    panel: dict[str, pd.DataFrame], spy: pd.Series, sectors: pd.Series, frame: pd.DataFrame
) -> None:
    """Features at a cutoff must not change when later prices and events are removed."""
    for cut in pd.to_datetime(LEAK_CUTOFFS):
        grid = frame.index.unique().sort_values()  # the cutoff must be a date the frame has
        cut = grid[grid.searchsorted(cut)]
        rows = frame[frame.index == cut][["ticker"]]
        full = all_features(panel, spy, sectors, rows, cut=cut)
        trunc = all_features(
            {k: v.loc[:cut] for k, v in panel.items()}, spy.loc[:cut], sectors, rows, cut=cut
        )
        a, b = full.sort_index(), trunc.sort_index()
        same = np.isclose(a.to_numpy(), b.to_numpy(), rtol=1e-6, atol=1e-9, equal_nan=True)
        bad = a.columns[~same.all(axis=0)].tolist()
        print(f"leak check at {cut.date()}: {len(a)} rows, mismatched columns {bad or 'none'}")
        if bad:
            sys.exit("LEAK: an earnings feature changed when future data was removed")


def main() -> None:
    """Append the earnings features to the frame and report coverage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("frame")
    parser.add_argument("out")
    parser.add_argument("--leak-check", action="store_true")
    args = parser.parse_args()
    frame_path = Path(args.frame)

    frame = pd.read_parquet(frame_path)
    tickers = sorted(frame["ticker"].unique())
    spy = load_cached("SPY", "1d", auto_adjust=True)["Close"].astype(float)
    sectors = load_sectors(frame_path.with_name("frame_sp900_sectors.csv"))
    sectors = sectors[sectors.index.isin(tickers)]
    panel = load_panel(tickers, spy.index)
    print(f"panel {panel['Close'].shape}, rss {rss_mb():.0f} MB", flush=True)

    if args.leak_check:
        leak_check(panel, spy, sectors, frame)

    new = all_features(panel, spy, sectors, frame[["ticker"]])
    keys = pd.MultiIndex.from_arrays([frame.index, frame["ticker"]])
    new = new.reindex(keys)
    out = frame.copy()
    for c in NEW_COLS:
        out[c] = new[c].to_numpy()
    assert len(out) == len(frame)
    print("\ncoverage (share of rows not missing):")
    print(out[NEW_COLS].notna().mean().round(3).to_string())
    print("\nsummary:")
    print(out[NEW_COLS].describe().T[["mean", "std", "min", "50%", "max"]].round(4).to_string())
    out.to_parquet(args.out)
    print(f"wrote {args.out}: {out.shape}")


if __name__ == "__main__":
    main()
