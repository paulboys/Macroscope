"""Phase A features for the S&P 500 + MidCap 400 frame: residual, sector, overnight.

Everything here comes from the daily prices already in cache/ohlc; nothing is
downloaded except the GICS sector of each ticker, read from the same Wikipedia
tables build_sp900.py uses. The output is frame_sp900.parquet with new columns
appended, row for row, so the meta file still lines up.

Feature groups, by prefix:

  res_  residuals from a rolling 252-day regression of each stock's daily return
        on the market (SPY) and on its sector's excess return (leave-one-out
        equal-weight mean of the other members, minus SPY). Betas are lagged 21
        days so the recent residuals are out of sample. Residual returns over
        5/10/20 days, the 20-day residual reversal scaled by residual volatility,
        12-1 month residual momentum (Blitz, Huij & Martens 2011), both betas,
        residual volatility and its share of total volatility.
  sec_  stock return minus the sector median over 5/20/60 days, and the sector's
        own 1- and 6-month return over SPY (Moskowitz & Grinblatt 1999).
  on_   cumulative overnight (close to next open) and intraday (open to close)
        log returns over 20 and 60 days (Lou, Polk & Skouras 2019).
  oth_  largest daily return in the last month (MAX effect), Amihud illiquidity,
        log 60-day median dollar volume.

Point in time: every value at date t uses prices through t only, and the sector
statistics use only members with a price that day. The --leak-check option
proves this mechanically by recomputing on panels cut off at several dates.

Not point in time: the GICS sector is today's classification, and the universe is
today's index members (survivorship bias, as in build_sp900.py).

usage (stockcharts env):
  python build_features_a.py frame_sp900.parquet frame_sp900_a.parquet [--leak-check]
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from build_sp900 import SOURCES, USER_AGENT, rss_mb  # noqa: E402

from stockcharts.data.cache import load_cached  # noqa: E402

BETA_WINDOW, BETA_MIN, BETA_LAG = 252, 126, 21
LEAK_CUTOFFS = ("2009-06-15", "2016-03-01", "2024-11-01")


def load_sectors(path: Path) -> pd.Series:
    """GICS sector per ticker, from the Wikipedia constituent tables (cached to csv)."""
    if path.exists():
        return pd.read_csv(path).set_index("ticker")["sector"]
    rows = []
    for url in SOURCES.values():
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            table = pd.read_html(io.StringIO(response.read().decode("utf-8")))[0]
        for sym, sector in zip(table["Symbol"], table["GICS Sector"], strict=True):
            rows.append((str(sym).strip().upper().replace(".", "-"), str(sector).strip()))
    out = pd.DataFrame(rows, columns=["ticker", "sector"]).drop_duplicates("ticker")
    out.to_csv(path, index=False)
    return out.set_index("ticker")["sector"]


def load_panel(tickers: list[str], index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Wide adjusted Open/Close/Volume (dates x tickers) from the local cache only."""
    cols: dict[str, dict[str, pd.Series]] = {"Open": {}, "Close": {}, "Volume": {}}
    for t in tickers:
        df = load_cached(t, "1d", auto_adjust=True)
        if df is None or df.empty:
            continue
        df = df[~df.index.duplicated(keep="last")]
        for k in cols:
            cols[k][t] = df[k].astype(float)
    return {k: pd.DataFrame(v).reindex(index).replace(0.0, np.nan) for k, v in cols.items()}


def rolling_two_factor(
    y: pd.DataFrame, x1: pd.Series, x2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling OLS slopes of y on (x1, x2) per column, from moments over BETA_WINDOW.

    x1 is shared by every column; x2 differs per column. Both are masked to the
    days y exists so each moment uses the same observations.
    """
    mask = y.notna()
    a = pd.DataFrame(np.broadcast_to(x1.to_numpy()[:, None], y.shape), y.index, y.columns)
    a, b = a.where(mask), x2.where(mask)

    def m(s: pd.DataFrame) -> pd.DataFrame:
        return s.rolling(BETA_WINDOW, min_periods=BETA_MIN).mean()

    ma, mb, my = m(a), m(b), m(y)
    caa, cbb, cab = m(a * a) - ma * ma, m(b * b) - mb * mb, m(a * b) - ma * mb
    cay, cby = m(a * y) - ma * my, m(b * y) - mb * my
    det = (caa * cbb - cab * cab).where(lambda d: d > 0)
    return (cay * cbb - cby * cab) / det, (cby * caa - cay * cab) / det


def sector_features(
    close: pd.DataFrame, open_: pd.DataFrame, volume: pd.DataFrame, spy: pd.Series
) -> dict[str, pd.DataFrame]:
    """All Phase A features for the members of one sector, as wide frames."""
    f: dict[str, pd.DataFrame] = {}
    r = close.pct_change(fill_method=None)
    m = spy.pct_change(fill_method=None)

    # Leave-one-out sector factor in excess of the market.
    n = r.notna().sum(axis=1)
    loo = (r.sum(axis=1).to_numpy()[:, None] - r.fillna(0.0)) / (n - 1).replace(
        0, np.nan
    ).to_numpy()[:, None]
    sec_x = loo.where(r.notna()).sub(m, axis=0)

    b_m, b_s = rolling_two_factor(r, m, sec_x)
    b_m, b_s = b_m.shift(BETA_LAG), b_s.shift(BETA_LAG)
    e = r - b_m.mul(m, axis=0) - b_s * sec_x

    for k in (5, 10, 20):
        f[f"res_ret_{k}d"] = e.rolling(k, min_periods=k).sum()
    res_vol = e.rolling(60, min_periods=40).std()
    f["res_vol_60"] = res_vol
    f["res_rev_z"] = f["res_ret_20d"] / (res_vol * np.sqrt(20))
    e_old = e.shift(21)
    f["res_mom_12_1"] = e_old.rolling(231, min_periods=200).sum() / (
        e_old.rolling(231, min_periods=200).std() * np.sqrt(231)
    )
    f["res_beta_mkt"], f["res_beta_sec"] = b_m, b_s
    f["res_ivol_share"] = res_vol / r.rolling(60, min_periods=40).std()

    for k in (5, 20, 60):
        rk = close / close.shift(k) - 1.0
        f[f"sec_rel_{k}d"] = rk.sub(rk.median(axis=1), axis=0)
    for k, name in ((21, "sec_mom_1m"), (126, "sec_mom_6m")):
        rk = close / close.shift(k) - 1.0
        excess = rk.median(axis=1) - (spy / spy.shift(k) - 1.0)
        f[name] = pd.DataFrame(
            np.broadcast_to(excess.to_numpy()[:, None], close.shape), close.index, close.columns
        ).where(close.notna())

    overnight = np.log(open_ / close.shift(1))
    intraday = np.log(close / open_)
    for k in (20, 60):
        f[f"on_ret_{k}d"] = overnight.rolling(k, min_periods=k).sum()
        f[f"id_ret_{k}d"] = intraday.rolling(k, min_periods=k).sum()

    dollar = close * volume
    f["oth_max_21"] = r.rolling(21, min_periods=21).max()
    f["oth_amihud_20"] = np.log((r.abs() / dollar).rolling(20, min_periods=15).mean() * 1e9)
    f["oth_log_dv60"] = np.log(dollar.rolling(60, min_periods=40).median())
    return f


def long_rows(feats: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Stack wide features to (date, ticker) rows on the requested dates."""
    parts = {k: v.reindex(dates).stack(future_stack=True) for k, v in feats.items()}
    out = pd.DataFrame(parts)
    out.index.names = ["Date", "ticker"]
    return out.astype(np.float32)


def all_features(
    panel: dict[str, pd.DataFrame], spy: pd.Series, sectors: pd.Series, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Phase A rows on `dates` for every ticker, computed one sector at a time."""
    out = []
    for sector, members in sectors.groupby(sectors):
        cols = [t for t in members.index if t in panel["Close"].columns]
        if len(cols) < 5:
            continue
        feats = sector_features(
            panel["Close"][cols], panel["Open"][cols], panel["Volume"][cols], spy
        )
        out.append(long_rows(feats, dates))
        print(f"  {sector:<24} {len(cols):>4} names  rss {rss_mb():.0f} MB", flush=True)
    return pd.concat(out)


def leak_check(panel: dict[str, pd.DataFrame], spy: pd.Series, sectors: pd.Series) -> None:
    """Recompute on panels cut off at each cutoff; the values at the cutoff must not move."""
    for cut in pd.to_datetime(LEAK_CUTOFFS):
        cut = spy.index[spy.index.searchsorted(cut)]
        full = all_features(panel, spy, sectors, pd.DatetimeIndex([cut]))
        trunc = all_features(
            {k: v.loc[:cut] for k, v in panel.items()},
            spy.loc[:cut],
            sectors,
            pd.DatetimeIndex([cut]),
        )
        a, b = full.sort_index(), trunc.sort_index()
        same = np.isclose(a.to_numpy(), b.to_numpy(), rtol=1e-6, atol=1e-9, equal_nan=True)
        bad = a.columns[~same.all(axis=0)].tolist()
        print(f"leak check at {cut.date()}: {len(a)} rows, mismatched columns {bad or 'none'}")
        if bad:
            sys.exit("LEAK: a feature at the cutoff changed when future data was removed")


def main() -> None:
    """Compute Phase A features and append them to the frame, row for row."""
    parser = argparse.ArgumentParser()
    parser.add_argument("frame")
    parser.add_argument("out")
    parser.add_argument("--leak-check", action="store_true")
    args = parser.parse_args()
    frame_path = Path(args.frame)

    frame = pd.read_parquet(frame_path)
    tickers = sorted(frame["ticker"].unique())
    spy = load_cached("SPY", "1d", auto_adjust=True)["Close"].astype(float)
    sectors = load_sectors(frame_path.with_name(frame_path.stem + "_sectors.csv"))
    sectors = sectors[sectors.index.isin(tickers)]
    missing = sorted(set(tickers) - set(sectors.index))
    print(
        f"{len(tickers)} tickers, {sectors.nunique()} sectors, no sector for {len(missing)}: "
        f"{missing[:10]}"
    )
    print(sectors.value_counts().to_string())

    panel = load_panel(tickers, spy.index)
    print(f"panel {panel['Close'].shape}, rss {rss_mb():.0f} MB", flush=True)

    if args.leak_check:
        leak_check(panel, spy, sectors)

    dates = pd.DatetimeIndex(sorted(frame.index.unique()))
    feats = all_features(panel, spy, sectors, dates)
    del panel

    keys = pd.MultiIndex.from_arrays([frame.index, frame["ticker"]], names=["Date", "ticker"])
    new = feats.reindex(keys)
    out = frame.copy()
    for c in new.columns:
        out[c] = new[c].to_numpy()
    assert len(out) == len(frame) and (out["ticker"] == frame["ticker"]).all()

    cov = out[new.columns].notna().mean().sort_values()
    print("\ncoverage of new features (share of rows not missing):")
    print(cov.round(3).to_string())
    out.to_parquet(args.out)
    print(f"wrote {args.out}: {out.shape}")


if __name__ == "__main__":
    main()
