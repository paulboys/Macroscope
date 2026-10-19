"""Pre-registered evaluation of new feature groups against the 32-feature baseline.

Fixed design (decided before any result was seen; do not tune it per variant):

  data      dates from 2010-01-01 on, so every variant (including later XBRL
            fundamentals, which start ~2009-2011) trains and tests on the same span
  target    train on target_open demeaned per date; score Spearman IC against
            target_open (entry at the next open, exit at the close of t+15)
  features  stock-level features are ranked per date to [-0.5, 0.5]; the five
            market-level features stay raw (they are constant within a date)
  models    one fixed GBM (GBM_PARAMS) and a ridge on the ranked stock features
  schemes   walk-forward: refit each January on all earlier dates (30-day
            embargo), test years 2013..last
            frozen:       fit once on 2010-2016, score each later year unrefitted
  metrics   IC overall / 2019+ / 2023+, paired dIC vs baseline (t > 2.5 needed,
            several variants are compared), top-20 minus universe mean return
            per 15 days before and after 10 bps, top-20 turnover

The gate before building the prospective simulation (GATE) is printed at the end.

usage (nori env):
  python eval_features.py frame_sp900_a.parquet frame_sp900_meta.parquet
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

START, FIRST_TEST_YEAR, FROZEN_TRAIN_END = "2010-01-01", 2013, 2016
EMBARGO = pd.Timedelta(days=30)
TOP_K, COST = 20, 0.0010
HELPER = ("target", "naive_momentum", "ticker", "bar")
MARKET = ("bench_ret_5d", "bench_ret_20d", "bench_vol_20", "vix", "vix_chg_5d")
GROUPS = {"res": ("res_",), "sec": ("sec_",), "on": ("on_", "id_"), "oth": ("oth_",)}
GBM_PARAMS = dict(
    loss="absolute_error",
    learning_rate=0.03,
    max_iter=300,
    max_depth=4,
    max_leaf_nodes=15,
    min_samples_leaf=500,
    l2_regularization=1.0,
    max_features=0.7,
    early_stopping=False,
    random_state=0,
)
RIDGE_ALPHA = 10.0
GATE = "IC >= 0.03 with t > 3; 2019+ IC > 0 with t > 2; 2023+ IC > 0; net top-20 spread > 0"


def tstat(a: np.ndarray) -> float:
    """t-statistic of the mean, treating entries (dates) as independent."""
    a = np.asarray(a, float)
    return a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))


def rank_by_date(x: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Per-date percentile ranks of each column, centred on 0; NaN stays NaN."""
    out = np.full_like(x, np.nan, dtype=np.float32)
    for g in groups:
        block = x[g]
        for j in range(block.shape[1]):
            col = block[:, j]
            ok = np.isfinite(col)
            if ok.sum() > 1:
                out[g[ok], j] = (rankdata(col[ok]) - 0.5) / ok.sum() - 0.5
    return out


def main() -> None:
    """Fit every variant under both schemes and print the pre-registered table."""
    frame = pd.read_parquet(sys.argv[1])
    meta = pd.read_parquet(sys.argv[2])
    assert (
        len(frame) == len(meta) and (frame["ticker"].to_numpy() == meta["ticker"].to_numpy()).all()
    )
    frame["target_open"] = meta["target_open"].to_numpy()
    frame = frame[(frame.index >= START) & frame["target_open"].notna()].sort_index(kind="stable")

    base = [c for c in frame.columns[:36] if c not in HELPER]
    new = [c for c in frame.columns[36:] if c != "target_open"]
    group_cols = {g: [c for c in new if c.startswith(p)] for g, p in GROUPS.items()}
    variants = {"baseline": base, **{f"+{g}": base + cols for g, cols in group_cols.items()}}
    variants["+A"] = base + new
    stock_cols = [c for c in base + new if c not in MARKET]

    dates = frame.index.to_numpy()
    uniq = np.unique(dates)
    groups = [np.flatnonzero(dates == d) for d in uniq]
    y = frame["target_open"].to_numpy(float)
    y_dm = y - pd.Series(y).groupby(dates).transform("mean").to_numpy()
    ranked = pd.DataFrame(
        rank_by_date(frame[stock_cols].to_numpy(np.float32), groups),
        index=frame.index,
        columns=stock_cols,
    )
    for c in MARKET:
        ranked[c] = frame[c].to_numpy(np.float32)
    years = pd.DatetimeIndex(uniq).year
    print(
        f"{len(frame):,} rows, {len(uniq)} dates {pd.Timestamp(uniq[0]):%Y-%m-%d}..{pd.Timestamp(uniq[-1]):%Y-%m-%d}, "
        f"baseline {len(base)} features, new {len(new)}: "
        + ", ".join(f"{g} {len(c)}" for g, c in group_cols.items()),
        flush=True,
    )

    def fit_predict(model: str, cols: list[str], train: np.ndarray, test: np.ndarray) -> np.ndarray:
        if model == "gbm":
            X = ranked[cols].to_numpy(np.float32)
            reg = HistGradientBoostingRegressor(**GBM_PARAMS).fit(X[train], y_dm[train])
            return reg.predict(X[test])
        lin = [c for c in cols if c not in MARKET]
        X = np.nan_to_num(ranked[lin].to_numpy(np.float64))  # missing rank -> the median
        return Ridge(alpha=RIDGE_ALPHA).fit(X[train], y_dm[train]).predict(X[test])

    def date_mask(lo: pd.Timestamp | None, hi: pd.Timestamp) -> np.ndarray:
        m = dates < hi
        return m if lo is None else m & (dates >= lo)

    results: dict[tuple[str, str, str], pd.Series] = {}
    picks: dict[tuple[str, str, str], pd.Series] = {}
    t0 = time.perf_counter()
    for scheme in ("walk", "frozen"):
        for model in ("gbm", "ridge"):
            for name, cols in variants.items():
                pred = np.full(len(frame), np.nan)
                last_year = int(years.max())
                if scheme == "walk":
                    blocks = [(yr, yr + 1) for yr in range(FIRST_TEST_YEAR, last_year + 1)]
                else:
                    blocks = [(FROZEN_TRAIN_END + 1, last_year + 1)]
                for lo_y, hi_y in blocks:
                    lo = pd.Timestamp(f"{lo_y}-01-01")
                    test = np.flatnonzero(date_mask(lo, pd.Timestamp(f"{hi_y}-01-01")))
                    if not len(test):
                        continue
                    train = np.flatnonzero(dates <= dates[test[0]] - EMBARGO)
                    pred[test] = fit_predict(model, cols, train, test)
                key = (scheme, model, name)
                ics, spreads = {}, {}
                for d, g in zip(uniq, groups, strict=True):
                    p = pred[g]
                    if not np.isfinite(p).all():
                        continue
                    ics[d] = (
                        np.corrcoef(rankdata(p), rankdata(y[g]))[0, 1] if np.ptp(p) > 0 else 0.0
                    )
                    top = g[np.argsort(-p, kind="stable")[:TOP_K]]
                    spreads[d] = (
                        y[top].mean() - y[g].mean(),
                        frozenset(frame["ticker"].to_numpy()[top]),
                    )
                results[key] = pd.Series(ics)
                picks[key] = pd.Series(spreads)
                print(
                    f"  {scheme:>6} {model:>5} {name:>8}  IC {results[key].mean():+.4f}  "
                    f"({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )

    def row(key: tuple[str, str, str]) -> dict[str, float]:
        s = results[key]
        yr = pd.DatetimeIndex(s.index).year
        out = {
            "IC": s.mean(),
            "t": tstat(s),
            "IC_2019+": s[yr >= 2019].mean(),
            "t_2019+": tstat(s[yr >= 2019]),
            "IC_2023+": s[yr >= 2023].mean(),
        }
        base_s = results[(key[0], key[1], "baseline")]
        d = (s - base_s).dropna()
        out["dIC"], out["t_dIC"] = (d.mean(), tstat(d)) if key[2] != "baseline" else (0.0, 0.0)
        sp = picks[key]
        spread = np.array([v[0] for v in sp])
        names = [v[1] for v in sp]
        turnover = np.mean(
            [1 - len(a & b) / TOP_K for a, b in zip(names[:-1], names[1:], strict=False)]
        )
        out["top20"] = spread.mean()
        out["top20_net"] = spread.mean() - COST * turnover  # each replaced name is one round trip
        out["turnover"] = turnover
        return out

    for scheme in ("walk", "frozen"):
        print(f"\n=== {scheme} " + "=" * 90)
        table = pd.DataFrame(
            {f"{m}:{n}": row((scheme, m, n)) for m in ("gbm", "ridge") for n in variants}
        ).T
        print(table.to_string(float_format=lambda v: f"{v:+.4f}"))

    print("\n=== frozen model by year (fit 2010-2016, no refit) " + "=" * 40)
    by_year = pd.DataFrame(
        {
            f"{m}:{n}": results[("frozen", m, n)]
            .groupby(pd.DatetimeIndex(results[("frozen", m, n)].index).year)
            .mean()
            for m in ("gbm", "ridge")
            for n in ("baseline", "+A")
        }
    )
    print(by_year.to_string(float_format=lambda v: f"{v:+.4f}"))

    print(f"\nGATE (walk-forward): {GATE}")
    for m in ("gbm", "ridge"):
        r = row(("walk", m, "+A"))
        ok = (
            r["IC"] >= 0.03
            and r["t"] > 3
            and r["IC_2019+"] > 0
            and r["t_2019+"] > 2
            and r["IC_2023+"] > 0
            and r["top20_net"] > 0
        )
        print(f"  {m}:+A  {'PASS' if ok else 'FAIL'}")
    pd.DataFrame({f"{k[0]}:{k[1]}:{k[2]}": v for k, v in results.items()}).to_parquet(
        "eval_features_ic.parquet"
    )


if __name__ == "__main__":
    main()
