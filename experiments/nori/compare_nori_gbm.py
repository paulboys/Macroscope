"""Score Nori against the tuned GBM from the saved prediction files.

No model is fit and Nori is not re-run; this only reads two parquet files:

  preds_seed{S}_n{N}.parquet       written by nori_fair.py (all models, one test set)
  gbm_tuned_preds_n{N}.parquet     written by gbm_tune.py  (same test rows)

Both are built from the same seeded subsample, so rows align one to one. That is
asserted rather than assumed, because a misaligned merge would score noise.

usage: python compare_nori_gbm.py [seed=1] [names_per_date=200]
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
N_BOOT = 5000

nori = pd.read_parquet(f"preds_seed{SEED}_n{N}.parquet")
tuned = pd.read_parquet(f"gbm_tuned_preds_n{N}.parquet")
assert len(nori) == len(tuned), "row counts differ: the test sets are not the same"
assert (nori.index == tuned.index).all(), "dates differ"
assert (nori["ticker"].to_numpy() == tuned["ticker"].to_numpy()).all(), "tickers differ"
assert np.allclose(nori["target"], tuned["target"]), "targets differ"

df = nori[["target", "naive-reversal", "ridge", "gbm", "nori-6m"]].copy()
df["gbm-tuned"] = tuned["gbm-tuned"].to_numpy()
df["gbm-tuned-bag"] = tuned[
    [c for c in tuned.columns if c.startswith("gbm-tuned-bag")][0]
].to_numpy()

dates = df.index.to_numpy()
UNIQ = np.unique(dates)
GROUPS = [np.flatnonzero(dates == d) for d in UNIQ]
TARGET = df["target"].to_numpy()


def per_date_ic(pred: np.ndarray) -> pd.Series:
    """Spearman IC of pred against the target within each date."""
    p = np.asarray(pred, float)
    return pd.Series(
        [np.corrcoef(rankdata(p[g]), rankdata(TARGET[g]))[0, 1] for g in GROUPS], index=UNIQ
    )


def tstat(a: pd.Series) -> float:
    """t-statistic of the mean, treating the per-date entries as independent."""
    return a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))


def rank_average(*cols: str) -> np.ndarray:
    """Per-date sum of ranks across models: an equal-weight ensemble."""
    out = np.zeros(len(df))
    for g in GROUPS:
        out[g] = sum(rankdata(df[c].to_numpy()[g]) for c in cols)
    return out


ic = {
    c: per_date_ic(df[c])
    for c in ("naive-reversal", "ridge", "gbm", "gbm-tuned", "gbm-tuned-bag", "nori-6m")
}
ic["nori+tuned"] = per_date_ic(rank_average("nori-6m", "gbm-tuned"))

print(f"{len(df):,} rows, {len(UNIQ)} dates, {N} names/date, nori seed {SEED}\n")
for k, s in ic.items():
    print(f"{k:>15}  IC {s.mean():+.4f}  t {tstat(s):+.2f}  dates>0 {np.mean(s > 0):.1%}")

d = ic["nori-6m"] - ic["gbm-tuned"]
rng = np.random.default_rng(0)
boot = [rng.choice(d.to_numpy(), len(d)).mean() for _ in range(N_BOOT)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(
    f"\npaired nori - gbm-tuned : dIC {d.mean():+.4f}  t {tstat(d):+.2f}  "
    f"95% CI [{lo:+.4f}, {hi:+.4f}]"
)
dd = ic["nori-6m"] - ic["gbm"]
print(f"paired nori - gbm default: dIC {dd.mean():+.4f}  t {tstat(dd):+.2f}")

print("\n4-year sub-periods (nori - gbm-tuned)")
years = pd.DatetimeIndex(UNIQ).year
for start in range(int(years.min()), int(years.max()) + 1, 4):
    m = (years >= start) & (years <= start + 3)
    if m.sum() < 5:
        continue
    print(
        f"  {start}-{min(start + 3, int(years.max()))}  n={m.sum():>3}  "
        f"nori {ic['nori-6m'][m].mean():+.4f}  tuned {ic['gbm-tuned'][m].mean():+.4f}  "
        f"diff {d[m].mean():+.4f}  t {tstat(d[m]):+.2f}"
    )

agree = np.mean(
    [
        np.corrcoef(rankdata(df["nori-6m"].to_numpy()[g]), rankdata(df["gbm-tuned"].to_numpy()[g]))[
            0, 1
        ]
        for g in GROUPS
    ]
)
print(f"\nmean per-date rank correlation, nori vs gbm-tuned: {agree:.3f}")
