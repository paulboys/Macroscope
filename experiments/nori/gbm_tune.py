"""Tune GBM on the TRAINING period only, then freeze it for the Nori comparison.

The 2007+ test set never influences a choice here. Otherwise the tuned GBM would
have a look-ahead edge over Nori and the comparison would mean nothing.

  1. Walk-forward folds over the pre-test dates (expanding window, a gap of
     GAP_DATES between train and validation, matching the 30-day embargo).
  2. Random search; each config is scored by mean per-date Spearman IC across
     folds, the same metric the test uses (not MSE).
  3. n_iter is chosen by scoring staged predictions, not by sklearn's random
     early-stopping split, which would leak across time.
  4. The winner is refit on all training rows (bagged over N_BAG seeds when it
     subsamples features) and scored once on the test set built exactly as
     nori_fair.py builds it.

usage: python gbm_tune.py FRAME [names_per_date=200] [n_configs=60] [--xs] [--tag T]
                          [--meta PATH]

  names_per_date  names sampled per test date; 0 scores every name
  --xs      rank features within date and demean the target within date. Reported
            as a separate line: Nori gets raw inputs, so this is not like for like.
  --tag T   prefix for the output files, so a new frame cannot overwrite old runs
  --meta P  next-open meta file written by build_sp900.py (default: FRAME_meta.parquet
            beside the frame, if present). When found, the test is also scored against
            target_open, entering at the next day's open instead of today's close.
            That is information only: no choice is made from it.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor

parser = argparse.ArgumentParser()
parser.add_argument("frame")
parser.add_argument("names_per_date", nargs="?", type=int, default=200)
parser.add_argument("n_configs", nargs="?", type=int, default=60)
parser.add_argument("--xs", action="store_true")
parser.add_argument("--demean", action="store_true")
parser.add_argument("--tag", default="")
parser.add_argument("--meta", default=None)
_a = parser.parse_args()
XS = _a.xs
DEMEAN = _a.demean or XS  # --xs implies a demeaned target
PATH = _a.frame
NAMES_PER_DATE = _a.names_per_date
N_CONFIGS = _a.n_configs
TAG = (f"{_a.tag}_" if _a.tag else "") + (f"n{NAMES_PER_DATE}" if NAMES_PER_DATE else "nall")
TAG += "_xs" if XS else ("_dm" if DEMEAN else "")
MODE = "xs" if XS else ("demean" if DEMEAN else "raw")
META_PATH = Path(_a.meta) if _a.meta else Path(PATH).with_name(Path(PATH).stem + "_meta.parquet")

HORIZON, TEST_FROM, MIN_NAMES, SEED = 15, pd.Timestamp("2007-01-01"), 8, 0
N_FOLDS, GAP_DATES, N_BAG = 5, 2, 5
ITER_GRID = [25, 50, 100, 150, 200, 300, 450, 600, 900]
MSL_FRAC = (0.0015, 0.06)  # min_samples_leaf as a share of training rows (50-2000 at 32k)

frame = pd.read_parquet(PATH).sort_index()
helper = ("target", "naive_momentum", "ticker", "bar")
feat_cols = [c for c in frame.columns if c not in helper]

# --- identical split and test-set construction to nori_fair.py -----------------
dates = np.array(sorted(frame.index.unique()))
test_pool = dates[dates >= TEST_FROM]
embargo_cut = test_pool[0] - pd.Timedelta(days=HORIZON * 2)
train = frame[frame.index <= embargo_cut]
test_all = frame[frame.index.isin(set(test_pool))]
counts = test_all.groupby(level=0).size()
test_all = test_all[test_all.index.isin(counts[counts >= MIN_NAMES].index)].copy()
test_all["_row"] = np.arange(len(test_all))  # to find the subsample after --xs

rng = np.random.default_rng(SEED)
if NAMES_PER_DATE > 0:
    test = test_all.groupby(level=0, group_keys=False).apply(
        lambda g: g.iloc[
            np.sort(rng.choice(len(g), size=min(NAMES_PER_DATE, len(g)), replace=False))
        ]
    )
else:
    test = test_all  # every name on every test date
N_TRAIN = len(train)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Under --xs rank features within each date; under --xs or --demean demean the target.

    Demeaning removes the market-wide move, so the model is trained on relative
    performance, which is what the rank metric scores. Uses only same-date rows,
    so it cannot leak across time. With neither flag the frame is returned unchanged.
    """
    if not DEMEAN:
        return df
    out = df.copy()
    if XS:
        for c in feat_cols:
            out[c] = out.groupby(level=0)[c].rank(pct=True) - 0.5
    out["target"] = out["target"] - out.groupby(level=0)["target"].transform("mean")
    return out


# Rank the test features against the whole universe on each date, as production
# would, then keep the subsampled rows. Ranking within the 200-name subsample
# would give the test a different feature distribution from training.
train_t = transform(train)
test_t = transform(test_all).set_index("_row", append=True)
test_t = test_t[test_t.index.get_level_values("_row").isin(test["_row"])]
test_t = test_t.reset_index(level="_row", drop=True)
print(
    f"train {len(train):,} rows / {train.index.nunique()} dates   "
    f"test {len(test):,} rows / {test.index.nunique()} dates   mode {MODE}",
    flush=True,
)


def date_groups(idx: pd.Index) -> list[np.ndarray]:
    """Row positions for each date with at least 4 rows, computed once per row set."""
    _, inv = np.unique(np.asarray(idx), return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
    return [g for g in groups if len(g) >= 4]


def per_date_ic(pred: np.ndarray, truth: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Spearman correlation of pred with truth within each date group.

    One entry per group. A date where the prediction is constant ranks nothing,
    so it counts as IC 0, not as missing. Dropping those dates lets a model that
    only splits on market-wide features (identical for every stock on a date)
    score on the few dates where it happens to vary, and tuning then selects it.
    """
    out = np.zeros(len(groups))
    for k, g in enumerate(groups):
        p = pred[g]
        if np.ptp(p) > 0:  # np.std of a constant float array can be ~1e-17, not 0
            out[k] = np.corrcoef(rankdata(p), rankdata(truth[g]))[0, 1]
    return out


def flat_dates(pred: np.ndarray, groups: list[np.ndarray]) -> int:
    """How many date groups the prediction does not rank at all (constant within the date)."""
    return int(sum(np.ptp(pred[g]) == 0 for g in groups))


def tstat(a: np.ndarray) -> float:
    """t-statistic of the mean of a, ignoring NaN, treating entries as independent."""
    a = a[np.isfinite(a)]
    return a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))


# --- walk-forward folds over training dates ------------------------------------
tr_dates = np.array(sorted(train.index.unique()))
first_val = len(tr_dates) // 3  # first third is training only
bounds = np.linspace(first_val, len(tr_dates), N_FOLDS + 1).astype(int)
fold_data = []
for k in range(N_FOLDS):
    v0, v1 = bounds[k], bounds[k + 1]
    fit_d, val_d = tr_dates[: v0 - GAP_DATES], tr_dates[v0:v1]
    a = train_t[train_t.index.isin(set(fit_d))]
    b = train_t[train_t.index.isin(set(val_d))]
    fold_data.append(
        (
            a[feat_cols].to_numpy(float),
            a["target"].to_numpy(float),
            b[feat_cols].to_numpy(float),
            b["target"].to_numpy(float),
            date_groups(b.index),
        )
    )
    print(
        f"  fold {k + 1}: fit {len(fit_d)} dates, val {len(val_d)} dates "
        f"({pd.Timestamp(val_d[0]).date()} .. {pd.Timestamp(val_d[-1]).date()})"
    )


def sample_config(r: np.random.Generator) -> dict:
    """Draw one hyperparameter set.

    max_leaf_nodes is capped at 2**max_depth: a depth-3 tree cannot have more
    than 8 leaves, so larger caps would only duplicate configs.
    """
    depth = int(r.choice([3, 4, 5, 6]))
    leaves = [n for n in (7, 15, 31, 63) if n <= 2**depth]
    return {
        "learning_rate": float(np.exp(r.uniform(np.log(0.01), np.log(0.1)))),
        "max_leaf_nodes": int(r.choice(leaves)),
        "max_depth": depth,
        "min_samples_leaf": int(
            np.exp(r.uniform(np.log(MSL_FRAC[0] * N_TRAIN), np.log(MSL_FRAC[1] * N_TRAIN)))
        ),
        "l2_regularization": float(r.choice([0.0, 0.1, 1.0, 10.0])),
        "max_features": float(r.choice([0.5, 0.7, 1.0])),
        "loss": str(r.choice(["squared_error", "absolute_error"])),
    }


def make(cfg: dict, n_iter: int, seed: int) -> HistGradientBoostingRegressor:
    """A GBM with early stopping off, so n_iter means what it says."""
    return HistGradientBoostingRegressor(
        max_iter=n_iter, early_stopping=False, random_state=seed, **cfg
    )


def score_config(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fold IC at every ITER_GRID count, from one fit per fold.

    Returns (mean over folds per iteration count, the folds x counts array).
    """
    per_iter = np.zeros((N_FOLDS, len(ITER_GRID)))
    col = {n: j for j, n in enumerate(ITER_GRID)}
    for k, (Xa, ya, Xb, yb, gb) in enumerate(fold_data):
        m = make(cfg, max(ITER_GRID), SEED).fit(Xa, ya)
        for i, p in enumerate(m.staged_predict(Xb), start=1):
            if i in col:
                per_iter[k, col[i]] = np.nanmean(per_date_ic(p, yb, gb))
    return per_iter.mean(axis=0), per_iter


# nori_fair.py's settings with early stopping off. Its actual model stops early
# on a random 10% holdout (sklearn's default above 10k rows); that one is scored
# on the test below as gbm-default.
default_cfg = {
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_features": 1.0,
    "loss": "squared_error",
}

r = np.random.default_rng(SEED + 1)
results, t0 = [], time.perf_counter()
for i in range(N_CONFIGS):
    cfg = sample_config(r)
    mean_by_iter, per_iter = score_config(cfg)
    j = int(np.argmax(mean_by_iter))
    results.append(
        {
            "cfg": cfg,
            "n_iter": ITER_GRID[j],
            "cv_ic": float(mean_by_iter[j]),
            "fold_ic": per_iter[:, j].tolist(),
        }
    )
    print(
        f"  cfg {i + 1:>2}/{N_CONFIGS}  cv IC {mean_by_iter[j]:+.4f} @ {ITER_GRID[j]:>3} it "
        f"  lr {cfg['learning_rate']:.3f} leaves {cfg['max_leaf_nodes']:>2} "
        f"depth {cfg['max_depth']} msl {cfg['min_samples_leaf']:>4} "
        f"l2 {cfg['l2_regularization']:<4} mf {cfg['max_features']} {cfg['loss']}  "
        f"({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )

d_mean, _ = score_config(default_cfg)
dj = int(np.argmax(d_mean))
print(
    f"\nnori_fair's settings, same CV: best {d_mean[dj]:+.4f} @ {ITER_GRID[dj]} it; "
    f"at 300 it {d_mean[ITER_GRID.index(300)]:+.4f}"
)

results.sort(key=lambda x: -x["cv_ic"])
print("\nTop 5 by CV IC (folds shown, to see how stable they are).")
print("The top score is the max of N_CONFIGS noisy estimates, so it is optimistic.")
for x in results[:5]:
    print(
        f"  {x['cv_ic']:+.4f}  folds {np.round(x['fold_ic'], 3).tolist()}  "
        f"{x['n_iter']} it  {x['cfg']}"
    )

best = results[0]
with open(f"gbm_tuned_{TAG}.json", "w") as fh:
    json.dump(
        {
            "cfg": best["cfg"],
            "n_iter": best["n_iter"],
            "cv_ic": best["cv_ic"],
            "fold_ic": best["fold_ic"],
            "default_cv_ic": float(d_mean[dj]),
            "mode": MODE,
            "frame": str(PATH),
            "n_train_rows": N_TRAIN,
        },
        fh,
        indent=2,
    )

# --- the only time the test set is touched -------------------------------------
Xtr, ytr = train_t[feat_cols].to_numpy(float), train_t["target"].to_numpy(float)
Xte, yte = test_t[feat_cols].to_numpy(float), test["target"].to_numpy(float)
te_groups = date_groups(test.index)

m_default = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED).fit(
    Xtr, ytr
)
models = {
    "gbm-default": m_default.predict(Xte),
    "gbm-tuned": make(best["cfg"], best["n_iter"], SEED).fit(Xtr, ytr).predict(Xte),
}
# With max_features=1.0 the fit is deterministic and every seed gives the same
# trees, so a bag would be the single model under another name.
if best["cfg"]["max_features"] < 1.0:
    models[f"gbm-tuned-bag{N_BAG}"] = np.mean(
        [
            make(best["cfg"], best["n_iter"], SEED + s).fit(Xtr, ytr).predict(Xte)
            for s in range(N_BAG)
        ],
        axis=0,
    )
else:
    print("\nbag skipped: the tuned config uses every feature, so it is deterministic")

print("\n" + "=" * 70)
names_label = f"{NAMES_PER_DATE}" if NAMES_PER_DATE else "all"
print(f"TEST ({len(test):,} rows, {names_label} names/date, {MODE} inputs)")
print(f"gbm-default early-stopped at {m_default.n_iter_} of 300 iterations")
out = test[["ticker", "target"]].copy()
base = per_date_ic(models["gbm-default"], yte, te_groups)
for name, p in models.items():
    a = per_date_ic(p, yte, te_groups)
    line = (
        f"  {name:>16}  IC {np.nanmean(a):+.4f}  t {tstat(a):+.2f}  "
        f"flat dates {flat_dates(p, te_groups)}/{len(te_groups)}"
    )
    if name != "gbm-default":
        dd = a - base
        line += f"   vs default: dIC {np.nanmean(dd):+.4f} t {tstat(dd):+.2f}"
    print(line)
    out[name] = p
print("=" * 70)

# --- information only: the same predictions, entered at the next day's open ----
if META_PATH.exists():
    meta = pd.read_parquet(META_PATH).set_index("ticker", append=True)["target_open"]
    meta = meta[~meta.index.duplicated()]
    key = pd.MultiIndex.from_arrays([test.index, test["ticker"]])
    t_open = meta.reindex(key).to_numpy(float)
    ok = np.isfinite(t_open)
    og = date_groups(test.index[ok])
    out["target_open"] = t_open
    print(f"\nENTRY AT NEXT OPEN ({ok.sum():,} rows with target_open); no choice depends on this")
    print(f"  {'':>16}  {'IC vs close-entry':>20}  {'IC vs open-entry':>20}  {'change':>8}")
    for name, p in models.items():
        a = per_date_ic(p[ok], test["target"].to_numpy(float)[ok], og)
        b = per_date_ic(p[ok], t_open[ok], og)
        print(
            f"  {name:>16}  {np.nanmean(a):+.4f} (t {tstat(a):+5.2f})  "
            f"{np.nanmean(b):+.4f} (t {tstat(b):+5.2f})  {np.nanmean(b) - np.nanmean(a):+.4f}"
        )
else:
    print(f"\n(no meta file at {META_PATH}; next-open scoring skipped)")
out.to_parquet(f"gbm_tuned_preds_{TAG}.parquet")
print(f"saved gbm_tuned_{TAG}.json and gbm_tuned_preds_{TAG}.parquet")
