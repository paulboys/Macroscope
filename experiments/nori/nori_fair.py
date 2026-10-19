"""Nori vs ridge/GBM/naive on an IDENTICAL subsampled test set.

Two constraints force the subsample, both measured rather than assumed:

  - Query memory. Nori's RBF encoder materialises the whole query set at
    once, about 17.6 KB per query row at 32 features, so 92,599 rows asked
    for 1.63 GB in one tensor and died. Ceiling is ~50k rows on 8 GB.
  - Query time. ~0.08 s per query row after the context pass.

Everything is scored on the same rows so the comparison is not confounded by
who saw what.
"""

import sys
import time

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(2)
from pathlib import Path  # noqa: E402

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from synthefy_nori import NoriRegressor  # noqa: E402

# Random seed for context subsampling
CONTEXT_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1

# Names sampled per test date. Changing it changes the test row order, so the
# chunk cache must be keyed on it.
NAMES_PER_DATE = int(sys.argv[3]) if len(sys.argv) > 3 else 45

# Cache directory for chunked Nori predictions, unique per context seed and test
# set. The original 45-name layout keeps its old name so those chunks still load.
CACHE = Path(
    f"nori_chunks_seed{CONTEXT_SEED}" + ("" if NAMES_PER_DATE == 45 else f"_n{NAMES_PER_DATE}")
)
CACHE.mkdir(exist_ok=True)

HORIZON, TEST_FROM, MIN_NAMES = 15, pd.Timestamp("2007-01-01"), 8
CONTEXT_CAP, CHUNK, SEED = 2500, 1000, 0

frame = pd.read_parquet(sys.argv[1]).sort_index()
helper = ("target", "naive_momentum", "ticker", "bar")
feat_cols = [c for c in frame.columns if c not in helper]

dates = np.array(sorted(frame.index.unique()))
test_pool = dates[dates >= TEST_FROM]
embargo_cut = test_pool[0] - pd.Timedelta(days=HORIZON * 2)

train = frame[frame.index <= embargo_cut]
test_all = frame[frame.index.isin(set(test_pool))]
counts = test_all.groupby(level=0).size()
test_all = test_all[test_all.index.isin(counts[counts >= MIN_NAMES].index)]

# One subsample, used by every model.
rng = np.random.default_rng(SEED)
test = test_all.groupby(level=0, group_keys=False).apply(
    lambda g: g.iloc[np.sort(rng.choice(len(g), size=min(NAMES_PER_DATE, len(g)), replace=False))]
)

print(f"train {len(train):,} rows")
print(
    f"test  {len(test):,} rows over {test.index.nunique()} dates, "
    f"{NAMES_PER_DATE} names/date (subsampled from {len(test_all):,})"
)
print(
    f"expected Nori time at ~0.5 s/row: {len(test) * 0.5 / 3600:.1f} h " f"(cache {CACHE})",
    flush=True,
)

X_tr, y_tr = train[feat_cols].to_numpy(float), train["target"].to_numpy(float)
X_te, y_te = test[feat_cols].to_numpy(float), test["target"].to_numpy(float)
te_dates = test.index.to_numpy()


def ic_series(pred: np.ndarray) -> pd.Series:
    """Spearman IC per date, indexed by date."""
    out = {}
    for d in np.unique(te_dates):
        m = te_dates == d
        if m.sum() < 4 or np.std(pred[m]) == 0:
            continue
        v = pd.Series(pred[m]).corr(pd.Series(y_te[m]), method="spearman")
        if np.isfinite(v):
            out[d] = v
    return pd.Series(out)


def ic(pred: np.ndarray) -> tuple[float, float, int, float]:
    """Mean per-date IC, its t-stat across dates, the date count and the share above 0."""
    a = ic_series(pred).to_numpy()
    return a.mean(), a.mean() / (a.std(ddof=1) / np.sqrt(len(a))), len(a), float(np.mean(a > 0))


all_preds = {}
rows = []
for name, pred in (
    ("naive-reversal", -test["naive_momentum"].to_numpy()),
    (
        "ridge",
        make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25), cv=TimeSeriesSplit(5))
        )
        .fit(X_tr, y_tr)
        .predict(X_te),
    ),
    (
        "gbm",
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED)
        .fit(X_tr, y_tr)
        .predict(X_te),
    ),
):
    all_preds[name] = np.asarray(pred, float)
    m, t, n, pos = ic(pred)
    rows.append(
        {
            "model": name,
            "IC": m,
            "IC_t": t,
            "n_dates": n,
            "dates>0": pos,
            "train_rows": len(X_tr),
            "secs": 0.0,
        }
    )
    print(f"  {name:>14}  IC {m:+.4f}  t {t:+.2f}", flush=True)

# --- Nori, chunked to stay inside memory ----------------------------------
# Subsample the training context for Nori using CONTEXT_SEED and CONTEXT_CAP.
ctx_rng = np.random.default_rng(CONTEXT_SEED)
# Select CONTEXT_CAP rows from the training set using the context RNG.
idx = np.sort(ctx_rng.choice(len(train), size=min(CONTEXT_CAP, len(train)), replace=False))
# Extract the context rows from the training set.
ctx = train.iloc[idx]
# Fit the Nori model on the subsampled context.
reg = NoriRegressor(model="nori-6m", device="cpu")
# fit() stores the context; no weights are updated
reg.fit(ctx[feat_cols].to_numpy(np.float32), ctx["target"].to_numpy(np.float32))

preds, t0 = [], time.perf_counter()
for start in range(0, len(X_te), CHUNK):
    f = CACHE / f"chunk_{start:06d}.npy"
    if f.exists():
        p = np.load(f)
    else:
        block = X_te[start : start + CHUNK].astype(np.float32)
        p = np.asarray(reg.predict(block, output_type="median")).ravel()
        np.save(f, p)
    preds.append(p)
    print(
        f"    chunk {start // CHUNK + 1}/{-(-len(X_te) // CHUNK)} "
        f"({time.perf_counter() - t0:.0f}s elapsed)",
        flush=True,
    )
nori_pred = np.concatenate(preds)
dt = time.perf_counter() - t0

all_preds["nori-6m"] = np.asarray(nori_pred, float)
m, t, n, pos = ic(nori_pred)
rows.append(
    {
        "model": "nori-6m",
        "IC": m,
        "IC_t": t,
        "n_dates": n,
        "dates>0": pos,
        "train_rows": len(ctx),
        "secs": round(dt, 1),
    }
)

print("\n" + "=" * 86)
print(pd.DataFrame(rows).set_index("model").to_string(float_format=lambda v: f"{v:.4f}"))
print("=" * 86)
print("train_rows shows the handicap: Nori sees CONTEXT_CAP rows, the others all of them.")

# Paired difference against GBM on the same dates: the number that decides it.
gbm_ic = ic_series(all_preds["gbm"])
print("\nPaired vs gbm, per-date IC difference")
for name in ("naive-reversal", "ridge", "nori-6m"):
    d = (ic_series(all_preds[name]) - gbm_ic).dropna().to_numpy()
    print(
        f"  {name:>14}  dIC {d.mean():+.4f}  t {d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.2f}"
    )

# Save every model's predictions so rescoring never needs Nori re-run.
out = test[["ticker", "target"]].copy()
for name, p in all_preds.items():
    out[name] = p
out_path = f"preds_seed{CONTEXT_SEED}_n{NAMES_PER_DATE}.parquet"
out.to_parquet(out_path)
print(f"\nsaved predictions to {out_path}")
