# Nori vs a tuned GBM: 15-day forward return ranking

Written for: anyone picking this experiment up later (you in a month included).

## Setup

- **Data:** `frame_names.parquet`, 300 tickers, 32 features, target = forward 15-day return. Dates are
  spaced about 15 trading days apart, so target windows do not overlap.
- **Split:** train on dates up to 2006-12-02 (32,303 rows, 231 dates); test 2007-01-01 onward
  (331 dates). The 30-day embargo is applied between them.
- **Test set:** 200 names sampled per date, 66,200 rows. Every model is scored on the same rows.
- **Metric:** Spearman rank correlation (IC) of prediction against target within each date, averaged
  over dates; t-stat across dates.
- **Nori:** `nori-6m`, CPU, context capped at 2,500 randomly drawn training rows (context seed 1).
  Inference took 32,878 s (9.1 h).
- **Tuned GBM:** scikit-learn `HistGradientBoostingRegressor`, best of 60 random configurations,
  chosen by 5 walk-forward folds on the pre-2007 data only. The test set was used once, after the
  choice. Selected config: absolute-error loss, learning rate 0.021, 15 leaves, depth 4,
  `min_samples_leaf` 419, `max_features` 0.5, 200 iterations. Raw feature inputs, as Nori receives.

## Result

| Model | IC | t | Dates with IC > 0 |
|---|---|---|---|
| naive-reversal | +0.0267 | 3.95 | 58.9% |
| ridge | +0.0045 | 0.66 | 52.9% |
| gbm, default settings | +0.0290 | 4.76 | 59.5% |
| **gbm, tuned** | **+0.0750** | **10.67** | **74.3%** |
| gbm, tuned, 5-seed bag | +0.0751 | 10.47 | 71.9% |
| nori-6m | +0.0575 | 7.62 | 70.1% |
| nori + tuned gbm, rank-averaged | +0.0739 | 9.81 | 74.3% |

Paired, per-date IC difference:

| Comparison | dIC | t |
|---|---|---|
| nori - tuned gbm | -0.0175 | -3.02 (95% bootstrap CI -0.030 to -0.007) |
| nori - default gbm | +0.0285 | +3.39 |

Sub-periods, nori minus tuned gbm: 2007-10 -0.045 (t -4.25), 2011-14 -0.004 (t -0.38),
2015-18 -0.024 (t -1.64), 2019-22 -0.004 (t -0.27), 2023-26 -0.009 (t -0.74).
Per-date rank correlation between the two models' predictions: 0.515.

## Reading it

- Nori carries a real signal (t 7.6) but a tuned GBM scores higher, and the gap is significant.
- The earlier "Nori beats GBM" result (45 names, Nori 0.061-0.070 vs GBM 0.040) compared against an
  untuned GBM. It is true only of that baseline.
- The tuned GBM leads in all five 4-year periods. The gap is clearly significant only in 2007-10; from
  2011 it is small and not distinguishable from zero.
- Rank-averaging Nori with the tuned GBM does not beat the tuned GBM alone.

## Caveats

- One Nori context seed. In the 45-name runs, three seeds varied by about 0.01 IC, well under the
  0.0175 gap, so another seed is unlikely to reverse it. Not checked at 200 names.
- The best-of-60 CV score (+0.091) overstates the tuned config; the test number (+0.075) is the honest
  one, since the test set took no part in selection. The top five configs were close in CV.
- Nori sees 2,500 context rows; GBM sees all 32,303. That is how the model is run here, not a like for
  like data budget.
- Valid for this feature set, universe, and 15-day horizon only.
- `nori_fair.py`'s untuned GBM early-stops on sklearn's automatic random 10% holdout (132 of 300
  iterations), so it is not the 300-iteration model the code suggests.

## Reproduce

```
python gbm_tune.py frame_names.parquet 200 60                      # about 11 min
python nori_fair.py frame_names.parquet 1 200 > out_fair_seed1_n200.txt 2>&1   # about 9 h
python compare_nori_gbm.py 1 200
```

Files: `gbm_tuned_n200.json` (chosen params), `gbm_tuned_preds_n200.parquet`,
`preds_seed1_n200.parquet`, `out_gbm_tune_n200.txt`, `out_fair_seed1_n200.txt`.
