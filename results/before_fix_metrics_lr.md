# Before-fix (pre-leakage-fix) LR URL model metrics — REGENERATED, not original

**These numbers were regenerated on 2026-08-03, not recovered from a saved file.** As with
the XGBoost before-fix run in `results/before_fix_metrics.md`, the original pre-fix
Logistic Regression metrics were never written to disk anywhere in this repo or its git
history (confirmed by exhaustive search — see the prior conversation turn). This file is
a **faithful reconstruction**, not a recovery of an original saved number.

## What this reproduces

Pre-fix commit reproduced: **`98958b5`** ("add fastapi app and frontend"), the true git
DAG parent of **`880337e`** ("fix data leakage, switch to real benign urls from
curlie"), same commit identified for the XGBoost before-fix run.

Unlike the XGBoost run, **no code reconstruction was needed** for Logistic Regression:
`train_baseline.py::run_url_baseline()` has used a plain (non-grouped) `train_test_split`
natively since commit 1, and is still unmodified today. This run is therefore closer to a
literal historical checkout than the XGBoost reconstruction was.

| Component | Source |
|---|---|
| Model | `train_baseline.py::run_url_baseline()`'s pipeline — `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))` — **unchanged code, verbatim** |
| Features | `features.py::build_feature_frame()` — **unchanged code, verbatim**, full 19-column feature set |
| Phishing data | `data/phishing_urls.csv` — **not reconstructed**, hash-verified byte-identical (`ce235b2ced4126c0`) to the file that trained the currently-deployed model |
| Benign data | `data/benign_urls_bare_domains.csv` — the actual original pre-fix data (raw Tranco bare-domain download), preserved on disk unchanged, not reconstructed |
| Split | `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)` — **verbatim, native to the script**, no modification |

## Step 2 gate — leakage confirmed before training

```
8,645 / 20,000 (43.2%) test-set domains also appear in train
```

Identical to the XGBoost before-fix run's overlap count — expected, since both runs use
the same merged dataset, same random_state, and the same split call (`train_baseline.py`'s
native split is what the XGBoost run's split was reconstructed to match). Same order of
magnitude and mechanism as `results/review_experiments.txt`'s documented 57.9%.
**Leakage confirmed present** — proceeded to train per the gate condition.

## Results (single run, not tuned, not repeated)

| Metric | Before — LR (this reconstruction) | Before — XGBoost (prior reconstruction) | After — XGBoost, honest (reference) |
|---|---:|---:|---:|
| Accuracy | **0.9940** | 0.9986 | 0.7213 |
| Precision | **0.9990** | 0.9998 | 0.7823 |
| Recall | **0.9889** | 0.9974 | 0.4182 |
| F1 | **0.9939** | 0.9986 | 0.545 |
| AUC-ROC | **0.9991** | 0.9992 | 0.6878 |

n_train=79,996, n_test=20,000 (80/20 stratified split, class-balanced 50/50 source data;
same split indices as the XGBoost before-fix run).

## Honest comparison against the dissertation's claimed LR "before" numbers

Claimed: accuracy 0.8695, recall 0.8462, F1 0.8664, AUC 0.9313.

**None of the four match.** The regenerated numbers are 10–15 points higher on accuracy,
recall, and F1, and about 7 points higher on AUC, than claimed:

- Accuracy: claimed 86.95% vs regenerated **99.40%** — does not match.
- Recall: claimed 84.62% vs regenerated **98.89%** — does not match.
- F1: claimed 86.64% vs regenerated **99.39%** — does not match.
- AUC: claimed 93.13% vs regenerated **99.91%** — does not match.

This is a larger and more uniform gap than the XGBoost case (where accuracy matched and
only recall diverged). Here all four figures diverge in the same direction (claimed
numbers lower across the board), which reads more like a different run configuration
entirely than transcription noise on one digit. Do not cite the claimed LR numbers as
verified. As with the XGBoost recall discrepancy, I have not chased alternate
configurations (e.g. the intermediate shape-matched benign set, a filtered feature
subset, or a different split) to try to match the claimed figures — that would risk
fitting the reconstruction to a target rather than reporting the honest one. If you want
a specific alternate configuration tried, it should be requested and gated the same way
as this run.
