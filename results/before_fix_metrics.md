# Before-fix (pre-leakage-fix) URL model metrics — REGENERATED, not original

**These numbers were regenerated on 2026-08-03, not recovered from a saved file.** The
original pre-fix run's metrics were never written to disk anywhere in this repo or its
git history (confirmed by an exhaustive `git log --all -p` search). This file is a
**faithful reconstruction** of that run, built and verified as described below — treat
it as the honest regenerated evidence, not as a recovery of the original number.

## What this reproduces, and what it does not

Pre-fix commit identified: **`98958b5`** ("add fastapi app and frontend"), the true git
DAG parent of **`880337e`** ("fix data leakage, switch to real benign urls from
curlie" — the fix commit named in `results/model_comparison.md`).

Investigation before training (full detail in conversation) found that **`src/features.py`
and `src/train_xgboost.py` have never changed in this repo's history** — byte-identical
from the first commit through HEAD — and `train_xgboost.py` has used a **domain-grouped
split since commit 1**. Checking out the pre-fix commit and retraining verbatim therefore
reproduces **zero domain overlap**, not leakage. The only script ever committed with a
plain (non-grouped) random split is `train_baseline.py`'s Logistic Regression baseline,
also unchanged since commit 1.

This run is therefore a **deliberate, labelled reconstruction** ("Option B"), not a
literal historical checkout:

| Component | Source |
|---|---|
| Model | `train_xgboost.py::make_model()` — **unchanged code**, XGBoost, same hyperparameters as always |
| Features | `features.py::build_feature_frame()` — **unchanged code**, full 19-column feature set (no filtering; `clean_columns()` did not exist pre-fix) |
| Phishing data | `data/phishing_urls.csv` — **not reconstructed**, hash-verified byte-identical (`ce235b2ced4126c0`) to the file that trained the currently-deployed model |
| Benign data | `data/benign_urls_bare_domains.csv` — the **actual original pre-fix data** (raw Tranco bare-domain download), preserved on disk unchanged by `rebuild_benign_urls.py`'s own backup step, not reconstructed |
| Split | **Reconstructed**: plain `train_test_split(stratify=y, random_state=42)`, in place of `train_xgboost.py`'s actual (always-on) `grouped_train_test_split()`. This one line is the only authored change — it matches the split `train_baseline.py`'s LR baseline has always used, which is the only leaky split ever committed to this repo. |

## Step 2 gate — leakage confirmed before training

Domain-overlap check on this run's actual train/test indices, same methodology as
`results/review_experiments.txt`:

```
8,645 / 20,000 (43.2%) test-set domains also appear in train
(cf. results/review_experiments.txt documented: 11,574/20,000 = 57.9%, same source data,
 old LR-baseline split)
```

Same order of magnitude and same mechanism (phishing-side domain repetition — shortener
domains repeated thousands of times) as the documented figure; not identical because this
run uses the bare-domain benign set rather than whatever exact snapshot the original
review numbers were computed against, and benign-side sampling introduces some variance.
**Leakage confirmed present** — proceeded to train per the gate condition.

## Results (single run, not tuned, not repeated)

| Metric | Before (this reconstruction) | After (honest, domain-grouped — for reference) |
|---|---:|---:|
| Accuracy | **0.9986** | 0.7213 |
| Precision | **0.9998** | 0.7823 |
| Recall | **0.9974** | 0.4182 |
| F1 | **0.9986** | 0.545 |
| AUC-ROC | **0.9992** | 0.6878 |

n_train=79,996, n_test=20,000 (80/20 stratified split, class-balanced 50/50 source data).
"After" column is unchanged, sourced from `results/model_comparison.md` /
`models/deployed_model.metadata.json`, included here only for side-by-side reference.

## Honest comparison against the dissertation's claimed headline numbers

- **Accuracy**: claimed "99.9%" vs regenerated **99.86%** — a match within normal
  rounding (99.86% rounds to 99.9% at one decimal place).
- **Recall**: claimed "93.7%" vs regenerated **99.74%** — **this does NOT match.** The
  gap (≈6 points) is too large to be rounding or run-to-run noise. Do not cite "93.7%"
  as verified; either the original claim used a different benign-data generation, a
  different feature subset, or was misremembered. If you need this figure resolved
  further (e.g. by re-running against `data/benign_urls_generated_shaped.csv`, the
  intermediate shape-matched benign set, as a second reconstruction), that is a further
  authorized training run, not something already covered by this file — ask before I run
  it, per the same faithfulness standard as above.
