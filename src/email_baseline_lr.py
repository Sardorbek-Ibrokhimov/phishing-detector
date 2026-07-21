"""Logistic Regression on TF-IDF, the simple baseline the
transformer must beat to justify its complexity. Same leak-aware grouped
split as the DistilBERT run (src/email_data.py).

Trains on a fixed 20k-email subset (the shared head-to-head training set, so
LR and DistilBERT see identical data — fair on the CPU budget) and also on
the full training set as a reference. Evaluates both on the full held-out
test set and saves per-example predictions for McNemar.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from email_data import RANDOM_STATE, get_split

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_TRAIN_HEADTOHEAD = 8000  # shared with DistilBERT (sized for CPU fine-tuning)


def make_lr():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0, random_state=RANDOM_STATE),
    )


def metrics(y, pred, proba):
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "auc_roc": roc_auc_score(y, proba),
    }


def run(train, test, label):
    model = make_lr()
    model.fit(train["text"], train["label"])
    proba = model.predict_proba(test["text"])[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = metrics(test["label"].values, pred, proba)
    print(f"\n=== LR-TFIDF ({label}, train={len(train)}) ===")
    for k, v in m.items():
        print(f"  {k:10s} {v:.4f}")
    return model, pred, proba, m


def top_features(model, k=18):
    vec = model.named_steps["tfidfvectorizer"]
    lr = model.named_steps["logisticregression"]
    names = np.array(vec.get_feature_names_out())
    coef = lr.coef_[0]
    order = np.argsort(coef)
    print("\nTop LEGIT-leaning tokens:", list(names[order[:k]]))
    print("Top PHISH-leaning tokens:", list(names[order[-k:]][::-1]))


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)

    # Head-to-head: same 20k train + full test as DistilBERT.
    tr_h2h, test = get_split(cap_train=N_TRAIN_HEADTOHEAD, cap_test=None)
    model_h, pred_h, proba_h, m_h = run(tr_h2h, test, f"head-to-head {N_TRAIN_HEADTOHEAD//1000}k")
    top_features(model_h)

    # Reference: LR on the full training set (to show it isn't handicapped).
    tr_full, _ = get_split(cap_train=None, cap_test=None)
    _, _, _, m_full = run(tr_full, test, "full train ref")

    out = test[["label"]].copy()
    out["lr_pred"] = pred_h
    out["lr_proba"] = proba_h
    out.to_csv(RESULTS / "email_lr_headtohead.csv", index=False)
    pd.DataFrame([{"model": "LR-TFIDF (20k)", **m_h},
                 {"model": "LR-TFIDF (full)", **m_full}]).to_csv(
        RESULTS / "email_lr_metrics.csv", index=False)
    print(f"\nSaved: {RESULTS/'email_lr_headtohead.csv'}, {RESULTS/'email_lr_metrics.csv'}")
