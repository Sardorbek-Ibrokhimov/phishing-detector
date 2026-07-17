"""Step 5 (user instruction): check whether path_is_opaque_short is doing
honest work. Break out accuracy/recall for opaque-short-path URLs vs. not,
for both classes, on the held-out domain-grouped test set. If benign
opaque-short URLs are being flagged phishing, the feature is harmful.

Also explains the LR-vs-XGBoost divergence seen after adding the feature:
LR jumped 0.667->0.839 accuracy but XGBoost barely moved (0.721->0.722).

Historical results — documented as a
negative result. path_is_opaque_short was subsequently reverted from
features.py (the harm check + LR/XGBoost divergence were reasons why), so
this script will now raise a KeyError if re-run as-is â€” it depends on a
column that no longer exists. Kept for transparency of what was analysed.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_models import clean_columns
from features import build_feature_frame
from train_baseline import grouped_train_test_split, load_merged_url_dataset
from train_xgboost import make_model

RANDOM_STATE = 42


def breakdown(name, y_test, pred, opaque_mask):
    print(f"\n--- {name} ---")
    for cls, cls_name in [(1, "phishing"), (0, "benign")]:
        for flag, flag_name in [(True, "opaque-short path"), (False, "normal path")]:
            mask = (y_test == cls) & (opaque_mask == flag)
            n = mask.sum()
            if n == 0:
                print(f"  {cls_name:10s} | {flag_name:18s} | n=0")
                continue
            correct_label = "phishing" if cls == 1 else "legitimate"
            acc = (pred[mask] == cls).mean()
            print(f"  {cls_name:10s} | {flag_name:18s} | n={n:5d} | "
                  f"correctly classified as {correct_label}: {acc:.1%}")


def main():
    df = load_merged_url_dataset()
    X = build_feature_frame(df["url"])
    y = df["label"]
    cols = clean_columns(X.columns)
    assert "path_is_opaque_short" in cols

    train_idx, test_idx = grouped_train_test_split(df)
    X_train, X_test = X.loc[train_idx, cols], X.loc[test_idx, cols]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    opaque_mask = X_test["path_is_opaque_short"].astype(bool)

    print(f"Test set: {len(X_test)} rows, {opaque_mask.sum()} ({opaque_mask.mean():.1%}) "
          f"opaque-short-path")
    print(f"  of which phishing: {(opaque_mask & (y_test==1)).sum()}, "
          f"benign: {(opaque_mask & (y_test==0)).sum()}")

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    breakdown("Logistic Regression", y_test.values, lr_pred, opaque_mask.values)
    print(f"\n  Overall LR: acc={accuracy_score(y_test, lr_pred):.4f} recall={recall_score(y_test, lr_pred):.4f}")

    xgb = make_model()
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    breakdown("XGBoost", y_test.values, xgb_pred, opaque_mask.values)
    print(f"\n  Overall XGB: acc={accuracy_score(y_test, xgb_pred):.4f} recall={recall_score(y_test, xgb_pred):.4f}")

    # Direct explanation of the LR vs XGBoost divergence: how many of the
    # opaque-short phishing rows does each model actually catch?
    op_phish_mask = opaque_mask & (y_test == 1)
    print(f"\n=== Opaque-short phishing rows specifically ({op_phish_mask.sum()} rows) ===")
    print(f"  LR catches:  {(lr_pred[op_phish_mask.values]==1).sum()}/{op_phish_mask.sum()} "
          f"({(lr_pred[op_phish_mask.values]==1).mean():.1%})")
    print(f"  XGB catches: {(xgb_pred[op_phish_mask.values]==1).sum()}/{op_phish_mask.sum()} "
          f"({(xgb_pred[op_phish_mask.values]==1).mean():.1%})")

    non_op_phish_mask = (~opaque_mask) & (y_test == 1)
    print(f"\n=== Non-opaque phishing rows ({non_op_phish_mask.sum()} rows) ===")
    print(f"  LR catches:  {(lr_pred[non_op_phish_mask.values]==1).mean():.1%}")
    print(f"  XGB catches: {(xgb_pred[non_op_phish_mask.values]==1).mean():.1%}")


if __name__ == "__main__":
    main()
