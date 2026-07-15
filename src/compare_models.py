"""Day 2 (part 2): clean comparison of LR vs XGBoost on the ablated
feature set (no length features, no uses_https), same train/test split.
Runs McNemar's exact test on the two models' correctness on the shared
test set and saves the metrics table.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_feature_frame
from train_baseline import grouped_train_test_split, load_merged_url_dataset
from train_xgboost import LENGTH_FEATURES, make_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

# Same exclusions as the train_xgboost.py ablation: URL-length features
# and uses_https both leak the PhishTank-vs-Tranco source split rather
# than real phishing signal.
EXCLUDED_FEATURES = LENGTH_FEATURES + ["uses_https"]


def clean_columns(all_columns) -> list:
    return [c for c in all_columns if c not in EXCLUDED_FEATURES]


def compute_metrics(y_test, pred, proba) -> dict:
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc_roc": roc_auc_score(y_test, proba),
    }


def mcnemar_exact(y_test, pred_a, pred_b):
    correct_a = pred_a == y_test
    correct_b = pred_b == y_test
    b = int((correct_a & ~correct_b).sum())  # A right, B wrong
    c = int((~correct_a & correct_b).sum())  # A wrong, B right
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return b, c, p


def main():
    df = load_merged_url_dataset()
    X = build_feature_frame(df["url"])
    y = df["label"]
    cols = clean_columns(X.columns)

    # Domain-grouped split  — no
    # registrable domain appears in both train and test.
    train_idx, test_idx = grouped_train_test_split(df)
    X_train, X_test = X.loc[train_idx, cols], X.loc[test_idx, cols]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    from train_baseline import registrable_domain
    train_domains = set(df.loc[train_idx, "url"].map(registrable_domain))
    test_domains = set(df.loc[test_idx, "url"].map(registrable_domain))
    overlap = train_domains & test_domains
    print(f"Domain overlap check: {len(overlap)} domains present in both "
          f"train ({len(train_domains)} distinct) and test ({len(test_domains)} distinct)")
    assert len(overlap) == 0, f"grouped split leaked {len(overlap)} domains: {list(overlap)[:5]}"

    lr = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    )
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]

    xgb = make_model()
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    rows = {
        "Logistic Regression": compute_metrics(y_test, lr_pred, lr_proba),
        "XGBoost": compute_metrics(y_test, xgb_pred, xgb_proba),
    }
    table = pd.DataFrame(rows).T.round(4)
    print(f"\n=== Clean comparison ({len(cols)} features, no length, no uses_https) ===")
    print(table)

    b, c, p = mcnemar_exact(y_test.to_numpy(), lr_pred, xgb_pred)
    sig = "significant (p < 0.05)" if p < 0.05 else "not significant (p >= 0.05)"
    print(f"\nMcNemar's exact test: LR-only-correct={b}, XGB-only-correct={c}, p={p:.4g} -> {sig}")

    out_csv = RESULTS_DIR / "model_comparison.csv"
    table.to_csv(out_csv)

    out_md = RESULTS_DIR / "model_comparison.md"
    with open(out_md, "w") as f:
        f.write("# LR vs XGBoost, clean feature set (no URL-length, no uses_https)\n\n")
        f.write(f"Features used ({len(cols)}): {', '.join(cols)}\n\n")
        f.write(table.to_markdown())
        f.write(
            f"\n\nMcNemar's exact test: LR-only-correct={b}, XGB-only-correct={c}, "
            f"p={p:.4g} -> {sig}\n"
        )

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
