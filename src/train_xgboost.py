"""Day 2: XGBoost on the merged URL dataset, same split as the LR baseline.

Reports accuracy, precision, recall, F1, AUC-ROC and the top-10 features
by gain importance. Then re-runs with length-driven features removed to
check whether the high score is a URL-length artifact of the
PhishTank-vs-Tranco pairing. Finally runs XGBoost on the UCI dataset as
the richer-features reference.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from features import build_feature_frame
from train_baseline import grouped_train_test_split, load_merged_url_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 42

# Features that directly encode URL length. PhishTank URLs are long
# (full phishing pages); Tranco entries are bare root domains, so any
# length signal separates the two sources rather than phish vs benign.
LENGTH_FEATURES = [
    "url_length", "host_length", "path_length", "query_length",
    "num_slashes", "num_dots", "num_query_params", "url_entropy",
]


def make_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def report(name: str, model: XGBClassifier, X_test, y_test) -> None:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred):.4f}")
    print(f"Recall:    {recall_score(y_test, pred):.4f}")
    print(f"F1:        {f1_score(y_test, pred):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, proba):.4f}")


def top_features(model: XGBClassifier, columns, k: int = 10) -> pd.Series:
    imp = pd.Series(
        model.get_booster().get_score(importance_type="gain")
    ).sort_values(ascending=False)
    # get_score keys are feature names since we fit on a DataFrame
    return imp.head(k)


def run_merged() -> None:
    df = load_merged_url_dataset()
    X = build_feature_frame(df["url"])
    y = df["label"]

    # Domain-grouped split — no domain appears in both train and test.
    train_idx, test_idx = grouped_train_test_split(df)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    model = make_model()
    model.fit(X_train, y_train)
    report(f"XGBoost, merged URL set ({len(df)} URLs), all features", model, X_test, y_test)

    print("\nTop 10 features by gain:")
    top = top_features(model, X.columns)
    for name, gain in top.items():
        tag = "  <- length-driven" if name in LENGTH_FEATURES else ""
        print(f"  {name:24s} {gain:12.1f}{tag}")

    # Ablation: drop length-driven features, same split indices
    keep = [c for c in X.columns if c not in LENGTH_FEATURES]
    model2 = make_model()
    model2.fit(X_train[keep], y_train)
    report("XGBoost, merged URL set, length features removed", model2, X_test[keep], y_test)

    print("\nTop 10 features by gain (ablated model):")
    for name, gain in top_features(model2, keep).items():
        print(f"  {name:24s} {gain:12.1f}")


def run_uci() -> None:
    df = pd.read_csv(DATA_DIR / "uci_phishing.csv")
    y = df.iloc[:, -1].replace(-1, 0)
    X = df.iloc[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    model = make_model()
    model.fit(X_train, y_train)
    report(f"XGBoost, UCI Phishing Websites ({len(df)} rows)", model, X_test, y_test)

    print("\nTop 10 features by gain:")
    for name, gain in top_features(model, X.columns).items():
        print(f"  {name:24s} {gain:12.1f}")


if __name__ == "__main__":
    run_merged()
    run_uci()
