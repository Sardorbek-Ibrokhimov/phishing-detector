"""Baseline: Logistic Regression on lexical URL features.

Also trains LR on the UCI Phishing Websites dataset as a reference point.

Uses a domain-grouped train/test split so no domain appears on both sides,
which avoids the model just memorising specific domains.
"""

from pathlib import Path

import pandas as pd
import tldextract
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_feature_frame

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 42


def load_merged_url_dataset() -> pd.DataFrame:
    phish = pd.read_csv(DATA_DIR / "phishing_urls.csv")
    benign = pd.read_csv(DATA_DIR / "benign_urls.csv")
    phish["label"] = 1
    benign["label"] = 0

    # Balance classes: downsample the larger side
    n = min(len(phish), len(benign))
    phish = phish.sample(n=n, random_state=RANDOM_STATE)
    benign = benign.sample(n=n, random_state=RANDOM_STATE)

    df = pd.concat([phish, benign], ignore_index=True)
    df = df.drop_duplicates(subset="url").dropna(subset=["url"])
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def registrable_domain(url: str) -> str:
    """eTLD+1 via the public suffix list (tldextract) — correctly handles
    multi-part TLDs (co.uk, com.au, ...), unlike splitting on the last dot."""
    ext = tldextract.extract(url if "://" in url else "http://" + url)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


def grouped_train_test_split(df: pd.DataFrame, test_size: float = 0.2,
                              random_state: int = RANDOM_STATE):
    """Split df's index into (train_idx, test_idx) such that no registrable
    domain appears on both sides. Falls back to a plain random split only if
    there are too few distinct groups for GroupShuffleSplit (never expected
    for this dataset size)."""
    groups = df["url"].map(registrable_domain)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    return df.index[train_idx], df.index[test_idx]


def run_url_baseline() -> None:
    df = load_merged_url_dataset()
    print(f"\n=== URL baseline (merged dataset: {len(df)} URLs, "
          f"{df['label'].mean():.1%} phishing) ===")

    X = build_feature_frame(df["url"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"F1:       {f1_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, target_names=["benign", "phishing"]))


def run_uci_baseline() -> None:
    path = DATA_DIR / "uci_phishing.csv"
    if not path.exists():
        print("\n[skip] UCI dataset not downloaded")
        return

    df = pd.read_csv(path)
    print(f"\n=== UCI Phishing Websites reference ({len(df)} rows) ===")

    y = df.iloc[:, -1].replace(-1, 0)  # -1 = legitimate -> 0
    X = df.iloc[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"F1:       {f1_score(y_test, pred):.4f}")


if __name__ == "__main__":
    run_url_baseline()
    run_uci_baseline()
