"""Email classifier comparison: data prep and leak-aware split.

SEPARATE from the deployed URL system. Classifies email BODY TEXT, not URLs.
Nothing here is imported by api.py / features.py / the served model.

Applies the leakage checks to the Kaggle email corpus
(data/phishing_emails.csv). Findings ():
  * 408 exact-duplicate emails -> deduplicated before splitting.
  * ~7.9% near-duplicate template families (spam campaigns; mailing-list
    ham) -> a naive split would leak them across train/test. We group by a
    text-prefix signature and use GroupShuffleSplit so a template family
    never straddles the split.
  * Sender/domain is NOT recoverable: preprocessing stripped every "@", so a
    sender-grouped split (the ideal, as in the URL work) is impossible here.
  * Source/date confound (NOT fixable by any split — it is baked into the
    labels): legit is Enron trading email (2001) + mailing-list ham; phishing
    is spam/phishing campaigns (2004-2008). The token 'enron' is a near-
    perfect legit marker (7205 legit vs 3 phishing). Year strongly predicts
    label. This is the reason the off-corpus sanity test matters more than
    the held-out score.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA = Path(__file__).resolve().parent.parent / "data" / "phishing_emails.csv"
RANDOM_STATE = 42
PREFIX_TOKENS = 10  # near-duplicate grouping signature length


def _group_signature(text: str) -> str:
    """Signature that keeps near-duplicate template families together: the
    first PREFIX_TOKENS tokens, hashed. Emails with an identical opening
    (the spam campaigns / mailing-list ham seen in EDA) share a group and
    are kept on the same side of the split."""
    toks = text.split()[:PREFIX_TOKENS]
    return hashlib.md5(" ".join(toks).encode()).hexdigest()[:16]


def load_dedup() -> pd.DataFrame:
    df = pd.read_csv(DATA).fillna({"text_combined": ""})
    df = df.rename(columns={"text_combined": "text"})
    df["text"] = df["text"].astype(str)
    n0 = len(df)
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"[dedup] {n0} -> {len(df)} rows ({n0 - len(df)} exact-duplicate/empty removed)")
    df["group"] = df["text"].map(_group_signature)
    return df


def grouped_split(df: pd.DataFrame, test_size: float = 0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    tr, te = next(gss.split(df, df["label"], groups=df["group"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def report_split(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("\n=== split leakage checks ===")
    print(f"train: {len(train)}  test: {len(test)}")
    print(f"train %phish: {train['label'].mean():.3f}   test %phish: {test['label'].mean():.3f}")

    exact = set(train["text"]) & set(test["text"])
    print(f"exact-duplicate emails across train/test: {len(exact)} (want 0)")

    grp_overlap = set(train["group"]) & set(test["group"])
    print(f"near-duplicate template groups in BOTH splits: {len(grp_overlap)} (want 0)")

    # Source-confound marker: does the 'enron' near-perfect legit token leak?
    for name, sub in [("train", train), ("test", test)]:
        en = sub["text"].str.contains("enron", regex=False)
        print(f"  {name}: {en.sum()} emails mention 'enron' "
              f"({sub.loc[en, 'label'].mean() if en.any() else float('nan'):.3f} phish rate)")


def get_split(cap_train: int | None = None, cap_test: int | None = None,
              seed: int = RANDOM_STATE):
    """Deterministic dedup + grouped split. Optional stratified caps let both
    models train/eval on the SAME reduced set for a fair CPU-feasible
    head-to-head (used for the DistilBERT comparison)."""
    df = load_dedup()
    train, test = grouped_split(df)

    def cap(frame, n):
        if n is None or n >= len(frame):
            return frame
        frac = n / len(frame)
        parts = [g.sample(n=int(round(len(g) * frac)), random_state=seed)
                 for _, g in frame.groupby("label")]
        return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    return cap(train, cap_train), cap(test, cap_test)


if __name__ == "__main__":
    df = load_dedup()
    train, test = grouped_split(df)
    report_split(train, test)
