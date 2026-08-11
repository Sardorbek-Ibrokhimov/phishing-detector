"""Train and save the email-text (TF-IDF + Logistic Regression) model so the
API can load it at startup instead of retraining every time.

Uses the exact same pipeline as email_baseline_lr.py::make_lr() — the
recipe already evaluated in results/email_comparison.md, where it beat
DistilBERT (98.05% vs 97.23% accuracy, McNemar p=1.1e-8). That evaluation
capped training at 8k emails for a fair head-to-head against DistilBERT's
CPU fine-tuning budget; this script trains on the FULL available training
set instead, since the deployed artifact isn't CPU-time-constrained the way
the comparison was. Expect somewhat better numbers than the 8k comparison
figures as a result — that's a different (larger-data), not incomparable,
run, and is recorded as such in the metadata below.

Run: .venv/Scripts/python src/persist_email_model.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from email_baseline_lr import make_lr
from email_data import RANDOM_STATE, get_split

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "deployed_email_model.joblib"
METADATA_PATH = MODELS_DIR / "deployed_email_model.metadata.json"

# The full training set (65,713 rows) makes TfidfVectorizer's bigram
# vocabulary-counting step exceed available memory on the training machine
# (MemoryError in CountVectorizer._count_vocab, before max_features pruning
# even applies). Capped via the existing stratified get_split(cap_train=...)
# helper (email_data.py) rather than writing new sampling logic. Still ~5x
# the 8k head-to-head comparison set.
TRAIN_CAP = 40_000


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Loading train/test split (train capped at {TRAIN_CAP:,} for memory)...")
    train, test = get_split(cap_train=TRAIN_CAP, cap_test=None)
    print(f"train={len(train)}  test={len(test)}")

    print("Training LR-TFIDF...")
    model = make_lr()
    model.fit(train["text"], train["label"])

    proba = model.predict_proba(test["text"])[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": round(accuracy_score(test["label"], pred), 4),
        "precision": round(precision_score(test["label"], pred), 4),
        "recall": round(recall_score(test["label"], pred), 4),
        "f1": round(f1_score(test["label"], pred), 4),
        "auc_roc": round(roc_auc_score(test["label"], proba), 4),
    }
    print(f"Held-out test metrics (full-train run): {metrics}")

    joblib.dump({"model": model}, MODEL_PATH)
    print(f"Saved model bundle: {MODEL_PATH}")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "n_train": len(train),
        "n_test": len(test),
        "test_metrics": metrics,
        "model_type": "TfidfVectorizer + LogisticRegression (make_lr, email_baseline_lr.py)",
        "split_method": "dedup + near-duplicate-template-grouped (email_data.get_split)",
        "note": f"Trained on a {TRAIN_CAP:,}-row stratified subset of the training set "
                 "(not the full 65,713 rows, and not the 8k head-to-head subset used in "
                 "results/email_comparison.md's DistilBERT comparison) — capped due to a "
                 "MemoryError fitting TF-IDF bigrams on the full set on the training "
                 "machine. Expect numbers between the 8k comparison figures and the "
                 "full-train reference in results/email_lr_metrics.csv, not an exact "
                 "match to either.",
        "known_limitation": "The training data has a source/date confound baked into the "
                 "labels (see src/email_data.py docstring): legitimate = Enron corporate "
                 "email (2001) + mailing-list ham; phishing = spam/phishing campaigns "
                 "(2004-2008). The model may partly be learning corpus era/genre rather "
                 "than phishing intent specifically. Not fixable by re-splitting — it is "
                 "baked into the labels themselves.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata: {METADATA_PATH}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
