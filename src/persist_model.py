"""Train and save the model to disk so the API can load it at startup
instead of retraining every time.

Saves the model + feature list as a joblib bundle, plus a metadata JSON
with training data hash, features, seed, and test metrics.

Run: .venv/Scripts/python src/persist_model.py
"""

import hashlib
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
from shap_explain import train_final_model
from train_baseline import RANDOM_STATE, grouped_train_test_split, load_merged_url_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "deployed_model.joblib"
METADATA_PATH = MODELS_DIR / "deployed_model.metadata.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def hash_training_data(df) -> str:
    """Order-independent hash of the exact (url, label) pairs trained on —
    changes if the dataset content changes, even if row order doesn't."""
    canon = "\n".join(sorted(f"{u}|{l}" for u, l in zip(df["url"], df["label"])))
    return hashlib.sha256(canon.encode()).hexdigest()


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("Training model...")
    model, X_test, cols = train_final_model()

    # Re-derive y_test deterministically (same seed/logic as train_final_model)
    # without touching shap_explain.py's training function.
    df = load_merged_url_dataset()
    train_idx, test_idx = grouped_train_test_split(df)
    assert list(X_test.index) == list(test_idx), (
        "X_test index doesn't match a freshly-derived grouped split — "
        "training pipeline may have changed; investigate before trusting metrics."
    )
    y_test = df.loc[test_idx, "label"]

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "auc_roc": round(roc_auc_score(y_test, proba), 4),
    }
    print(f"Held-out test metrics: {metrics}")

    # Only store model + feature list (no SHAP explainer — we use XGBoost's
    # native TreeSHAP at inference instead).
    bundle = {"model": model, "cols": list(cols)}
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle: {MODEL_PATH}")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "feature_list": list(cols),
        "n_features": len(cols),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "test_metrics": metrics,
        "model_type": type(model).__name__,
        "split_method": "domain-grouped (GroupShuffleSplit, eTLD+1 via tldextract)",
        "training_data_hash": hash_training_data(df),
        "source_file_hashes": {
            "phishing_urls.csv": hash_file(DATA_DIR / "phishing_urls.csv"),
            "benign_urls.csv": hash_file(DATA_DIR / "benign_urls.csv"),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata: {METADATA_PATH}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
