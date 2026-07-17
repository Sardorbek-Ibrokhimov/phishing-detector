"""Day 2 (part 2b): SHAP explainability for the final XGBoost model,
trained on the clean feature set (no URL-length, no uses_https).

Produces:
  - results/shap_summary.png : global feature-importance summary plot
  - explain_prediction(url)  : per-URL verdict + plain-English reasons,
                                the function the API will call later.
"""

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from compare_models import clean_columns
from features import build_feature_frame, extract_url_features
from train_baseline import grouped_train_test_split, load_merged_url_dataset
from train_xgboost import make_model

# Neither `shap` nor `matplotlib` is imported at module load — and therefore
# not by the API, which imports this module. Rationale (see DEPLOYMENT.md):
#   * The `shap` library eagerly imports matplotlib; together they add ~80 MB
#     to the process, pushing a cold boot to ~536 MB — over Render's 512 MB
#     free tier. Deferring an import used on every request cannot lower the
#     *peak*, so instead explain_prediction() computes SHAP values via
#     XGBoost's built-in TreeSHAP (`pred_contribs`), which is the identical
#     algorithm and gives byte-identical values (verified diff = 0.0) with no
#     `shap` dependency at runtime.
#   * `shap` (for the global summary plot) and `matplotlib` are imported
#     lazily inside save_global_summary(), a training-time-only helper.

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

# Thresholds above which a continuous feature reads as phishing-leaning in
# the plain-English wording only — they never affect the model's verdict or
# confidence, only which side of "raises"/"lowers" a reason is phrased on.
#
# Derivation (percentiles computed on the frozen training data;
#  for how that data was built):
#
#   host_entropy   benign:   p25=3.18  p50=3.43  p75=3.61  p90=3.77
#                  phishing: p25=2.87  p50=3.46  p75=3.88  p90=4.11
#   digit_ratio    benign:   p25=0.00  p50=0.00  p75=0.00  p90=0.03
#                  phishing: p25=0.00  p50=0.05  p75=0.11  p90=0.21
#
# These are NOT clean decision boundaries — the two classes overlap heavily.
# HOST_ENTROPY_THRESHOLD=3.6 sits at roughly the benign 75th percentile and
# the phishing 50th percentile: about a quarter of genuinely benign URLs and
# roughly half of phishing URLs will land on the "looks randomly generated"
# side. DIGIT_RATIO_THRESHOLD=0.05 is a modestly cleaner split (benign 90th
# percentile vs. phishing 50th), but still only around half of phishing mass
# exceeds it. Both are best read as rough anchors picked to word individual
# examples sensibly, not as a claim that these values cleanly separate the
# classes — the model's actual verdict comes from all 11 features jointly,
# not from either threshold.
HOST_ENTROPY_THRESHOLD = 3.6
DIGIT_RATIO_THRESHOLD = 0.05


def _count(n: int, singular: str) -> str:
    """'1 hyphen' / '2 hyphens' — correct singular/plural."""
    n = int(n)
    return f"{n} {singular}" if n == 1 else f"{n} {singular}s"


def describe_feature(feat: str, v):
    """Return (direction, phrase) for a feature's actual value, where
    direction is +1 if the value leans phishing or -1 if it leans
    legitimate. The phrase is self-consistent with that direction, so a
    zero/benign value is never worded as 'raising' phishing likelihood.
    Returns None for features with no plain-English phrasing.
    """
    if feat == "num_hyphens":
        return (-1, "no hyphens in the URL") if v == 0 else (1, f"{_count(v, 'hyphen')} in the URL")
    if feat == "num_digits":
        return (-1, "no digits in the URL") if v == 0 else (1, f"{_count(v, 'digit')} in the URL")
    if feat == "num_special":
        return ((-1, "no unusual special characters (@!#$%&*?=~)") if v == 0
                else (1, f"{_count(v, 'special character')} (@!#$%&*?=~)"))
    if feat == "has_at":
        return ((1, "an '@' symbol in the URL, which can hide the real destination") if v
                else (-1, "no '@' symbol in the URL"))
    if feat == "has_ip_host":
        return ((1, "a raw IP address used instead of a domain name") if v
                else (-1, "a domain name rather than a raw IP address"))
    if feat == "num_subdomains":
        return (-1, "no extra subdomains") if v == 0 else (1, _count(v, "subdomain"))
    if feat == "has_port":
        return ((1, "a non-standard network port in the URL") if v
                else (-1, "no non-standard port in the URL"))
    if feat == "host_entropy":
        return ((1, f"a domain name that looks randomly generated (entropy {v:.2f})")
                if v >= HOST_ENTROPY_THRESHOLD
                else (-1, f"a normal-looking domain name (entropy {v:.2f})"))
    if feat == "num_suspicious_tokens":
        return ((-1, "no phishing-related keywords (e.g. 'login', 'verify', 'secure')") if v == 0
                else (1, f"{_count(v, 'phishing-related keyword')} (e.g. 'login', 'verify', 'secure')"))
    if feat == "digit_ratio":
        return ((1, f"digits making up {v:.0%} of the URL") if v >= DIGIT_RATIO_THRESHOLD
                else (-1, "few or no digits in the URL"))
    if feat == "tld_length":
        return ((-1, "a common top-level domain length (3 characters)") if int(v) == 3
                else (1, f"an unusual top-level domain length ({_count(v, 'character')})"))
    return None


def train_final_model():
    df = load_merged_url_dataset()
    X = build_feature_frame(df["url"])
    y = df["label"]
    cols = clean_columns(X.columns)

    # Domain-grouped split  — no
    # registrable domain appears in both train and test.
    train_idx, test_idx = grouped_train_test_split(df)
    X_train, X_test = X.loc[train_idx, cols], X.loc[test_idx, cols]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    model = make_model()
    model.fit(X_train, y_train)
    return model, X_test, cols


def save_global_summary(model, X_test, cols, sample_size: int = 5000) -> Path:
    # shap + matplotlib imported here, not at module load, so the API (which
    # imports this module) never pays their memory cost — this helper only
    # runs during training/analysis.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    out_path = RESULTS_DIR / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def shap_values_via_xgboost(model, row: pd.DataFrame):
    """Per-feature SHAP values for one row using XGBoost's built-in TreeSHAP
    (`pred_contribs`) instead of the `shap` library. Identical algorithm,
    byte-identical values (verified diff = 0.0), but no `shap`/`matplotlib`
    dependency at runtime — the memory win that fits Render's free tier."""
    contribs = model.get_booster().predict(xgb.DMatrix(row), pred_contribs=True)
    return contribs[0][:-1]  # last column is the bias/base term


def explain_prediction(url: str, model, explainer, cols: list) -> dict:
    """Verdict + confidence + top plain-English reasons for a single URL.
    This is the function the API layer calls.

    NB: `explainer` is retained in the signature for backward compatibility
    with callers that still pass a shap.TreeExplainer, but it is IGNORED —
    SHAP values are computed natively via XGBoost (see
    shap_values_via_xgboost). Pass None."""
    feats = extract_url_features(url)
    row = pd.DataFrame([feats])[cols]

    proba_phish = float(model.predict_proba(row)[0, 1])
    verdict = "phishing" if proba_phish >= 0.5 else "legitimate"
    confidence = proba_phish if verdict == "phishing" else 1 - proba_phish

    shap_row = shap_values_via_xgboost(model, row)
    direction = 1 if verdict == "phishing" else -1
    suffix = "raises phishing likelihood" if direction == 1 else "lowers phishing likelihood"

    # Rank features by how strongly SHAP pushes them toward the verdict.
    ranked = sorted(zip(cols, shap_row), key=lambda kv: direction * kv[1], reverse=True)

    reasons = []
    for feat, shap_val in ranked:
        if direction * shap_val <= 0:
            continue  # SHAP pushes this feature the other way; not a reason
        described = describe_feature(feat, feats[feat])
        if described is None:
            continue
        sem_dir, phrase = described
        if sem_dir != direction:
            continue  # value itself doesn't support the verdict; suppress it
        reasons.append(f"{phrase} ({suffix})")
        if len(reasons) >= 5:
            break

    # Fallback: if no single feature value cleanly supports the verdict
    # (rare — usually a borderline case), state it honestly without
    # attaching a contradictory feature.
    if not reasons:
        reasons.append(f"the overall combination of URL characteristics ({suffix})")

    return {
        "url": url,
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "reasons": reasons,
    }


if __name__ == "__main__":
    model, X_test, cols = train_final_model()
    save_global_summary(model, X_test, cols)

    # explain_prediction ignores the explainer arg (computes via XGBoost); pass None.
    explainer = None

    phishing_df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "phishing_urls.csv")
    real_phish_url = phishing_df["url"].iloc[0]

    examples = [
        # obviously benign
        "https://www.wikipedia.org/",
        "https://mail.google.com/",
        "https://github.com/user/repo",
        # obviously phishing
        "http://paypal-secure-login-verify.account-update.tk/signin?user=1234",
        "http://192.168.1.1/wp-admin/login.php?redirect=account&id=88291",
        real_phish_url,
    ]
    print("\n=== Example explanations ===")
    for url in examples:
        result = explain_prediction(url, model, explainer, cols)
        print(json.dumps(result, indent=2))
