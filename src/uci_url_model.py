"""Day 2 (part 3): UCI Phishing Websites model restricted to the subset
of columns derivable purely from the URL string (no page fetch, no
WHOIS, no DNS, no third-party traffic/rank/index APIs). This is the
model the API will serve, and the one explain_prediction() now targets.

Rule definitions follow Mohammad, Thabtah & McCluskey's original
description of the dataset (the paper behind UCI id=327). They are a
best-effort reproduction from the published feature descriptions, not
the authors' original (unpublished) extraction code, so a handful of
borderline URLs may land on a different -1/0/1 value than the authors'
own tool would have assigned. The model itself is trained on the real
downloaded UCI rows either way; only *this* URL-string encoder is an
approximation of how those columns were originally computed.

KEPT (9, URL-string only):
  having_ip_address, url_length, shortining_service, having_at_symbol,
  double_slash_redirecting, prefix_suffix, having_sub_domain, port,
  https_token

DROPPED (21, need a live page fetch / WHOIS / DNS / third-party API):
  sslfinal_state, domain_registration_length, favicon, request_url,
  url_of_anchor, links_in_tags, sfh, submitting_to_email, abnormal_url,
  redirect, on_mouseover, rightclick, popupwindow, iframe, age_of_domain,
  dnsrecord, web_traffic, page_rank, google_index, links_pointing_to_page,
  statistical_report
"""

import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from train_xgboost import make_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 42

KEPT_COLUMNS = [
    "having_ip_address", "url_length", "shortining_service", "having_at_symbol",
    "double_slash_redirecting", "prefix_suffix", "having_sub_domain", "port",
    "https_token",
]

DROPPED_COLUMNS = [
    "sslfinal_state", "domain_registration_length", "favicon", "request_url",
    "url_of_anchor", "links_in_tags", "sfh", "submitting_to_email", "abnormal_url",
    "redirect", "on_mouseover", "rightclick", "popupwindow", "iframe",
    "age_of_domain", "dnsrecord", "web_traffic", "page_rank", "google_index",
    "links_pointing_to_page", "statistical_report",
]

_SHORTENERS = re.compile(
    r"(bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|buff\.ly|adf\.ly|bit\.do|"
    r"mcaf\.ee|cutt\.ly|tiny\.cc|rebrand\.ly|shorte\.st|soo\.gd|s2r\.co|clck\.ru|"
    r"ity\.im|q\.gs|po\.st|bc\.vc|tr\.im|v\.gd)",
    re.IGNORECASE,
)


def extract_uci_url_features(url: str) -> dict:
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc.split("@")[-1]  # drop userinfo before host for port/IP checks
    host_no_port = host.split(":")[0]

    # having_ip_address
    is_ip = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host_no_port))
    having_ip_address = 1 if is_ip else -1

    # url_length
    n = len(url)
    if n < 54:
        url_length = -1
    elif n <= 75:
        url_length = 0
    else:
        url_length = 1

    # shortining_service
    shortining_service = 1 if _SHORTENERS.search(url) else -1

    # having_at_symbol
    having_at_symbol = 1 if "@" in url else -1

    # double_slash_redirecting: "//" reappearing after the scheme separator
    double_slash_redirecting = 1 if url.find("//", 8) != -1 else -1

    # prefix_suffix: hyphen in the host
    prefix_suffix = 1 if "-" in host_no_port else -1

    # having_sub_domain: dot count in host (excluding IP hosts)
    dots = host_no_port.count(".")
    if is_ip:
        having_sub_domain = -1
    elif dots <= 1:
        having_sub_domain = -1
    elif dots == 2:
        having_sub_domain = 0
    else:
        having_sub_domain = 1

    # port: explicit, non-default port
    if ":" in host:
        try:
            port_num = int(host.split(":")[1])
        except ValueError:
            port_num = None
        default = 443 if parsed.scheme == "https" else 80
        port = 1 if (port_num is not None and port_num != default) else -1
    else:
        port = -1

    # https_token: "https" used deceptively inside the host itself
    https_token = 1 if "https" in host_no_port.lower() else -1

    return {
        "having_ip_address": having_ip_address,
        "url_length": url_length,
        "shortining_service": shortining_service,
        "having_at_symbol": having_at_symbol,
        "double_slash_redirecting": double_slash_redirecting,
        "prefix_suffix": prefix_suffix,
        "having_sub_domain": having_sub_domain,
        "port": port,
        "https_token": https_token,
    }


# Plain-English phrasing per feature, keyed by its -1/0/1 value. Every
# value each feature can take is covered so explain_prediction() never
# has to fall back to a raw feature name.
FEATURE_PHRASES = {
    "having_ip_address": {
        1: "the domain is a raw IP address rather than a name",
        -1: "the domain uses a normal hostname rather than a raw IP",
    },
    "url_length": {
        1: "the URL is unusually long (over 75 characters)",
        0: "the URL is a suspicious length (54-75 characters)",
        -1: "the URL is a typical, short length (under 54 characters)",
    },
    "shortining_service": {
        1: "the URL uses a link-shortening service",
        -1: "the URL does not use a link-shortening service",
    },
    "having_at_symbol": {
        1: "the URL contains an '@' symbol, which can hide the real destination",
        -1: "the URL does not contain a hiding '@' symbol",
    },
    "double_slash_redirecting": {
        1: "the URL path contains '//', which can silently redirect to another site",
        -1: "the URL path does not contain a hidden '//' redirect",
    },
    "prefix_suffix": {
        1: "the domain contains a hyphen, often used to imitate a brand name",
        -1: "the domain does not contain a hyphen",
    },
    "having_sub_domain": {
        1: "the domain has multiple extra subdomains",
        0: "the domain has one extra subdomain",
        -1: "the domain has no extra subdomains",
    },
    "port": {
        1: "the URL specifies a non-standard port",
        -1: "the URL uses the standard port for its protocol",
    },
    "https_token": {
        1: "the domain contains the word 'https' as a deceptive trick, not as the real protocol",
        -1: "the domain does not contain a deceptive 'https' token",
    },
}


def load_url_only_uci():
    df = pd.read_csv(DATA_DIR / "uci_phishing.csv")
    y = df.iloc[:, -1].replace(-1, 0)
    X_full = df.iloc[:, :-1]
    X_kept = X_full[KEPT_COLUMNS]
    return X_kept, X_full, y


def compute_full_feature_uci_metrics(X_full: pd.DataFrame, y: pd.Series) -> dict:
    """Trains XGBoost on all 30 UCI columns, same split params as the
    URL-only run below (identical row split, since train_test_split
    partitions by index/stratify(y), not by X content) — so the two
    metric sets are directly comparable. Computed fresh rather than a
    hardcoded literal : a hardcoded number
    silently drifts out of sync if the data or split ever changes."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    model = make_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc_roc": roc_auc_score(y_test, proba),
    }


def train_and_report():
    X_kept, X_full, y = load_url_only_uci()

    # Same split params as the full-feature UCI run -> identical row split,
    # since train_test_split partitions by index/stratify(y), not by X content.
    X_train, X_test, y_train, y_test = train_test_split(
        X_kept, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = make_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc_roc": roc_auc_score(y_test, proba),
    }

    full_metrics = compute_full_feature_uci_metrics(X_full, y)

    print(f"\n=== UCI, URL-string-only features ({len(KEPT_COLUMNS)}/{len(X_full.columns)} columns) ===")
    print(f"Kept:    {KEPT_COLUMNS}")
    print(f"Dropped: {DROPPED_COLUMNS}")
    table = pd.DataFrame({"full_feature_uci": full_metrics, "url_only_uci": metrics}).round(4)
    table["drop"] = (table["full_feature_uci"] - table["url_only_uci"]).round(4)
    print(f"\n{table}")

    return model, X_test, KEPT_COLUMNS


def explain_prediction(url: str, model, explainer, cols: list = KEPT_COLUMNS) -> dict:
    """Verdict + confidence + top plain-English reasons for a single URL,
    using the URL-only UCI model. Secondary/comparison model — the deployed
    API uses shap_explain.explain_prediction (the lexical model)."""
    feats = extract_uci_url_features(url)
    row = pd.DataFrame([feats])[cols]

    proba_phish = float(model.predict_proba(row)[0, 1])
    verdict = "phishing" if proba_phish >= 0.5 else "legitimate"
    confidence = proba_phish if verdict == "phishing" else 1 - proba_phish

    shap_row = explainer.shap_values(row)[0]
    direction = 1 if verdict == "phishing" else -1
    suffix = "raises phishing likelihood" if direction == 1 else "lowers phishing likelihood"
    ranked = sorted(zip(cols, shap_row), key=lambda kv: direction * kv[1], reverse=True)

    reasons = []
    for feat, shap_val in ranked:
        if direction * shap_val <= 0:
            continue  # SHAP pushes the other way
        raw_value = feats[feat]
        # UCI encoding: value >= 0 leans phishing/suspicious, -1 leans
        # legitimate. Only show the reason if the value supports the verdict,
        # so a benign value (e.g. "does not use a shortener") is never
        # worded as raising phishing likelihood.
        value_dir = 1 if raw_value >= 0 else -1
        if value_dir != direction:
            continue
        reasons.append(f"{FEATURE_PHRASES[feat][raw_value]} ({suffix})")
        if len(reasons) >= 5:
            break

    if not reasons:
        reasons.append(f"the overall combination of URL characteristics ({suffix})")

    return {
        "url": url,
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "reasons": reasons,
    }


if __name__ == "__main__":
    import json

    import shap

    model, X_test, cols = train_and_report()
    explainer = shap.TreeExplainer(model)

    # A real, currently-listed PhishTank phishing URL, not a synthetic example.
    phishing_df = pd.read_csv(DATA_DIR / "phishing_urls.csv")
    example_url = phishing_df["url"].iloc[0]

    result = explain_prediction(example_url, model, explainer, cols)
    print("\n=== Example explanation (real PhishTank URL) ===")
    print(json.dumps(result, indent=2))
