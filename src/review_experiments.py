"""Adversarial experiments attacking the dissertation's CLAIMS.
Runs against the deployed lexical model directly (independent of the server).
Prints evidence and saves it to results/review_experiments.txt.
"""
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_models import clean_columns
from features import build_feature_frame, extract_url_features
from shap_explain import explain_prediction, train_final_model
import shap
from train_baseline import grouped_train_test_split, load_merged_url_dataset, registrable_domain

RESULTS = Path(__file__).resolve().parent.parent / "results"
out_lines = []


def log(s=""):
    print(s)
    out_lines.append(str(s))


# reg_domain() (naive last-two-labels) is kept only to reproduce the OLD,
# pre-fix leakage numbers for the record. registrable_domain() (tldextract,
# correct eTLD+1) is what the actual grouped split now uses.
def reg_domain(url):
    host = urlparse(url if "://" in url else "http://" + url).netloc.split(":")[0].lower()
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


# ============================================================ CLAIM 1: split leakage
def claim_split_leakage():
    log("\n" + "=" * 70)
    log("CLAIM 1 — '88.97% accuracy' / leak-free split? (C1 fix verification)")
    log("=" * 70)
    df = load_merged_url_dataset()

    log("\n--- BEFORE (old URL-level random split, for the record) ---")
    old_train_idx, old_test_idx = train_test_split(
        df.index, test_size=0.2, stratify=df["label"], random_state=42)
    tr, te = df.loc[old_train_idx], df.loc[old_test_idx]
    exact = set(tr["url"]) & set(te["url"])
    log(f"Exact duplicate URLs across train/test: {len(exact)}")
    tr_dom = set(tr["url"].map(reg_domain))
    te_doms = te["url"].map(reg_domain)
    overlap_rows = te_doms.isin(tr_dom).sum()
    log(f"Test rows whose registrable domain ALSO appears in train: "
        f"{overlap_rows}/{len(te)} ({overlap_rows/len(te):.1%})")
    for lbl, name in [(1, "phishing"), (0, "benign")]:
        sub = te[te["label"] == lbl]
        sub_dom = sub["url"].map(reg_domain)
        ov = sub_dom.isin(tr_dom).sum()
        log(f"  {name}: {ov}/{len(sub)} ({ov/len(sub):.1%}) test domains seen in train")
    ph = df[df["label"] == 1]["url"].map(reg_domain)
    log(f"Most common phishing registrable domains: {Counter(ph).most_common(5)}")

    log("\n--- AFTER (domain-grouped split, tldextract eTLD+1) ---")
    new_train_idx, new_test_idx = grouped_train_test_split(df)
    tr2, te2 = df.loc[new_train_idx], df.loc[new_test_idx]
    tr2_dom = set(tr2["url"].map(registrable_domain))
    te2_dom = te2["url"].map(registrable_domain)
    overlap2 = te2_dom.isin(tr2_dom).sum()
    log(f"Test rows whose registrable domain ALSO appears in train: "
        f"{overlap2}/{len(te2)} ({overlap2/len(te2):.1%})")
    for lbl, name in [(1, "phishing"), (0, "benign")]:
        sub = te2[te2["label"] == lbl]
        sub_dom = sub["url"].map(registrable_domain)
        ov = sub_dom.isin(tr2_dom).sum()
        log(f"  {name}: {ov}/{len(sub)} ({ov/len(sub):.1%}) test domains seen in train")
    log(f"Train/test class balance (grouped split): "
        f"train={tr2['label'].mean():.1%} phishing, test={te2['label'].mean():.1%} phishing")
    log(f"VERDICT: {'FIXED — zero domain overlap' if overlap2 == 0 else f'STILL LEAKING — {overlap2} rows'}")


# ============================================================ CLAIM 2: source confound
def claim_source_confound(model, explainer, cols):
    log("\n" + "=" * 70)
    log("CLAIM 2 — 'detects phishing' or 'detects PhishTank vs Tranco'?")
    log("Testing on URLs from NEITHER source.")
    log("=" * 70)

    legit = [
        "https://en.wikipedia.org/wiki/Phishing",
        "https://github.com/torvalds/linux/blob/master/README",
        "https://stackoverflow.com/questions/292357/what-is-the-difference",
        "https://www.amazon.com/dp/B08N5WRWNW",
        "https://docs.python.org/3/library/os.path.html",
        "https://www.bbc.co.uk/news/technology-68000000",
        "https://www.reddit.com/r/programming/comments/abc123/some_title/",
        "https://www.nytimes.com/2024/01/15/technology/ai-chips.html",
        "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT",
        "https://www.gov.uk/government/organisations/hm-revenue-customs",
    ]
    phish = [
        "http://secure-paypal-login.com/verify",
        "http://appleid-verify.support-account.com/login",
        "http://bankofamerica.account-alert.info/signin",
        "http://amaz0n-security.tk/update",
        "http://micros0ft-support.xyz/reset-password",
        "http://www.paypa1.com/webscr?cmd=login-run",
        "http://verify-account-facebook.gq/confirm",
        "http://dhl-tracking-parcel.top/track?id=99281",
        "http://netflix-billing-update.online/account",
        "http://coinbase-wallet-verify.live/unlock",
    ]

    def run(urls, want):
        rows, correct = [], 0
        for u in urls:
            r = explain_prediction(u, model, explainer, cols)
            ok = r["verdict"] == want
            correct += ok
            rows.append((u, r["verdict"], r["confidence"], ok))
        return rows, correct

    lrows, lc = run(legit, "legitimate")
    prows, pc = run(phish, "phishing")
    log(f"Legitimate real URLs correct: {lc}/{len(legit)}")
    for u, v, c, ok in lrows:
        if not ok:
            log(f"  FP [{v}@{c}] {u}")
    log(f"Phishing-pattern URLs correct: {pc}/{len(phish)}")
    for u, v, c, ok in prows:
        if not ok:
            log(f"  FN [{v}@{c}] {u}")
    log(f"OUT-OF-DISTRIBUTION accuracy: {(lc+pc)}/{len(legit)+len(phish)} "
        f"({(lc+pc)/(len(legit)+len(phish)):.1%}) vs claimed ~89% in-distribution")

    # vocabulary confound: same domain, vocab path vs random path
    log("\nPath-vocabulary confound (same domain, benign-vocab vs random path):")
    for u in ["https://example-corp.com/products/category/view",
              "https://example-corp.com/x7f2a9q/z1p8k"]:
        r = explain_prediction(u, model, explainer, cols)
        log(f"  {r['verdict']}@{r['confidence']}  {u}")


# ============================================================ CLAIM 3: SHAP faithfulness
def claim_shap_faithfulness(model, explainer, cols):
    log("\n" + "=" * 70)
    log("CLAIM 3 — are SHAP reasons faithful? (perturb a cited feature)")
    log("=" * 70)
    df = load_merged_url_dataset()
    Xall = build_feature_frame(df["url"])[cols]
    benign_median = Xall[df["label"] == 0].median()

    for url in ["http://allegro.id-38247ns4.click",
                "http://secure-login-verify.account-update.tk/signin?id=9"]:
        feats = extract_url_features(url)
        row = pd.DataFrame([feats])[cols]
        base = float(model.predict_proba(row)[0, 1])
        sv = explainer.shap_values(row)[0]
        ranked = sorted(zip(cols, sv), key=lambda kv: kv[1], reverse=True)
        log(f"\n{url}\n  base phishing proba = {base:.3f}")
        for feat, s in ranked[:3]:
            perturbed = row.copy()
            perturbed[feat] = benign_median[feat]
            new = float(model.predict_proba(perturbed)[0, 1])
            log(f"  top-SHAP '{feat}' (shap={s:+.2f}, val={feats[feat]}): "
                f"set to benign median {benign_median[feat]:.2f} -> proba {new:.3f} "
                f"(delta={new-base:+.3f}) {'FAITHFUL' if new < base - 0.02 else 'weak/none'}")


# ============================================================ CLAIM 4: feedback loop
def claim_feedback_loop():
    log("\n" + "=" * 70)
    log("CLAIM 4 — is the 'feedback loop' actually a loop?")
    log("=" * 70)
    src = Path(__file__).resolve().parent
    consumers = []
    for f in src.glob("*.py"):
        txt = f.read_text(encoding="utf-8")
        if "FROM feedback" in txt or "SELECT" in txt and "feedback" in txt:
            # find reads of the feedback table
            if "feedback" in txt and ("SELECT" in txt and "feedback" in txt):
                consumers.append(f.name)
    log("Files that READ the feedback table:")
    for c in consumers:
        log(f"  {c}")
    log("Manual check: the only reader is recent_history() -> DISPLAY only.")
    log("Nothing retrains the model, adjusts a threshold, or maintains an")
    log("allow/deny list from feedback. => It is a feedback LOG, not a LOOP.")


if __name__ == "__main__":
    log("Training deployed model for experiments...")
    model, _X, cols = train_final_model()
    explainer = shap.TreeExplainer(model)

    claim_split_leakage()
    claim_source_confound(model, explainer, cols)
    claim_shap_faithfulness(model, explainer, cols)
    claim_feedback_loop()

    (RESULTS / "review_experiments.txt").write_text("\n".join(out_lines), encoding="utf-8")
    log(f"\nSaved: {RESULTS / 'review_experiments.txt'}")
