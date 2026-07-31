"""One-off EXPORT utility for the dissertation data-analysis appendix.

Produces, for each of the four data sources, a raw file (sampled to 10k where
the source is large) and a cleaned file. The cleaned files are produced by
CALLING THE EXISTING PIPELINE FUNCTIONS — no new cleaning/split/feature logic
is written here; this script only invokes the pipeline and serialises the
result, plus samples the raws and writes a manifest.

Output: data_export_for_supervisor/  (local, not committed).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"
OUT = ROOT / "data_export_for_supervisor"
SAMPLE_N = 10000
SEED = 42

# existing pipeline functions (cleaning/split/features live here, not below)
from features import build_feature_frame
from train_baseline import grouped_train_test_split, load_merged_url_dataset
import email_data
import uci_url_model
import build_benign_urls_curlie

manifest = []  # (filename, rows, full_count_or_None, description)


def save(df, name, desc, full_count=None):
    df.to_csv(OUT / name, index=False)
    manifest.append((name, len(df), full_count, desc))
    tag = f" (sample of {full_count:,})" if full_count and full_count != len(df) else ""
    print(f"  wrote {name}: {len(df):,} rows{tag}")


def stratified_sample(df, n, by, seed=SEED):
    if n >= len(df):
        return df
    frac = n / len(df)
    parts = [g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed)
             for _, g in df.groupby(by)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    OUT.mkdir(exist_ok=True)

    # ================= RAW files (exactly as downloaded; sampled if large) =====
    print("RAW:")
    phish_raw = pd.read_csv(DATA / "phishing_urls.csv")
    save(phish_raw.sample(min(SAMPLE_N, len(phish_raw)), random_state=SEED),
         "01_phishtank_raw_sample.csv",
         "PhishTank phishing URLs, exactly as downloaded (random 10k sample).",
         len(phish_raw))

    # Curlie raw = URLs from the downloaded directory dump (.tsv), via the
    # existing reader; sample 10k across a few categories.
    curlie_urls = []
    for f in ["rdf-Top-c.tsv", "rdf-Business-c.tsv", "rdf-Society-c.tsv", "rdf-Arts-c.tsv"]:
        p = DATA / "curlie" / "curlie-rdf" / f
        if p.exists():
            curlie_urls += build_benign_urls_curlie.read_curlie_urls(p)
    curlie_raw = pd.DataFrame({"url": curlie_urls})
    tsv_total = 2739037  # measured: wc -l of all rdf-*-c.tsv (see manifest note)
    save(curlie_raw.sample(min(SAMPLE_N, len(curlie_raw)), random_state=SEED),
         "02_curlie_benign_raw_sample.csv",
         "Curlie (curlie.org) directory URLs, as downloaded (10k sample; full "
         f"dump ~{tsv_total:,} rows across all category .tsv files).",
         tsv_total)

    uci_raw = pd.read_csv(DATA / "uci_phishing.csv")
    save(uci_raw, "03_uci_raw.csv",
         "UCI Phishing Websites dataset, exactly as downloaded (full, 30 "
         "pre-extracted features + label).")

    email_raw = pd.read_csv(DATA / "phishing_emails.csv")
    save(email_raw.sample(SAMPLE_N, random_state=SEED),
         "04_email_raw_sample.csv",
         "Kaggle phishing-email corpus, as downloaded (random 10k sample; "
         "text is pre-lowercased/punctuation-stripped by the source).",
         len(email_raw))

    # ================= CLEANED files (via existing pipeline) ==================
    print("CLEANED (calling existing pipeline):")

    # --- URL model: merged, deduped, feature-extracted, domain-grouped split ---
    df = load_merged_url_dataset()                 # dedup + balance (existing)
    X = build_feature_frame(df["url"])             # feature extraction (existing)
    train_idx, test_idx = grouped_train_test_split(df)   # grouped split (existing)
    clean = X.copy()
    clean.insert(0, "url", df["url"].values)
    clean.insert(1, "label", df["label"].values)  # 1=phishing, 0=benign
    if "source" in df.columns:
        clean.insert(2, "source", df["source"].values)
    clean["split"] = np.where(df.index.isin(test_idx), "test", "train")

    # per-source cleaned files (subsets of the single merged feature table)
    save(stratified_sample(clean[clean["label"] == 1], SAMPLE_N, "split"),
         "05_phishtank_cleaned_features_with_split.csv",
         "PhishTank rows (label=1) of the cleaned URL feature table: 20 lexical "
         "features + label + train/test split. 10k sample.",
         int((clean["label"] == 1).sum()))
    save(stratified_sample(clean[clean["label"] == 0], SAMPLE_N, "split"),
         "06_curlie_cleaned_features_with_split.csv",
         "Curlie benign rows (label=0) of the same cleaned URL feature table. "
         "10k sample.",
         int((clean["label"] == 0).sum()))

    # --- UCI: URL-only 9-column subset + same stratified split as the pipeline ---
    X_kept, X_full, y = uci_url_model.load_url_only_uci()   # existing
    Xtr, Xte, ytr, yte = train_test_split(
        X_kept, y, test_size=0.2, stratify=y, random_state=uci_url_model.RANDOM_STATE)
    uci_clean = X_kept.copy()
    uci_clean["label"] = y.values                  # 1=phishing, 0=legit
    uci_clean["split"] = "train"
    uci_clean.loc[Xte.index, "split"] = "test"
    save(uci_clean, "07_uci_cleaned_url_features_with_split.csv",
         "UCI cleaned: the 9 URL-derivable columns the deployed comparison uses "
         "+ label + train/test split (full 11k).")

    # --- Email: deduped + domain(near-dup)-grouped split via existing pipeline ---
    edf = email_data.load_dedup()                  # existing dedup
    etr, ete = email_data.grouped_split(edf)       # existing grouped split
    etr = etr.assign(split="train"); ete = ete.assign(split="test")
    email_clean = pd.concat([etr, ete])[["text", "label", "group", "split"]]

    # PII CHECK before writing: confirm no email addresses in the exported text
    email_re = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
    at_rows = int(email_clean["text"].str.contains("@", regex=False).sum())
    addr_rows = int(email_clean["text"].str.contains(email_re).sum())
    print(f"  [PII check] '@' rows={at_rows}, email-address-pattern rows={addr_rows} (both must be 0)")
    assert at_rows == 0 and addr_rows == 0, "PII check FAILED: email addresses present"

    email_sample = stratified_sample(email_clean, SAMPLE_N, "split")
    save(email_sample, "08_email_cleaned_deduped_split_sample.csv",
         "Kaggle email corpus cleaned: exact-deduplicated, with the near-duplicate "
         "group id and the grouped train/test split. 10k sample. Verified: no "
         "email addresses.", len(email_clean))

    # ================= manifest ==============================================
    lines = ["# Data export — manifest",
             "",
             "Datasets behind the phishing-detection dissertation, in raw (as "
             "downloaded) and cleaned (as the pipeline trains/tests on) form.",
             "Cleaned files were produced by running the project's own pipeline "
             "functions, not by ad-hoc cleaning.",
             "",
             "| File | Rows | Full source count | What it is |",
             "|------|-----:|------------------:|------------|"]
    for name, rows, full, desc in manifest:
        fc = f"{full:,}" if full else "(full)"
        lines.append(f"| `{name}` | {rows:,} | {fc} | {desc} |")
    lines += ["",
              "Notes:",
              "- label: 1 = phishing, 0 = benign/legitimate.",
              "- URL cleaned files share one merged, deduplicated, "
              "feature-extracted table; the split is **domain-grouped** "
              "(no registrable domain in both train and test).",
              "- The email corpus text is pre-lowercased and punctuation-stripped "
              "by the original Kaggle source; the cleaned export was checked to "
              "contain **no email addresses**. It does contain ordinary words "
              "including names from the public Enron research corpus (a standard, "
              "published dataset), but no addresses/credentials.",
              "- **`04_email_raw_sample.csv` is not the untouched component data.** "
              "It samples the source author's pre-combined, address-stripped "
              "`phishing_email.csv` (columns `text_combined,label`). The truly-raw "
              "per-source component files (CEAS_08, Nazario, Nigerian_Fraud, "
              "SpamAssasin, Enron, Ling) still contain real From/To email "
              "addresses and are deliberately **not** exported.",
              "- Email corpus source: Kaggle *Phishing Email Dataset*, "
              "https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset "
              "(no download slug/command was recorded in the repo at build time; "
              "identified from the component-file fingerprint).",
              "- No API keys, passwords, or other secrets are included.",
              ]
    (OUT / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nManifest: {OUT/'MANIFEST.md'}")
    print(f"All files in: {OUT}")


if __name__ == "__main__":
    main()
