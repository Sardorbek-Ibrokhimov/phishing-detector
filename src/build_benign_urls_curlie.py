"""Build benign URL dataset from Curlie (https://curlie.org), a human-edited
web directory (successor to DMOZ).

Replaces the old template-based benign set which used a fixed 18-word
dictionary for paths. The model learned those patterns as a benign signal
and then flagged real sites with opaque paths as phishing. Curlie URLs
have real-world path diversity (query strings, numeric IDs, etc).

Source: data/curlie/curlie-rdf/*.tsv (bulk download from curlie.org/download)
Output: data/benign_urls.csv
"""

import csv
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CURLIE_DIR = DATA_DIR / "curlie" / "curlie-rdf"
RANDOM_STATE = 42
TARGET_TOTAL = 50000

EXCLUDED_FILES = {"rdf-Adult-c.tsv"}  # not appropriate for this dataset


def read_curlie_urls(path: Path) -> list:
    urls = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            url = row[0].strip()
            if not url:
                continue
            urls.append(url)
    return urls


def is_valid_web_url(url: str) -> bool:
    try:
        p = urlparse(url)
    except ValueError:
        return False
    return p.scheme in ("http", "https") and bool(p.netloc) and len(url) < 500


def main():
    csv_files = sorted(
        f for f in CURLIE_DIR.glob("rdf-*-c.tsv") if f.name not in EXCLUDED_FILES
    )
    print(f"Reading {len(csv_files)} Curlie category files (excluding {EXCLUDED_FILES})...")

    all_urls = []
    for f in csv_files:
        urls = read_curlie_urls(f)
        all_urls.extend(urls)
        print(f"  {f.name}: {len(urls)} rows")

    print(f"\nTotal raw rows: {len(all_urls)}")
    valid = [u for u in all_urls if is_valid_web_url(u)]
    print(f"Valid http(s) URLs: {len(valid)}")

    df = pd.DataFrame({"url": valid})
    df = df.drop_duplicates(subset="url").dropna(subset=["url"])
    print(f"After de-duplication: {len(df)}")

    sample = df.sample(n=min(TARGET_TOTAL, len(df)), random_state=RANDOM_STATE)
    sample = sample.copy()
    sample["source"] = "curlie"

    out_path = DATA_DIR / "benign_urls.csv"
    sample.to_csv(out_path, index=False)
    print(f"\n[ok] Wrote {len(sample)} real Curlie-listed benign URLs to {out_path}")

    # Path-diversity spot check: are these genuinely varied, not templated?
    paths = sample["url"].map(lambda u: urlparse(u).path)
    non_root = (paths.str.len() > 1).sum()
    print(f"\nURLs with a non-trivial path: {non_root}/{len(sample)} ({non_root/len(sample):.1%})")
    has_query = sample["url"].str.contains(r"\?").sum()
    print(f"URLs with a query string: {has_query}/{len(sample)} ({has_query/len(sample):.1%})")

    print("\nSample URLs (should show genuine, non-templated, human-curated diversity):")
    for u in sample["url"].sample(min(15, len(sample)), random_state=RANDOM_STATE):
        print(f"  {u}")


if __name__ == "__main__":
    main()
