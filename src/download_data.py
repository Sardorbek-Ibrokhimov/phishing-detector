"""Day 1 data acquisition.

Downloads:
  1. Phishing URLs  - PhishTank public CSV (fallback: OpenPhish free feed)
  2. Benign URLs    - Tranco top-1M domain list (free, no auth)
  3. UCI Phishing Websites dataset (id=327) via ucimlrepo

The Kaggle email corpus requires API credentials (~/.kaggle/kaggle.json)
and is downloaded separately once those exist.
"""

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "phishing-detection-dissertation/0.1 (academic research)"}


def download_phishing_urls() -> Path:
    out = DATA_DIR / "phishing_urls.csv"
    if out.exists():
        print(f"[skip] {out} already exists")
        return out

    # PhishTank public dump
    try:
        url = "http://data.phishtank.com/data/online-valid.csv"
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df = df[["url"]].dropna()
        df["source"] = "phishtank"
        df.to_csv(out, index=False)
        print(f"[ok] PhishTank: {len(df)} phishing URLs")
        return out
    except Exception as e:
        print(f"[warn] PhishTank download failed ({e}); falling back to OpenPhish")

    # OpenPhish free feed (plain text, one URL per line)
    url = "https://openphish.com/feed.txt"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    urls = [line.strip() for line in r.text.splitlines() if line.strip()]
    df = pd.DataFrame({"url": urls})
    df["source"] = "openphish"
    df.to_csv(out, index=False)
    print(f"[ok] OpenPhish: {len(df)} phishing URLs")
    return out


def download_benign_urls(n: int = 50000) -> Path:
    out = DATA_DIR / "benign_urls.csv"
    if out.exists():
        print(f"[skip] {out} already exists")
        return out

    url = "https://tranco-list.eu/top-1m.csv.zip"
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = zf.namelist()[0]
        df = pd.read_csv(zf.open(name), names=["rank", "domain"])
    df = df.head(n)
    df["url"] = "http://" + df["domain"].astype(str) + "/"
    df = df[["url"]]
    df["source"] = "tranco"
    df.to_csv(out, index=False)
    print(f"[ok] Tranco: {len(df)} benign URLs")
    return out


def download_uci() -> Path:
    out = DATA_DIR / "uci_phishing.csv"
    if out.exists():
        print(f"[skip] {out} already exists")
        return out

    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=327)  # Phishing Websites
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.to_csv(out, index=False)
    print(f"[ok] UCI Phishing Websites: {df.shape[0]} rows, {df.shape[1]} cols")
    return out


if __name__ == "__main__":
    failures = []
    for fn in (download_phishing_urls, download_benign_urls, download_uci):
        try:
            fn()
        except Exception as e:
            print(f"[error] {fn.__name__}: {e}")
            failures.append(fn.__name__)
    if failures:
        sys.exit(f"Failed: {', '.join(failures)}")
    print("All downloads complete.")
