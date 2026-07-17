"""Step 2 of the shortener-feature work (per user instruction): check the
base rate BEFORE adding anything. If shorteners appear almost exclusively
in one class in the current training data, an is_shortener feature would
just re-encode source/class correlation, not real signal — report that
honestly rather than adding it blind.

Shortener list source: union of two independently maintained, published
lists (not hand-picked from our own PhishTank/Curlie data):
  - PeterDaveHello/url-shorteners (used by NextDNS, ControlD, RethinkDNS,
    dnslow.me for allowlist/blocklist purposes) — 1,490 domains
  - korlabsio/urlshortener — 499 domains
Union, deduplicated, lowercased: 1,593 domains. Saved to
data/shortener_domains.txt for reuse by features.py.

Note: `ead.me` (the domain behind most of our phishing-recall misses,
`l.ead.me`) is NOT in either list — reported honestly below, not patched.
"""

from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import load_merged_url_dataset, registrable_domain

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SHORTENER_LIST_PATH = DATA_DIR / "shortener_domains.txt"


SOURCE_DIR = DATA_DIR / "shortener_sources"


def build_shortener_set() -> set:
    domains = set()
    for p in SOURCE_DIR.glob("*.txt"):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip().lower()
            if not line or line.startswith("#"):
                continue
            domains.add(line)
    return domains


def main():
    shortener_domains = build_shortener_set()
    SHORTENER_LIST_PATH.write_text("\n".join(sorted(shortener_domains)), encoding="utf-8")
    print(f"Shortener list: {len(shortener_domains)} domains "
          f"(union of PeterDaveHello/url-shorteners + korlabsio/urlshortener)")
    print(f"Saved to: {SHORTENER_LIST_PATH}")
    print(f"'ead.me' in list: {'ead.me' in shortener_domains}  "
          f"(this is the domain behind most phishing-recall misses — confirmed absent)")

    df = load_merged_url_dataset()
    df["reg_domain"] = df["url"].map(registrable_domain)
    df["is_shortener"] = df["reg_domain"].isin(shortener_domains)

    print(f"\n=== Base rate in current training data ({len(df)} URLs) ===")
    for label, name in [(1, "phishing (PhishTank)"), (0, "benign (Curlie)")]:
        sub = df[df["label"] == label]
        rate = sub["is_shortener"].mean()
        n_short = sub["is_shortener"].sum()
        print(f"{name}: {n_short}/{len(sub)} ({rate:.2%}) are on a listed shortener domain")

    # which shortener domains actually appear, per class
    for label, name in [(1, "phishing"), (0, "benign")]:
        sub = df[(df["label"] == label) & (df["is_shortener"])]
        top = sub["reg_domain"].value_counts().head(8)
        print(f"\nTop shortener domains used by {name} URLs in this data:")
        for dom, cnt in top.items():
            print(f"  {dom}: {cnt}")

    p_rate = df[df["label"] == 1]["is_shortener"].mean()
    b_rate = df[df["label"] == 0]["is_shortener"].mean()
    print(f"\n=== Verdict ===")
    print(f"Phishing shortener rate: {p_rate:.2%}, Benign shortener rate: {b_rate:.2%}")
    if b_rate < 0.001:
        print("WARNING: shorteners appear almost exclusively in phishing in THIS data. "
              "An is_shortener feature would likely re-encode source/class correlation, "
              "not real generalisable signal, even though real-world benign shortener use "
              "(e.g. bit.ly in legitimate marketing email) exists outside this sample.")
    else:
        print(f"Shorteners appear in BOTH classes ({b_rate:.2%} of benign, {p_rate:.2%} of "
              f"phishing). Provided the ratio isn't ~exclusively one class, is_shortener "
              f"encodes a real behavioural signal (destination obscured) rather than pure "
              f"source correlation.")


if __name__ == "__main__":
    main()
