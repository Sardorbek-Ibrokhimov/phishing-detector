"""Base-rate check for the second proposed feature, path_is_opaque_short,
BEFORE wiring it into features.py — same discipline as the is_shortener
check (shortener_baserate.py). This feature is domain-list-independent (it
looks at path shape, not domain identity), which is exactly why it's worth
checking separately: it could behave very differently from is_shortener.

Definition (principled, not fit to this dataset): the URL has exactly one
path segment (mimics a flat redirect/tracking key, not a multi-level site
hierarchy), that segment is 4-10 characters, contains no separator
(hyphen/underscore/dot — real CMS slugs almost always use these for SEO),
and looks base62-like (mixed case or contains a digit, i.e. not a plain
lowercase dictionary word or human-readable slug).
"""

from pathlib import Path
from urllib.parse import urlparse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import load_merged_url_dataset


def path_is_opaque_short(url: str) -> bool:
    parsed = urlparse(url if "://" in url else "http://" + url)
    segments = [s for s in parsed.path.split("/") if s]
    if len(segments) != 1:
        return False
    seg = segments[0]
    if not (4 <= len(seg) <= 10):
        return False
    if any(c in seg for c in "-_."):
        return False
    has_upper = any(c.isupper() for c in seg)
    has_digit = any(c.isdigit() for c in seg)
    has_lower = any(c.islower() for c in seg)
    return seg.isalnum() and has_lower and (has_upper or has_digit)


def main():
    df = load_merged_url_dataset()
    df["opaque_short"] = df["url"].map(path_is_opaque_short)

    print("=== Base rate: path_is_opaque_short ===")
    for label, name in [(1, "phishing (PhishTank)"), (0, "benign (Curlie)")]:
        sub = df[df["label"] == label]
        rate = sub["opaque_short"].mean()
        n = sub["opaque_short"].sum()
        print(f"{name}: {n}/{len(sub)} ({rate:.2%})")

    print("\nSample phishing matches:")
    for u in df[(df["label"] == 1) & df["opaque_short"]]["url"].head(8):
        print(f"  {u}")
    print("\nSample benign matches:")
    for u in df[(df["label"] == 0) & df["opaque_short"]]["url"].head(8):
        print(f"  {u}")

    p_rate = df[df["label"] == 1]["opaque_short"].mean()
    b_rate = df[df["label"] == 0]["opaque_short"].mean()
    print(f"\nVerdict: phishing={p_rate:.2%}, benign={b_rate:.2%}")
    if b_rate < 0.005:
        print("WARNING: also near-exclusive to phishing in this data.")
    else:
        print("Appears in both classes at non-trivial rates.")


if __name__ == "__main__":
    main()
