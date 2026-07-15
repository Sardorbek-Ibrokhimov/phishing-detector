"""Day 3: rebuild the benign URL set so its shape (subdomain count, path
depth, query presence) matches the phishing set's empirical distribution,
instead of being bare root domains with none of the above. Fixes Artifact
2 in (num_subdomains leaking the Tranco/PhishTank
source split rather than encoding real phishing signal).

Proportions are not hand-picked: they're sampled from the actual
PhishTank URL set's own shape distribution, so shape can no longer
separate the two classes by construction.

Source of domains: data/benign_urls_bare_domains.csv (original Tranco
bare-domain download, preserved unchanged).
Output: data/benign_urls.csv (what train_baseline.py reads), overwritten
with generated realistic-shaped URLs.
"""

import random
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 42

SUBDOMAIN_TOKENS = [
    "www", "mail", "blog", "shop", "cdn", "app", "api", "m",
    "support", "help", "docs", "news", "dev", "static", "images",
]
PATH_TOKENS = [
    "products", "about", "articles", "posts", "docs", "search",
    "category", "user", "view", "index", "page", "item", "profile",
    "settings", "help", "news", "2024", "2025",
]
QUERY_KEYS = ["id", "page", "ref", "category", "sort", "q", "utm_source", "lang"]


def url_shape(url: str) -> dict:
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc.split(":")[0].lower()
    num_subdomains = max(host.count(".") - 1, 0)
    path_segments = [p for p in parsed.path.split("/") if p]
    return {
        "num_subdomains": min(num_subdomains, 2),  # bucket: 0 / 1 / 2+
        "path_depth": min(len(path_segments), 3),  # bucket: 0 / 1 / 2 / 3+
        "has_query": bool(parsed.query),
    }


def compute_phishtank_shape_distribution() -> dict:
    df = pd.read_csv(DATA_DIR / "phishing_urls.csv")
    shapes = pd.DataFrame(df["url"].map(url_shape).tolist())
    return {
        "num_subdomains": shapes["num_subdomains"].value_counts(normalize=True).sort_index().to_dict(),
        "path_depth": shapes["path_depth"].value_counts(normalize=True).sort_index().to_dict(),
        "has_query": shapes["has_query"].value_counts(normalize=True).to_dict(),
    }


def weighted_choice(rng: random.Random, dist: dict):
    return rng.choices(list(dist.keys()), weights=list(dist.values()), k=1)[0]


def generate_realistic_url(rng: random.Random, domain: str, dist: dict) -> str:
    subdomain_bucket = weighted_choice(rng, dist["num_subdomains"])
    num_subdomains = rng.randint(2, 3) if subdomain_bucket == 2 else subdomain_bucket

    path_bucket = weighted_choice(rng, dist["path_depth"])
    path_depth = rng.randint(3, 5) if path_bucket == 3 else path_bucket

    has_query = rng.random() < dist["has_query"].get(True, 0.0)

    host = domain
    if num_subdomains >= 1:
        subs = rng.sample(SUBDOMAIN_TOKENS, k=min(num_subdomains, len(SUBDOMAIN_TOKENS)))
        host = ".".join(subs) + "." + domain

    path = ""
    if path_depth > 0:
        segments = [rng.choice(PATH_TOKENS) for _ in range(path_depth)]
        path = "/" + "/".join(segments)

    query = ""
    if has_query:
        params = [f"{rng.choice(QUERY_KEYS)}={rng.randint(1, 9999)}" for _ in range(rng.randint(1, 3))]
        query = "?" + "&".join(params)

    return f"https://{host}{path}{query}"


def main():
    raw_path = DATA_DIR / "benign_urls_bare_domains.csv"
    current_path = DATA_DIR / "benign_urls.csv"
    if not raw_path.exists():
        current_path.rename(raw_path)
        print(f"[backup] preserved raw bare-domain set as {raw_path}")

    domains_df = pd.read_csv(raw_path)
    dist = compute_phishtank_shape_distribution()

    print("PhishTank empirical shape distribution (source of the proportions used below):")
    print(f"  num_subdomains (0/1/2+): {dist['num_subdomains']}")
    print(f"  path_depth (0/1/2/3+):   {dist['path_depth']}")
    print(f"  has_query (True/False):  {dist['has_query']}")

    rng = random.Random(RANDOM_STATE)
    urls = []
    for domain_url in domains_df["url"]:
        domain = urlparse(domain_url).netloc.rstrip("/")
        urls.append(generate_realistic_url(rng, domain, dist))

    out = pd.DataFrame({"url": urls, "source": "tranco_realistic"})
    out.to_csv(current_path, index=False)
    print(f"\n[ok] Wrote {len(out)} realistic-shaped benign URLs to {current_path}")

    gen_shapes = pd.DataFrame(out["url"].map(url_shape).tolist())
    print("\nGenerated benign set shape distribution (should roughly match PhishTank's above):")
    print(f"  num_subdomains: {gen_shapes['num_subdomains'].value_counts(normalize=True).sort_index().to_dict()}")
    print(f"  path_depth:     {gen_shapes['path_depth'].value_counts(normalize=True).sort_index().to_dict()}")
    print(f"  has_query:      {gen_shapes['has_query'].value_counts(normalize=True).to_dict()}")

    print("\nSample generated benign URLs:")
    for u in out["url"].sample(5, random_state=RANDOM_STATE):
        print(f"  {u}")


if __name__ == "__main__":
    main()
