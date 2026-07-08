"""Lexical URL feature extraction.

Deliberately host-independent: no DNS/WHOIS lookups, so feature extraction
is fast, reproducible, and works offline. These are the standard lexical
features used in the phishing-URL literature.
"""

import math
import re
from urllib.parse import urlparse

import pandas as pd

SUSPICIOUS_TOKENS = (
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "paypal", "password", "webscr", "ebayisapi", "wallet",
)

IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {c: s.count(c) / len(s) for c in set(s)}
    return -sum(p * math.log2(p) for p in freq.values())


def extract_url_features(url: str) -> dict:
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
    except ValueError:
        parsed = urlparse("http://invalid.invalid/")
    host = parsed.netloc.split(":")[0].lower()
    path = parsed.path or ""
    query = parsed.query or ""
    url_l = url.lower()

    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_slashes": url.count("/"),
        "num_special": sum(url.count(c) for c in "@!#$%&*?=~"),
        "has_at": int("@" in url),
        "has_ip_host": int(bool(IP_RE.match(host))),
        "num_subdomains": max(host.count(".") - 1, 0),
        "uses_https": int(parsed.scheme == "https"),
        "has_port": int(":" in parsed.netloc),
        "query_length": len(query),
        "num_query_params": query.count("=") if query else 0,
        "host_entropy": shannon_entropy(host),
        "url_entropy": shannon_entropy(url),
        "num_suspicious_tokens": sum(t in url_l for t in SUSPICIOUS_TOKENS),
        "digit_ratio": sum(c.isdigit() for c in url) / max(len(url), 1),
        "tld_length": len(host.rsplit(".", 1)[-1]) if "." in host else 0,
    }


def build_feature_frame(urls: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([extract_url_features(u) for u in urls])
