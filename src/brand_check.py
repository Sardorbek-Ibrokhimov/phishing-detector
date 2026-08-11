"""Brand-impersonation cross-check: does a URL's domain actually belong to
the brand the surrounding text claims to be from?

Deliberately NOT a trained model — a small curated lexicon plus exact
domain matching. This is the one signal neither the URL model nor the
email-text model can see on its own, since each only ever looks at its own
input in isolation: the URL model has no idea what brand the text claims to
be; the text model has no idea what domain the link actually points to. A
link to a domain that isn't the brand it claims to be is one of the most
reliable real phishing tells there is, and it needs no training data at all
to check — just an "is this string equal to that string" comparison.
"""

import re

import tldextract

# Commonly-impersonated brands (per typical phishing-target lists, e.g.
# APWG/PhishLabs annual reports) mapped to their real registrable domain(s).
# Deliberately short and deliberately excludes brand names that double as
# ordinary English words (e.g. "chase", "target") — a false brand-mention
# match here would turn an unrelated email into a false "mismatch" alarm,
# which is worse than silently missing a less common brand.
KNOWN_BRANDS: dict[str, list[str]] = {
    "paypal": ["paypal.com"],
    "amazon": ["amazon.com"],
    "microsoft": ["microsoft.com", "live.com", "outlook.com", "office.com"],
    "apple": ["apple.com", "icloud.com"],
    "google": ["google.com", "gmail.com"],
    "netflix": ["netflix.com"],
    "facebook": ["facebook.com", "fb.com"],
    "instagram": ["instagram.com"],
    "linkedin": ["linkedin.com"],
    "docusign": ["docusign.net", "docusign.com"],
    "dropbox": ["dropbox.com"],
    "dhl": ["dhl.com"],
    "fedex": ["fedex.com"],
    "wells fargo": ["wellsfargo.com"],
    "bank of america": ["bankofamerica.com"],
    "hsbc": ["hsbc.com", "hsbc.co.uk"],
    "coinbase": ["coinbase.com"],
    "binance": ["binance.com"],
    "steam": ["steampowered.com", "steamcommunity.com"],
    "spotify": ["spotify.com"],
    "adobe": ["adobe.com"],
    "wetransfer": ["wetransfer.com"],
    "zoom": ["zoom.us"],
}

_BRAND_PATTERNS = {b: re.compile(r"\b" + re.escape(b) + r"\b") for b in KNOWN_BRANDS}


def registrable_domain(url: str) -> str:
    ext = tldextract.extract(url if "://" in url else "http://" + url)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


def find_brand_mismatches(text: str, urls: list[str]) -> list[dict]:
    """For every known brand mentioned in the text, check whether ANY of the
    URLs in the same message actually resolves to that brand's real domain.
    Mentioned but not linked (or linked somewhere else entirely) -> flag it.

    Returns a list of dicts (empty if nothing suspicious), one per
    mismatched brand: {"brand", "linked_domain", "example_url"}.
    """
    if not urls:
        return []
    text_l = text.lower()
    url_domains = [(u, registrable_domain(u)) for u in urls]

    mismatches = []
    for brand, real_domains in KNOWN_BRANDS.items():
        if not _BRAND_PATTERNS[brand].search(text_l):
            continue
        if any(d in real_domains for _, d in url_domains):
            continue  # text and at least one link agree — not a mismatch
        example_url, example_domain = url_domains[0]
        mismatches.append({
            "brand": brand,
            "linked_domain": example_domain,
            "example_url": example_url,
        })
    return mismatches
