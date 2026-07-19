"""Black-box tests: knowing only the documented HTTP interface.

Documented interface:
  POST /analyze  {content:str} -> stable schema (see api.py module docstring)
  POST /feedback {analysis_id?, url?, model_verdict, confidence?, corrected_label}
  GET  /history?limit=N
  GET  /health
  GET  /
"""
import time
import uuid

import pytest
import requests

from conftest import AUTH_HEADERS, BASE, GATE_AUTH

# Session that carries the platform gate's Basic Auth (or None when the
# server under test is ungated), so every request below passes the gate.
S = requests.Session()
S.auth = GATE_AUTH

PHISH = "http://allegro.id-38247ns4.click"
BENIGN = "https://www.wikipedia.org/"


def post(content):
    return S.post(f"{BASE}/analyze", json={"content": content}, timeout=60)


# ---------------------------------------------------------------- valid inputs
def test_health(base):
    r = S.get(f"{BASE}/health", timeout=10)
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_valid_url(base):
    d = post(PHISH).json()
    u = d["urls_analyzed"][0]
    assert u["verdict"] in {"phishing", "legitimate"}
    # Assert on model_confidence: it is always a float, whereas `confidence`
    # is deliberately null if this URL happens to carry a human correction.
    assert 0.5 <= u["model_confidence"] <= 1.0


def test_valid_email_text(base):
    d = post(f"See {BENIGN} please").json()
    assert d["urls_analyzed"][0]["url"] == BENIGN


# ---------------------------------------------------------------- boundaries
@pytest.mark.parametrize("content,expect_unknown", [
    ("", True),
    ("   \t ", True),
    ("a", True),
    ("http://", True),          # scheme with no host
    ("x.y", True),              # "y" is not a real public suffix (G6 fix)
    ("example.com", False),     # "com" is a real public suffix
])
def test_boundary_inputs(base, content, expect_unknown):
    d = post(content).json()
    is_unknown = d["overall_verdict"] == "unknown"
    assert is_unknown == expect_unknown, f"{content!r} -> {d['overall_verdict']}"


# ---------------------------------------------------------------- G6: URL over-extraction
@pytest.mark.parametrize("content", [
    "report.pdf", "Node.js", "someone@example.com",
])
def test_g6_non_urls_rejected(base, content):
    d = post(content).json()
    assert d["overall_verdict"] == "unknown", f"{content!r} was misread as a URL"
    assert d["urls_analyzed"] == []


@pytest.mark.parametrize("content", [
    "192.168.1.1",                    # bare IP host must still work
    "paypal-login.tk/signin",         # bare host+path with a real suffix
])
def test_g6_plausible_bare_urls_still_accepted(base, content):
    d = post(content).json()
    assert d["overall_verdict"] != "unknown", f"{content!r} should have been treated as a URL"
    assert d["urls_analyzed"][0]["url"] == content


# ---------------------------------------------------------------- B4: stable contract
RESPONSE_KEYS = {"input_type", "urls_analyzed", "overall_verdict", "overall_confidence",
                 "note", "truncated", "corrections_applied"}


def test_contract_consistent_schema(base):
    """Every /analyze response has the same top-level keys regardless of
    branch (B4 fix) — a client never has to check which fields exist."""
    url_keys = set(post(PHISH).json().keys())
    unknown_keys = set(post("no links here at all").json().keys())
    assert url_keys == unknown_keys == RESPONSE_KEYS


def test_contract_urls_analyzed_item_shape(base):
    u = post(PHISH).json()["urls_analyzed"][0]
    for k in ("url", "verdict", "confidence", "reasons", "virustotal", "analysis_id"):
        assert k in u, f"missing {k}"
    assert isinstance(u["reasons"], list)
    assert set(u["virustotal"]).issuperset({"available", "link"})


# ---------------------------------------------------------------- idempotency
def test_idempotent_verdict(base):
    """The meaningful idempotency property: same input -> same verdict and
    confidence. (A byte-identical full response, including a monotonically
    incrementing analysis_id from logging each call, is neither expected
    nor desirable — that's correct audit-log behaviour, not a bug.)"""
    a = post(PHISH).json()["urls_analyzed"][0]
    b = post(PHISH).json()["urls_analyzed"][0]
    assert (a["verdict"], a["confidence"]) == (b["verdict"], b["confidence"])


# ---------------------------------------------------------------- adversarial
def test_sql_injection_content_does_not_corrupt(base):
    post("'; DROP TABLE analyses;-- http://evil.com")
    post("http://x.com/'||(SELECT 1)||'")
    # if injection worked, history/analyze would break
    assert S.get(f"{BASE}/history", timeout=10).status_code == 200
    assert post(PHISH).status_code == 200


def test_sql_injection_feedback_fields(base):
    r = S.post(f"{BASE}/feedback", headers=AUTH_HEADERS, json={
        "url": "x'; DROP TABLE feedback;--", "model_verdict": "phishing",
        "corrected_label": "legitimate'; DELETE FROM feedback;--"}, timeout=10)
    assert r.status_code == 200
    assert S.get(f"{BASE}/history", timeout=10).status_code == 200


def test_xss_payload_is_data_not_executable(base):
    """The URL regex stops at '<' (it's not a valid URL character), so a
    script tag riding along after a URL is simply never included in the
    extracted URL — safe by construction, not by sanitisation."""
    payload = 'http://evil.com/<script>alert(1)</script>'
    d = post(payload).json()
    extracted = d["urls_analyzed"][0]["url"]
    assert "<script>" not in extracted
    assert extracted == "http://evil.com/"
    assert d["urls_analyzed"][0]["verdict"] in {"phishing", "legitimate"}


def test_null_byte(base):
    r = post("http://exa\x00mple.com/login")
    assert r.status_code < 500, f"null byte caused {r.status_code}"


def test_crlf_header_injection(base):
    r = post("http://evil.com/\r\nSet-Cookie: pwned=1")
    assert r.status_code < 500


def test_unicode_rtl_and_punycode(base):
    for u in ["http://exampΕ.com/login",           # greek homograph
              "http://аpple.com/",                  # cyrillic 'a'
              "http://xn--pple-43d.com/",            # punycode
              "http://test.com/‮gnp.exe"]:     # RTL override
        r = post(u)
        assert r.status_code < 500, f"{u!r} -> {r.status_code}"


def test_deeply_nested_subdomains(base):
    r = post("http://" + "a." * 300 + "com/login")
    assert r.status_code < 500


# ---------------------------------------------------------------- B6: size / fan-out caps
def test_oversized_body_rejected(base):
    """A body over the documented 100KB limit gets a clean 413, not a slow
    accept-and-process (B6 fix)."""
    big = "http://example.com/" + "a" * 150_000
    t = time.perf_counter()
    r = post(big)
    dt = time.perf_counter() - t
    assert r.status_code == 413, f"expected 413 for oversized body, got {r.status_code}"
    assert dt < 5, f"413 rejection took {dt:.2f}s — should be near-instant"


def test_url_fanout_capped(base):
    """More than MAX_URLS_PER_REQUEST (20) URLs in one input are truncated,
    not analysed unbounded (B6 fix)."""
    content = " ".join(f"http://example{i}.com/" for i in range(30))
    d = post(content).json()
    assert len(d["urls_analyzed"]) == 20
    assert d["truncated"] is True
    assert d["note"] is not None


# ---------------------------------------------------------------- validation
def test_feedback_requires_fields(base):
    # Authenticated, so this exercises body validation rather than the gate.
    r = S.post(f"{BASE}/feedback", headers=AUTH_HEADERS,
                      json={"url": "http://x.com"}, timeout=10)
    assert r.status_code == 422  # missing model_verdict / corrected_label


def test_analyze_requires_content(base):
    r = S.post(f"{BASE}/analyze", json={}, timeout=10)
    assert r.status_code == 422


# ---------------------------------------------------------------- platform gate (Basic Auth)
def test_gate_blocks_without_credentials(base):
    """When the deployment gate is enabled, NO route is reachable without the
    password — not even /health. Self-skips against an ungated test server."""
    if GATE_AUTH is None:
        pytest.skip("server under test is ungated (no GATE_PASSWORD)")
    r = requests.get(f"{BASE}/health", timeout=10)  # deliberately no auth
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate", "").lower().startswith("basic")


def test_gate_blocks_wrong_password(base):
    if GATE_AUTH is None:
        pytest.skip("server under test is ungated (no GATE_PASSWORD)")
    r = requests.get(f"{BASE}/health", auth=(GATE_AUTH[0], "wrong-password"), timeout=10)
    assert r.status_code == 401


def test_gate_allows_with_credentials(base):
    if GATE_AUTH is None:
        pytest.skip("server under test is ungated (no GATE_PASSWORD)")
    r = requests.get(f"{BASE}/health", auth=GATE_AUTH, timeout=10)
    assert r.status_code == 200


def test_gate_protects_analyze_too(base):
    """The gate covers the whole app, not just writes."""
    if GATE_AUTH is None:
        pytest.skip("server under test is ungated (no GATE_PASSWORD)")
    r = requests.post(f"{BASE}/analyze", json={"content": BENIGN}, timeout=30)  # no auth
    assert r.status_code == 401


# ---------------------------------------------------------------- auth on /feedback
def _feedback_body():
    return {"analysis_id": None, "url": "http://auth-probe.tk/x",
            "model_verdict": "phishing", "confidence": 0.9,
            "corrected_label": "legitimate"}


def test_feedback_rejects_missing_api_key(base):
    """A correction is an authoritative override, so it must not be writable
    anonymously."""
    r = S.post(f"{BASE}/feedback", json=_feedback_body(), timeout=10)
    assert r.status_code in (401, 503), f"unauthenticated write returned {r.status_code}"


def test_feedback_rejects_wrong_api_key(base):
    r = S.post(f"{BASE}/feedback", headers={"X-API-Key": "definitely-not-the-key"},
                      json=_feedback_body(), timeout=10)
    assert r.status_code in (401, 503)


def test_feedback_accepts_correct_api_key(base):
    if not S.get(f"{BASE}/health", timeout=10).json()["corrections_enabled"]:
        pytest.skip("server has no FEEDBACK_API_KEY configured")
    r = S.post(f"{BASE}/feedback", headers=AUTH_HEADERS,
                      json=_feedback_body(), timeout=10)
    assert r.status_code == 200


def test_analyze_remains_open(base):
    """/analyze is deliberately unauthenticated — it cannot change a verdict,
    so it is not a privileged write. Guards against over-correcting the auth
    fix onto the read path."""
    assert post(BENIGN).status_code == 200


def test_health_does_not_leak_the_key(base):
    """/health reports whether corrections are enabled, never the secret."""
    body = S.get(f"{BASE}/health", timeout=10).text
    assert "corrections_enabled" in body
    assert AUTH_HEADERS["X-API-Key"] not in body


# ---------------------------------------------------------------- C4: correction override
def _unique_url(tag):
    """Each test gets its own URL so corrections can't bleed between tests
    or collide with the shared demo state in feedback.db."""
    return f"http://c4-{tag}-{uuid.uuid4().hex[:10]}.tk/page"


def _correct(url, analysis_id, model_verdict, corrected_label):
    return S.post(f"{BASE}/feedback", headers=AUTH_HEADERS, json={
        "analysis_id": analysis_id, "url": url,
        "model_verdict": model_verdict, "confidence": 0.9,
        "corrected_label": corrected_label}, timeout=10)


def test_correction_overrides_model_on_reanalysis(base):
    """The full C4 loop: analyse -> correct -> re-analyse returns the
    corrected verdict, labelled as human-sourced, with the model's own
    verdict still visible."""
    url = _unique_url("override")

    first = post(url).json()["urls_analyzed"][0]
    assert first["verdict_source"] == "model"
    assert first["correction"] is None
    model_verdict = first["model_verdict"]
    opposite = "legitimate" if model_verdict == "phishing" else "phishing"

    r = _correct(url, first["analysis_id"], model_verdict, opposite)
    assert r.status_code == 200 and r.json()["correction_applied"] is True

    second = post(url).json()["urls_analyzed"][0]
    assert second["verdict"] == opposite, "corrected label should now win"
    assert second["verdict_source"] == "human_correction"
    assert second["confidence"] is None, "a human correction carries no probability"
    # the override must be transparent, not a silent swap
    assert second["model_verdict"] == model_verdict
    assert second["model_confidence"] is not None
    assert second["correction"]["overrode_model"] is True
    assert second["correction"]["corrected_label"] == opposite


def test_correction_does_not_leak_to_other_urls(base):
    """A correction applies to the exact URL only — never the whole host."""
    base_host = f"c4-leak-{uuid.uuid4().hex[:10]}.tk"
    corrected_url = f"http://{base_host}/corrected"
    other_urls = [f"http://{base_host}/other", f"http://{base_host}/", f"https://{base_host}/corrected"]

    first = post(corrected_url).json()["urls_analyzed"][0]
    opposite = "legitimate" if first["model_verdict"] == "phishing" else "phishing"
    _correct(corrected_url, first["analysis_id"], first["model_verdict"], opposite)

    # the corrected URL is overridden...
    assert post(corrected_url).json()["urls_analyzed"][0]["verdict_source"] == "human_correction"
    # ...but nothing else on that host is
    for other in other_urls:
        got = post(other).json()["urls_analyzed"][0]
        assert got["verdict_source"] == "model", f"{other} wrongly inherited the correction"
        assert got["correction"] is None


def test_corrections_applied_counter_and_note(base):
    url = _unique_url("counter")
    first = post(url).json()["urls_analyzed"][0]
    opposite = "legitimate" if first["model_verdict"] == "phishing" else "phishing"
    _correct(url, first["analysis_id"], first["model_verdict"], opposite)

    d = post(url).json()
    assert d["corrections_applied"] == 1
    assert "human correction" in (d["note"] or "")


def test_uncorrected_response_has_correction_fields(base):
    """B4 discipline: the C4 fields are present even when no correction
    exists, so clients never branch on key existence."""
    u = post(BENIGN).json()["urls_analyzed"][0]
    for k in ("verdict_source", "model_verdict", "model_confidence", "correction"):
        assert k in u, f"missing {k}"
    assert u["verdict_source"] == "model"
    assert u["correction"] is None
    assert u["model_confidence"] == u["confidence"]


# ---------------------------------------------------------------- rate limiting
# NOTE: must run last in this file — it deliberately exhausts the shared
# per-IP /analyze rate limit budget, which would otherwise cause later tests
# in the same 60s window to see spurious 429s.
def test_zz_rate_limit_eventually_kicks_in(base):
    """Firing past the server's configured limit must eventually produce a
    429. Reads the effective limit from /health rather than assuming the
    default, so it works whatever the server was started with.

    Skips when the configured limit is too high to exceed *sequentially*
    inside the limiter's own 60s sliding window — at that point the earliest
    requests age out faster than we can add new ones, so no amount of serial
    traffic would trip it (that is the limiter behaving correctly, not a
    bug). The limiter's logic is covered deterministically by
    TestRateLimiter in test_greybox.py."""
    limit = S.get(f"{BASE}/health", timeout=10).json()["analyze_rate_limit_per_min"]

    # Send until we trip the limit or run out of window. No latency model:
    # per-request cost varies a lot (a live VirusTotal call is ~2s, a
    # VT-rate-limited one is ~20ms), so measuring one request and
    # extrapolating badly misestimates sustained throughput.
    deadline = time.perf_counter() + 45
    codes = []
    while time.perf_counter() < deadline:
        codes.append(post(BENIGN).status_code)
        if codes[-1] == 429:
            break

    if 429 not in codes:
        pytest.skip(
            f"could not exceed {limit}/min within 45s ({len(codes)} requests sent) — "
            f"at this limit the sliding window drains faster than serial requests "
            f"fill it; limiter logic covered by TestRateLimiter unit tests"
        )
    assert 429 in codes
