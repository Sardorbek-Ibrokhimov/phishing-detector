"""Part 4: edge-case degradation tests.

For each edge input, capture the actual API response (not just the status
code) so we can judge whether the system degrades gracefully. The
'API unreachable from the frontend' case is tested separately in the
browser (frontend fetch-failure handling).
"""

import json
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
RESULTS = Path(__file__).resolve().parent.parent / "results"

CASES = {
    "empty input": "",
    "whitespace only": "   \t  \n ",
    "malformed URL": "ht!tp:/bad url .com//??",
    "very long input (~20k chars)": "http://example.com/" + "a" * 20000,
    "URL with unusual characters": "http://xn--e1afmkfd.xn--p1ai/пример?q=<script>alert(1)</script>&x=@#%",
}


def summarize(resp):
    try:
        body = resp.json()
    except Exception:
        return {"status": resp.status_code, "body": resp.text[:200]}
    out = {"status": resp.status_code, "overall_verdict": body.get("overall_verdict")}
    if body.get("urls_analyzed"):
        u = body["urls_analyzed"][0]
        out["first_url_verdict"] = f"{u['verdict']}@{u['confidence']}"
        out["url_len_echoed"] = len(u["url"])
    if body.get("note"):
        out["note"] = body["note"][:80]
    return out


def main():
    print("=== Part 4: edge-case degradation ===\n")
    report = {}
    for name, payload in CASES.items():
        try:
            r = requests.post(f"{BASE}/analyze", json={"content": payload}, timeout=60)
            summary = summarize(r)
            graceful = r.status_code < 500
        except Exception as e:
            summary = {"error": f"{type(e).__name__}: {e}"}
            graceful = False
        report[name] = {"graceful": graceful, **summary}
        flag = "graceful" if graceful else "CRASH"
        print(f"[{flag}] {name}\n    {json.dumps(summary)}\n")

    # also test a totally empty JSON body (missing 'content' field) -> should be a clean 422, not 500
    try:
        r = requests.post(f"{BASE}/analyze", json={}, timeout=10)
        report["missing content field"] = {"graceful": r.status_code < 500, "status": r.status_code}
        print(f"[{'graceful' if r.status_code < 500 else 'CRASH'}] missing 'content' field -> HTTP {r.status_code} (expect 422 validation)\n")
    except Exception as e:
        report["missing content field"] = {"graceful": False, "error": str(e)}

    out = RESULTS / "edge_cases.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    crashed = [k for k, v in report.items() if not v.get("graceful")]
    print(f"Saved: {out}")
    print(f"Result: {len(report) - len(crashed)}/{len(report)} degraded gracefully; "
          f"crashed: {crashed or 'none'}")


if __name__ == "__main__":
    main()
