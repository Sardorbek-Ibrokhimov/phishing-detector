"""Regenerate the dissertation screenshots in docs/screenshots/.

Self-contained: starts its OWN uvicorn server on a fresh, throwaway database
(so the History tab shows only these demo analyses and nothing else), drives
the four UI states with Playwright, saves a PNG for each, then shuts the
server down. Re-run it any time the UI changes:

    .venv/Scripts/python docs/capture_screenshots.py

Requirements (dev only):
    pip install playwright && playwright install chromium

Notes
-----
* The app is started WITHOUT a GATE_PASSWORD, so it is open locally (no Basic
  Auth prompt). Your real feedback.db is never touched — a temp DB is used.
* VirusTotal: if VIRUSTOTAL_API_KEY is set in the environment it is passed
  through, so the VT panel shows live reputation data for URLs VT has indexed
  (e.g. wikipedia.org). The constructed demo phishing URLs are deliberately
  ones VirusTotal has not seen, so their VT panel reads "URL not previously
  seen" — the honest, real behaviour of a lexical model catching novel URLs.
* Viewport width is 1280 CSS px (DEVICE_SCALE_FACTOR=2 gives crisp 2560px
  PNGs for print; set it to 1 if you want literal 1280px-wide files).
"""

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
VIEWPORT_WIDTH = 1280
DEVICE_SCALE_FACTOR = 2          # 2 = retina-crisp; 1 = literal 1280px-wide PNGs
COLOR_SCHEME = "light"           # "light" or "dark"

PHISH_URL = "http://paypal-secure-login-verify.account-update.tk/signin?user=1234"
LEGIT_URL = "https://www.wikipedia.org/"
EMAIL_TEXT = """From: PayPal Service <service@paypaI-security.com>
Subject: Your account access has been limited

Dear customer,

We noticed unusual sign-in activity on your account. To restore full access,
please verify your identity within 24 hours at the link below:

    http://paypal-secure-login-verify.account-update.tk/signin?user=1234

If you did not request this, secure your account here:

    http://account-verify.secure-signin.confirm-update.ml/login?id=88213

Thank you,
PayPal Security Team"""

# Brand-mismatch showcase: claims Microsoft, links to a domain that isn't
# Microsoft's — exercises all three signals (brand check, email wording,
# URL model) at once, on a different brand than EMAIL_TEXT above.
BRAND_MISMATCH_TEXT = """From: Microsoft Account Team <security@microsoft-alerts.com>
Subject: Action required: unusual sign-in activity on your Microsoft account

Dear Customer,

We detected an unusual sign-in attempt on your Microsoft account from an
unrecognized device in Lagos, Nigeria. To protect your account, please
verify your identity immediately.

If you do not confirm your identity within 24 hours, your account will be
temporarily locked for your security.

Verify your account now:
http://microsoft-account-verify.secure-login.tk/confirm?user=88213

Thank you for helping us keep your account secure.

Microsoft Account Team"""

# Wording-only showcase: phishing intent with NO link at all — a case the
# URL model literally cannot see, and only the email-text model can catch.
WORDING_ONLY_TEXT = """Congratulations! You have been selected to receive a free
$500 gift card. This is a limited-time offer available only to a select
number of recipients. To claim your reward, simply reply to this email with
your full name, date of birth, and mailing address within 48 hours. Act now
before this exclusive offer expires!"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_health(base: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.4)
    raise RuntimeError(f"server did not become healthy at {base}")


def start_server(port: int, db_path: Path) -> subprocess.Popen:
    env = {**os.environ}
    env.pop("GATE_PASSWORD", None)            # keep the app open locally
    env["PHISHING_DB_PATH"] = str(db_path)    # throwaway DB -> clean History
    env["ANALYZE_RATE_LIMIT_PER_MIN"] = "200"
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--app-dir", "src",
         "--port", str(port), "--log-level", "warning"],
        cwd=str(ROOT), env=env,
    )


def analyze(page, text: str) -> None:
    """Type input, click Analyze, wait for a rendered result card + VT panel."""
    page.fill("#input", text)
    page.click("#analyzeBtn")
    page.wait_for_selector("#results .card .badge", timeout=20000)
    page.wait_for_selector("#results .vt", timeout=20000)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1000)               # let the rise/grow animations settle


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    db_path = Path(tempfile.gettempdir()) / "phishing_screenshots.db"
    db_path.unlink(missing_ok=True)

    server = start_server(port, db_path)
    console_errors: list[str] = []
    try:
        wait_for_health(base)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(
                viewport={"width": VIEWPORT_WIDTH, "height": 900},
                device_scale_factor=DEVICE_SCALE_FACTOR,
                color_scheme=COLOR_SCHEME,
            )
            page = ctx.new_page()
            page.on("console",
                    lambda m: console_errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: console_errors.append(str(e)))

            page.goto(base, wait_until="networkidle")

            # 1 — phishing verdict (verdict + confidence bar + reasons + VT)
            analyze(page, PHISH_URL)
            page.screenshot(path=str(OUT / "01-phishing.png"), full_page=True)
            print("saved 01-phishing.png")

            # 2 — legitimate verdict (VT panel shows live data for a known site)
            page.fill("#input", "")
            analyze(page, LEGIT_URL)
            page.screenshot(path=str(OUT / "02-legitimate.png"), full_page=True)
            print("saved 02-legitimate.png")

            # 3 — email path: pasted email -> every extracted link is checked
            page.fill("#input", "")
            analyze(page, EMAIL_TEXT)
            page.wait_for_selector("#results .card:nth-child(3)", timeout=20000)
            page.screenshot(path=str(OUT / "03-email.png"), full_page=True)
            print("saved 03-email.png")

            # 4 — history tab (summary tiles + recent rows)
            page.click("#tab-history")
            page.wait_for_selector("#histList .hist-row", timeout=20000)
            page.wait_for_timeout(600)
            page.screenshot(path=str(OUT / "04-history.png"), full_page=True)
            print("saved 04-history.png")

            # 5 — brand-impersonation banner + email wording + URL, all three
            # signals firing together on one input.
            page.click("#tab-analyze")
            page.fill("#input", "")
            analyze(page, BRAND_MISMATCH_TEXT)
            page.wait_for_selector(".brand-alert", timeout=20000)
            page.screenshot(path=str(OUT / "05-brand-mismatch.png"), full_page=True)
            print("saved 05-brand-mismatch.png")

            # 6 — email wording alone: phishing intent with zero links, a
            # case the URL model has nothing to check at all.
            page.fill("#input", "")
            page.fill("#input", WORDING_ONLY_TEXT)
            page.click("#analyzeBtn")
            page.wait_for_selector("#results .card .badge", timeout=20000)
            page.wait_for_timeout(1000)
            page.screenshot(path=str(OUT / "06-wording-only.png"), full_page=True)
            print("saved 06-wording-only.png")

            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    if console_errors:
        print("\nWARNING — browser console errors during capture:", file=sys.stderr)
        for e in console_errors:
            print("  -", e, file=sys.stderr)
        return 1
    print(f"\nAll screenshots written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
