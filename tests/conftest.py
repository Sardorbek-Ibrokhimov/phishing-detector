"""Shared fixtures. Black-box tests hit the live server as a stranger would.

Start the server for a test run with a throwaway database, a raised rate
limit, and a known feedback key, so the suite neither pollutes production
data nor throttles itself:

    PHISHING_DB_PATH=/tmp/test_feedback.db \
    ANALYZE_RATE_LIMIT_PER_MIN=200 \
    FEEDBACK_API_KEY=test-key-for-suite \
    .venv/Scripts/python -m uvicorn api:app --app-dir src --port 8000

Against a default-configured server the C4/rate-limit tests will still pass,
but functional tests may see 429s once the 30/min budget is exhausted.
"""
import os
import socket
from urllib.parse import urlparse

import pytest
import requests

BASE = os.environ.get("TEST_BASE_URL", "http://127.0.0.1:8000")

# Must match the FEEDBACK_API_KEY the server under test was started with.
FEEDBACK_API_KEY = os.environ.get("FEEDBACK_API_KEY", "test-key-for-suite")
AUTH_HEADERS = {"X-API-Key": FEEDBACK_API_KEY}

# Platform gate (Basic Auth). If the server under test was started with a
# GATE_PASSWORD, tests must present it on every request. When unset the tuple
# is None, which requests treats as "no auth" — so the same suite passes
# against both a gated (deployment-config) and an ungated server.
GATE_AUTH = (
    (os.environ.get("GATE_USERNAME", "admin"), os.environ["GATE_PASSWORD"])
    if os.environ.get("GATE_PASSWORD") else None
)


def _host_port():
    parsed = urlparse(BASE)
    return parsed.hostname or "127.0.0.1", parsed.port or 8000


def _server_up() -> bool:
    try:
        s = socket.create_connection(_host_port(), timeout=1)
        s.close()
        return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def base():
    if not _server_up():
        pytest.skip(f"API server not running at {BASE}")
    return BASE


def analyze(content):
    return requests.post(f"{BASE}/analyze", json={"content": content},
                         auth=GATE_AUTH, timeout=60)
