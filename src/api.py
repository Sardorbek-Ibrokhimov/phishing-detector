"""FastAPI service for phishing detection.

Single endpoint for URL analysis plus health check. Frontend served from
the same process so no CORS needed.

The model is a pre-trained XGBoost loaded from models/deployed_model.joblib
at startup. Explanations use XGBoost's native TreeSHAP (pred_contribs).
VirusTotal lookup is optional and degrades gracefully when unavailable.

Run:  .venv/Scripts/python -m uvicorn api:app --app-dir src --port 8000
"""

import base64
import hmac
import logging
import os
import re
import sqlite3
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
import joblib
import tldextract
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from brand_check import find_brand_mismatches
from compare_models import clean_columns
from features import extract_url_features
from shap_explain import explain_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("phishing_api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_INDEX = PROJECT_ROOT / "frontend" / "index.html"
MODEL_PATH = PROJECT_ROOT / "models" / "deployed_model.joblib"
EMAIL_MODEL_PATH = PROJECT_ROOT / "models" / "deployed_email_model.joblib"

FEEDBACK_DB = Path(os.environ.get("PHISHING_DB_PATH", PROJECT_ROOT / "feedback.db"))

MAX_BODY_BYTES = 100_000
MAX_URLS_PER_REQUEST = 20
ANALYZE_RATE_LIMIT_PER_MIN = int(os.environ.get("ANALYZE_RATE_LIMIT_PER_MIN", "30"))

# HTTP Basic Auth gate for the whole app
GATE_USERNAME = os.environ.get("GATE_USERNAME", "admin")
GATE_PASSWORD = os.environ.get("GATE_PASSWORD")

# API key for the correction endpoint. Fail-closed: no key = corrections disabled.
FEEDBACK_API_KEY = os.environ.get("FEEDBACK_API_KEY")


def require_feedback_key(x_api_key: str | None = Header(default=None)) -> None:
    """Check API key for /feedback endpoint."""
    if not FEEDBACK_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Corrections are disabled: no FEEDBACK_API_KEY is configured "
                   "on the server. Set it to enable the correction endpoint.",
        )
    if not x_api_key or not hmac.compare_digest(x_api_key, FEEDBACK_API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")

# --- URL extraction ---------------------------------------------------------

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"'\]\)]+", re.IGNORECASE)
_TRAILING = ".,;:!?)]}<>\"'"
_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _is_plausible_bare_url(candidate: str) -> bool:
    """Check if a bare token (no scheme) looks like a real URL."""
    if "@" in candidate:
        return False
    host = candidate.split("/")[0].split(":")[0]
    if _IP_RE.match(host):
        return True
    ext = tldextract.extract(candidate)
    return bool(ext.domain and ext.suffix)


def extract_urls(text: str) -> list[str]:
    text = text.strip()
    found = [u.rstrip(_TRAILING) for u in _URL_RE.findall(text)]
    if found:
        seen, out = set(), []
        for u in found:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out
    if text and " " not in text and "\n" not in text:
        candidate = text.rstrip(_TRAILING)
        if _is_plausible_bare_url(candidate):
            return [candidate]
    return []


# --- VirusTotal client ------------------------------------------------------

class VirusTotalClient:
    """Best-effort VT v3 URL-reputation lookup with free-tier rate limiting."""

    API = "https://www.virustotal.com/api/v3/urls/{vt_id}"
    GUI = "https://www.virustotal.com/gui/url/{vt_id}"
    FREE_TIER_PER_MIN = 4

    def __init__(self) -> None:
        self.api_key = os.environ.get("VIRUSTOTAL_API_KEY")
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    @staticmethod
    def _vt_id(url: str) -> str:
        return base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")

    def _reserve_slot(self) -> bool:
        """Sliding-window rate limiter for the free tier."""
        now = time.monotonic()
        with self._lock:
            while self._calls and now - self._calls[0] >= 60:
                self._calls.popleft()
            if len(self._calls) >= self.FREE_TIER_PER_MIN:
                return False
            self._calls.append(now)
            return True

    def lookup(self, url: str) -> dict:
        vt_id = self._vt_id(url)
        gui_link = self.GUI.format(vt_id=vt_id)

        if not self.api_key:
            return {
                "available": False,
                "note": "reputation data unavailable: VIRUSTOTAL_API_KEY not configured",
                "link": gui_link,
            }
        if not self._reserve_slot():
            return {
                "available": False,
                "note": "reputation data unavailable: VirusTotal free-tier rate "
                        "limit (4/min) reached, skipped to avoid blocking",
                "link": gui_link,
            }

        try:
            resp = httpx.get(
                self.API.format(vt_id=vt_id),
                headers={"x-apikey": self.api_key},
                timeout=10.0,
            )
        except httpx.HTTPError as e:
            return {
                "available": False,
                "note": f"reputation data unavailable: VirusTotal request failed ({type(e).__name__})",
                "link": gui_link,
            }

        if resp.status_code == 404:
            return {
                "available": False,
                "note": "reputation data unavailable: URL not previously seen by VirusTotal",
                "link": gui_link,
            }
        if resp.status_code == 429:
            return {
                "available": False,
                "note": "reputation data unavailable: VirusTotal rate-limited the request (HTTP 429)",
                "link": gui_link,
            }
        if resp.status_code != 200:
            return {
                "available": False,
                "note": f"reputation data unavailable: VirusTotal returned HTTP {resp.status_code}",
                "link": gui_link,
            }

        try:
            stats = resp.json()["data"]["attributes"]["last_analysis_stats"]
        except (KeyError, TypeError, ValueError):
            return {
                "available": False,
                "note": "reputation data unavailable: unexpected VirusTotal response shape",
                "link": gui_link,
            }

        return {
            "available": True,
            "malicious": int(stats.get("malicious", 0)),
            "suspicious": int(stats.get("suspicious", 0)),
            "harmless": int(stats.get("harmless", 0)),
            "undetected": int(stats.get("undetected", 0)),
            "link": gui_link,
        }


# --- Persistence (SQLite) ---------------------------------------------------

_ANALYSES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS analyses (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        url        TEXT NOT NULL,
        verdict    TEXT NOT NULL,
        confidence REAL
    )
"""

_FEEDBACK_SCHEMA = """
    CREATE TABLE IF NOT EXISTS feedback (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at      TEXT NOT NULL,
        analysis_id     INTEGER UNIQUE,
        url             TEXT,
        model_verdict   TEXT NOT NULL,
        confidence      REAL,
        corrected_label TEXT NOT NULL
    )
"""

_CORRECTIONS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS corrections (
        url                TEXT PRIMARY KEY,
        corrected_label    TEXT NOT NULL,
        first_corrected_at TEXT NOT NULL,
        updated_at         TEXT NOT NULL,
        times_corrected    INTEGER NOT NULL DEFAULT 1
    )
"""


def init_db() -> None:
    conn = sqlite3.connect(FEEDBACK_DB)
    try:
        # Handle schema migrations for older databases
        acols = [r[1] for r in conn.execute("PRAGMA table_info(analyses)").fetchall()]
        if acols and "input_preview" in acols:
            conn.execute("ALTER TABLE analyses RENAME TO analyses_old")
            conn.execute(_ANALYSES_SCHEMA)
            conn.execute(
                "INSERT INTO analyses (id, created_at, url, verdict, confidence) "
                "SELECT id, created_at, url, verdict, confidence FROM analyses_old"
            )
            conn.execute("DROP TABLE analyses_old")
        else:
            conn.execute(_ANALYSES_SCHEMA)

        fcols = [r[1] for r in conn.execute("PRAGMA table_info(feedback)").fetchall()]
        if fcols and "input" in fcols and "analysis_id" not in fcols:
            conn.execute("ALTER TABLE feedback RENAME TO feedback_old")
            conn.execute(_FEEDBACK_SCHEMA)
            conn.execute(
                "INSERT INTO feedback (id, created_at, analysis_id, url, "
                "model_verdict, confidence, corrected_label) "
                "SELECT id, created_at, NULL, url, model_verdict, confidence, "
                "corrected_label FROM feedback_old"
            )
            conn.execute("DROP TABLE feedback_old")
        else:
            conn.execute(_FEEDBACK_SCHEMA)

        # Add UNIQUE constraint if missing
        existing_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='feedback'"
        ).fetchone()
        if existing_sql and "UNIQUE" not in existing_sql[0].upper():
            logger.info("Migrating feedback table to add UNIQUE constraint...")
            conn.execute("ALTER TABLE feedback RENAME TO feedback_predupe")
            conn.execute(_FEEDBACK_SCHEMA)
            conn.execute(
                "INSERT INTO feedback (created_at, analysis_id, url, model_verdict, confidence, "
                "corrected_label) "
                "SELECT created_at, analysis_id, url, model_verdict, confidence, "
                "corrected_label FROM feedback_predupe "
                "WHERE analysis_id IS NULL "
                "   OR id IN (SELECT MAX(id) FROM feedback_predupe "
                "             WHERE analysis_id IS NOT NULL GROUP BY analysis_id)"
            )
            conn.execute("DROP TABLE feedback_predupe")

        conn.execute(_CORRECTIONS_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def insert_analysis(url: str, verdict: str, confidence) -> int | None:
    """Log an analysis to the database. Returns None on failure."""
    try:
        conn = sqlite3.connect(FEEDBACK_DB)
        try:
            cur = conn.execute(
                "INSERT INTO analyses (created_at, url, verdict, confidence) "
                "VALUES (?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 url, verdict, confidence),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as e:
        logger.warning("insert_analysis failed: %s", e)
        return None


def insert_feedback(row: "FeedbackRequest") -> int | None:
    """Log feedback, upsert on analysis_id to avoid duplicates."""
    try:
        conn = sqlite3.connect(FEEDBACK_DB)
        try:
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            params = (now, row.analysis_id, row.url, row.model_verdict,
                      row.confidence, row.corrected_label)
            cur = conn.execute(
                "INSERT INTO feedback (created_at, analysis_id, url, model_verdict, "
                "confidence, corrected_label) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(analysis_id) DO UPDATE SET "
                "  created_at = excluded.created_at, "
                "  url = excluded.url, "
                "  model_verdict = excluded.model_verdict, "
                "  confidence = excluded.confidence, "
                "  corrected_label = excluded.corrected_label",
                params,
            )
            conn.commit()
            if row.analysis_id is not None:
                found = conn.execute(
                    "SELECT id FROM feedback WHERE analysis_id = ?", (row.analysis_id,)
                ).fetchone()
                return found[0] if found else cur.lastrowid
            return cur.lastrowid
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as e:
        logger.warning("insert_feedback failed: %s", e)
        return None


def upsert_correction(url: str, corrected_label: str) -> bool:
    """Store a human correction for this URL so future analyses are overridden."""
    if not url:
        return False
    try:
        conn = sqlite3.connect(FEEDBACK_DB)
        try:
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            conn.execute(
                "INSERT INTO corrections (url, corrected_label, first_corrected_at, "
                "                          updated_at, times_corrected) "
                "VALUES (?, ?, ?, ?, 1) "
                "ON CONFLICT(url) DO UPDATE SET "
                "  corrected_label = excluded.corrected_label, "
                "  updated_at = excluded.updated_at, "
                "  times_corrected = times_corrected + 1",
                (url, corrected_label, now, now),
            )
            conn.commit()
            return True
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as e:
        logger.warning("upsert_correction failed for %r: %s", url, e)
        return False


def lookup_correction(url: str) -> dict | None:
    """Look up any stored human correction for this URL."""
    try:
        conn = sqlite3.connect(FEEDBACK_DB)
        try:
            row = conn.execute(
                "SELECT corrected_label, updated_at, times_corrected "
                "FROM corrections WHERE url = ?", (url,)
            ).fetchone()
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as e:
        logger.warning("lookup_correction failed for %r: %s", url, e)
        return None
    if not row:
        return None
    return {"corrected_label": row[0], "corrected_at": row[1], "times_corrected": row[2]}


def recent_history(limit: int = 20) -> dict:
    conn = sqlite3.connect(FEEDBACK_DB)
    try:
        rows = conn.execute(
            """
            SELECT a.id, a.created_at, a.url, a.verdict, a.confidence,
                   MAX(f.corrected_label) AS corrected_label
            FROM analyses a
            LEFT JOIN feedback f ON f.analysis_id = a.id
            GROUP BY a.id
            ORDER BY a.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        entries = [
            {"id": r[0], "created_at": r[1], "url": r[2], "verdict": r[3],
             "confidence": r[4], "corrected_label": r[5]}
            for r in rows
        ]
        total = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        phishing = conn.execute("SELECT COUNT(*) FROM analyses WHERE verdict='phishing'").fetchone()[0]
        legit = conn.execute("SELECT COUNT(*) FROM analyses WHERE verdict='legitimate'").fetchone()[0]
        flagged = conn.execute(
            "SELECT COUNT(DISTINCT analysis_id) FROM feedback WHERE analysis_id IS NOT NULL"
        ).fetchone()[0]
        return {
            "summary": {"total": total, "phishing": phishing,
                        "legitimate": legit, "flagged_incorrect": flagged},
            "entries": entries,
        }
    finally:
        conn.close()


# --- Rate limiting -----------------------------------------------------------

class RateLimiter:
    """Sliding-window rate limiter."""

    def __init__(self, limit_per_min: int) -> None:
        self.limit_per_min = limit_per_min
        self._calls: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            calls = self._calls.setdefault(key, deque())
            while calls and now - calls[0] >= 60:
                calls.popleft()
            if len(calls) >= self.limit_per_min:
                return False
            calls.append(now)
            return True


analyze_limiter = RateLimiter(ANALYZE_RATE_LIMIT_PER_MIN)


# --- App state / lifecycle --------------------------------------------------

class _State:
    model = None
    cols = None
    email_model = None
    vt = None


state = _State()


def _load_persisted_model() -> None:
    """Load the pre-trained model from disk."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No persisted model found at {MODEL_PATH}. Run "
            f"`.venv/Scripts/python src/persist_model.py` to train and save one "
            f"before starting the API."
        )
    bundle = joblib.load(MODEL_PATH)
    model, cols = bundle["model"], bundle["cols"]

    # Check that features still match
    current_cols = clean_columns(list(extract_url_features("http://example.com/").keys()))
    if list(cols) != list(current_cols):
        raise RuntimeError(
            f"Persisted model's feature list does not match the current "
            f"feature extractor — the model is stale.\n"
            f"  persisted ({len(cols)}): {cols}\n"
            f"  current   ({len(current_cols)}): {current_cols}\n"
            f"Re-run `.venv/Scripts/python src/persist_model.py` to retrain."
        )

    state.model, state.cols = model, cols


def _load_persisted_email_model() -> None:
    """Load the pre-trained email-text (TF-IDF + LR) model from disk.

    Soft-optional, unlike the URL model: this is a secondary capability
    layered on top of the original URL analysis, so a missing/broken
    artifact degrades to "email-text analysis unavailable" rather than
    refusing to start the whole app.
    """
    if not EMAIL_MODEL_PATH.exists():
        logger.warning(
            "No persisted email-text model at %s — email wording analysis "
            "disabled (URL analysis is unaffected). Run "
            "`.venv/Scripts/python src/persist_email_model.py` to enable it.",
            EMAIL_MODEL_PATH,
        )
        return
    try:
        bundle = joblib.load(EMAIL_MODEL_PATH)
        state.email_model = bundle["model"]
    except Exception:
        logger.exception("Failed to load email-text model — email wording analysis disabled.")
        state.email_model = None


EMAIL_REASON_TOP_K = 5


def explain_email_text(text: str, model, k: int = EMAIL_REASON_TOP_K) -> dict:
    """Verdict + confidence + top contributing tokens/phrases for pasted
    email text, using the TF-IDF + Logistic Regression pipeline. Mirrors
    explain_prediction()'s shape (verdict/confidence/reasons) so the
    frontend can render both consistently.

    Per-instance explanation: (this text's TF-IDF weight) x (that token's
    LR coefficient) for every token present in the text, ranked by how
    strongly each pushes toward the predicted class — the same idea as
    SHAP contribution ranking, computed directly from a linear model
    instead of needing a separate library.
    """
    proba_phish = float(model.predict_proba([text])[0, 1])
    verdict = "phishing" if proba_phish >= 0.5 else "legitimate"
    confidence = proba_phish if verdict == "phishing" else 1 - proba_phish
    direction = 1 if verdict == "phishing" else -1
    suffix = "raises phishing likelihood" if direction == 1 else "lowers phishing likelihood"

    vec = model.named_steps["tfidfvectorizer"]
    clf = model.named_steps["logisticregression"]
    x = vec.transform([text]).toarray()[0]
    contrib = x * clf.coef_[0]
    names = vec.get_feature_names_out()

    # URL/protocol fragments end up in the vocabulary (raw links appear inline
    # in training emails) but read as noise in a "wording" explanation, e.g.
    # "the phrase 'http' in the text" — filter them out of the shown reasons.
    NOT_WORDING = {"http", "https", "www", "com", "org", "net"}

    present = [i for i in range(len(x)) if x[i] != 0 and names[i] not in NOT_WORDING]
    present.sort(key=lambda i: direction * contrib[i], reverse=True)

    reasons = []
    for i in present:
        if direction * contrib[i] <= 0:
            break  # ranked descending — once contributions stop supporting the verdict, stop
        reasons.append(f"the phrase ‘{names[i]}’ in the text ({suffix})")
        if len(reasons) >= k:
            break
    if not reasons:
        reasons.append(f"the overall combination of wording ({suffix})")

    return {"verdict": verdict, "confidence": round(confidence, 3), "reasons": reasons}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from %s ...", MODEL_PATH)
    _load_persisted_model()
    _load_persisted_email_model()
    state.vt = VirusTotalClient()
    init_db()
    logger.info("Model ready (%d features). Email-text model: %s. VT key: %s. DB: %s",
                len(state.cols), "loaded" if state.email_model else "unavailable",
                "configured" if state.vt.api_key else "not configured",
                FEEDBACK_DB)
    if FEEDBACK_API_KEY:
        logger.info("Corrections enabled.")
    else:
        logger.warning("FEEDBACK_API_KEY not set — corrections disabled.")
    if GATE_PASSWORD:
        logger.info("Password gate enabled (user '%s').", GATE_USERNAME)
    else:
        logger.warning("GATE_PASSWORD not set — app is open.")
    yield


app = FastAPI(title="Phishing Detection API", version="0.1", lifespan=lifespan)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Reject oversized request bodies."""
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body exceeds the {MAX_BODY_BYTES}-byte limit."},
            )
    return await call_next(request)


@app.middleware("http")
async def basic_auth_gate(request: Request, call_next):
    """Password gate for the whole app using HTTP Basic Auth."""
    if GATE_PASSWORD:
        auth = request.headers.get("authorization", "")
        ok = False
        if auth.startswith("Basic "):
            try:
                user, _, pw = base64.b64decode(auth[6:]).decode("utf-8").partition(":")
                ok = (hmac.compare_digest(user, GATE_USERNAME)
                      and hmac.compare_digest(pw, GATE_PASSWORD))
            except (ValueError, UnicodeDecodeError):
                ok = False
        if not ok:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required."},
                headers={"WWW-Authenticate": 'Basic realm="Phishing Detector"'},
            )
    return await call_next(request)


# --- Schemas ----------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    content: str = Field(..., description="A URL or pasted email text to analyze.")


class FeedbackRequest(BaseModel):
    analysis_id: int | None = Field(None, description="Id of the analysis being corrected.")
    url: str | None = Field(None, description="The specific URL the verdict was about.")
    model_verdict: str = Field(..., description="What the model predicted.")
    confidence: float | None = Field(None, description="The model's confidence.")
    corrected_label: str = Field(..., description="The label the user says is correct.")


# --- Endpoints --------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND_INDEX)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "email_model_loaded": state.email_model is not None,
        "analyze_rate_limit_per_min": ANALYZE_RATE_LIMIT_PER_MIN,
        "corrections_enabled": bool(FEEDBACK_API_KEY),
    }


@app.get("/history")
def history(limit: int = 20):
    return recent_history(limit)


@app.post("/feedback")
def feedback(req: FeedbackRequest, _auth: None = Depends(require_feedback_key)):
    row_id = insert_feedback(req)
    applied = upsert_correction(req.url, req.corrected_label) if req.url else False
    return {
        "status": "logged" if row_id is not None else "not_logged",
        "id": row_id,
        "correction_applied": applied,
    }


def _classify_input_type(content: str, urls: list[str]) -> str:
    """Determine if input is a URL or email text."""
    stripped = content.strip()
    if len(urls) == 1 and urls[0] == stripped.rstrip(_TRAILING):
        return "url"
    if urls:
        return "email_text"
    return "url" if stripped and not re.search(r"\s", stripped) else "email_text"


def _stable_response(input_type, urls_analyzed, overall_verdict, overall_confidence,
                      note=None, truncated=False, corrections_applied=0,
                      email_text_verdict=None, brand_mismatches=None) -> dict:
    """Consistent response shape for every /analyze branch.

    email_text_verdict and brand_mismatches are new, additive, always-optional
    fields (None/[] unless applicable) — existing consumers of
    urls_analyzed/overall_verdict are unaffected."""
    return {
        "input_type": input_type,
        "urls_analyzed": urls_analyzed,
        "overall_verdict": overall_verdict,
        "overall_confidence": overall_confidence,
        "note": note,
        "truncated": truncated,
        "corrections_applied": corrections_applied,
        "email_text_verdict": email_text_verdict,
        "brand_mismatches": brand_mismatches or [],
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest, request: Request):
    client_key = request.client.host if request.client else "unknown"
    if not analyze_limiter.allow(client_key):
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded ({ANALYZE_RATE_LIMIT_PER_MIN} "
                                f"requests/minute). Try again shortly."},
        )

    content = req.content.strip()
    urls = extract_urls(content)
    input_type = _classify_input_type(content, urls)

    # Email-text wording analysis: only for genuine pasted text (not a lone
    # URL, where there's no "wording" to speak of), only when the model
    # loaded successfully at startup.
    email_text_verdict = None
    if input_type == "email_text" and state.email_model is not None:
        email_text_verdict = explain_email_text(content, state.email_model)

    if not urls:
        if email_text_verdict is not None:
            return _stable_response(
                input_type, [], email_text_verdict["verdict"], email_text_verdict["confidence"],
                note="No links found in the text — verdict is based on the wording only.",
                email_text_verdict=email_text_verdict,
            )
        return _stable_response(
            input_type, [], "unknown", None,
            note=("No URLs found, and email-wording analysis is unavailable right now."
                  if input_type == "email_text" else
                  "No URLs found in the input. Paste a URL or an email containing links."),
        )

    truncated = len(urls) > MAX_URLS_PER_REQUEST
    urls = urls[:MAX_URLS_PER_REQUEST]

    # Cross-reference: does the text claim a brand that none of the links
    # actually go to? Deterministic, not model-based — see brand_check.py.
    brand_mismatches = find_brand_mismatches(content, urls)

    results = []
    corrections_applied = 0
    for url in urls:
        feats = extract_url_features(url)
        feats_clean = {k: feats[k] for k in state.cols}
        import pandas as pd
        row = pd.DataFrame([feats_clean])
        prob = float(state.model.predict_proba(row)[0][1])
        model_verdict = "phishing" if prob >= 0.5 else "legitimate"
        model_confidence = round(prob if model_verdict == "phishing" else 1 - prob, 4)

        # Check for human corrections
        correction_info = lookup_correction(url)
        if correction_info:
            verdict = correction_info["corrected_label"]
            confidence = None
            verdict_source = "human_correction"
            corrections_applied += 1
        else:
            verdict = model_verdict
            confidence = model_confidence
            verdict_source = "model"
            correction_info = None

        reasons = explain_prediction(url, state.model, None, state.cols)["reasons"]

        analysis_id = insert_analysis(url, verdict, confidence)

        vt = state.vt.lookup(url)

        results.append({
            "url": url,
            "verdict": verdict,
            "confidence": confidence,
            "verdict_source": verdict_source,
            "model_verdict": model_verdict,
            "model_confidence": model_confidence,
            "correction": {
                "corrected_label": correction_info["corrected_label"],
                "corrected_at": correction_info["corrected_at"],
                "times_corrected": correction_info["times_corrected"],
                "overrode_model": correction_info["corrected_label"] != model_verdict,
            } if correction_info else None,
            "reasons": reasons,
            "virustotal": vt,
            "analysis_id": analysis_id,
        })

    # Roll the email-wording verdict into the worst-case alongside the
    # per-URL results, so phishing wording with a clean/no link still
    # surfaces as the overall verdict rather than being silently dropped.
    candidates = list(results)
    if email_text_verdict is not None:
        candidates.append({
            "verdict": email_text_verdict["verdict"],
            "confidence": email_text_verdict["confidence"],
            "model_confidence": email_text_verdict["confidence"],
        })
    if brand_mismatches:
        # Deterministic, not a model probability — but it's a near-certain
        # phishing tell, so it outranks both models' confidence in the
        # worst-case rollup rather than being averaged away.
        candidates.append({"verdict": "phishing", "confidence": 0.99, "model_confidence": 0.99})
    worst = max(candidates, key=lambda r: (r["verdict"] == "phishing", r.get("model_confidence", 0)))
    return _stable_response(
        input_type, results, worst["verdict"], worst["confidence"],
        truncated=truncated, corrections_applied=corrections_applied,
        email_text_verdict=email_text_verdict, brand_mismatches=brand_mismatches,
    )
