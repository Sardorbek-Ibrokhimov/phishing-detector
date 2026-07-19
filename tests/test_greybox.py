"""Grey-box tests: using knowledge of internals to attack the seams.

Imports the app module directly (no server). Importing `api` does NOT train
or load a model (that only happens inside the lifespan, and since G7,
loading is a fast joblib.load rather than a retrain), so these are fast
unit tests.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import api  # noqa: E402


# =================================================================== URL extractor
class TestUrlExtractor:
    def test_http_in_html_href(self):
        assert api.extract_urls('<a href="http://evil.com/login">click</a>') == ["http://evil.com/login"]

    def test_url_in_parens(self):
        assert api.extract_urls("(see http://x.com/a)") == ["http://x.com/a"]

    def test_trailing_punctuation_stripped(self):
        assert api.extract_urls("Go to http://x.com/path. Now.") == ["http://x.com/path"]

    def test_two_urls(self):
        got = api.extract_urls("a http://one.com and http://two.com b")
        assert got == ["http://one.com", "http://two.com"]

    # --- G6 fix: bare tokens require a scheme or a real public suffix ---
    def test_bare_filename_not_url(self):
        assert api.extract_urls("report.pdf") == [], "filename misread as URL"

    def test_library_name_not_url(self):
        assert api.extract_urls("Node.js") == [], "'.js' is not a real TLD"

    def test_bare_email_not_url(self):
        assert api.extract_urls("someone@example.com") == [], "email misread as URL"

    def test_fake_tld_not_url(self):
        assert api.extract_urls("x.y") == [], "'y' is not a real public suffix"

    # --- and genuinely plausible bare tokens must still work ---
    def test_bare_ip_still_url(self):
        assert api.extract_urls("192.168.1.1") == ["192.168.1.1"]

    def test_bare_host_with_real_tld_still_url(self):
        assert api.extract_urls("paypal-login.tk/signin") == ["paypal-login.tk/signin"]

    def test_bare_host_with_multipart_suffix_still_url(self):
        assert api.extract_urls("example.co.uk") == ["example.co.uk"]


# =================================================================== VirusTotal seams
class _FakeResp:
    def __init__(self, status, payload=None, raise_json=False):
        self.status_code = status
        self._payload = payload
        self._raise = raise_json

    def json(self):
        if self._raise:
            raise ValueError("malformed json")
        return self._payload


@pytest.fixture
def vt():
    c = api.VirusTotalClient()
    c.api_key = "TEST_KEY"          # force the "configured" path
    c._calls.clear()
    return c


class TestVirusTotalSeams:
    def test_malformed_json(self, vt, monkeypatch):
        monkeypatch.setattr(api.httpx, "get", lambda *a, **k: _FakeResp(200, raise_json=True))
        out = vt.lookup("http://x.com")
        assert out["available"] is False and "note" in out

    def test_missing_stats_key(self, vt, monkeypatch):
        monkeypatch.setattr(api.httpx, "get", lambda *a, **k: _FakeResp(200, {"data": {}}))
        assert vt.lookup("http://x.com")["available"] is False

    def test_http_500(self, vt, monkeypatch):
        monkeypatch.setattr(api.httpx, "get", lambda *a, **k: _FakeResp(500))
        assert vt.lookup("http://x.com")["available"] is False

    def test_http_404_not_seen(self, vt, monkeypatch):
        monkeypatch.setattr(api.httpx, "get", lambda *a, **k: _FakeResp(404))
        assert vt.lookup("http://x.com")["available"] is False

    def test_timeout_hang(self, vt, monkeypatch):
        def boom(*a, **k):
            raise api.httpx.TimeoutException("timed out")
        monkeypatch.setattr(api.httpx, "get", boom)
        assert vt.lookup("http://x.com")["available"] is False

    def test_rate_limiter_blocks_after_4(self, vt, monkeypatch):
        monkeypatch.setattr(api.httpx, "get",
                            lambda *a, **k: _FakeResp(200, {"data": {"attributes": {"last_analysis_stats": {}}}}))
        outs = [vt.lookup(f"http://x{i}.com") for i in range(6)]
        available = [o["available"] for o in outs]
        assert available[:4] == [True] * 4
        assert available[4] is False and "rate limit" in outs[4]["note"].lower()


# =================================================================== DB seams
class TestDbSeams:
    def test_insert_analysis_on_unwritable_path_is_best_effort(self, monkeypatch, caplog):
        """G3 fix: a DB failure (missing dir / read-only / locked) must not
        raise out of insert_analysis — it logs a warning and returns None,
        so /analyze can still return the verdict."""
        monkeypatch.setattr(api, "FEEDBACK_DB", ROOT / "no_such_dir_zzz" / "x.db")
        with caplog.at_level("WARNING"):
            result = api.insert_analysis("http://x.com", "phishing", 0.9)
        assert result is None
        assert any("insert_analysis failed" in r.message for r in caplog.records)

    def test_insert_feedback_on_unwritable_path_is_best_effort(self, monkeypatch, caplog):
        """Same guarantee for insert_feedback (G3)."""
        monkeypatch.setattr(api, "FEEDBACK_DB", ROOT / "no_such_dir_zzz" / "x.db")
        req = api.FeedbackRequest(analysis_id=1, url="http://x.com",
                                  model_verdict="phishing", confidence=0.9,
                                  corrected_label="legitimate")
        with caplog.at_level("WARNING"):
            result = api.insert_feedback(req)
        assert result is None
        assert any("insert_feedback failed" in r.message for r in caplog.records)


# =================================================================== G7: model persistence
class TestModelPersistence:
    def test_persisted_model_file_exists(self):
        assert api.MODEL_PATH.exists(), (
            f"no persisted model at {api.MODEL_PATH} — run "
            f"src/persist_model.py before running the API/tests"
        )

    def test_missing_model_fails_clearly(self, monkeypatch):
        monkeypatch.setattr(api, "MODEL_PATH", ROOT / "models" / "does_not_exist.joblib")
        with pytest.raises(RuntimeError, match="No persisted model found"):
            api._load_persisted_model()

    def test_feature_mismatch_fails_clearly(self, monkeypatch):
        """If the persisted model's feature list no longer matches what
        features.py currently produces, startup must refuse to serve it
        rather than silently running an incompatible model."""
        import joblib as _joblib

        real_load = _joblib.load

        def fake_load(path):
            bundle = dict(real_load(path))
            bundle["cols"] = bundle["cols"] + ["a_feature_that_no_longer_exists"]
            return bundle

        monkeypatch.setattr(api.joblib, "load", fake_load)
        with pytest.raises(RuntimeError, match="feature list does not match"):
            api._load_persisted_model()

    def test_loaded_model_can_predict(self):
        """The persisted bundle actually works end-to-end, not just loads."""
        bundle = api.joblib.load(api.MODEL_PATH)
        from shap_explain import explain_prediction
        # explainer is None: SHAP values come from XGBoost's native TreeSHAP.
        result = explain_prediction(
            "http://allegro.id-38247ns4.click",
            bundle["model"], None, bundle["cols"],
        )
        assert result["verdict"] in {"phishing", "legitimate"}

    def test_bundle_has_no_shap_object(self):
        """The deployed bundle must NOT contain a shap explainer — that would
        force the runtime to import shap (+ matplotlib) to unpickle it, the
        memory cost the pred_contribs switch exists to avoid."""
        bundle = api.joblib.load(api.MODEL_PATH)
        assert "explainer" not in bundle
        assert set(bundle) == {"model", "cols"}

    def test_duplicate_feedback_updates_not_duplicates(self, tmp_path, monkeypatch):
        """G4 fix: UNIQUE(analysis_id) means re-submitting feedback for the
        same analysis updates that row instead of appending a duplicate."""
        import sqlite3
        db = tmp_path / "t.db"
        monkeypatch.setattr(api, "FEEDBACK_DB", db)
        api.init_db()
        for label in ("legitimate", "phishing", "legitimate"):
            api.insert_feedback(api.FeedbackRequest(
                analysis_id=1, url="http://x.com", model_verdict="phishing",
                confidence=0.9, corrected_label=label))
        conn = sqlite3.connect(db)
        n = conn.execute("SELECT COUNT(*) FROM feedback WHERE analysis_id=1").fetchone()[0]
        label = conn.execute("SELECT corrected_label FROM feedback WHERE analysis_id=1").fetchone()[0]
        assert n == 1, "duplicate feedback should update, not append"
        assert label == "legitimate", "the row should hold the most recent correction"

    def test_null_analysis_id_feedback_still_appends(self, tmp_path, monkeypatch):
        """SQLite allows multiple NULLs in a UNIQUE column, so feedback with
        no analysis_id has no identity to dedupe on and still appends. This
        documents the accepted limit of the G4 constraint."""
        import sqlite3
        db = tmp_path / "t.db"
        monkeypatch.setattr(api, "FEEDBACK_DB", db)
        api.init_db()
        for _ in range(3):
            api.insert_feedback(api.FeedbackRequest(
                analysis_id=None, url="http://x.com", model_verdict="phishing",
                confidence=0.9, corrected_label="legitimate"))
        n = sqlite3.connect(db).execute(
            "SELECT COUNT(*) FROM feedback WHERE analysis_id IS NULL").fetchone()[0]
        assert n == 3


# =================================================================== rate limiter
class TestRateLimiter:
    """Unit-level so the limiter logic is covered deterministically without
    burning the HTTP suite's shared per-IP budget."""

    def test_allows_up_to_limit_then_blocks(self):
        rl = api.RateLimiter(3)
        assert [rl.allow("ip") for _ in range(3)] == [True, True, True]
        assert rl.allow("ip") is False

    def test_keys_are_independent(self):
        rl = api.RateLimiter(2)
        assert rl.allow("a") and rl.allow("a")
        assert rl.allow("a") is False
        assert rl.allow("b") is True, "one client must not consume another's budget"

    def test_window_expiry_frees_budget(self, monkeypatch):
        rl = api.RateLimiter(2)
        clock = [1000.0]
        monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
        assert rl.allow("ip") and rl.allow("ip")
        assert rl.allow("ip") is False
        clock[0] += 61  # advance past the 60s sliding window
        assert rl.allow("ip") is True


# =================================================================== C4: correction store
class TestCorrectionStore:
    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        path = tmp_path / "corrections.db"
        monkeypatch.setattr(api, "FEEDBACK_DB", path)
        api.init_db()
        return path

    def test_lookup_empty_when_no_correction(self, db):
        assert api.lookup_correction("http://x.com/a") is None

    def test_upsert_then_lookup(self, db):
        assert api.upsert_correction("http://x.com/a", "legitimate") is True
        got = api.lookup_correction("http://x.com/a")
        assert got["corrected_label"] == "legitimate"
        assert got["times_corrected"] == 1

    def test_repeat_correction_updates_and_counts(self, db):
        api.upsert_correction("http://x.com/a", "legitimate")
        api.upsert_correction("http://x.com/a", "phishing")
        got = api.lookup_correction("http://x.com/a")
        assert got["corrected_label"] == "phishing", "latest correction should win"
        assert got["times_corrected"] == 2

    def test_correction_is_exact_url_only_not_domain(self, db):
        """The headline safety property: correcting one URL must not
        whitelist the whole host."""
        api.upsert_correction("http://x.com/a", "legitimate")
        assert api.lookup_correction("http://x.com/a") is not None
        for other in ["http://x.com/b", "http://x.com/", "http://x.com",
                      "https://x.com/a", "http://sub.x.com/a"]:
            assert api.lookup_correction(other) is None, f"{other} must not inherit the correction"

    def test_empty_url_not_stored(self, db):
        assert api.upsert_correction("", "legitimate") is False

    def test_lookup_is_best_effort_on_db_failure(self, monkeypatch, caplog):
        """G3 discipline applies here too: an unreadable corrections store
        must fall through to the model verdict, not break analysis."""
        monkeypatch.setattr(api, "FEEDBACK_DB", ROOT / "no_such_dir_zzz" / "x.db")
        with caplog.at_level("WARNING"):
            assert api.lookup_correction("http://x.com/a") is None
        assert any("lookup_correction failed" in r.message for r in caplog.records)
