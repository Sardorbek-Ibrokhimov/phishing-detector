# Phishing Detector

A web app that checks whether a URL or an email is a phishing attempt and explains why — three independent, explainable signals (a URL model, an email-wording model, and a brand-impersonation cross-check) rather than one black box. Built with XGBoost + FastAPI for my final year dissertation on explainable ML in cybersecurity.

![Analysing a phishing URL: the verdict, a confidence bar, plain-English reasons, and the VirusTotal reputation panel](docs/screenshots/01-phishing.png)

> **Live demo:** https://phishing-detector-nmyw.onrender.com/
> Password-protected, hosted on Render free tier so first load takes ~30-60s.

## What it does

Paste a URL or email, and it returns:
- A **verdict** (phishing or legitimate)
- A **confidence score**
- **Plain-English reasons** (e.g. "the domain looks randomly generated")
- **VirusTotal reputation** check for URLs

For pasted email text, two extra signals run alongside the URL checks:
- An **email-wording verdict** (TF-IDF + Logistic Regression) — catches phishing intent even when there's no link at all
- A **brand-impersonation check** — flags when the text claims a known brand (PayPal, Microsoft, a bank, ...) but the link goes to a different domain entirely

Each signal is independent and shown separately, with its own reasons — the overall verdict is the worst case across all of them, so a suspicious wording or a brand mismatch surfaces even if the URL alone looks clean.

You can also report wrong verdicts, which override the model for that URL.

## How it works

```
Browser ──HTTP──> FastAPI app (src/api.py)
                   ├── password gate (optional)
                   ├── extract URLs from input
                   ├── per URL:
                   │     ├── XGBoost model → verdict + confidence
                   │     ├── TreeSHAP → plain-English reasons
                   │     └── VirusTotal lookup (optional)
                   ├── if email text: TF-IDF + LR model → wording verdict + reasons
                   ├── brand-mismatch check (deterministic, no model)
                   └── human-correction override
                         │                    │
              VirusTotal API          SQLite (analyses,
              (optional)              feedback, corrections)
```

Single FastAPI process serves both API and frontend — no CORS, no separate frontend deploy.

## Performance

**URL model** — tested on a domain-disjoint split (strict, no data leakage):

| Metric | Value |
|---|---|
| Accuracy | 72.1% |
| Recall | 41.8% |
| Precision | 78.2% |

The model misses about half of phishing URLs but when it flags something it's usually right. It's an explainable second opinion, not a production filter.

**Email-wording model** — held-out accuracy 98.7%, but read that number with real caution: the training labels have a source/era confound (legitimate = 2001 Enron corporate email, phishing = 2004–2008 spam campaigns — see `src/email_data.py`), so it may partly be learning corpus era rather than phishing intent. Adversarial testing confirmed this: a modern, subtly-worded business-email-compromise attempt with none of the spam-era vocabulary was misread as legitimate (93% confidence), and an ordinary corporate security notice was misread as phishing (65% confidence). The app discloses this directly on the wording-verdict card rather than hiding it — see `results/before_fix_metrics.md` / `results/before_fix_metrics_lr.md` for the URL model's equivalent honesty exercise (regenerated pre-fix metrics, gated on confirming real leakage before training).

## Quickstart

Need Python 3.11+ and git.

### Windows
```powershell
git clone https://github.com/Sardorbek-Ibrokhimov/phishing-detector.git
cd phishing-detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn api:app --app-dir src --port 8000
```

### macOS / Linux
```bash
git clone https://github.com/Sardorbek-Ibrokhimov/phishing-detector.git
cd phishing-detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn api:app --app-dir src --port 8000
```

Then open http://localhost:8000.

## Usage

1. **Analyse a URL** — paste a URL and click Analyze
2. **Analyse an email** — paste email text or upload a .eml file. Links get extracted and checked by the URL model; the wording itself is checked separately, even with zero links
3. **Brand mismatch** — if the text claims a known brand but the link goes elsewhere, a banner flags it
4. **History** — see recent analyses
5. **Report wrong verdict** — click "Report incorrect verdict" to override the model

![Brand mismatch, email wording, and URL verdicts firing together on one message](docs/screenshots/05-brand-mismatch.png)

![Phishing wording caught with zero links present — a case the URL model alone cannot see](docs/screenshots/06-wording-only.png)

![History tab](docs/screenshots/04-history.png)

## Environment Variables

All optional — app works without any of them.

| Variable | Purpose |
|---|---|
| `VIRUSTOTAL_API_KEY` | Enables VirusTotal reputation lookup |
| `GATE_PASSWORD` | Password-protects the whole app |
| `GATE_USERNAME` | Username for auth (default: admin) |
| `FEEDBACK_API_KEY` | Enables the correction endpoint |

## Running Tests

```bash
pip install -r requirements-dev.txt
# start server in one terminal:
PHISHING_DB_PATH=./test.db ANALYZE_RATE_LIMIT_PER_MIN=200 FEEDBACK_API_KEY=test-key \
  python -m uvicorn api:app --app-dir src --port 8000
# run tests in another:
pytest tests/
```

## Project Structure

```
src/           - API, feature extraction, model training, SHAP explanations,
                  email-wording model, brand-mismatch check
frontend/      - Single-page UI (no build step)
tests/         - Black-box and grey-box test suite
models/        - Pre-trained XGBoost + email-wording models (committed, loaded at boot)
data/          - Raw datasets (~1.2GB, not in repo, only needed for retraining)
results/       - Experiment results and evaluation data
```

## Known Limitations

- Free tier cold starts take 30-60s
- Corrections/history wiped on restart (ephemeral storage)
- VirusTotal free API: 4 req/min limit
- URL model has 41.8% recall — misses most phishing
- URL shorteners bypass the URL model (can't follow redirects)
- Email-wording model has a source/era confound baked into its training labels (2001 corporate email vs. 2004-2008 spam) — confirmed by adversarial testing to miss modern, subtly-worded phishing and to false-positive on ordinary corporate security emails. Shown as a caution on the card, not silently trusted
- Brand-mismatch check only knows a curated list of ~24 common brands — anything outside that list is invisible to it
- The three signals (URL, wording, brand check) are independent and combined by a simple worst-case rule, not a unified model — none of them shares information with the others

## Deployment

See `DEPLOYMENT.md` for hosting setup (Render, Docker, env vars).

## Disclaimer

Research project for a university dissertation. Not a production security tool.
