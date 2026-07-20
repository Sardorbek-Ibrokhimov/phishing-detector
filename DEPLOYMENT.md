# Deployment

Deployed on **Render** (free tier, Docker web service).

## Why Render

- Free indefinitely (750 hrs/month)
- Supports Docker
- HF Spaces Docker needs a paid plan, Fly.io has no free tier since 2024

## Memory

The app fits within Render's 512MB limit:
- Original stack with shap library: 536MB (too big)
- After switching to XGBoost native TreeSHAP: 476MB (fits)

The shap library was replaced with XGBoost's built-in `pred_contribs` which gives the same results without importing shap/matplotlib/numba.

## Auth

HTTP Basic Auth middleware in FastAPI since Render doesn't have built-in password protection.

- Set `GATE_PASSWORD` env var to enable it
- Browser shows native login dialog, credentials auto-attach to API calls
- Corrections require both gate password + `FEEDBACK_API_KEY` (X-API-Key header)

## Env Vars

Set these in the Render dashboard:

| Variable | Required | Purpose |
|---|---|---|
| `GATE_PASSWORD` | yes | app password |
| `GATE_USERNAME` | no (default: admin) | app username |
| `FEEDBACK_API_KEY` | for corrections | enables /feedback endpoint |
| `VIRUSTOTAL_API_KEY` | no | enables VT reputation lookup |

## Storage

The correction database is **ephemeral** on the free tier — wiped on restart/redeploy. The model itself is baked into the Docker image so verdicts are stable.

## Deploy Steps

1. Render dashboard → New → Web Service → connect GitHub repo
2. It auto-detects the Dockerfile, pick Free tier
3. Add env vars under Environment
4. Deploy

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Python 3.12-slim, non-root user, uvicorn |
| `.dockerignore` | Excludes data/, venv, tests from build |
| `requirements.txt` | Runtime deps only |
| `requirements-dev.txt` | Adds pytest, shap, matplotlib for dev |
