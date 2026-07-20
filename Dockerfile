# Deployment image for the phishing detector (see DEPLOYMENT.md).
# Platform-neutral: reads $PORT (Render injects it) and defaults to 7860.
# Primary target is Render's free Docker web service.
FROM python:3.12-slim

# Run as a non-root UID-1000 user with a writable home, to avoid permission
# errors on caches and the SQLite file (and required by some platforms).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    TLDEXTRACT_CACHE=/home/user/.cache/tldextract
WORKDIR $HOME/app

# Install deps first so the layer caches across code changes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Ship code, frontend, and the PERSISTED model (~3.4 MB). The model is loaded
# at startup, never retrained (G7) — so models/ must be present in the image.
COPY --chown=user src/ ./src/
COPY --chown=user frontend/ ./frontend/
COPY --chown=user models/ ./models/

EXPOSE 7860
# Shell form so ${PORT} is expanded at runtime (HF sets app_port=7860;
# other platforms inject $PORT).
CMD ["sh", "-c", "uvicorn api:app --app-dir src --host 0.0.0.0 --port ${PORT:-7860}"]
