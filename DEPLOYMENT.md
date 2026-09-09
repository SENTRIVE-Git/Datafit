# DataFit — Deployment Guide

## What's actually built and proven

- **Backend**: FastAPI app (`app/main.py`) wrapping the DataFit engine (`datafit/`)
  - Tested end-to-end with real HTTP requests: file upload → readiness score → findings
  - Tested security controls with real (not assumed) requests: wrong file type, empty file,
    oversized file (25MB limit), malformed CSV, invalid target column — all correctly rejected
- **Frontend**: single-page app (`app/frontend/index.html`) styled to the Sentrive brand,
  talks to the backend via `fetch`

## Honest security posture (read this before calling it "secure")

What's actually in place:
- File type whitelist (`.csv` only)
- File size limit (25MB, configurable in `main.py`)
- No data persisted to disk — processed in memory per-request, discarded after response
- No user data logged
- CORS restricted to an explicit origin allowlist (edit `ALLOWED_ORIGINS` in `main.py`
  before deploying — currently only allows localhost)
- Internal errors return a generic message to the client, not a stack trace
- Container runs as a non-root user
- **Rate limiting**: 20 requests/minute per IP on all analysis endpoints (tested with real
  requests — confirmed 20 succeed, 21st+ return 429). Adjust `RATE_LIMIT` in `main.py` based
  on real traffic once deployed.
- **Optional API key gate**: set the `DATAFIT_API_KEY` environment variable to require an
  `X-API-Key` header on all analysis endpoints. Unset by default (open access), matching the
  current free/no-accounts product state. This is a single shared secret for gating early/beta
  access — not a per-user accounts system. Tested with real requests: no key → 401, wrong key →
  401, correct key → 200.

What is **not** yet in place, and matters before handling real customer data:
- **No per-user accounts** — the optional API key above is a single shared secret, not
  individual user authentication. Building real accounts is a separate, larger feature.
- **No HTTPS configured here** — this must be terminated at your hosting provider or load
  balancer (Render/Railway/Fly.io all do this automatically).
- **No encryption-at-rest consideration needed yet**, since nothing is persisted — but the
  moment you add any database or file storage (e.g. to save past reports), this needs
  explicit design.
- **Dependency vulnerability scanning** — run `pip-audit` periodically against `requirements.txt`.

None of the above is exotic or expensive to add — they're just not built yet. Treat "no auth,
no rate limiting" as a hard blocker before any public/production deployment, not a nice-to-have.

## Local test (proven working)

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open `app/frontend/index.html` in a browser (or serve it statically) — it points at
`http://127.0.0.1:8000` by default.

## Deploying for real (recommended: Render, ~10 minutes)

1. Push this project to a GitHub repo.
2. Go to https://render.com → New → Web Service → connect your repo.
3. Render will detect the `Dockerfile` automatically. Set:
   - Instance type: Starter is fine to begin
   - Port: 8000
4. Once deployed, Render gives you a URL like `https://datafit-xyz.onrender.com`.
5. Edit `app/frontend/index.html`: set `window.DATAFIT_API_BASE` to that URL, and in
   `app/main.py`, add your frontend's deployed origin to `ALLOWED_ORIGINS`, then redeploy.
6. Host the frontend separately (e.g. Vercel, Netlify, or Render static site) — it's a single
   static HTML file with no build step needed.

Alternative platforms that work the same way with this Dockerfile: Railway, Fly.io, Google
Cloud Run.

### Required production environment variables

Set these on the backend service; do not put them in frontend code:

```text
DATAFIT_API_KEY=<long-random-secret>
DATAFIT_ALLOWED_ORIGINS=https://www.datafit.co.in
DATAFIT_ALLOWED_HOSTS=api.datafit.co.in
```

The API key is required for all analysis and experiment endpoints when set. The
health endpoint remains available for uptime checks. Configure HTTPS at the
hosting provider, and use its secret manager rather than committing the key to
the repository.

## Before charging real customers or handling their real data

At minimum, add:
1. API authentication (API keys or OAuth)
2. Rate limiting
3. A real privacy policy matching what the code actually does (this doc can be the source
   of truth for that copy, since it's accurate to the implementation)
4. Basic uptime monitoring (e.g. UptimeRobot, free tier is enough to start)
