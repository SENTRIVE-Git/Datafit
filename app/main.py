"""
DataFit — web API.

Security posture for v1 (honest, not "100%"):
- File type + extension whitelist (CSV only for now)
- File size limit enforced before processing
- Uploaded data is processed in-memory / temp file and deleted
  immediately after the response is built — nothing persisted to disk
  beyond the request lifecycle
- No user data logged
- CORS restricted to explicit origins (edit ALLOWED_ORIGINS for your
  deployed frontend domain)
- Per-IP rate limiting on all analysis endpoints (see RATE_LIMIT below)
- Optional shared API key gate (see DATAFIT_API_KEY below) — this is
  NOT a user-accounts system. It's a single shared secret you can set
  before a public/beta launch to require a key without building full
  per-user auth, which would need an accounts system we don't have yet.
"""

from __future__ import annotations

import io
import os
import secrets

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from datafit.readiness import assess_readiness
from datafit.experiments import run_imbalance_experiment, run_synthetic_data_experiment, preview_cleanup, run_regression_baseline_experiment

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB — generous for a CSV, prevents abuse/DoS
MAX_CSV_ROWS = 1_000_000
MAX_CSV_COLUMNS = 500
ALLOWED_EXTENSIONS = {".csv"}
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:4173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:4173",
]
ALLOWED_ORIGINS = [origin.strip().rstrip("/") for origin in os.environ.get(
    "DATAFIT_ALLOWED_ORIGINS", ",".join(DEFAULT_ALLOWED_ORIGINS)
).split(",") if origin.strip()]
ALLOWED_HOSTS = [host.strip() for host in os.environ.get(
    "DATAFIT_ALLOWED_HOSTS", "127.0.0.1,localhost"
).split(",") if host.strip()]

# Rate limit for analysis endpoints — generous enough for real use, tight
# enough to block abuse/scraping. Adjust based on real traffic patterns
# once deployed; this is a starting point, not a tuned production value.
RATE_LIMIT = "20/minute"

# Optional shared API key. If DATAFIT_API_KEY is set in the environment,
# all analysis endpoints require an `X-API-Key` header matching it. If
# unset (the default), the API is open — appropriate for local dev and
# the current no-accounts free tier, not for a monetized or
# invite-only launch. Set this env var before that changes.
REQUIRED_API_KEY = os.environ.get("DATAFIT_API_KEY", "").strip()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="DataFit API", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def _validate_and_read_csv(file: UploadFile, raw_bytes: bytes) -> pd.DataFrame:
    """Validates an uploaded file before doing anything with it."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Only .csv is supported.")

    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max size is {MAX_FILE_SIZE_BYTES // (1024*1024)}MB.")

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), nrows=MAX_CSV_ROWS + 1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty or len(df.columns) == 0:
        raise HTTPException(status_code=400, detail="CSV parsed but contains no usable data.")
    if len(df) > MAX_CSV_ROWS:
        raise HTTPException(status_code=413, detail=f"CSV has too many rows. Max is {MAX_CSV_ROWS:,}.")
    if len(df.columns) > MAX_CSV_COLUMNS:
        raise HTTPException(status_code=413, detail=f"CSV has too many columns. Max is {MAX_CSV_COLUMNS}.")

    return df


async def _read_bounded_upload(request: Request, file: UploadFile) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE_BYTES + 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large.")
    return await file.read(MAX_FILE_SIZE_BYTES + 1)


def _verify_api_key(x_api_key: str | None) -> None:
    """No-op when DATAFIT_API_KEY isn't set (current open/free tier).
    When it IS set, requires a matching X-API-Key header, compared in
    constant time to avoid timing side-channels on the comparison."""
    if not REQUIRED_API_KEY:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, REQUIRED_API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/assess")
@limiter.limit(RATE_LIMIT)
async def assess(
    request: Request,
    file: UploadFile = File(...),
    task: str = Form("generic"),
    target_column: str | None = Form(None),
    x_api_key: str | None = Header(None),
):
    """Runs the read-only readiness assessment. Never modifies or
    retains the uploaded data beyond this request."""
    _verify_api_key(x_api_key)
    raw_bytes = await _read_bounded_upload(request, file)
    df = _validate_and_read_csv(file, raw_bytes)

    if target_column and target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target_column}' not found in uploaded data. "
                                                       f"Available columns: {list(df.columns)}")

    try:
        report = assess_readiness(df, task=task, target_column=target_column)
    except Exception as e:
        # Don't leak internal stack traces to the client; log server-side only if needed.
        raise HTTPException(status_code=500, detail="Assessment failed due to an internal error.")

    return JSONResponse({
        "task": report.task,
        "analyzable": report.analyzable,
        "analyzability_reason": report.analyzability_reason,
        "score": report.score,
        "status_label": report.status_label(),
        "n_rows": report.n_rows,
        "n_cols": report.n_cols,
        "findings": [
            {"check": f.check, "severity": f.severity, "message": f.message, "points_deducted": f.points_deducted}
            for f in report.findings
        ],
        "recommended_next_steps": report.recommended_next_steps,
        "scope_disclaimer": report.scope_disclaimer,
        "columns": list(df.columns),
    })


@app.post("/experiment/imbalance")
@limiter.limit(RATE_LIMIT)
async def experiment_imbalance(request: Request, file: UploadFile = File(...), target_column: str = Form(...),
                                x_api_key: str | None = Header(None)):
    _verify_api_key(x_api_key)
    raw_bytes = await _read_bounded_upload(request, file)
    df = _validate_and_read_csv(file, raw_bytes)
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target_column}' not found.")

    try:
        result = run_imbalance_experiment(df, target_column)
    except Exception:
        raise HTTPException(status_code=500, detail="Experiment failed due to an internal error.")

    return JSONResponse({
        "experiment": result.experiment,
        "ran_successfully": result.ran_successfully,
        "evidence_summary": result.evidence_summary,
        "metrics": result.metrics,
        "caveat": result.caveat,
    })


@app.post("/experiment/synthetic")
@limiter.limit(RATE_LIMIT)
async def experiment_synthetic(request: Request, file: UploadFile = File(...), target_column: str = Form(...),
                                x_api_key: str | None = Header(None)):
    _verify_api_key(x_api_key)
    raw_bytes = await _read_bounded_upload(request, file)
    df = _validate_and_read_csv(file, raw_bytes)
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target_column}' not found.")

    try:
        result = run_synthetic_data_experiment(df, target_column)
    except Exception:
        raise HTTPException(status_code=500, detail="Experiment failed due to an internal error.")

    return JSONResponse({
        "experiment": result.experiment,
        "ran_successfully": result.ran_successfully,
        "evidence_summary": result.evidence_summary,
        "metrics": result.metrics,
        "caveat": result.caveat,
    })


@app.post("/experiment/cleanup-preview")
@limiter.limit(RATE_LIMIT)
async def experiment_cleanup_preview(request: Request, file: UploadFile = File(...),
                                      target_column: str | None = Form(None),
                                      x_api_key: str | None = Header(None)):
    _verify_api_key(x_api_key)
    raw_bytes = await _read_bounded_upload(request, file)
    df = _validate_and_read_csv(file, raw_bytes)

    try:
        result = preview_cleanup(df, target_column=target_column)
    except Exception:
        raise HTTPException(status_code=500, detail="Preview failed due to an internal error.")

    return JSONResponse({
        "experiment": result.experiment,
        "ran_successfully": result.ran_successfully,
        "evidence_summary": result.evidence_summary,
        "caveat": result.caveat,
    })


@app.post("/experiment/regression-baseline")
@limiter.limit(RATE_LIMIT)
async def experiment_regression_baseline(request: Request, file: UploadFile = File(...), target_column: str = Form(...),
                                         x_api_key: str | None = Header(None)):
    _verify_api_key(x_api_key)
    raw_bytes = await _read_bounded_upload(request, file)
    df = _validate_and_read_csv(file, raw_bytes)
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target_column}' not found.")

    try:
        result = run_regression_baseline_experiment(df, target_column)
    except Exception:
        raise HTTPException(status_code=500, detail="Regression baseline failed due to an internal error.")

    return JSONResponse({
        "experiment": result.experiment,
        "ran_successfully": result.ran_successfully,
        "evidence_summary": result.evidence_summary,
        "metrics": result.metrics,
        "caveat": result.caveat,
    })
