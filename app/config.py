import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_log = logging.getLogger(__name__)


def _float_env(name: str, default: float) -> float:
    """Parse a float from env with a safe default on missing/bad values.

    Returning the default (with a warning) is preferable to raising at
    import time because a misconfigured env var would otherwise prevent
    the FastAPI app from starting and give no actionable log context.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        _log.warning(
            "Env var %s=%r is not a number; falling back to %s",
            name, raw, default,
        )
        return default


def _str_env(name: str) -> str | None:
    """Return env value stripped; treat empty/whitespace-only as unset."""
    raw = os.getenv(name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped else None

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

ARTIFACTS_DIR = BASE_DIR / "artifacts"

DATABASE_URL = _str_env("DATABASE_URL") or f"sqlite:///{DATA_DIR}/bittorch.db"

# Artifact paths
MODEL_PATH = ARTIFACTS_DIR / "production_model.pth"
SCALER_PATH = ARTIFACTS_DIR / "feature_scaler.pkl"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
RIDGE_PATH = ARTIFACTS_DIR / "ridge_baseline.pkl"

# Model / feature hyperparameters
SEQUENCE_LENGTH = 30

# Prediction service tunables
LATEST_FETCH_DAYS = 60          # days of recent data pulled for /predict/next-day
CONFIDENCE_WINDOW = 30          # trailing returns used for confidence vol estimate

# Backfill tunables
BACKFILL_MAX_DAYS = 365                 # hard cap on range size
BACKFILL_MAX_DAILY_RESULTS = 120        # cap when include_daily=True
BACKFILL_FEATURE_BUFFER_DAYS = 90       # warmup history floor for feature rolling windows
BACKFILL_FEATURE_MAX_ROLLING = 30       # longest rolling window inside make_features()

# Data provider / HTTP
YFINANCE_HTTP_TIMEOUT = _float_env("YFINANCE_HTTP_TIMEOUT", 10.0)

# API security. If API_KEY is unset OR empty/whitespace-only (e.g. a bare
# `API_KEY=` line in .env), `require_api_key` fails closed: every protected
# route returns 503 (authentication not configured). Set API_KEY in .env
# for local dev. Empty strings are treated as unset to avoid an accidentally
# public API where `compare_digest("", "")` would otherwise succeed.
API_KEY = _str_env("API_KEY")
API_KEY_HEADER = "X-API-Key"

# BTC-USD yfinance coverage begins 2014-09-17
BTC_USD_START_DATE = "2014-09-17"
