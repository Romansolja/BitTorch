import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

ARTIFACTS_DIR = BASE_DIR / "artifacts"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/bittorch.db")

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
YFINANCE_HTTP_TIMEOUT = float(os.getenv("YFINANCE_HTTP_TIMEOUT", "10"))

# API security. If API_KEY is unset, `require_api_key` fails closed: every
# protected route returns 503 (authentication not configured) rather than
# being silently public. Set API_KEY in .env for local dev.
API_KEY = os.getenv("API_KEY")
API_KEY_HEADER = "X-API-Key"

# BTC-USD yfinance coverage begins 2014-09-17
BTC_USD_START_DATE = "2014-09-17"
