# BitTorch: Bitcoin Return Prediction with PyTorch

BitTorch is a Bitcoin prediction tool that uses machine learning to forecast next-day price movements.

## v3.0 Changes — Security & Robustness

- **API-key auth** on every mutating/expensive route via `X-API-Key` header (fail-closed if unset)
- **Upstream timeouts** on all yfinance calls; outages return clean `503`s via a normalized `UpstreamDataError` instead of hanging workers
- **Env-based config** (`.env` + `python-dotenv`, `.env.example` included)
- **Backfill cutoff guard** — `metadata.json` now records `training_cutoff`; backfill requests overlapping the training window are rejected so reported metrics can't be inflated by in-sample data
- **Ridge baseline** saved alongside the LSTM and exposed in every response (`ridge_*`)
- **Tz-aware datetimes** end-to-end; `DateTime(timezone=True)` on all columns
- **Atomic backfill persistence** — single `db.commit()` per range (was one-per-day)
- **Constant-time API-key compare** (`secrets.compare_digest`)
- **`confidence_valid` flag** distinguishes a real 0.5 from the degenerate-vol fallback
- Full story in [PR #6](https://github.com/Romansolja/BitTorch/pull/6)

## v2.1 Changes

- **Backfill endpoint**: validate model on historical periods with proper walk-forward simulation
- No future data leakage in backfill (uses only data up to day D to predict D+1)
- Compares against baselines (zero return, persistence)

## v2.0 Changes

- **Predicts returns** instead of price levels (more stationary target)
- **Feature engineering**: RSI, volatility, momentum, MA distances
- **Walk-forward validation**: proper out-of-sample testing
- **Correct baselines**: "predict 0" and "predict last return"
- **Directional accuracy**: track if we got up/down correct
- **Confidence metric**: signal strength relative to volatility

## Project Structure
```
BitTorch/
├── main.py                    # Training: walk-forward + production artifacts
├── inference.py               # Standalone inference module
├── artifacts/                 # Production model artifacts
│   ├── production_model.pth
│   ├── ridge_baseline.pkl
│   ├── feature_scaler.pkl
│   └── metadata.json          # includes training_cutoff
├── checkpoints/               # Per-fold LSTM weights
├── app/
│   ├── main.py                # FastAPI routes + lifespan
│   ├── auth.py                # X-API-Key dependency
│   ├── config.py              # dotenv-backed config + tunables
│   ├── database.py            # SQLAlchemy models (tz-aware)
│   ├── schemas.py             # Pydantic models + validators
│   ├── models/
│   │   └── ml_models.py       # PyTorch LSTM
│   └── services/
│       ├── prediction.py
│       ├── price_updater.py
│       └── yf_client.py       # timeout + UpstreamDataError wrapper
├── data/
│   └── bittorch.db            # SQLite (default)
├── .env.example
├── requirements.txt
└── requirements-dev.txt
```

## Installation

```bash
git clone https://github.com/Romansolja/BitTorch.git
cd BitTorch
pip install -r requirements.txt          # runtime
# or
pip install -r requirements-dev.txt      # runtime + pytest, black, mypy, etc.

cp .env.example .env
# Generate and paste an API key into .env:
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Usage

### 1. Train the Model

```bash
python main.py
```

This will:
- Download 5 years of BTC-USD data
- Run 14-fold walk-forward validation
- Train a Ridge baseline + LSTM production model on all data
- Save artifacts to `artifacts/` (including `training_cutoff` in `metadata.json`)

### 2. Standalone Inference

```bash
python inference.py
```

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

API at `http://127.0.0.1:8000`, interactive docs at `/docs`.

Public routes: `/` and `/health`. Every other route requires an `X-API-Key` header:

```bash
curl -H "X-API-Key: $API_KEY" http://127.0.0.1:8000/predict/next-day
```

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET  | `/`                                  |      | Welcome message |
| GET  | `/health`                            |      | Model + GPU status |
| GET  | `/predict/next-day`                  |  ✔   | LSTM + Ridge prediction |
| GET  | `/predictions/history`               |  ✔   | Recent predictions from DB |
| GET  | `/predictions/accuracy`              |  ✔   | Directional accuracy / MAE on stored predictions |
| POST | `/predictions/update-actual-prices`  |  ✔   | Fill in actuals for past predictions |
| POST | `/predictions/backfill`              |  ✔   | Walk-forward backtest on historical data |

### HTTP Semantics

| Code | Meaning |
|---|---|
| `200` | Success |
| `400` | Client error (bad dates, range > 365 days, `start_date` inside `training_cutoff`) |
| `401` | Missing or invalid `X-API-Key` |
| `422` | Pydantic schema validation (future `end_date`, pre-2014 `start_date`, reversed range) |
| `500` | Unexpected server/DB error (safe detail, no stack traces leaked) |
| `503` | Model not loaded, or upstream yfinance unavailable |

### Backfill Endpoint

Proper walk-forward simulation — for each day D, only data up to D is used to predict D+1.

**Request:**
```json
{
  "start_date": "2025-12-01",
  "end_date":   "2025-12-31",
  "store": false
}
```

**Validation (rejected with 400/422):**
- `end_date >= start_date`
- Range ≤ 365 inclusive days
- `start_date >= 2014-09-17` (BTC-USD yfinance coverage)
- `end_date <= today` (UTC)
- `start_date > training_cutoff` (prevents in-sample metric inflation)

**Response includes:**
- Model metrics: `mae`, `directional_accuracy`
- Ridge baseline: `ridge_mae`, `ridge_diracc`
- Reference baselines: `baseline_zero_mae`, `baseline_lastret_*`, `baseline_majority_diracc`
- `skipped` — per-cause counters (missing_market_day, insufficient_history, no_next_day_actual, nan_features)
- Optional `daily_results` (set `?include_daily=true`; capped at 120 rows with `daily_results_truncated` flag)

### Example Prediction Response

```json
{
  "current_price": 105000.00,
  "predicted_price": 105944.78,
  "predicted_return": 0.00895,
  "change_percent": 0.90,
  "direction": "up",
  "confidence": 0.63,
  "confidence_valid": true,
  "model_agreement": 0.80,
  "prediction_date": "2026-04-21",
  "ridge_predicted_return": 0.00412,
  "ridge_predicted_price": 105433.14,
  "ridge_direction": "up",
  "prediction_id": 42,
  "saved": true
}
```

## Model Details

**Architecture:** 2-layer LSTM → Dropout → Linear, plus a Ridge(alpha=1.0) baseline trained on the last-timestep features only.

**Features (12):**

| Feature | What it measures |
|---|---|
| `ret_1`, `ret_2`, `ret_3` | Lagged log returns |
| `vol_7`, `vol_14` | Rolling volatility |
| `mom_7`, `mom_14` | Momentum |
| `rsi_14` | Relative Strength Index |
| `ma10_dist`, `ma30_dist` | Distance from 10/30-day moving averages |
| `hl_range` | (High − Low) / Close |
| `vol_chg` | Log-change in volume |

**Target:** next-day log return
**Validation:** expanding-window walk-forward, 14 folds

### Latest walk-forward metrics (14 folds)

| Model                 | MAE    | DirAcc |
|-----------------------|--------|--------|
| LSTM                  | 0.0176 | 0.490  |
| Ridge (last timestep) | 0.0175 | 0.511  |
| Persistence baseline  | 0.0259 | 0.479  |

Ridge currently beats LSTM on directional accuracy — the LSTM's sequence context isn't adding signal on this feature set yet. Treat directional metrics with appropriate skepticism (a naïve-majority predictor can beat either on a trend-biased window).

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `API_KEY` | *(unset → 503)* | Required for protected routes. Empty / whitespace values are treated as unset (fail-closed) |
| `DATABASE_URL` | `sqlite:///data/bittorch.db` | Any SQLAlchemy URL. For Postgres/MySQL you must install a driver yourself (not pinned in `requirements.txt`) |
| `YFINANCE_HTTP_TIMEOUT` | `10` (seconds) | Upstream HTTP timeout. Bad values fall back to the default with a warning |

## Metrics Glossary

| Metric | Meaning |
|---|---|
| `predicted_return` | Log return (raw LSTM output) |
| `change_percent` | Human-readable % change |
| `direction` | `"up"` or `"down"` |
| `confidence` | Signal strength vs. recent volatility (**not** accuracy) |
| `confidence_valid` | `false` when recent vol is degenerate; `confidence` is a fallback |
| `model_agreement` | Fraction of recent overlapping windows that agree on sign |
| `directional_accuracy` | % of correct up/down calls (historical) |
| `mae_improvement_vs_zero` | `(zero_mae − mae) / zero_mae`; `null` when `zero_mae == 0` |
| `ridge_*` | Mirror of the above for the Ridge baseline |

## Requirements

- Python 3.10+
- PyTorch (CUDA optional)
- FastAPI, SQLAlchemy, Pydantic v2
- yfinance, scikit-learn, pandas, numpy, matplotlib

See `requirements.txt` for runtime and `requirements-dev.txt` for test/dev tooling.

**Disclaimer:** Research / educational tool. Not financial advice.
