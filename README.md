# BitTorch: Bitcoin Return Prediction with PyTorch

BitTorch is a Bitcoin prediction tool that uses machine learning to forecast next-day price movements.

## v2.1 Changes

- **Backfill endpoint**: Validate model on historical periods with proper walk-forward simulation
- No future data leakage in backfill (uses only data up to day D to predict D+1)
- Compares against baselines (zero return, persistence)

## v2.0 Changes

- **Predicts returns** instead of price levels (more stationary target)
- **Feature engineering**: RSI, volatility, momentum, MA distances
- **Walk-forward validation**: Proper out-of-sample testing
- **Correct baselines**: Compare against "predict 0" and "predict last return"
- **Directional accuracy**: Track if we got up/down correct
- **Confidence metric**: Signal strength relative to volatility

## Project Structure
```
BitTorch/
├── main.py                 # Training script (walk-forward + production)
├── inference.py            # Standalone inference module
├── artifacts/              # Production model artifacts
│   ├── production_model.pth
│   ├── feature_scaler.pkl
│   └── metadata.json
├── checkpoints/            # Per-fold checkpoints
├── app/
│   ├── main.py             # FastAPI application
│   ├── config.py           # Configuration
│   ├── database.py         # SQLAlchemy models
│   ├── schemas.py          # Pydantic models (incl. BackfillRequest)
│   ├── models/
│   │   └── ml_models.py    # PyTorch LSTM
│   └── services/
│       ├── prediction.py   # Prediction logic
│       └── price_updater.py # Actual price fetching + backfill
├── data/
│   └── bittorch.db         # SQLite database
└── requirements.txt
```

## Installation
```bash
git clone https://github.com/Romansolja/BitTorch.git
cd BitTorch
pip install -r requirements.txt
```

## Usage

### 1. Train the Model
```bash
python main.py
```

This will:
- Download 5 years of BTC data
- Run walk-forward validation (multiple folds)
- Train a production model on all data
- Save artifacts to `artifacts/`

### 2. Standalone Inference
```bash
python inference.py
```

### 3. Run the API
```bash
uvicorn app.main:app --reload
```

API at `http://127.0.0.1:8000`, docs at `/docs`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | API status |
| GET | `/predict/next-day` | Get prediction (return + price + direction) |
| GET | `/predictions/history` | View recent predictions |
| GET | `/predictions/accuracy` | Calculate directional accuracy |
| POST | `/predictions/update-actual-prices` | Update with actual prices |
| POST | `/predictions/backfill` | **NEW**: Run historical backtest |

### Backfill Endpoint

Run the model on historical data to see what it *would have predicted*. This is a proper walk-forward simulation with no future data leakage.

**Request:**
```json
{
  "start_date": "2024-12-01",
  "end_date": "2024-12-31",
  "store": false
}
```

**Response:**
```json
{
  "n_days": 30,
  "start_date": "2024-12-01",
  "end_date": "2024-12-31",
  "mae": 0.0234,
  "directional_accuracy": 0.533,
  "baseline_zero_mae": 0.0241,
  "baseline_zero_diracc": 0.0,
  "baseline_lastret_mae": 0.0312,
  "baseline_lastret_diracc": 0.467,
  "mae_improvement_vs_zero": 0.029,
  "diracc_improvement_vs_random": 0.033,
  "stored": false,
  "daily_results": [...]
}
```

**Backfill Rules (and why it's valid):**
1. For each day D, only uses data up to D to predict D+1
2. Uses the production scaler (no refitting per day)
3. All rolling features are backward looking (no centered windows)

### Example Prediction Response
```json
{
  "current_price": 105000.00,
  "predicted_price": 106050.00,
  "predicted_return": 0.00995,
  "change_percent": 1.0,
  "direction": "up",
  "confidence": 0.65,
  "model_agreement": 0.8,
  "prediction_date": "2025-01-27",
  "prediction_id": 1,
  "saved": true
}
```

## Model Details

**Architecture**: 2-layer LSTM → Dropout → Linear

**Features** (11 total):
- `ret_1`, `ret_2`, `ret_3`: Lagged returns
- `vol_7`, `vol_14`: Rolling volatility
- `mom_7`, `mom_14`: Momentum
- `rsi_14`: RSI indicator
- `ma10_dist`, `ma30_dist`: Distance from moving averages
- `hl_range`: High-low range proxy

**Target**: Next day log return

**Validation**: Walk-forward with expanding window

## Metrics Explained

| Metric | What it means |
|--------|---------------|
| `predicted_return` | Log return (raw model output) |
| `change_percent` | Human-readable % change |
| `direction` | "up" or "down" |
| `confidence` | Signal strength vs volatility (NOT accuracy) |
| `model_agreement` | Consistency across recent windows |
| `directional_accuracy` | % of correct up/down calls (historical) |
| `baseline_zero_mae` | MAE if we predicted 0 every day |
| `baseline_lastret_mae` | MAE if we predicted last observed return |

## Requirements

- Python 3.9+
- PyTorch
- FastAPI
- yfinance
- SQLAlchemy
- scikit-learn
- pandas, numpy, matplotlib


**Disclaimer**: This is a research/educational tool. Not financial advice.
