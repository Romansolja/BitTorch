# README.md (UPDATE)

# BitTorch: Bitcoin Return Prediction with PyTorch

An LSTM neural network built with PyTorch to predict next-day Bitcoin **returns** (not price levels).

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
│   ├── schemas.py          # Pydantic models
│   ├── models/
│   │   └── ml_models.py    # PyTorch LSTM
│   └── services/
│       ├── prediction.py   # Prediction logic
│       └── price_updater.py # Actual price fetching
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

### Example Response
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

**Target**: Next-day log return

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

## Requirements

- Python 3.9+
- PyTorch
- FastAPI
- yfinance
- SQLAlchemy
- scikit-learn
- pandas, numpy, matplotlib

## License

MIT License - see [LICENSE](LICENSE)

---

**Disclaimer**: This is a research/educational tool. Not financial advice.