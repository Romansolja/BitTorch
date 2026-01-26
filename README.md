# BitTorch: Bitcoin Price Prediction with PyTorch

A simple LSTM neural network built with PyTorch to predict the next-day closing price of Bitcoin (BTC-USD).

## Features

- **LSTM Neural Network**: 2-layer LSTM with dropout for time-series forecasting
- **FastAPI REST API**: Clean interface for making predictions
- **Prediction Tracking**: SQLite database stores all predictions for accuracy analysis
- **Accuracy Metrics**: Automatically calculates MAE and MAPE against actual prices

## Project Structure

```
BitTorch/
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── database.py         # SQLAlchemy models
│   ├── main.py             # FastAPI application
│   ├── schemas.py          # Pydantic models
│   ├── models/
│   │   └── ml_models.py    # PyTorch LSTM model
│   └── services/
│       ├── prediction.py   # Prediction logic
│       └── price_updater.py # Actual price fetching
├── data/
│   └── models/             # Trained model storage
├── main.py                 # Training script
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Romansolja/BitTorch.git
   cd BitTorch
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Model

```bash
python main.py
```

This downloads 2 years of BTC data, trains the LSTM model, and saves `best_model.pth` to `data/models/`.

### 2. Run the API

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

Interactive docs at `http://127.0.0.1:8000/docs`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | API status, GPU info, model status |
| GET | `/predict/next-day` | Get next-day price prediction |
| GET | `/predictions/history` | View recent predictions |
| GET | `/predictions/accuracy` | Calculate prediction accuracy |
| POST | `/predictions/update-actual-prices` | Update predictions with actual prices |

### Example: Get Prediction

```bash
curl http://127.0.0.1:8000/predict/next-day
```

Response:
```json
{
  "current_price": 105000.00,
  "predicted_price": 106250.00,
  "change_amount": 1250.00,
  "change_percent": 1.19,
  "prediction_date": "2025-01-26 10:30:00",
  "prediction_id": 1,
  "saved": true
}
```

## Model Performance

The LSTM model is trained with:
- 70/20/10 train/val/test split
- Early stopping with patience of 15 epochs
- Learning rate scheduling
- Gradient clipping

Typical results:
- Test MAPE: ~1-2%
- Improvement over baseline: 5-10%

## Requirements

- Python 3.8+
- PyTorch
- FastAPI
- yfinance
- SQLAlchemy
- scikit-learn

## License

MIT License - see [LICENSE](LICENSE)
