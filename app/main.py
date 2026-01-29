# app/main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import torch
from datetime import datetime

from app.services.prediction import prediction_service
from app.services.price_updater import price_updater
from app.database import SessionLocal, PricePrediction
from app.schemas import BackfillRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    if prediction_service.load_model():
        print("Model loaded successfully")
    else:
        print("No model found - run 'python main.py' to train first")
    yield
    print("Shutting down...")


app = FastAPI(
    title="BitTorch API",
    description="Bitcoin Return Prediction Service (v2.0)",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {"message": "BitTorch API v2.0 - Return-based prediction"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "model_loaded": prediction_service.model is not None,
        "model_type": "LSTM (return-based)"
    }


@app.get("/predict/next-day")
def predict_next_day(save_to_db: bool = True):
    """Get prediction for next day's Bitcoin return and price."""
    prediction = prediction_service.predict_next_day()

    if prediction is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded or insufficient data. Run 'python main.py' to train."
        )

    if save_to_db:
        prediction_id = prediction_service.save_prediction(prediction)
        prediction["prediction_id"] = prediction_id
        prediction["saved"] = True
    else:
        prediction["saved"] = False

    return prediction


@app.get("/predictions/history")
def get_prediction_history(limit: int = 10):
    """Get recent predictions from database."""
    history = prediction_service.get_prediction_history(limit)
    return {
        "count": len(history),
        "predictions": [
            {
                "id": p.id,
                "prediction_date": p.prediction_date.isoformat() if p.prediction_date else None,
                "current_price": p.current_price,
                "predicted_price": p.predicted_price,
                "predicted_return": p.predicted_return,
                "direction": p.predicted_direction,
                "confidence": p.confidence,
                "actual_price": p.actual_price,
                "actual_return": p.actual_return,
                "direction_correct": p.direction_correct,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in history
        ]
    }


@app.get("/predictions/accuracy")
def get_prediction_accuracy():
    """Calculate accuracy metrics for predictions with known actual prices."""
    metrics = price_updater.calculate_accuracy_metrics()
    if metrics is None:
        return {
            "message": "No predictions with actual prices yet.",
            "hint": "Run POST /predictions/update-actual-prices after prediction dates pass"
        }
    return metrics


@app.post("/predictions/update-actual-prices")
def update_actual_prices():
    """Fetch actual Bitcoin prices and update past predictions."""
    return price_updater.update_actual_prices()


@app.post("/predictions/backfill")
def backfill_predictions(req: BackfillRequest, include_daily: bool = True):
    """
    Run model on historical data to simulate what it would have predicted.

    IMPORTANT: This is a proper walk-forward backtest:
    - For each day D, only data up to D is used to predict D+1
    - Uses the production scaler (no refitting)
    - No future data leakage

    Args:
        req: BackfillRequest with start_date, end_date, and store flag
        include_daily: Whether to include per-day results (default True)

    Returns:
        Model metrics vs baselines, optionally with daily breakdown
    """
    if prediction_service.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python main.py' to train first."
        )

    # Validate date range
    if req.end_date < req.start_date:
        raise HTTPException(
            status_code=400,
            detail="end_date must be >= start_date"
        )

    # Cap range to prevent massive requests
    days_requested = (req.end_date - req.start_date).days
    if days_requested > 365:
        raise HTTPException(
            status_code=400,
            detail="Maximum backfill range is 365 days"
        )

    result = price_updater.backfill_historical(
        start_date=req.start_date,
        end_date=req.end_date,
        prediction_service=prediction_service,
        store=req.store,
        include_daily=include_daily,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result