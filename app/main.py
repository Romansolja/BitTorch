from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import torch
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from app.services.price_updater import price_updater

sys.path.append(str(Path(__file__).parent.parent))

from app.services.prediction import prediction_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    print("Loading model...")
    model_loaded = prediction_service.load_model()
    if model_loaded:
        print("Model loaded successfully")
    else:
        print("No saved model found, you need to train first")
    yield
    # Cleanup on shutdown
    print("Shutting down...")

app = FastAPI(
    title="BitTorch API",
    description="Bitcoin Price Prediction Service",
    version="0.1.0",
    lifespan=lifespan
)

@app.get("/")
def root():
    return {"message": "BitTorch API is running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "model_loaded": prediction_service.model is not None
    }

@app.get("/predict/next-day")
def predict_next_day(save_to_db: bool = True):
    """Get prediction for next day's Bitcoin price

    Args:
        save_to_db: Whether to save this prediction to database (default: True)
    """
    try:
        prediction = prediction_service.predict_next_day()
        if prediction is None:
            raise HTTPException(status_code=503,
                                detail="Model not loaded. Please train the model first.")

        # Save to database if requested
        if save_to_db:
            prediction_id = prediction_service.save_prediction(prediction)
            prediction["prediction_id"] = prediction_id
            prediction["saved"] = True
        else:
            prediction["saved"] = False

        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/history")
def get_prediction_history(limit: int = 10):
    """Get history of recent predictions

    Args:
        limit: Number of recent predictions to return (default: 10)
    """
    try:
        history = prediction_service.get_prediction_history(limit)
        return {
            "count": len(history),
            "predictions": [
                {
                    "id": p.id,
                    "prediction_date": p.prediction_date.isoformat(),
                    "current_price": p.current_price,
                    "predicted_price": p.predicted_price,
                    "actual_price": p.actual_price,
                    "created_at": p.created_at.isoformat()
                }
                for p in history
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/accuracy")
def get_prediction_accuracy():
    """Calculate accuracy of past predictions where actual prices are known"""
    metrics = price_updater.calculate_accuracy_metrics()
    if metrics is None:
        return {"message": "No predictions with actual prices yet. Run /predictions/update-actual-prices first"}
    return metrics

@app.post("/predictions/update-actual-prices")
def update_actual_prices():
    """Fetch actual Bitcoin prices and update past predictions"""
    result = price_updater.update_actual_prices()
    return result
