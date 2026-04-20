import logging
from contextlib import asynccontextmanager

import requests
import torch
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from app.auth import require_api_key
from app.database import get_db
from app.schemas import (
    AccuracyMetrics,
    BackfillRequest,
    BackfillResponse,
    HealthResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
    PredictionResponse,
    RootResponse,
    UpdateActualPricesResponse,
)
from app.services.prediction import prediction_service
from app.services.price_updater import price_updater

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    if prediction_service.load_model():
        logger.info("Model loaded successfully")
    else:
        logger.warning("No model found - run 'python main.py' to train first")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="BitTorch API",
    description="Bitcoin Return Prediction Service (v2.0)",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=RootResponse)
def root():
    return RootResponse(message="BitTorch API v2.0 - Return-based prediction")


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        gpu_available=torch.cuda.is_available(),
        torch_version=torch.__version__,
        model_loaded=prediction_service.model is not None,
        model_type="LSTM (return-based)",
    )


@app.get(
    "/predict/next-day",
    response_model=PredictionResponse,
    dependencies=[Depends(require_api_key)],
)
def predict_next_day(save_to_db: bool = True, db: Session = Depends(get_db)):
    """Get prediction for next day's Bitcoin return and price."""
    try:
        prediction = prediction_service.predict_next_day()
    except requests.RequestException as e:
        logger.warning("yfinance fetch failed: %s", e)
        raise HTTPException(status_code=503, detail="upstream data provider unavailable")
    except Exception:
        logger.exception("predict_next_day failed")
        raise HTTPException(status_code=500, detail="prediction failed")

    if prediction is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded or insufficient data. Run 'python main.py' to train.",
        )

    if save_to_db:
        try:
            prediction_id = prediction_service.save_prediction(prediction, db)
            prediction["prediction_id"] = prediction_id
            prediction["saved"] = True
        except Exception:
            logger.exception("Failed to save prediction")
            prediction["saved"] = False
    else:
        prediction["saved"] = False

    return prediction


@app.get(
    "/predictions/history",
    response_model=PredictionHistoryResponse,
    dependencies=[Depends(require_api_key)],
)
def get_prediction_history(limit: int = 10, db: Session = Depends(get_db)):
    """Get recent predictions from database."""
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")

    history = prediction_service.get_prediction_history(db, limit)
    items = [
        PredictionHistoryItem(
            id=p.id,
            prediction_date=p.prediction_date.isoformat() if p.prediction_date else None,
            current_price=p.current_price,
            predicted_price=p.predicted_price,
            predicted_return=p.predicted_return,
            direction=p.predicted_direction,
            confidence=p.confidence,
            actual_price=p.actual_price,
            actual_return=p.actual_return,
            direction_correct=p.direction_correct,
            created_at=p.created_at.isoformat() if p.created_at else None,
        )
        for p in history
    ]
    return PredictionHistoryResponse(count=len(items), predictions=items)


@app.get(
    "/predictions/accuracy",
    response_model=AccuracyMetrics,
    dependencies=[Depends(require_api_key)],
)
def get_prediction_accuracy(db: Session = Depends(get_db)):
    """Calculate accuracy metrics for predictions with known actual prices."""
    metrics = price_updater.calculate_accuracy_metrics(db)
    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail="No predictions with actual prices yet. POST /predictions/update-actual-prices first.",
        )
    return metrics


@app.post(
    "/predictions/update-actual-prices",
    response_model=UpdateActualPricesResponse,
    dependencies=[Depends(require_api_key)],
)
def update_actual_prices(db: Session = Depends(get_db)):
    """Fetch actual Bitcoin prices and update past predictions."""
    try:
        return price_updater.update_actual_prices(db)
    except requests.RequestException as e:
        logger.warning("yfinance fetch failed: %s", e)
        raise HTTPException(status_code=503, detail="upstream data provider unavailable")
    except Exception:
        logger.exception("update_actual_prices failed")
        raise HTTPException(status_code=500, detail="update failed")


@app.post(
    "/predictions/backfill",
    response_model=BackfillResponse,
    dependencies=[Depends(require_api_key)],
)
def backfill_predictions(
    req: BackfillRequest,
    include_daily: bool = False,
    db: Session = Depends(get_db),
):
    """Run model on historical data to simulate what it would have predicted.

    Walk-forward by construction: for each day D, only data up to D is used
    to predict D+1. Rejects date ranges overlapping the production training
    window.
    """
    if prediction_service.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python main.py' to train first.",
        )

    try:
        result = price_updater.backfill_historical(
            start_date=req.start_date,
            end_date=req.end_date,
            prediction_service=prediction_service,
            store=req.store,
            include_daily=include_daily,
            db=db,
        )
    except requests.RequestException as e:
        logger.warning("yfinance fetch failed during backfill: %s", e)
        raise HTTPException(status_code=503, detail="upstream data provider unavailable")
    except Exception:
        logger.exception("backfill_predictions failed")
        raise HTTPException(status_code=500, detail="backfill failed")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result
