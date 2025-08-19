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
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from app.schemas import (
    PredictionRequest, PredictionResponse,
    TrainingRequest, TrainingResponse,
    HistoryRequest, ErrorResponse,
    ModelMetricsResponse
)
from app.services.training import training_service

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


@app.get("/predictions/debug")
def debug_predictions():
    """Debug endpoint to see all prediction dates"""
    from app.database import SessionLocal, PricePrediction
    from datetime import datetime

    db = SessionLocal()
    try:
        predictions = db.query(PricePrediction).all()
        now = datetime.now()
        return {
            "current_time": now.isoformat(),
            "total_predictions": len(predictions),
            "predictions": [
                {
                    "id": p.id,
                    "prediction_date": p.prediction_date.isoformat(),
                    "is_past": p.prediction_date <= now,
                    "actual_price": p.actual_price,
                    "predicted_price": p.predicted_price,
                    "current_price": p.current_price
                }
                for p in predictions
            ]
        }
    finally:
        db.close()


@app.post("/test/create-past-prediction")
def create_test_past_prediction():
    """Create a test prediction for yesterday to test the update functionality"""
    from app.database import SessionLocal, PricePrediction
    from datetime import datetime, timedelta

    db = SessionLocal()
    try:
        # Create a prediction for yesterday
        yesterday = datetime.now() - timedelta(days=1)
        test_pred = PricePrediction(
            current_price=110000.0,
            predicted_price=112000.0,
            prediction_date=yesterday,
            created_at=yesterday
        )
        db.add(test_pred)
        db.commit()
        db.refresh(test_pred)
        return {
            "message": "Created test prediction for yesterday",
            "id": test_pred.id,
            "prediction_date": test_pred.prediction_date.isoformat()
        }
    finally:
        db.close()


@app.post("/train", response_model=TrainingResponse)
async def train_model(
        request: TrainingRequest,
        background_tasks: BackgroundTasks
):
    """Start model training with custom parameters"""
    try:
        # Check if training is already running
        running_jobs = [
            job for job in training_service.training_jobs.values()
            if job["status"] == "running"
        ]

        if running_jobs and not request.force_retrain:
            raise HTTPException(
                status_code=409,
                detail="Training already in progress. Set force_retrain=true to override"
            )

        # Start training
        training_id = await training_service.train_model_async(request.dict())

        return TrainingResponse(
            status="started",
            training_id=training_id,
            started_at=datetime.now(),
            parameters=request,
            message="Training started successfully. Check /train/{training_id}/status for progress"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/train/{training_id}/status")
def get_training_status(training_id: str):
    """Get status of a training job"""
    status = training_service.get_training_status(training_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Training job not found")
    return status


@app.post("/predict/next-day", response_model=PredictionResponse)
def predict_next_day_enhanced(request: PredictionRequest):
    """Enhanced prediction endpoint with validation"""
    try:
        prediction = prediction_service.predict_next_day()
        if prediction is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train the model first using /train endpoint"
            )

        # Convert to Pydantic response
        response = PredictionResponse(
            current_price=prediction["current_price"],
            predicted_price=prediction["predicted_price"],
            change_amount=prediction["change_amount"],
            change_percent=prediction["change_percent"],
            prediction_date=datetime.now(),
            saved=request.save_to_db,
            model_version="v1.0"
        )

        if request.save_to_db:
            prediction_id = prediction_service.save_prediction(prediction)
            response.prediction_id = prediction_id

        return response

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Prediction failed",
                detail=str(e)
            ).dict()
        )


@app.get("/models/metrics", response_model=List[ModelMetricsResponse])
def get_model_metrics(limit: int = 5):
    """Get metrics for recent model versions"""
    db = SessionLocal()
    try:
        metrics = db.query(ModelMetrics) \
            .order_by(ModelMetrics.train_date.desc()) \
            .limit(limit) \
            .all()

        return [
            ModelMetricsResponse(
                model_version=m.model_version,
                train_date=m.train_date,
                mse=m.mse,
                mae=m.mae,
                mape=m.mape,
                baseline_improvement=m.baseline_improvement,
                total_predictions=0  # Calculate from predictions table if needed
            )
            for m in metrics
        ]
    finally:
        db.close()