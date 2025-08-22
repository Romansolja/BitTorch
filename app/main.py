from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
import torch
import sys
import asyncio
from pathlib import Path
from typing import List, Optional
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from app.services.price_updater import price_updater
from app.services.prediction import prediction_service
from app.services.training import training_service
from app.schemas import (
    PredictionRequest, PredictionResponse,
    TrainingRequest, TrainingResponse,
    HistoryRequest, ErrorResponse,
    ModelMetricsResponse
)
from app.dependencies import require_api_key, require_write_permission, require_admin
from app.auth import auth_service
from app.database import SessionLocal, PricePrediction, ModelMetrics, APIKey
from app.middleware import rate_limiter

from app.websocket import manager, price_fetcher, chart_manager, handle_client_message, heartbeat_task

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    print("Loading model...")
    model_loaded = prediction_service.load_model()
    if model_loaded:
        print("Model loaded successfully")
    else:
        print("No saved model found, you need to train first")

    # Start WebSocket services
    print("Starting WebSocket services...")
    price_task = asyncio.create_task(price_fetcher.start())
    chart_task = asyncio.create_task(chart_manager.start())
    heartbeat = asyncio.create_task(heartbeat_task())

    yield

    # Cleanup on shutdown
    print("Shutting down...")
    await price_fetcher.stop()
    await chart_manager.stop()
    price_task.cancel()
    chart_task.cancel()
    heartbeat.cancel()


app = FastAPI(
    title="BitTorch API",
    description="Bitcoin Price Prediction Service",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/")
async def root(request: Request):
    # Check rate limit for anonymous users
    if not await rate_limiter.check_rate_limit(request):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please authenticate for higher limits"
        )
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
    """Get prediction for next day's Bitcoin price - No auth for testing"""
    try:
        prediction = prediction_service.predict_next_day()
        if prediction is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train the model first."
            )

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


@app.get("/predictions/history")
def get_prediction_history(limit: int = 10):
    """Get history of recent predictions"""
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
                total_predictions=0
            )
            for m in metrics
        ]
    finally:
        db.close()


# Protected endpoints (with authentication)
@app.get("/predict/next-day-protected", dependencies=[Depends(require_api_key)])
async def predict_next_day_protected(
    request: PredictionRequest = PredictionRequest(),
    api_key: str = Depends(require_api_key)
):
    """Protected prediction endpoint"""
    prediction = prediction_service.predict_next_day()
    if prediction is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    return prediction


@app.post("/train-protected", dependencies=[Depends(require_write_permission)])
async def train_model_protected(
    request: TrainingRequest,
    api_key: str = Depends(require_write_permission)
):
    """Training endpoint - requires write permission"""
    training_id = await training_service.train_model_async(request.dict())
    return {"training_id": training_id, "status": "started"}


# Admin endpoints
@app.post("/admin/api-keys", dependencies=[Depends(require_admin)])
async def create_api_key_endpoint(
    name: str,
    permissions: str = "read",
    rate_limit: int = 100,
    api_key: str = Depends(require_admin)
):
    """Create new API key - admin only"""
    new_key = auth_service.create_api_key(name, permissions, rate_limit)
    return {
        "api_key": new_key,
        "name": name,
        "permissions": permissions,
        "rate_limit": rate_limit,
        "message": "Save this key securely - it won't be shown again"
    }


@app.get("/admin/api-keys", dependencies=[Depends(require_admin)])
async def list_api_keys(api_key: str = Depends(require_admin)):
    """List all API keys - admin only"""
    db = SessionLocal()
    try:
        keys = db.query(APIKey).all()
        return [
            {
                "id": k.id,
                "name": k.name,
                "created_at": k.created_at,
                "last_used": k.last_used,
                "is_active": k.is_active,
                "permissions": k.permissions,
                "rate_limit": k.rate_limit
            }
            for k in keys
        ]
    finally:
        db.close()


@app.delete("/admin/api-keys/{key_id}", dependencies=[Depends(require_admin)])
async def revoke_api_key(
    key_id: int,
    api_key: str = Depends(require_admin)
):
    """Revoke API key - admin only"""
    db = SessionLocal()
    try:
        db_key = db.query(APIKey).filter(APIKey.id == key_id).first()
        if not db_key:
            raise HTTPException(status_code=404, detail="API key not found")

        db_key.is_active = False
        db.commit()
        return {"message": f"API key {db_key.name} revoked"}
    finally:
        db.close()


@app.get("/predictions/debug")
def debug_predictions():
    """Debug endpoint to see all prediction dates"""
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
    from datetime import timedelta

    db = SessionLocal()
    try:
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time BTC prices"""
    await manager.connect(websocket)
    try:
        while True:
            # Wait for messages from client
            message = await websocket.receive_text()
            await handle_client_message(websocket, message)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await manager.disconnect(websocket)


@app.websocket("/ws/simple")
async def simple_websocket(websocket: WebSocket):
    """Simplified WebSocket - just receives price updates, no commands"""
    await manager.connect(websocket)
    try:
        while True:
            # Just keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)


@app.get("/ws/status")
async def websocket_status():
    """Get WebSocket connection status"""
    return {
        "active_connections": len(manager.active_connections),
        "latest_price": manager.latest_price_data.get("price", 0),
        "last_update": manager.latest_price_data.get("timestamp", ""),
        "history_points": len(manager.price_history),
        "rapid_mode": price_fetcher.use_rapid_mode
    }


@app.post("/ws/broadcast")
async def manual_broadcast(message: dict, api_key: str = Depends(require_admin)):
    """Manually broadcast a message to all WebSocket clients (admin only)"""
    await manager.broadcast({
        "type": "admin_message",
        "data": message,
        "timestamp": datetime.now().isoformat()
    })
    return {"message": f"Broadcasted to {len(manager.active_connections)} clients"}


@app.post("/ws/rapid-mode")
async def toggle_rapid_mode(enabled: bool = True, api_key: str = Depends(require_admin)):
    """Toggle rapid price updates (5s instead of 30s) - admin only"""
    price_fetcher.set_rapid_mode(enabled)

    # Notify all clients
    await manager.broadcast({
        "type": "rapid_mode_change",
        "enabled": enabled,
        "interval": price_fetcher.quick_fetch_interval if enabled else price_fetcher.fetch_interval
    })

    return {
        "rapid_mode": enabled,
        "interval": price_fetcher.quick_fetch_interval if enabled else price_fetcher.fetch_interval
    }


@app.websocket("/ws/chart/{symbol}")
async def chart_websocket(websocket: WebSocket, symbol: str = "BTC-USD", interval: str = "1m"):
    """WebSocket endpoint for historical chart data streaming"""
    await manager.connect(websocket, connection_type="chart")

    # Auto-subscribe to requested symbol and interval
    await manager.subscribe_to_chart(websocket, symbol, interval)

    # Send initial chart data
    from app.websocket import TimeFrame
    if interval in [tf.value for tf in TimeFrame]:
        chart_data = await chart_manager.fetch_chart_data(symbol, TimeFrame(interval))
        await manager.send_personal_message({
            "type": "chart_initial",
            "symbol": symbol,
            "interval": interval,
            "data": chart_data
        }, websocket)

    try:
        while True:
            # Wait for messages from client
            message = await websocket.receive_text()
            await handle_client_message(websocket, message)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        print(f"Chart WebSocket error: {e}")
        await manager.disconnect(websocket)


# Add chart status endpoint
@app.get("/ws/chart/status")
async def chart_status():
    """Get chart streaming status"""
    subscriptions = []
    for ws, sub in manager.chart_subscriptions.items():
        subscriptions.append({
            "symbol": sub.get("symbol"),
            "interval": sub.get("interval"),
            "subscribed_at": sub.get("subscribed_at").isoformat() if sub.get("subscribed_at") else None
        })

    return {
        "total_chart_subscribers": len(manager.chart_subscriptions),
        "subscriptions": subscriptions,
        "cache_size": len(chart_manager.chart_cache),
        "supported_intervals": ["1m", "5m", "15m", "1h", "1d"]
    }