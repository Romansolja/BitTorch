from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import torch
import sys
from pathlib import Path

# Add parent directory to path
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
def predict_next_day():
    """Get prediction for next day's Bitcoin price"""
    try:
        prediction = prediction_service.predict_next_day()
        if prediction is None:
            raise HTTPException(status_code=503, 
                              detail="Model not loaded. Please train the model first.")
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))