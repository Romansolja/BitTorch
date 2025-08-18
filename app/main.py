from fastapi import FastAPI
import torch
import sys
from pathlib import Path

# Add parent directory to path so we can import our original model
sys.path.append(str(Path(__file__).parent.parent))

app = FastAPI(
    title="BitTorch API",
    description="Bitcoin Price Prediction Service",
    version="0.1.0"
)

@app.get("/")
def root():
    return {"message": "BitTorch API is running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "torch_version": torch.__version__
    }

@app.get("/predict/next-day")
def predict_next_day():
    # Placeholder for now
    return {
        "status": "not implemented yet",
        "message": "This will return the next day prediction"
    }