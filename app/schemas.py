from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    current_price: float = Field(..., gt=0, description="Current BTC price in USD")
    predicted_price: float = Field(..., gt=0, description="Predicted BTC price")
    change_amount: float = Field(..., description="Price change in USD")
    change_percent: float = Field(..., description="Percentage change")
    prediction_date: str
    prediction_id: Optional[int] = None
    saved: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "current_price": 105000.00,
                "predicted_price": 106500.00,
                "change_amount": 1500.00,
                "change_percent": 1.43,
                "prediction_date": "2025-01-26 10:30:00",
                "prediction_id": 1,
                "saved": True
            }
        }


class AccuracyMetrics(BaseModel):
    """Response model for accuracy calculations"""
    total_predictions: int
    mae: float = Field(..., description="Mean Absolute Error in USD")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    best_prediction_error: float
    worst_prediction_error: float