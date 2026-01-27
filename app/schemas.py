from pydantic import BaseModel, Field
from typing import Optional


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    current_price: float = Field(..., gt=0)
    predicted_price: float = Field(..., gt=0)
    predicted_return: float = Field(..., description="Log return")
    change_percent: float
    direction: str = Field(..., description="up or down")
    confidence: float = Field(..., ge=0, le=1, description="Signal strength (not accuracy)")
    model_agreement: float = Field(..., ge=0, le=1)
    prediction_date: str
    prediction_id: Optional[int] = None
    saved: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "current_price": 105000.00,
                "predicted_price": 106050.00,
                "predicted_return": 0.00995,
                "change_percent": 1.0,
                "direction": "up",
                "confidence": 0.65,
                "model_agreement": 0.8,
                "prediction_date": "2025-01-27",
                "prediction_id": 1,
                "saved": True
            }
        }


class AccuracyMetrics(BaseModel):
    """Response model for accuracy calculations"""
    total_predictions: int
    mae_return: float = Field(..., description="Mean Absolute Error (returns)")
    mae_price: float = Field(..., description="Mean Absolute Error (USD)")
    directional_accuracy: float = Field(..., description="% correct direction")
    avg_confidence_when_correct: Optional[float] = None
    avg_confidence_when_wrong: Optional[float] = None