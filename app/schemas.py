from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional, List
from enum import Enum


class PredictionTimeframe(str, Enum):
    ONE_DAY = "1d"
    THREE_DAYS = "3d"
    SEVEN_DAYS = "7d"


class PredictionRequest(BaseModel):
    timeframe: PredictionTimeframe = PredictionTimeframe.ONE_DAY
    save_to_db: bool = True
    include_confidence: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "timeframe": "1d",
                "save_to_db": True,
                "include_confidence": False
            }
        }


class PredictionResponse(BaseModel):
    current_price: float = Field(..., gt=0, description="Current BTC price in USD")
    predicted_price: float = Field(..., gt=0, description="Predicted BTC price")
    change_amount: float = Field(..., description="Price change in USD")
    change_percent: float = Field(..., description="Percentage change")
    prediction_date: datetime
    confidence_interval: Optional[tuple] = None
    prediction_id: Optional[int] = None
    saved: bool = False
    model_version: str = "v1.0"

    @validator('change_percent')
    def round_percent(cls, v):
        return round(v, 2)


class TrainingRequest(BaseModel):
    epochs: int = Field(default=150, ge=10, le=500)
    batch_size: int = Field(default=32, ge=8, le=128)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    sequence_length: int = Field(default=7, ge=3, le=30)
    force_retrain: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "epochs": 150,
                "batch_size": 32,
                "learning_rate": 0.001,
                "sequence_length": 7,
                "force_retrain": False
            }
        }


class TrainingResponse(BaseModel):
    status: str
    training_id: str
    started_at: datetime
    parameters: TrainingRequest
    message: str


class ModelMetricsResponse(BaseModel):
    model_version: str
    train_date: datetime
    mse: float
    mae: float
    mape: float
    baseline_improvement: float
    total_predictions: int


class HistoryRequest(BaseModel):
    limit: int = Field(default=10, ge=1, le=100)
    include_metrics: bool = False


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)