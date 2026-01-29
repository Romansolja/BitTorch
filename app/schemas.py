from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


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
    mae_return: Optional[float] = Field(None, description="Mean Absolute Error (returns)")
    mae_price: Optional[float] = Field(None, description="Mean Absolute Error (USD)")
    directional_accuracy: Optional[float] = Field(None, description="% correct direction")
    avg_confidence_when_correct: Optional[float] = None
    avg_confidence_when_wrong: Optional[float] = None


# --- Backfill models ---

class BackfillRequest(BaseModel):
    """Request model for backfill endpoint"""
    start_date: date
    end_date: date
    store: bool = Field(False, description="Save predictions to database")

    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2024-12-01",
                "end_date": "2024-12-31",
                "store": False
            }
        }


class BackfillDayResult(BaseModel):
    """Single day result from backfill"""
    date: str = Field(..., description="Date D (prediction made using data up to D)")
    target_date: str = Field(..., description="Date D+1 (what we predicted)")
    current_price: float
    predicted_price: float
    actual_price: float
    predicted_return: float
    actual_return: float
    error: float = Field(..., description="Absolute error in return")
    direction_correct: bool


class BackfillResponse(BaseModel):
    """Response model for backfill endpoint"""
    n_days: int
    start_date: str
    end_date: str

    # Model metrics
    mae: float = Field(..., description="Mean Absolute Error (returns)")
    directional_accuracy: float = Field(..., description="Fraction of correct up/down calls")

    # Baseline metrics (for honest comparison)
    baseline_zero_mae: float = Field(..., description="MAE if we predicted 0 every day")
    baseline_lastret_mae: float = Field(..., description="MAE if we predicted last return")
    baseline_lastret_diracc: float = Field(..., description="DirAcc if we predicted last return")
    baseline_majority_diracc: float = Field(..., description="DirAcc if we always predicted majority direction")
    majority_direction: str = Field(..., description="'up' or 'down' - which direction was majority")
    up_day_fraction: float = Field(..., description="Fraction of up days in window")

    # Improvement over baselines
    mae_improvement_vs_zero: float = Field(..., description="(baseline - model) / baseline")
    diracc_improvement_vs_majority: float = Field(..., description="diracc - majority_diracc (can be negative!)")

    stored: bool = Field(..., description="Whether predictions were saved to DB")

    # Optional detailed results
    daily_results: Optional[List[BackfillDayResult]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "n_days": 30,
                "start_date": "2024-12-01",
                "end_date": "2024-12-31",
                "mae": 0.0234,
                "directional_accuracy": 0.533,
                "baseline_zero_mae": 0.0241,
                "baseline_lastret_mae": 0.0312,
                "baseline_lastret_diracc": 0.467,
                "baseline_majority_diracc": 0.533,
                "majority_direction": "up",
                "up_day_fraction": 0.533,
                "mae_improvement_vs_zero": 0.029,
                "diracc_improvement_vs_majority": 0.0,
                "stored": False,
                "daily_results": None
            }
        }