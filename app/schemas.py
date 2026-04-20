from datetime import date, datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import BACKFILL_MAX_DAYS, BTC_USD_START_DATE


class PredictionResponse(BaseModel):
    """Response model for /predict/next-day."""

    current_price: float = Field(..., gt=0)
    predicted_price: float = Field(..., gt=0)
    predicted_return: float = Field(..., description="Log return")
    change_percent: float
    direction: str = Field(..., description="up or down")
    confidence: float = Field(..., ge=0, le=1, description="Signal strength (not accuracy)")
    confidence_valid: bool = Field(
        ...,
        description="False when vol estimate degenerate; confidence is a fallback.",
    )
    model_agreement: float = Field(..., ge=0, le=1)
    prediction_date: str

    ridge_predicted_return: Optional[float] = None
    ridge_predicted_price: Optional[float] = None
    ridge_direction: Optional[str] = None

    prediction_id: Optional[int] = None
    saved: bool = False


class AccuracyMetrics(BaseModel):
    """Response model for /predictions/accuracy."""

    total_predictions: int
    mae_return: Optional[float] = Field(None, description="Mean Absolute Error (returns)")
    mae_price: Optional[float] = Field(None, description="Mean Absolute Error (USD)")
    directional_accuracy: Optional[float] = Field(None, description="% correct direction")
    avg_confidence_when_correct: Optional[float] = None
    avg_confidence_when_wrong: Optional[float] = None


class BackfillRequest(BaseModel):
    """Request model for /predictions/backfill."""

    start_date: date
    end_date: date
    store: bool = Field(False, description="Save predictions to database")

    @field_validator("start_date")
    @classmethod
    def _start_after_btc_genesis(cls, v: date) -> date:
        btc_start = datetime.strptime(BTC_USD_START_DATE, "%Y-%m-%d").date()
        if v < btc_start:
            raise ValueError(f"start_date must be >= {BTC_USD_START_DATE} (yfinance BTC-USD coverage)")
        return v

    @field_validator("end_date")
    @classmethod
    def _end_not_in_future(cls, v: date) -> date:
        today = datetime.now(timezone.utc).date()
        if v > today:
            raise ValueError(f"end_date ({v}) cannot be in the future (today UTC = {today})")
        return v

    @model_validator(mode="after")
    def _range_valid(self) -> "BackfillRequest":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")
        # Inclusive count — backfill_historical iterates `while current <= end`,
        # so same-day [D, D] is 1 day, not 0.
        days = (self.end_date - self.start_date).days + 1
        if days > BACKFILL_MAX_DAYS:
            raise ValueError(
                f"Backfill range is {days} inclusive days; maximum is {BACKFILL_MAX_DAYS}"
            )
        return self


class BackfillDayResult(BaseModel):
    date: str = Field(..., description="Date D (prediction made using data up to D)")
    target_date: str = Field(..., description="Date D+1 (what we predicted)")
    current_price: float
    predicted_price: float
    actual_price: float
    predicted_return: float
    actual_return: float
    error: float = Field(..., description="Absolute error in return")
    direction_correct: bool
    ridge_predicted_return: Optional[float] = None
    ridge_direction_correct: Optional[bool] = None


class BackfillSkipped(BaseModel):
    missing_market_day: int
    insufficient_history: int
    no_next_day_actual: int
    nan_features: int


class BackfillResponse(BaseModel):
    """Response model for /predictions/backfill."""

    n_days: int
    start_date: str
    end_date: str

    mae: float
    directional_accuracy: float

    baseline_zero_mae: float
    baseline_lastret_mae: float
    baseline_lastret_diracc: float
    baseline_majority_diracc: float
    majority_direction: str
    up_day_fraction: float

    ridge_mae: Optional[float] = None
    ridge_diracc: Optional[float] = None

    mae_improvement_vs_zero: Optional[float] = Field(
        None,
        description="(zero_mae - mae) / zero_mae. Null when zero_mae == 0 (undefined).",
    )
    diracc_improvement_vs_majority: float

    stored: bool
    skipped: BackfillSkipped

    daily_results_truncated: Optional[bool] = None
    daily_results: Optional[List[BackfillDayResult]] = None


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    torch_version: str
    model_loaded: bool
    model_type: str


class RootResponse(BaseModel):
    message: str


class UpdateActualPricesResponse(BaseModel):
    updated: int
    message: str


class PredictionHistoryItem(BaseModel):
    id: int
    prediction_date: Optional[str]
    current_price: Optional[float]
    predicted_price: Optional[float]
    predicted_return: Optional[float]
    direction: Optional[str]
    confidence: Optional[float]
    actual_price: Optional[float]
    actual_return: Optional[float]
    direction_correct: Optional[bool]
    created_at: Optional[str]


class PredictionHistoryResponse(BaseModel):
    count: int
    predictions: List[PredictionHistoryItem]
