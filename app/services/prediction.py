import json
import logging
import pickle
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import torch

from app.config import (
    CONFIDENCE_WINDOW,
    LATEST_FETCH_DAYS,
    METADATA_PATH,
    MODEL_PATH,
    RIDGE_PATH,
    SCALER_PATH,
    SEQUENCE_LENGTH,
)
from app.database import PricePrediction
from app.models.ml_models import LSTMRegressor
from app.services.yf_client import download as yf_download, flatten_multiindex

logger = logging.getLogger(__name__)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index - backward-looking only."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature table (same as training). All rolling windows are backward-looking."""
    out = df.copy()
    out = flatten_multiindex(out)

    if "Date" in out.columns:
        out["date"] = pd.to_datetime(out["Date"]).dt.date

    out["log_close"] = np.log(out["Close"])
    out["ret1"] = out["log_close"].diff()
    out["target_ret_next"] = out["ret1"].shift(-1)

    out["ret_1"] = out["ret1"]
    out["ret_2"] = out["ret1"].shift(1)
    out["ret_3"] = out["ret1"].shift(2)

    out["vol_7"] = out["ret1"].rolling(7).std()
    out["vol_14"] = out["ret1"].rolling(14).std()

    out["mom_7"] = out["log_close"].diff(7)
    out["mom_14"] = out["log_close"].diff(14)

    out["rsi_14"] = rsi(out["Close"], 14)

    ma_10 = out["Close"].rolling(10).mean()
    ma_30 = out["Close"].rolling(30).mean()
    out["ma10_dist"] = (out["Close"] - ma_10) / (ma_10 + 1e-12)
    out["ma30_dist"] = (out["Close"] - ma_30) / (ma_30 + 1e-12)

    if {"High", "Low", "Close"}.issubset(out.columns):
        out["hl_range"] = (out["High"] - out["Low"]) / (out["Close"] + 1e-12)

    if "Volume" in out.columns:
        out["vol_chg"] = np.log((out["Volume"] + 1.0) / (out["Volume"].shift(1) + 1.0))

    return out


class PredictionService:
    """Loads production artifacts and serves next-day return predictions.

    Not thread-safe during load_model(); intended to load once in FastAPI lifespan.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LSTMRegressor] = None
        self.scaler = None
        self.feature_cols: list = []
        self.seq_len = SEQUENCE_LENGTH
        self.ridge_model = None
        self.training_cutoff: Optional[str] = None  # ISO date from metadata

    def load_model(self) -> bool:
        missing = [p for p in (MODEL_PATH, SCALER_PATH, METADATA_PATH) if not p.exists()]
        if missing:
            logger.error("Cannot load model; missing artifacts: %s", missing)
            return False

        with open(METADATA_PATH) as f:
            meta = json.load(f)

        self.feature_cols = meta["feature_cols"]
        self.seq_len = meta["seq_len"]
        self.training_cutoff = meta.get("training_cutoff")
        if self.training_cutoff is None:
            logger.warning(
                "metadata.json has no training_cutoff; backfill guard cannot detect "
                "overlap with training window. Retrain to populate this field.",
            )

        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        model_cfg = meta.get("model", {})
        self.model = LSTMRegressor(
            n_features=len(self.feature_cols),
            hidden=model_cfg.get("hidden", 64),
            layers=model_cfg.get("layers", 2),
            dropout=model_cfg.get("dropout", 0.2),
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        if RIDGE_PATH.exists():
            with open(RIDGE_PATH, "rb") as f:
                self.ridge_model = pickle.load(f)
            logger.info("Loaded Ridge baseline from %s", RIDGE_PATH)
        else:
            logger.warning(
                "Ridge baseline not found at %s; ridge metrics will be omitted",
                RIDGE_PATH,
            )

        logger.info(
            "Model loaded (features=%d, seq_len=%d, training_cutoff=%s, device=%s)",
            len(self.feature_cols),
            self.seq_len,
            self.training_cutoff,
            self.device,
        )
        return True

    def get_latest_data(self, days: int = LATEST_FETCH_DAYS) -> pd.DataFrame:
        df = yf_download("BTC-USD", period=f"{days}d", interval="1d")
        return df.reset_index()

    def predict_next_day(self) -> Optional[dict]:
        if self.model is None:
            return None

        df = self.get_latest_data(days=LATEST_FETCH_DAYS)
        df_feat = make_features(df)
        df_feat = df_feat.dropna(subset=self.feature_cols).reset_index(drop=True)

        if len(df_feat) < self.seq_len:
            logger.warning(
                "Not enough rows after feature build: %d < %d", len(df_feat), self.seq_len
            )
            return None

        current_price = float(df["Close"].iloc[-1])

        X = df_feat[self.feature_cols].values.astype(np.float32)
        X_scaled = self.scaler.transform(X)

        X_seq = X_scaled[-self.seq_len:].reshape(1, self.seq_len, -1)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred_return = float(self.model(X_tensor).cpu().numpy()[0, 0])

        predicted_price = current_price * np.exp(pred_return)
        pct_change = (np.exp(pred_return) - 1) * 100
        direction = "up" if pred_return > 0 else "down"

        confidence, confidence_valid = self._compute_confidence(df_feat, pred_return)
        agreement = self._compute_agreement(X_scaled)

        target_date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

        result = {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_return": pred_return,
            "change_percent": pct_change,
            "direction": direction,
            "confidence": confidence,
            "confidence_valid": confidence_valid,
            "model_agreement": agreement,
            "prediction_date": target_date,
        }

        if self.ridge_model is not None:
            X_ridge = X_scaled[-1:]
            ridge_pred_return = float(self.ridge_model.predict(X_ridge)[0])
            result["ridge_predicted_return"] = ridge_pred_return
            result["ridge_predicted_price"] = current_price * np.exp(ridge_pred_return)
            result["ridge_direction"] = "up" if ridge_pred_return > 0 else "down"

        return result

    def _compute_confidence(self, df_feat: pd.DataFrame, pred_return: float):
        recent_vol = df_feat["ret_1"].iloc[-CONFIDENCE_WINDOW:].std()
        if recent_vol < 1e-8:
            return 0.5, False
        z_score = abs(pred_return) / recent_vol
        value = float(np.clip(1 / (1 + np.exp(-z_score + 1)), 0.0, 1.0))
        return value, True

    def _compute_agreement(self, X_scaled: np.ndarray) -> float:
        n_windows = min(5, len(X_scaled) - self.seq_len)
        if n_windows < 2:
            return 0.5

        predictions = []
        for i in range(n_windows):
            start = len(X_scaled) - self.seq_len - (n_windows - 1 - i)
            end = start + self.seq_len
            X_seq = X_scaled[start:end].reshape(1, self.seq_len, -1)
            X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                pred = float(self.model(X_tensor).cpu().numpy()[0, 0])
            predictions.append(pred)

        final_sign = np.sign(predictions[-1])
        same_sign = sum(1 for p in predictions if np.sign(p) == final_sign)
        return float(same_sign / len(predictions))

    def save_prediction(self, prediction_data: dict, db) -> int:
        prediction_date = datetime.strptime(
            prediction_data["prediction_date"], "%Y-%m-%d"
        ).replace(tzinfo=timezone.utc)

        db_prediction = PricePrediction(
            current_price=prediction_data["current_price"],
            predicted_price=prediction_data["predicted_price"],
            predicted_return=prediction_data["predicted_return"],
            predicted_direction=prediction_data["direction"],
            confidence=prediction_data["confidence"],
            prediction_date=prediction_date,
            created_at=datetime.now(timezone.utc),
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        logger.info("Saved prediction id=%d for %s", db_prediction.id, prediction_date.date())
        return db_prediction.id

    def get_prediction_history(self, db, limit: int = 10):
        return (
            db.query(PricePrediction)
            .order_by(PricePrediction.created_at.desc())
            .limit(limit)
            .all()
        )


prediction_service = PredictionService()
