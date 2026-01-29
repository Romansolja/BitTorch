# app/services/prediction.py
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from app.database import SessionLocal, PricePrediction
from app.models.ml_models import LSTMRegressor
from app.config import MODEL_PATH, SCALER_PATH, METADATA_PATH, SEQUENCE_LENGTH


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index - backward-looking only."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature table (same as training).
    All rolling windows are backward-looking (past-only).
    """
    out = df.copy()

    # Flatten MultiIndex columns if present
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    # Preserve date for backfill alignment
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
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LSTMRegressor] = None
        self.scaler = None
        self.feature_cols: list = []
        self.seq_len = SEQUENCE_LENGTH

    def load_model(self) -> bool:
        """Load production model and artifacts."""
        if not all(p.exists() for p in [MODEL_PATH, SCALER_PATH, METADATA_PATH]):
            return False

        with open(METADATA_PATH) as f:
            meta = json.load(f)

        self.feature_cols = meta["feature_cols"]
        self.seq_len = meta["seq_len"]

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
        return True

    def get_latest_data(self, days: int = 60) -> pd.DataFrame:
        """Fetch latest Bitcoin data."""
        df = yf.download(
            "BTC-USD",
            period=f"{days}d",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df.reset_index()

    def predict_next_day(self) -> Optional[dict]:
        """Make prediction for next day's return and price."""
        if self.model is None:
            return None

        df = self.get_latest_data(days=60)
        df_feat = make_features(df)
        df_feat = df_feat.dropna(subset=self.feature_cols).reset_index(drop=True)

        if len(df_feat) < self.seq_len:
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

        confidence = self._compute_confidence(df_feat, pred_return)
        agreement = self._compute_agreement(X_scaled)

        return {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_return": pred_return,
            "change_percent": pct_change,
            "direction": direction,
            "confidence": confidence,
            "model_agreement": agreement,
            "prediction_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        }

    def _compute_confidence(self, df_feat: pd.DataFrame, pred_return: float) -> float:
        recent_vol = df_feat["ret_1"].iloc[-30:].std()
        if recent_vol < 1e-8:
            return 0.5
        z_score = abs(pred_return) / recent_vol
        return float(np.clip(1 / (1 + np.exp(-z_score + 1)), 0.0, 1.0))

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

    def save_prediction(self, prediction_data: dict) -> int:
        """Save prediction to database."""
        db = SessionLocal()
        try:
            db_prediction = PricePrediction(
                current_price=prediction_data["current_price"],
                predicted_price=prediction_data["predicted_price"],
                predicted_return=prediction_data["predicted_return"],
                predicted_direction=prediction_data["direction"],
                confidence=prediction_data["confidence"],
                prediction_date=datetime.strptime(prediction_data["prediction_date"], "%Y-%m-%d"),
                created_at=datetime.utcnow(),
            )
            db.add(db_prediction)
            db.commit()
            db.refresh(db_prediction)
            return db_prediction.id
        finally:
            db.close()

    def get_prediction_history(self, limit: int = 10):
        """Get recent predictions."""
        db = SessionLocal()
        try:
            return db.query(PricePrediction) \
                .order_by(PricePrediction.created_at.desc()) \
                .limit(limit) \
                .all()
        finally:
            db.close()


prediction_service = PredictionService()