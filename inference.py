# Load production artifacts and make predictions

import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from main import LSTMRegressor, make_features


@dataclass
class Prediction:
    predicted_return: float
    predicted_pct_change: float
    current_price: float
    predicted_price: float
    direction: str
    confidence: float
    model_agreement: float


class BitTorchPredictor:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[LSTMRegressor] = None
        self.scaler = None
        self.meta: dict = {}
        self.feature_cols: list = []
        self.seq_len: int = 30

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        meta_path = self.artifacts_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}")

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.feature_cols = self.meta["feature_cols"]
        self.seq_len = self.meta["seq_len"]

        scaler_path = self.artifacts_dir / "feature_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing {scaler_path}")

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        model_path = self.artifacts_dir / "production_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}")

        model_cfg = self.meta.get("model", {})
        self.model = LSTMRegressor(
            n_features=len(self.feature_cols),
            hidden=model_cfg.get("hidden", 64),
            layers=model_cfg.get("layers", 2),
            dropout=model_cfg.get("dropout", 0.2),
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def predict(self, df: pd.DataFrame) -> Prediction:
        """
        Make prediction from OHLCV dataframe.
        Expects at least seq_len + 30 rows (for feature warmup).
        """
        df_feat = make_features(df)

        missing = set(self.feature_cols) - set(df_feat.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        df_feat = df_feat.dropna(subset=self.feature_cols).reset_index(drop=True)

        if len(df_feat) < self.seq_len:
            raise ValueError(f"Need at least {self.seq_len} rows, got {len(df_feat)}")

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

        return Prediction(
            predicted_return=pred_return,
            predicted_pct_change=pct_change,
            current_price=current_price,
            predicted_price=predicted_price,
            direction=direction,
            confidence=confidence,
            model_agreement=agreement,
        )

    def _compute_confidence(self, df_feat: pd.DataFrame, pred_return: float) -> float:
        """Signal strength vs recent volatility."""
        recent_vol = df_feat["ret_1"].iloc[-30:].std()

        if recent_vol < 1e-8:
            return 0.5

        z_score = abs(pred_return) / recent_vol
        confidence = 1 / (1 + np.exp(-z_score + 1))

        return float(np.clip(confidence, 0.0, 1.0))

    def _compute_agreement(self, X_scaled: np.ndarray) -> float:
        """Consistency across recent overlapping windows."""
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


def predict_from_yahoo(
        symbol: str = "BTC-USD",
        period: str = "60d",
        artifacts_dir: str = "artifacts",
) -> Prediction:
    """Download fresh data and predict."""
    import yfinance as yf

    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    predictor = BitTorchPredictor(artifacts_dir=artifacts_dir)
    return predictor.predict(df)


def main():
    print("=" * 60)
    print("BitTorch Inference")
    print("=" * 60)

    pred = predict_from_yahoo()

    print(f"\nCurrent price:     ${pred.current_price:,.2f}")
    print(f"Predicted price:   ${pred.predicted_price:,.2f}")
    print(f"Expected change:   {pred.predicted_pct_change:+.2f}%")
    print(f"Direction:         {pred.direction.upper()}")
    print(f"Signal strength:   {pred.confidence:.2f}")
    print(f"Model agreement:   {pred.model_agreement:.2f}")

    print("\n" + "-" * 60)
    print("NOTE: Research tool only, not financial advice.")
    print("-" * 60)


if __name__ == "__main__":
    main()