import torch
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os

from app.models.ml_models import BitcoinLSTM
from app.config import MODEL_PATH, SEQUENCE_LENGTH


class PredictionService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = MinMaxScaler()
        self.seq_len = SEQUENCE_LENGTH

    def load_model(self):
        """Load the trained model from disk"""
        self.model = BitcoinLSTM().to(self.device)

        # Check if model file exists
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.eval()
            return True
        else:
            # For now, use the model from original main.py
            # Copy best_model.pth to data/models/ directory
            return False

    def get_latest_data(self, days=30):
        """Fetch latest Bitcoin data"""
        btc = yf.download("BTC-USD", period=f"{days}d", interval="1d",
                          auto_adjust=True, progress=False)
        return btc["Close"].values.reshape(-1, 1)

    def predict_next_day(self):
        """Make prediction for next day"""
        if self.model is None:
            return None

        # Get recent data
        prices = self.get_latest_data(days=30)

        # Fit scaler on all data, use last seq_len for prediction
        self.scaler.fit(prices)
        prices_scaled = self.scaler.transform(prices)

        # Get last sequence
        last_seq = prices_scaled[-self.seq_len:].reshape(1, self.seq_len, 1)
        inp = torch.tensor(last_seq).float().to(self.device)

        # Predict
        with torch.no_grad():
            pred_scaled = self.model(inp).cpu().numpy()

        # Inverse transform
        pred_price = self.scaler.inverse_transform(pred_scaled)[0][0]
        current_price = prices[-1][0]

        return {
            "current_price": float(current_price),
            "predicted_price": float(pred_price),
            "change_amount": float(pred_price - current_price),
            "change_percent": float((pred_price / current_price - 1) * 100),
            "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


# Create singleton instance
prediction_service = PredictionService()