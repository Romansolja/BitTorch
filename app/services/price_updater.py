from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import yfinance as yf

from app.database import SessionLocal, PricePrediction


class PriceUpdater:
    def update_actual_prices(self) -> dict:
        """Fetch actual prices and update past predictions."""
        db = SessionLocal()
        try:
            pending = db.query(PricePrediction) \
                .filter(PricePrediction.actual_price.is_(None)) \
                .filter(PricePrediction.prediction_date <= datetime.utcnow()) \
                .all()

            if not pending:
                return {"updated": 0, "message": "No pending predictions to update"}

            dates = [p.prediction_date.strftime("%Y-%m-%d") for p in pending]
            start = min(dates)
            end = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

            btc = yf.download("BTC-USD", start=start, end=end, progress=False)
            if btc.empty:
                return {"updated": 0, "message": "Could not fetch price data"}

            updated = 0
            for pred in pending:
                date_str = pred.prediction_date.strftime("%Y-%m-%d")
                if date_str in btc.index.strftime("%Y-%m-%d"):
                    actual = float(btc.loc[date_str, "Close"])
                    pred.actual_price = actual

                    if pred.current_price and pred.current_price > 0:
                        pred.actual_return = float(np.log(actual / pred.current_price))

                    if pred.predicted_direction and pred.actual_return is not None:
                        actual_dir = "up" if pred.actual_return > 0 else "down"
                        pred.direction_correct = (pred.predicted_direction == actual_dir)

                    updated += 1

            db.commit()
            return {"updated": updated, "message": f"Updated {updated} predictions"}

        finally:
            db.close()

    def calculate_accuracy_metrics(self) -> Optional[dict]:
        """Calculate accuracy for predictions with known actual prices."""
        db = SessionLocal()
        try:
            completed = db.query(PricePrediction) \
                .filter(PricePrediction.actual_price.isnot(None)) \
                .all()

            if not completed:
                return None

            errors_price = []
            errors_return = []
            directions = []
            conf_correct = []
            conf_wrong = []

            for p in completed:
                if p.predicted_price and p.actual_price:
                    errors_price.append(abs(p.predicted_price - p.actual_price))

                if p.predicted_return is not None and p.actual_return is not None:
                    errors_return.append(abs(p.predicted_return - p.actual_return))

                if p.direction_correct is not None:
                    directions.append(p.direction_correct)
                    if p.confidence:
                        if p.direction_correct:
                            conf_correct.append(p.confidence)
                        else:
                            conf_wrong.append(p.confidence)

            return {
                "total_predictions": len(completed),
                "mae_return": float(np.mean(errors_return)) if errors_return else None,
                "mae_price": float(np.mean(errors_price)) if errors_price else None,
                "directional_accuracy": float(np.mean(directions)) if directions else None,
                "avg_confidence_when_correct": float(np.mean(conf_correct)) if conf_correct else None,
                "avg_confidence_when_wrong": float(np.mean(conf_wrong)) if conf_wrong else None,
            }

        finally:
            db.close()


price_updater = PriceUpdater()