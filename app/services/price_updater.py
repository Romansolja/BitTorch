import yfinance as yf
from datetime import datetime, timedelta
from app.database import SessionLocal, PricePrediction
from sqlalchemy import and_


class PriceUpdater:
    def update_actual_prices(self):
        """Check and update actual prices for past predictions"""
        db = SessionLocal()
        try:
            # First, let's see ALL predictions in the database
            all_predictions = db.query(PricePrediction).all()
            print(f"Total predictions in database: {len(all_predictions)}")

            # Get predictions that need actual price updates
            # Changed to be more inclusive - any prediction from the past without actual price
            now = datetime.now()

            predictions = db.query(PricePrediction).filter(
                and_(
                    PricePrediction.actual_price.is_(None),
                    PricePrediction.prediction_date <= now  # Any past prediction
                )
            ).all()

            print(f"Found {len(predictions)} predictions needing updates")

            if not predictions:
                # Let's see what dates we have
                for p in all_predictions[:5]:  # Show first 5
                    print(f"  - Prediction ID {p.id}: pred_date={p.prediction_date}, actual_price={p.actual_price}")
                return {"message": "No predictions to update", "count": 0, "total_in_db": len(all_predictions)}

            # Get Bitcoin price history for the date range
            earliest_date = min(p.prediction_date for p in predictions)
            btc = yf.download("BTC-USD",
                              start=earliest_date.date(),
                              end=now.date() + timedelta(days=1),
                              interval="1d",
                              progress=False)

            updated_count = 0
            for pred in predictions:
                # Find the actual price for the prediction date
                pred_date = pred.prediction_date.date()
                if pred_date in btc.index.date:
                    actual_price = float(btc.loc[btc.index.date == pred_date, 'Close'].iloc[0])
                    pred.actual_price = actual_price
                    updated_count += 1
                    print(f"Updated prediction {pred.id} with actual price: ${actual_price:.2f}")

            db.commit()
            return {"message": f"Updated {updated_count} predictions", "count": updated_count}

        except Exception as e:
            db.rollback()
            print(f"Error in update_actual_prices: {e}")
            return {"error": str(e)}
        finally:
            db.close()

    def calculate_accuracy_metrics(self):
        """Calculate accuracy metrics for predictions with actual prices"""
        db = SessionLocal()
        try:
            # Get all predictions with actual prices
            predictions = db.query(PricePrediction).filter(
                PricePrediction.actual_price.isnot(None)
            ).all()

            if not predictions:
                return None

            errors = []
            percentage_errors = []

            for pred in predictions:
                error = abs(pred.predicted_price - pred.actual_price)
                errors.append(error)
                percentage_errors.append((error / pred.actual_price) * 100)

            return {
                "total_predictions": len(predictions),
                "mae": sum(errors) / len(errors),
                "mape": sum(percentage_errors) / len(percentage_errors),
                "best_prediction_error": min(errors),
                "worst_prediction_error": max(errors)
            }
        finally:
            db.close()


price_updater = PriceUpdater()