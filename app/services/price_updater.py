from datetime import datetime, timedelta, date
from typing import Optional, List, TYPE_CHECKING
import numpy as np
import pandas as pd
import yfinance as yf

from app.database import SessionLocal, PricePrediction

if TYPE_CHECKING:
    from app.services.prediction import PredictionService


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

            # Flatten MultiIndex if present
            if isinstance(btc.columns, pd.MultiIndex):
                btc.columns = btc.columns.get_level_values(0)

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

    def backfill_historical(
            self,
            start_date: date,
            end_date: date,
            prediction_service: "PredictionService",
            store: bool = False,
            include_daily: bool = True,
    ) -> dict:
        """
        Run model on historical data day-by-day, simulating live predictions.

        CRITICAL: For each day D, we only use data up to and including D
        to predict the return for D+1. No future data leakage.

        Args:
            start_date: First day D to make predictions from
            end_date: Last day D to make predictions from
            prediction_service: Service with loaded model and scaler
            store: Whether to save predictions to database
            include_daily: Whether to include per-day results in response

        Returns:
            Metrics comparing model to baselines
        """
        from app.services.prediction import make_features
        import torch

        if prediction_service.model is None:
            return {"error": "Model not loaded"}

        # Dynamic buffer for feature warmup
        # Must be >= max rolling window in make_features() + seq_len
        # Current features use: rolling(30) for MA, rolling(14) for vol/RSI, diff(14) for mom
        # SAFETY NOTE: If you add longer rolling windows to make_features(), update this!
        max_rolling_window = 30  # from ma30_dist
        buffer_days = max(90, prediction_service.seq_len + max_rolling_window + 10)

        fetch_start = start_date - timedelta(days=buffer_days)
        fetch_end = end_date + timedelta(days=2)  # Need D+1 for actuals

        # Download full range once
        df = yf.download(
            "BTC-USD",
            start=fetch_start.strftime("%Y-%m-%d"),
            end=fetch_end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            return {"error": "Could not fetch price data"}

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        # Build features on the FULL downloaded range
        # CRITICAL ASSUMPTION: make_features() uses ONLY backward-looking transforms:
        #   - .diff() looks back
        #   - .rolling() defaults to right-aligned (past only)
        #   - .shift(n) with n>0 looks back
        # If you ever add a feature that uses shift(-n), rolling(center=True),
        # or any forward-looking transform, this backfill becomes invalid!
        # Consider adding a unit test that verifies no features leak future data.
        df_feat = make_features(df)

        # Date column is now set inside make_features for alignment safety

        seq_len = prediction_service.seq_len
        feature_cols = prediction_service.feature_cols
        scaler = prediction_service.scaler
        model = prediction_service.model
        device = prediction_service.device

        results = []

        # Iterate through each day D in [start_date, end_date]
        current_date = start_date
        while current_date <= end_date:
            # Find index of day D in dataframe
            day_mask = df_feat["date"] == current_date
            if not day_mask.any():
                current_date += timedelta(days=1)
                continue

            day_idx = df_feat[day_mask].index[0]

            # Check we have enough history for seq_len
            if day_idx < seq_len - 1:
                current_date += timedelta(days=1)
                continue

            # Check D+1 exists for actual comparison
            next_date = current_date + timedelta(days=1)
            next_mask = df_feat["date"] == next_date
            if not next_mask.any():
                current_date += timedelta(days=1)
                continue

            next_idx = df_feat[next_mask].index[0]

            # --- NO FUTURE DATA BEYOND HERE ---
            # Slice features up to and including day D
            X_slice = df_feat[feature_cols].iloc[:day_idx + 1].values.astype(np.float32)

            # Check for NaNs in the sequence we'll use
            X_seq_raw = X_slice[-seq_len:]
            if np.isnan(X_seq_raw).any():
                current_date += timedelta(days=1)
                continue

            # Scale with PRODUCTION scaler (no refitting!)
            X_scaled = scaler.transform(X_seq_raw)

            # Get current price (Close on day D)
            current_price = float(df_feat["Close"].iloc[day_idx])

            # Model prediction
            X_tensor = torch.tensor(
                X_scaled.reshape(1, seq_len, -1),
                dtype=torch.float32
            ).to(device)

            with torch.no_grad():
                pred_return = float(model(X_tensor).cpu().numpy()[0, 0])

            predicted_price = current_price * np.exp(pred_return)

            # --- Get D+1 actual ---
            actual_price = float(df_feat["Close"].iloc[next_idx])
            actual_return = float(np.log(actual_price / current_price))

            # Last observed return (for baseline)
            # This is ret_1 on day D, which is log(Close[D]/Close[D-1])
            last_return = float(df_feat["ret_1"].iloc[day_idx])

            # Record result
            error = abs(pred_return - actual_return)
            direction_correct = bool(np.sign(pred_return) == np.sign(actual_return))

            results.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "target_date": next_date.strftime("%Y-%m-%d"),
                "current_price": current_price,
                "predicted_price": predicted_price,
                "actual_price": actual_price,
                "predicted_return": pred_return,
                "actual_return": actual_return,
                "last_return": last_return,  # for baseline calc
                "error": error,
                "direction_correct": direction_correct,
            })

            # Optionally store to database
            if store:
                self._store_backfill_prediction(
                    current_date=current_date,
                    target_date=next_date,
                    current_price=current_price,
                    predicted_price=predicted_price,
                    predicted_return=pred_return,
                    actual_price=actual_price,
                    actual_return=actual_return,
                    direction_correct=direction_correct,
                )

            current_date += timedelta(days=1)

        if not results:
            return {"error": "No valid prediction days in range"}

        # Compute metrics
        pred_returns = np.array([r["predicted_return"] for r in results])
        actual_returns = np.array([r["actual_return"] for r in results])
        last_returns = np.array([r["last_return"] for r in results])

        mae = float(np.mean(np.abs(pred_returns - actual_returns)))
        diracc = float(np.mean([r["direction_correct"] for r in results]))

        # Baseline: predict 0 (MAE only - diracc is meaningless for "predict 0")
        zero_mae = float(np.mean(np.abs(actual_returns)))

        # Baseline: predict last return (persistence)
        lastret_mae = float(np.mean(np.abs(last_returns - actual_returns)))
        lastret_diracc = float(np.mean(np.sign(last_returns) == np.sign(actual_returns)))

        # Baseline: always predict majority direction in this window
        # This is the "naive classifier" baseline - better than coin flip if imbalanced
        up_fraction = float(np.mean(actual_returns > 0))
        majority_is_up = up_fraction >= 0.5
        majority_direction = "up" if majority_is_up else "down"
        majority_diracc = up_fraction if majority_is_up else (1 - up_fraction)

        # Improvement metrics
        mae_improvement = (zero_mae - mae) / zero_mae if zero_mae > 0 else 0
        diracc_improvement_vs_majority = diracc - majority_diracc

        response = {
            "n_days": len(results),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "mae": mae,
            "directional_accuracy": diracc,
            "baseline_zero_mae": zero_mae,
            "baseline_lastret_mae": lastret_mae,
            "baseline_lastret_diracc": lastret_diracc,
            "baseline_majority_diracc": majority_diracc,
            "majority_direction": majority_direction,
            "up_day_fraction": up_fraction,
            "mae_improvement_vs_zero": mae_improvement,
            "diracc_improvement_vs_majority": diracc_improvement_vs_majority,
            "stored": store,
        }

        if include_daily:
            response["daily_results"] = [
                {
                    "date": r["date"],
                    "target_date": r["target_date"],
                    "current_price": r["current_price"],
                    "predicted_price": r["predicted_price"],
                    "actual_price": r["actual_price"],
                    "predicted_return": r["predicted_return"],
                    "actual_return": r["actual_return"],
                    "error": r["error"],
                    "direction_correct": r["direction_correct"],
                }
                for r in results
            ]

        return response

    def _store_backfill_prediction(
            self,
            current_date: date,
            target_date: date,
            current_price: float,
            predicted_price: float,
            predicted_return: float,
            actual_price: float,
            actual_return: float,
            direction_correct: bool,
    ) -> None:
        """Store a backfilled prediction to the database."""
        db = SessionLocal()
        try:
            db_prediction = PricePrediction(
                current_price=current_price,
                predicted_price=predicted_price,
                predicted_return=predicted_return,
                predicted_direction="up" if predicted_return > 0 else "down",
                confidence=None,  # Not computed in backfill
                prediction_date=datetime.combine(target_date, datetime.min.time()),
                actual_price=actual_price,
                actual_return=actual_return,
                direction_correct=direction_correct,
                model_version="v2.0-backfill",
                created_at=datetime.utcnow(),
            )
            db.add(db_prediction)
            db.commit()
        finally:
            db.close()


price_updater = PriceUpdater()