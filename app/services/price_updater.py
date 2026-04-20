import logging
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import torch

from app.config import (
    BACKFILL_FEATURE_BUFFER_DAYS,
    BACKFILL_FEATURE_MAX_ROLLING,
    BACKFILL_MAX_DAILY_RESULTS,
)
from app.database import PricePrediction
from app.services.yf_client import download as yf_download

if TYPE_CHECKING:
    from app.services.prediction import PredictionService

logger = logging.getLogger(__name__)


class PriceUpdater:
    def update_actual_prices(self, db) -> dict:
        pending = (
            db.query(PricePrediction)
            .filter(PricePrediction.actual_price.is_(None))
            .filter(PricePrediction.prediction_date <= datetime.now(timezone.utc))
            .all()
        )

        if not pending:
            return {"updated": 0, "message": "No pending predictions to update"}

        dates = [p.prediction_date.strftime("%Y-%m-%d") for p in pending]
        start = min(dates)
        end = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

        btc = yf_download("BTC-USD", start=start, end=end)
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
                    pred.direction_correct = pred.predicted_direction == actual_dir

                updated += 1

        db.commit()
        logger.info("Updated actuals for %d predictions", updated)
        return {"updated": updated, "message": f"Updated {updated} predictions"}

    def calculate_accuracy_metrics(self, db) -> Optional[dict]:
        completed = (
            db.query(PricePrediction)
            .filter(PricePrediction.actual_price.isnot(None))
            .all()
        )

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
                    (conf_correct if p.direction_correct else conf_wrong).append(p.confidence)

        return {
            "total_predictions": len(completed),
            "mae_return": float(np.mean(errors_return)) if errors_return else None,
            "mae_price": float(np.mean(errors_price)) if errors_price else None,
            "directional_accuracy": float(np.mean(directions)) if directions else None,
            "avg_confidence_when_correct": float(np.mean(conf_correct)) if conf_correct else None,
            "avg_confidence_when_wrong": float(np.mean(conf_wrong)) if conf_wrong else None,
        }

    def backfill_historical(
        self,
        start_date: date,
        end_date: date,
        prediction_service: "PredictionService",
        store: bool = False,
        include_daily: bool = True,
        db=None,
    ) -> dict:
        """Run model on historical data day-by-day, simulating live predictions.

        For each day D, only data up to and including D is used to predict D+1.
        No future data leakage.
        """
        from app.services.prediction import make_features

        if prediction_service.model is None:
            return {"error": "Model not loaded"}

        cutoff_iso = prediction_service.training_cutoff
        if cutoff_iso:
            try:
                cutoff = datetime.strptime(cutoff_iso, "%Y-%m-%d").date()
                if start_date <= cutoff:
                    return {
                        "error": (
                            f"start_date ({start_date}) is inside the production training "
                            f"window (training_cutoff={cutoff_iso}). Backfill is only valid "
                            "on data the model has not seen. Request start_date > cutoff."
                        )
                    }
            except ValueError:
                logger.warning("training_cutoff in metadata is not ISO-8601: %r", cutoff_iso)

        max_rolling_window = BACKFILL_FEATURE_MAX_ROLLING
        buffer_days = max(
            BACKFILL_FEATURE_BUFFER_DAYS,
            prediction_service.seq_len + max_rolling_window + 10,
        )

        fetch_start = start_date - timedelta(days=buffer_days)
        fetch_end = end_date + timedelta(days=2)

        df = yf_download(
            "BTC-USD",
            start=fetch_start.strftime("%Y-%m-%d"),
            end=fetch_end.strftime("%Y-%m-%d"),
            interval="1d",
        )

        if df.empty:
            return {"error": "Could not fetch price data"}

        df = df.reset_index()
        df_feat = make_features(df)

        seq_len = prediction_service.seq_len
        feature_cols = prediction_service.feature_cols
        scaler = prediction_service.scaler
        model = prediction_service.model
        device = prediction_service.device

        results = []
        skipped_missing_day = 0
        skipped_insufficient_history = 0
        skipped_no_next_day = 0
        skipped_nan_features = 0

        current_date = start_date
        while current_date <= end_date:
            day_mask = df_feat["date"] == current_date
            if not day_mask.any():
                skipped_missing_day += 1
                current_date += timedelta(days=1)
                continue

            day_idx = df_feat[day_mask].index[0]

            if day_idx < seq_len - 1:
                skipped_insufficient_history += 1
                current_date += timedelta(days=1)
                continue

            next_date = current_date + timedelta(days=1)
            next_mask = df_feat["date"] == next_date
            if not next_mask.any():
                skipped_no_next_day += 1
                current_date += timedelta(days=1)
                continue

            next_idx = df_feat[next_mask].index[0]

            X_slice = df_feat[feature_cols].iloc[: day_idx + 1].values.astype(np.float32)
            X_seq_raw = X_slice[-seq_len:]
            if np.isnan(X_seq_raw).any():
                skipped_nan_features += 1
                current_date += timedelta(days=1)
                continue

            X_scaled = scaler.transform(X_seq_raw)
            current_price = float(df_feat["Close"].iloc[day_idx])

            X_tensor = torch.tensor(
                X_scaled.reshape(1, seq_len, -1),
                dtype=torch.float32,
            ).to(device)

            with torch.no_grad():
                pred_return = float(model(X_tensor).cpu().numpy()[0, 0])

            predicted_price = current_price * np.exp(pred_return)

            ridge_pred_return = None
            if prediction_service.ridge_model is not None:
                X_ridge = X_scaled[-1:]
                ridge_pred_return = float(prediction_service.ridge_model.predict(X_ridge)[0])

            actual_price = float(df_feat["Close"].iloc[next_idx])
            actual_return = float(np.log(actual_price / current_price))

            last_return = float(df_feat["ret_1"].iloc[day_idx])

            error = abs(pred_return - actual_return)
            direction_correct = bool(np.sign(pred_return) == np.sign(actual_return))

            ridge_direction_correct = None
            if ridge_pred_return is not None:
                ridge_direction_correct = bool(
                    np.sign(ridge_pred_return) == np.sign(actual_return)
                )

            results.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "target_date": next_date.strftime("%Y-%m-%d"),
                "current_price": current_price,
                "predicted_price": predicted_price,
                "actual_price": actual_price,
                "predicted_return": pred_return,
                "actual_return": actual_return,
                "last_return": last_return,
                "error": error,
                "direction_correct": direction_correct,
                "ridge_predicted_return": ridge_pred_return,
                "ridge_direction_correct": ridge_direction_correct,
            })

            if store and db is not None:
                self._store_backfill_prediction(
                    db=db,
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

        pred_returns = np.array([r["predicted_return"] for r in results])
        actual_returns = np.array([r["actual_return"] for r in results])
        last_returns = np.array([r["last_return"] for r in results])

        mae = float(np.mean(np.abs(pred_returns - actual_returns)))
        diracc = float(np.mean([r["direction_correct"] for r in results]))

        zero_mae = float(np.mean(np.abs(actual_returns)))
        lastret_mae = float(np.mean(np.abs(last_returns - actual_returns)))
        lastret_diracc = float(np.mean(np.sign(last_returns) == np.sign(actual_returns)))

        up_fraction = float(np.mean(actual_returns > 0))
        majority_is_up = up_fraction >= 0.5
        majority_direction = "up" if majority_is_up else "down"
        majority_diracc = up_fraction if majority_is_up else (1 - up_fraction)

        ridge_mae = None
        ridge_diracc = None
        if prediction_service.ridge_model is not None:
            ridge_pred_returns = np.array([r["ridge_predicted_return"] for r in results])
            ridge_mae = float(np.mean(np.abs(ridge_pred_returns - actual_returns)))
            ridge_diracc = float(np.mean([r["ridge_direction_correct"] for r in results]))

        # mae_improvement_vs_zero: None if the zero-prediction baseline is
        # exactly 0 (all-zero actual returns). NaN would serialize as invalid
        # JSON for strict parsers, so we surface the undefined case explicitly.
        mae_improvement: Optional[float] = (
            (zero_mae - mae) / zero_mae if zero_mae > 0 else None
        )
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
            "ridge_mae": ridge_mae,
            "ridge_diracc": ridge_diracc,
            "majority_direction": majority_direction,
            "up_day_fraction": up_fraction,
            "mae_improvement_vs_zero": mae_improvement,
            "diracc_improvement_vs_majority": diracc_improvement_vs_majority,
            "stored": store,
            "skipped": {
                "missing_market_day": skipped_missing_day,
                "insufficient_history": skipped_insufficient_history,
                "no_next_day_actual": skipped_no_next_day,
                "nan_features": skipped_nan_features,
            },
        }

        if include_daily:
            trimmed = results[:BACKFILL_MAX_DAILY_RESULTS]
            response["daily_results_truncated"] = len(results) > len(trimmed)
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
                    "ridge_predicted_return": r["ridge_predicted_return"],
                    "ridge_direction_correct": r["ridge_direction_correct"],
                }
                for r in trimmed
            ]

        logger.info(
            "Backfill complete: n_days=%d diracc=%.3f mae=%.6f skipped=%s",
            len(results), diracc, mae, response["skipped"],
        )
        return response

    def _store_backfill_prediction(
        self,
        db,
        current_date: date,
        target_date: date,
        current_price: float,
        predicted_price: float,
        predicted_return: float,
        actual_price: float,
        actual_return: float,
        direction_correct: bool,
    ) -> None:
        db_prediction = PricePrediction(
            current_price=current_price,
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            predicted_direction="up" if predicted_return > 0 else "down",
            confidence=None,
            prediction_date=datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc),
            actual_price=actual_price,
            actual_return=actual_return,
            direction_correct=direction_correct,
            model_version="v2.0-backfill",
            created_at=datetime.now(timezone.utc),
        )
        db.add(db_prediction)
        db.commit()


price_updater = PriceUpdater()
