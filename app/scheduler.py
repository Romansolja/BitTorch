from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import logging
from app.services.prediction import prediction_service
from app.services.training import training_service

logger = logging.getLogger(__name__)


class TaskScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()

    def start(self):
        """Start the scheduler with configured jobs"""

        # Daily prediction at 9 AM UTC
        self.scheduler.add_job(
            self.daily_prediction,
            CronTrigger(hour=9, minute=0),
            id="daily_prediction",
            name="Daily BTC Prediction",
            misfire_grace_time=3600
        )

        # Weekly model retraining on Sundays at 2 AM UTC
        self.scheduler.add_job(
            self.weekly_retrain,
            CronTrigger(day_of_week=6, hour=2, minute=0),
            id="weekly_retrain",
            name="Weekly Model Retrain",
            misfire_grace_time=7200
        )

        # Update actual prices daily at 10 AM UTC
        self.scheduler.add_job(
            self.update_prices,
            CronTrigger(hour=10, minute=0),
            id="update_prices",
            name="Update Actual Prices",
            misfire_grace_time=3600
        )

        self.scheduler.start()
        logger.info("Scheduler started with daily predictions and weekly retraining")

    async def daily_prediction(self):
        """Make and save daily prediction"""
        try:
            logger.info("Running daily prediction...")
            prediction = prediction_service.predict_next_day()
            if prediction:
                prediction_id = prediction_service.save_prediction(prediction)
                logger.info(f"Daily prediction saved with ID: {prediction_id}")

                # Optional: Send notification (email, Slack, etc.)
                await self.send_prediction_notification(prediction)
        except Exception as e:
            logger.error(f"Daily prediction failed: {e}")

    async def weekly_retrain(self):
        """Retrain model weekly with latest data"""
        try:
            logger.info("Starting weekly model retraining...")
            params = {
                "epochs": 150,
                "batch_size": 32,
                "learning_rate": 0.001,
                "sequence_length": 7
            }
            training_id = await training_service.train_model_async(params)
            logger.info(f"Weekly retraining started with ID: {training_id}")
        except Exception as e:
            logger.error(f"Weekly retraining failed: {e}")

    async def update_prices(self):
        """Update actual prices for past predictions"""
        try:
            from app.services.price_updater import price_updater
            result = price_updater.update_actual_prices()
            logger.info(f"Price update completed: {result}")
        except Exception as e:
            logger.error(f"Price update failed: {e}")

    async def send_prediction_notification(self, prediction):
        """Send prediction notification (implement based on your needs)"""
        # Example: Send to Slack, Discord, Email, etc.
        pass

    def shutdown(self):
        """Gracefully shutdown the scheduler"""
        self.scheduler.shutdown()


# Initialize scheduler
task_scheduler = TaskScheduler()