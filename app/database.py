from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.config import DATABASE_URL

Base = declarative_base()


class PricePrediction(Base):
    """Stores each prediction made by the model"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_date = Column(DateTime)
    current_price = Column(Float)
    predicted_price = Column(Float)
    predicted_return = Column(Float, nullable=True)  # NEW: log return
    predicted_direction = Column(String, nullable=True)  # NEW: up/down
    confidence = Column(Float, nullable=True)  # NEW
    actual_price = Column(Float, nullable=True)
    actual_return = Column(Float, nullable=True)  # NEW
    direction_correct = Column(Boolean, nullable=True)  # NEW
    model_version = Column(String, default="v2.0")
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelMetrics(Base):
    """Stores training metrics for each model version"""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String)
    train_date = Column(DateTime)
    mse = Column(Float)
    mae = Column(Float)
    mape = Column(Float, nullable=True)
    directional_accuracy = Column(Float, nullable=True)  # NEW
    baseline_improvement = Column(Float)


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()