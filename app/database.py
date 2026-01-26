from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.config import DATABASE_URL

Base = declarative_base()


class PricePrediction(Base):
    """Stores each prediction made by the model"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_date = Column(DateTime)  # The date we're predicting FOR
    current_price = Column(Float)       # Price when prediction was made
    predicted_price = Column(Float)     # What the model predicted
    actual_price = Column(Float, nullable=True)  # Filled in later
    model_version = Column(String, default="v1.0")
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelMetrics(Base):
    """Stores training metrics for each model version"""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String)
    train_date = Column(DateTime)
    mse = Column(Float)
    mae = Column(Float)
    mape = Column(Float)
    baseline_improvement = Column(Float)


# Database setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for FastAPI endpoints"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()