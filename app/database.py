from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()


class PricePrediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    current_price = Column(Float)
    predicted_price = Column(Float)
    actual_price = Column(Float, nullable=True)  # Updated later
    model_version = Column(String, default="v1.0")
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String)
    train_date = Column(DateTime)
    mse = Column(Float)
    mae = Column(Float)
    mape = Column(Float)
    baseline_improvement = Column(Float)


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key_hash = Column(String, unique=True, index=True)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    rate_limit = Column(Integer, default=100)  # requests per hour
    permissions = Column(String, default="read")  # read, write, admin


class RateLimitTracker(Base):
    __tablename__ = "rate_limits"

    id = Column(Integer, primary_key=True, index=True)
    key_hash = Column(String, index=True)
    ip_address = Column(String, index=True)
    endpoint = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Database setup
from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()