from datetime import datetime, timezone

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    DateTime,
    String,
    Boolean,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import DATABASE_URL

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PricePrediction(Base):
    """Stores each prediction made by the model."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_date = Column(DateTime(timezone=True))
    current_price = Column(Float)
    predicted_price = Column(Float)
    predicted_return = Column(Float, nullable=True)
    predicted_direction = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    actual_price = Column(Float, nullable=True)
    actual_return = Column(Float, nullable=True)
    direction_correct = Column(Boolean, nullable=True)
    model_version = Column(String, default="v2.0")
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class ModelMetrics(Base):
    """Stores training metrics for each model version."""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String)
    train_date = Column(DateTime(timezone=True))
    mse = Column(Float)
    mae = Column(Float)
    mape = Column(Float, nullable=True)
    directional_accuracy = Column(Float, nullable=True)
    baseline_improvement = Column(Float)


# check_same_thread=False is a SQLite-only tuning (FastAPI may route a request
# through a worker thread different from the one that opened the connection).
# For any other driver (Postgres, MySQL, etc.) passing it raises at engine init.
_is_sqlite = DATABASE_URL.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yield a session, rollback on error, always close."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
