from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class PricePrediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    prediction_date = Column(DateTime)
    current_price = Column(Float)
    predicted_price = Column(Float)
    actual_price = Column(Float, nullable=True)  # Updated later
    model_version = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime)


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True)
    model_version = Column(String)
    train_date = Column(DateTime)
    mse = Column(Float)
    mae = Column(Float)
    mape = Column(Float)
    baseline_improvement = Column(Float)


# THIS PART SHOULD NOT BE INDENTED - It's at module level, not inside the class
from .config import DATABASE_URL

# Create engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()