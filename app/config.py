from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

ARTIFACTS_DIR = BASE_DIR / "artifacts"

DATABASE_URL = f"sqlite:///{DATA_DIR}/bittorch.db"

# Model settings (new)
MODEL_PATH = ARTIFACTS_DIR / "production_model.pth"
SCALER_PATH = ARTIFACTS_DIR / "feature_scaler.pkl"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
SEQUENCE_LENGTH = 30