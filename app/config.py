from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Model directory
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Database
DATABASE_URL = f"sqlite:///{DATA_DIR}/bittorch.db"

# Model settings
MODEL_PATH = MODELS_DIR / "best_model.pth"
SEQUENCE_LENGTH = 7