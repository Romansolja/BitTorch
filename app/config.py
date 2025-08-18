import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Database
DATABASE_URL = f"sqlite:///{BASE_DIR}/data/bittorch.db"

# Model settings
MODEL_PATH = BASE_DIR / "data" / "models" / "best_model.pth"
SEQUENCE_LENGTH = 7

# Ensure directories exist
(BASE_DIR / "data").mkdir(exist_ok=True)
(BASE_DIR / "data" / "models").mkdir(exist_ok=True)