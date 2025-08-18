import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Create data directory if it doesn't exist
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Database - putting it directly in data folder
DATABASE_URL = f"sqlite:///{DATA_DIR}/bittorch.db"

# Model settings
MODEL_PATH = DATA_DIR / "models" / "best_model.pth"
SEQUENCE_LENGTH = 7

# Ensure models directory exists
(DATA_DIR / "models").mkdir(exist_ok=True)

print(f"Database location: {DATA_DIR}/bittorch.db")
print(f"Model location: {MODEL_PATH}")