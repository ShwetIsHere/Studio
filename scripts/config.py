import os
from pathlib import Path

# Project Root Discovery
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# --- AWS S3 CONFIGURATION ---
# Set these or use Environment Variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "YOUR_ACCESS_KEY_HERE")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "YOUR_SECRET_KEY_HERE")
S3_BUCKET = "cctv-alerts-17"

# Choose your storage: "LOCAL" or "S3"
STORAGE_TYPE = "S3" 

if STORAGE_TYPE == "S3":
    ALERTS_DIR = f"s3a://{S3_BUCKET}/logs/"
    CHECKPOINT_DIR = f"s3a://{S3_BUCKET}/checkpoints/"
else:
    ALERTS_DIR = str(PROJECT_ROOT / "alerts" / "logs")
    CHECKPOINT_DIR = str(PROJECT_ROOT / "alerts" / "mongo_checkpoints")

# --- MONGODB CONFIGURATION ---
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "cctv"
MONGO_COLLECTION = "threat_alerts"
