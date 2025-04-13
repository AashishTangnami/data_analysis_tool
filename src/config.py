"""
Configuration settings for the Dynamic Data Analysis Platform.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.environ.get("DATA_STORAGE_PATH", os.path.join(BASE_DIR, "data"))
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# API settings
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 8000))
API_WORKERS = int(os.environ.get("API_WORKERS", 4))

# File upload settings
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", 100000000))  # 100MB default
ALLOWED_EXTENSIONS = [".csv", ".json", ".xlsx", ".xls"]

# Processing settings
DEFAULT_CHUNK_SIZE = int(os.environ.get("DEFAULT_CHUNK_SIZE", 10000))

# Security
SECRET_KEY = os.environ.get("SECRET_KEY", "default_insecure_key")
