# shared/config.py
import os
from typing import Dict, Any

# API configuration
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "debug": os.getenv("API_DEBUG", "True").lower() == "true",
    "reload": os.getenv("API_RELOAD", "True").lower() == "true",
}

# Frontend configuration
FRONTEND_CONFIG = {
    "host": os.getenv("STREAMLIT_HOST", "0.0.0.0"),
    "port": int(os.getenv("STREAMLIT_PORT", "8501")),
    "debug": os.getenv("STREAMLIT_DEBUG", "True").lower() == "true",
}

# Storage settings
STORAGE_CONFIG = {
    "temp_file_expire_seconds": int(os.getenv("TEMP_FILE_EXPIRE_SECONDS", "3600")),
}

# Default engine settings
ENGINE_SETTINGS = {
    "pandas": {
        "chunk_size": int(os.getenv("PANDAS_CHUNK_SIZE", "10000")),
        "low_memory": os.getenv("PANDAS_LOW_MEMORY", "True").lower() == "true",
    },
    "polars": {
        "use_pyarrow": os.getenv("POLARS_USE_PYARROW", "True").lower() == "true",
    },
    "pyspark": {
        "executor_memory": os.getenv("PYSPARK_EXECUTOR_MEMORY", "2g"),
        "driver_memory": os.getenv("PYSPARK_DRIVER_MEMORY", "4g"),
        "cores_max": int(os.getenv("PYSPARK_CORES_MAX", "2")),
    }
}

# Analysis settings
ANALYSIS_CONFIG = {
    "max_categorical_values": int(os.getenv("MAX_CATEGORICAL_VALUES", "50")),
    "correlation_threshold": float(os.getenv("CORRELATION_THRESHOLD", "0.5")),
}