
# src/api/dependencies.py
from logging import config
from fastapi import Header, HTTPException, Request
from typing import Optional
from src.core.engine_context import EngineContext
from src.core.engine_base import EngineBase
import time
from src.core.config import config

# Rate limiting configuration
RATE_LIMIT_DURATION = 60  # seconds
MAX_REQUESTS = 100  # requests per duration
request_history: dict[str, list[float]] = {}

def check_rate_limit(request: Request):
    client_ip = request.client.host if request.client else config.api_host
    current_time = time.time()
    
    # Clean old entries
    request_history[client_ip] = [
        timestamp for timestamp in request_history.get(client_ip, [])
        if current_time - timestamp < RATE_LIMIT_DURATION
    ]
    
    # Check rate limit
    if len(request_history.get(client_ip, [])) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
    
    # Add current request
    request_history.setdefault(client_ip, []).append(current_time)

def get_engine(x_engine_type: str = Header(default="pandas")) -> EngineContext:
    """
    Get the appropriate engine based on the X-Engine-Type header.
    """
    if x_engine_type not in ["pandas", "polars", "pyspark"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported engine type: {x_engine_type}. Supported types are: pandas, polars and pyspark."
        )
    
    return EngineContext(engine=EngineBase.get_engine(x_engine_type))
