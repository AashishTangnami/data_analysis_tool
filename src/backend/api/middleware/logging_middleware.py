"""
Logging middleware for FastAPI.

This middleware automatically logs API requests with:
- Request method and endpoint
- Response status code
- Request duration
- Correlation ID for request tracking
"""

import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.shared.logging_config import get_context_logger, set_correlation_id, set_request_context

# Create a logger for the middleware
logger = get_context_logger("api.middleware")

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging API requests and responses."""
    
    def __init__(
        self,
        app: ASGIApp,
        correlation_id_header: str = "X-Correlation-ID"
    ):
        super().__init__(app)
        self.correlation_id_header = correlation_id_header
        
    async def dispatch(self, request: Request, call_next):
        # Extract or generate correlation ID
        correlation_id = request.headers.get(
            self.correlation_id_header, str(uuid.uuid4())
        )
        
        # Set correlation ID for this request
        set_correlation_id(correlation_id)
        
        # Extract request information
        method = request.method
        path = request.url.path
        
        # Set request context for logging
        set_request_context(endpoint=path, method=method)
        
        # Log the request
        start_time = time.time()
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"Request started: {method} {path} from {client_host}")
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Add correlation ID to response headers
            response.headers[self.correlation_id_header] = correlation_id
            
            # Calculate request duration
            duration_ms = round((time.time() - start_time) * 1000, 2)
            
            # Log the response
            if 200 <= response.status_code < 400:
                logger.success(f"Request completed: {method} {path} - {response.status_code} in {duration_ms}ms")
            else:
                logger.warning(f"Request completed with non-success status: {method} {path} - {response.status_code} in {duration_ms}ms")
                
            return response
            
        except Exception as e:
            # Calculate request duration
            duration_ms = round((time.time() - start_time) * 1000, 2)
            
            # Log the exception
            logger.exception(f"Request failed: {method} {path} in {duration_ms}ms")
            logger.failure(f"Error processing request: {str(e)}")
            
            # Re-raise the exception
            raise
