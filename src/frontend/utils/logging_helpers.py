"""
Utility functions for logging in the frontend.
"""
from typing import Dict, Any, Optional
from src.shared.logging_config import get_context_logger

# Get logger for this module
logger = get_context_logger(__name__)

def log_file_operation(operation: str, file_info: Dict[str, Any]) -> None:
    """
    Log a file operation with appropriate context.

    Args:
        operation: Operation being performed (e.g., "upload", "preview", "analyze")
        file_info: Dictionary with file information
    """
    # Add context to the log message
    logger.add_context(**file_info).info(
        f"File {operation}: {file_info.get('file_name', 'unknown')} "
        f"({file_info.get('file_size', 0)} bytes) "
        f"with engine {file_info.get('engine_type', 'unknown')}"
    )
    logger.clear_context()

def log_api_request(endpoint: str, method: str, params: Optional[Dict[str, Any]] = None,
                   duration_ms: Optional[int] = None) -> None:
    """
    Log an API request with appropriate context.

    Args:
        endpoint: API endpoint
        method: HTTP method
        params: Optional request parameters
        duration_ms: Optional request duration in milliseconds
    """
    context: Dict[str, Any] = {
        "endpoint": endpoint,
        "method": method
    }

    if params:
        context["params"] = params

    if duration_ms:
        context["duration_ms"] = duration_ms

    # Add context to the log message
    logger.add_context(**context).info(
        f"API request: {method} {endpoint}" +
        (f" in {duration_ms}ms" if duration_ms else "")
    )
    logger.clear_context()

def log_error(error_message: str, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an error with appropriate context.

    Args:
        error_message: Error message
        error: Exception object
        context: Optional additional context
    """
    error_context = {
        "error_type": type(error).__name__,
        "error_message": str(error)
    }

    if context:
        error_context.update(context)

    # Add context to the log message
    logger.add_context(**error_context).error(f"{error_message}: {str(error)}")
    logger.clear_context()
