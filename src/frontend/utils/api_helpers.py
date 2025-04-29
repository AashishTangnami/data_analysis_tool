"""
Utility functions for API interactions.
"""
import time
from typing import Dict, Any, Optional
import requests
from src.shared.logging_config import get_context_logger
from .logging_helpers import log_api_request, log_error

# Get logger for this module
logger = get_context_logger(__name__)

def handle_api_response(response, error_message: str = "API request failed", 
                       request_info: Optional[Dict[str, Any]] = None):
    """
    Process API response and handle errors consistently.
    
    Args:
        response: Response object from requests
        error_message: Custom error message prefix
        request_info: Additional request information for logging
        
    Returns:
        Parsed JSON response
        
    Raises:
        Exception: If response indicates an error
    """
    # Add request info to context
    context = {
        'status_code': response.status_code,
        'url': response.url,
        'method': response.request.method,
        'elapsed_ms': int(response.elapsed.total_seconds() * 1000)
    }
    
    # Add any additional request info
    if request_info:
        context.update(request_info)
    
    try:
        response.raise_for_status()
        result = response.json()
        
        # Log successful response with context
        log_api_request(
            endpoint=response.url,
            method=response.request.method,
            duration_ms=context['elapsed_ms']
        )
        
        return result
        
    except requests.exceptions.HTTPError as e:
        # Try to get error details from response
        try:
            error_detail = response.json().get('detail', str(e)) if response.content else str(e)
        except ValueError:
            error_detail = response.text if response.content else str(e)
        
        # Add error details to context
        context.update({
            'error_type': type(e).__name__,
            'error_detail': error_detail
        })
        
        # Log error with context
        log_error(error_message, e, context)
        
        raise Exception(f"{error_message}: {error_detail}")
        
    except Exception as e:
        # Log error with context
        log_error(error_message, e, context)
        
        raise Exception(f"{error_message}: {str(e)}")

def make_api_request(method: str, url: str, **kwargs) -> Dict[str, Any]:
    """
    Make an API request with consistent error handling and logging.
    
    Args:
        method: HTTP method (get, post, put, delete)
        url: API endpoint URL
        **kwargs: Additional arguments to pass to requests
        
    Returns:
        Parsed JSON response
        
    Raises:
        Exception: If request fails
    """
    try:
        # Start timing the request
        start_time = time.time()
        
        # Make the request
        response = requests.request(method, url, **kwargs)
        
        # Calculate request duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Handle the response
        return handle_api_response(
            response,
            f"{method.upper()} request to {url} failed",
            {'request_duration_ms': duration_ms}
        )
        
    except Exception as e:
        # Log error
        log_error(f"{method.upper()} request to {url} failed", e)
        raise
