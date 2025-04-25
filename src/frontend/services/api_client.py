"""
API Client for communicating with the backend API.
"""
import requests
import time
from typing import Dict, Any, Optional, List
from src.shared.logging_config import get_context_logger

# Get context logger for this module
logger = get_context_logger(__name__)

class ApiClient:
    """
    Client for the backend API.
    Handles request formatting, error handling, and response processing.
    """

    def __init__(self, base_url: str = "http://localhost:8000/api"):
        """Initialize the API client with the base URL."""
        self.base_url = base_url

    def _handle_response(self, response, error_message: str = "API request failed", request_info: Dict[str, Any] = None):
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
            logger.add_context(**context).info(
                f"API request successful: {response.request.method} {response.url} "
                f"({response.status_code}) in {context['elapsed_ms']}ms"
            )
            logger.clear_context()

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
            logger.add_context(**context).error(f"{error_message}: {error_detail}")
            logger.clear_context()

            raise Exception(f"{error_message}: {error_detail}")

        except Exception as e:
            # Add error details to context
            context.update({
                'error_type': type(e).__name__,
                'error_message': str(e)
            })

            # Log error with context
            logger.add_context(**context).exception(f"{error_message}: {str(e)}")
            logger.clear_context()

            raise Exception(f"{error_message}: {str(e)}")

    # File Operations
    def upload_file(self, file, engine_type: str) -> Dict[str, Any]:
        """
        Upload a file to the backend.

        Args:
            file: File object to upload
            engine_type: Type of engine to use for processing

        Returns:
            Dict with upload result

        Raises:
            Exception: If upload fails
        """
        try:
            # Start timing the request
            start_time = time.time()

            # Prepare request data
            files = {"file": file}
            data = {"engine_type": engine_type}

            # Log the request
            file_info = {
                'file_name': file.name if hasattr(file, 'name') else 'unknown',
                'file_type': file.type if hasattr(file, 'type') else 'unknown',
                'file_size': len(file.getvalue()) if hasattr(file, 'getvalue') else 0,
                'engine_type': engine_type
            }

            logger.add_context(**file_info).info(
                f"Uploading file {file_info['file_name']} ({file_info['file_size']} bytes) "
                f"with engine {engine_type}"
            )
            logger.clear_context()

            # Make the request
            response = requests.post(
                f"{self.base_url}/ingestion/upload",
                files=files,
                data=data
            )

            # Calculate request duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Handle the response with additional context
            return self._handle_response(
                response,
                "File upload failed",
                {**file_info, 'request_duration_ms': duration_ms}
            )

        except Exception as e:
            # Log error with context
            logger.add_context(
                file_name=file.name if hasattr(file, 'name') else 'unknown',
                engine_type=engine_type,
                error_type=type(e).__name__
            ).exception(f"Error uploading file: {str(e)}")
            logger.clear_context()
            raise

    # Preprocessing Operations
    def get_preprocessing_operations(self, engine_type: str) -> Dict[str, Any]:
        """Get available preprocessing operations for the specified engine."""
        try:
            response = requests.get(
                f"{self.base_url}/preprocessing/operations/{engine_type}"
            )

            result = self._handle_response(response, "Failed to get preprocessing operations")
            return result.get("operations", {})
        except Exception as e:
            logger.error(f"Error getting preprocessing operations: {str(e)}")
            raise

    def preprocess_data(self, file_id: str, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply preprocessing operations to data."""
        try:
            payload = {
                "file_id": file_id,
                "operations": operations
            }

            response = requests.post(
                f"{self.base_url}/preprocessing/process",
                json=payload
            )

            return self._handle_response(response, "Preprocessing failed")
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    # Analysis Operations
    def analyze_data(self, file_id: str, analysis_type: str,
                    params: Dict[str, Any], use_preprocessed: bool = False) -> Dict[str, Any]:
        """Analyze data according to specified analysis type."""
        try:
            payload = {
                "file_id": file_id,
                "analysis_type": analysis_type,
                "params": params,
                "use_preprocessed": use_preprocessed
            }

            response = requests.post(
                f"{self.base_url}/analysis/analyze",
                json=payload
            )

            return self._handle_response(response, f"{analysis_type.capitalize()} analysis failed")
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            raise

    # Data Operations
    def get_data_preview(self, file_id: str, use_preprocessed: bool = False) -> Dict[str, Any]:
        """Get a preview of the data."""
        try:
            response = requests.get(
                f"{self.base_url}/data/{file_id}",
                params={"use_preprocessed": use_preprocessed}
            )

            return self._handle_response(response, "Failed to get data preview")
        except Exception as e:
            logger.error(f"Error getting data preview: {str(e)}")
            raise

    # Transformation Operations
    def transform_data(self, file_id: str, transformation_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data according to specified transformation type."""
        try:
            payload = {
                "file_id": file_id,
                "transformation_type": transformation_type,
                "params": params
            }

            response = requests.post(
                f"{self.base_url}/transformation/transform",
                json=payload
            )

            return self._handle_response(response, "Transformation failed")
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise
