"""
API Client for communicating with the backend API.
"""
import requests
import time
from typing import Dict, Any, List, Optional
from src.shared.logging_config import get_context_logger
from src.frontend.utils.api_helpers import handle_api_response, make_api_request
from src.frontend.utils.logging_helpers import log_file_operation, log_api_request, log_error

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

    # This method is no longer needed as we use the centralized utility function directly

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

            # Create file info for logging
            file_info = {
                'file_name': file.name if hasattr(file, 'name') else 'unknown',
                'file_type': file.type if hasattr(file, 'type') else 'unknown',
                'file_size': len(file.getvalue()) if hasattr(file, 'getvalue') else 0,
                'engine_type': engine_type
            }

            # Log the file operation
            log_file_operation("upload", file_info)

            # Make the request
            response = requests.post(
                f"{self.base_url}/ingestion/upload",
                files=files,
                data=data
            )

            # Calculate request duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Handle the response with additional context
            return handle_api_response(
                response,
                "File upload failed",
                {**file_info, 'request_duration_ms': duration_ms}
            )

        except Exception as e:
            # Log error with context
            log_error("Error uploading file", e, {
                'file_name': file.name if hasattr(file, 'name') else 'unknown',
                'engine_type': engine_type
            })
            raise

    def preview_operation(self, file_id: str, operation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preview a preprocessing operation via the API.

        Args:
            file_id: ID of the file to preprocess
            operation: Operation to preview

        Returns:
            Dict with preview information
        """
        try:
            # Use the centralized API request function
            return make_api_request(
                "post",
                f"{self.base_url}/preprocessing/preview_operation",
                json={
                    "file_id": file_id,
                    "operation": operation
                }
            )
        except Exception as e:
            log_error("Failed to preview operation", e, {
                'file_id': file_id,
                'operation_type': operation.get('type', 'unknown')
            })
            raise Exception(f"Failed to preview operation: {str(e)}")

    def apply_single_operation(self, file_id: str, operation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a single preprocessing operation to data via the API.

        Args:
            file_id: ID of the file to preprocess
            operation: Single preprocessing operation to apply
AaTaMg113
        Returns:
            Dict with operation result information
        """
        try:
            # Use the centralized API request function
            return make_api_request(
                "post",
                f"{self.base_url}/preprocessing/apply_operation",
                json={
                    "file_id": file_id,
                    "operation": operation
                }
            )
        except Exception as e:
            log_error("Failed to apply operation", e, {
                'file_id': file_id,
                'operation_type': operation.get('type', 'unknown')
            })
            raise Exception(f"Failed to apply operation: {str(e)}")

    # Preprocessing Operations
    def get_preprocessing_operations(self, engine_type: str) -> Dict[str, Any]:
        """Get available preprocessing operations for the specified engine."""
        try:
            # Use the centralized API request function
            result = make_api_request(
                "get",
                f"{self.base_url}/preprocessing/operations/{engine_type}"
            )
            return result.get("operations", {})
        except Exception as e:
            log_error("Failed to get preprocessing operations", e, {'engine_type': engine_type})
            raise

    def preprocess_data(self, file_id: str, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply preprocessing operations to data."""
        try:
            # Use the centralized API request function
            return make_api_request(
                "post",
                f"{self.base_url}/preprocessing/process",
                json={
                    "file_id": file_id,
                    "operations": operations
                }
            )
        except Exception as e:
            log_error("Failed to preprocess data", e, {
                'file_id': file_id,
                'operation_count': len(operations)
            })
            raise

    # Analysis Operations
    def analyze_data(self, file_id: str, analysis_type: str,
                    params: Dict[str, Any], use_preprocessed: bool = False) -> Dict[str, Any]:
        """Analyze data according to specified analysis type."""
        try:
            # Use the centralized API request function
            return make_api_request(
                "post",
                f"{self.base_url}/analysis/analyze",
                json={
                    "file_id": file_id,
                    "analysis_type": analysis_type,
                    "params": params,
                    "use_preprocessed": use_preprocessed
                }
            )
        except Exception as e:
            log_error(f"{analysis_type.capitalize()} analysis failed", e, {
                'file_id': file_id,
                'analysis_type': analysis_type,
                'use_preprocessed': use_preprocessed
            })
            raise

    # Data Operations
    def get_data_preview(self, file_id: str, use_preprocessed: bool = False) -> Dict[str, Any]:
        """Get a preview of the data."""
        try:
            # Use the centralized API request function
            return make_api_request(
                "get",
                f"{self.base_url}/data/{file_id}",
                params={"use_preprocessed": use_preprocessed}
            )
        except Exception as e:
            log_error("Failed to get data preview", e, {
                'file_id': file_id,
                'use_preprocessed': use_preprocessed
            })
            raise

    # Transformation Operations
    def transform_data(self, file_id: str, transformation_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data according to specified transformation type."""
        try:
            # Use the centralized API request function
            return make_api_request(
                "post",
                f"{self.base_url}/transformation/transform",
                json={
                    "file_id": file_id,
                    "transformation_type": transformation_type,
                    "params": params
                }
            )
        except Exception as e:
            log_error("Transformation failed", e, {
                'file_id': file_id,
                'transformation_type': transformation_type
            })
            raise
