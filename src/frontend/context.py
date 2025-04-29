"""
Frontend context for managing engine selection and API communication.
Implements the Strategy pattern for the frontend.
"""
import streamlit as st
from typing import Dict, Any, List
from services.api_client import ApiClient
from src.shared.logging_config import get_context_logger

# Get context logger for this module
logger = get_context_logger(__name__)

class FrontendContext:
    """
    Context class for the Strategy Pattern in the frontend.
    Manages the currently selected engine and delegates API operations.
    """

    def __init__(self):
        """Initialize the frontend context with the engine from session state."""
        # Initialize session state if needed
        if "engine_type" not in st.session_state:
            st.session_state.engine_type = "pandas"

        # Initialize API client
        self.api_client = ApiClient()

    def get_current_engine(self) -> str:
        """
        Get the currently selected engine type.

        Returns:
            str: Current engine type
        """
        return st.session_state.engine_type

    def set_engine(self, engine_type: str) -> None:
        """
        Set the current engine type.

        Args:
            engine_type: Type of engine to use
        """
        previous_engine = st.session_state.engine_type if "engine_type" in st.session_state else None
        st.session_state.engine_type = engine_type

        # Add context to the log message
        logger.add_context(
            previous_engine=previous_engine,
            new_engine=engine_type,
            user_session_id=id(st.session_state)
        ).info(f"Engine changed from {previous_engine} to {engine_type}")

        # Clear context after use
        logger.clear_context()

    def upload_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Upload a file to the backend API.

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            Dict with file_id, preview, and summary

        Raises:
            Exception: If upload fails
        """
        try:
            # Use the API client to upload the file
            result = self.api_client.upload_file(
                uploaded_file,
                self.get_current_engine()
            )

            # Save results to session state
            st.session_state.file_id = result["file_id"]
            st.session_state.data_preview = result["preview"]
            st.session_state.data_summary = result["summary"]

            # Reset any previous analysis or preprocessing
            if "preprocessing_applied" in st.session_state:
                del st.session_state.preprocessing_applied
            if "analysis_completed" in st.session_state:
                del st.session_state.analysis_completed
            if "preprocessing_operations" in st.session_state:
                del st.session_state.preprocessing_operations

            # Generate a unique key for this upload to prevent duplicate logs
            upload_log_key = f"file_upload_{result['file_id']}"

            # Only log the first time we upload this file
            if upload_log_key not in st.session_state:
                # Log with context
                logger.add_context(
                    file_id=result['file_id'],
                    engine=self.get_current_engine(),
                    file_size=len(uploaded_file.getvalue()),
                    file_name=uploaded_file.name,
                    file_type=uploaded_file.type,
                    user_session_id=id(st.session_state)
                ).info(f"File uploaded successfully: {result['file_id']}")
                logger.clear_context()

                # Mark that we've logged this upload
                st.session_state[upload_log_key] = True

            return result

        except Exception as e:
            # Log error with context
            logger.add_context(
                engine=self.get_current_engine(),
                file_name=uploaded_file.name if hasattr(uploaded_file, 'name') else 'unknown',
                error_type=type(e).__name__,
                user_session_id=id(st.session_state)
            ).exception(f"Upload error: {str(e)}")
            logger.clear_context()
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
            # Delegate to the API client
            return self.api_client.preview_operation(file_id, operation)
        except Exception as e:
            logger.error(f"Error previewing operation: {str(e)}")
            raise

    def apply_single_operation(self, file_id: str, operation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a single preprocessing operation via the API.

        Args:
            file_id: ID of the file to preprocess
            operation: Operation to apply

        Returns:
            Dict with operation result
        """
        try:
            # Delegate to the API client
            return self.api_client.apply_single_operation(file_id, operation)
        except Exception as e:
            logger.error(f"Error applying operation: {str(e)}")
            raise

    # These methods were duplicated and have been removed
    # The implementations at lines 373-408 and 410-438 are kept instead


    def preprocess_data(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply preprocessing operations to data.

        Args:
            operations: List of preprocessing operations to apply

        Returns:
            Dict with preprocessed data information

        Raises:
            Exception: If preprocessing fails
        """
        try:
            if "file_id" not in st.session_state or not st.session_state.file_id:
                raise Exception("No file loaded. Please upload a file first.")

            # Use the API client to preprocess the data
            result = self.api_client.preprocess_data(
                st.session_state.file_id,
                operations
            )

            # Update session state
            st.session_state.data_preview = result["preview"]
            st.session_state.preprocessing_applied = True
            st.session_state.preprocessing_operations = operations

            logger.info(f"Preprocessing applied successfully for file: {st.session_state.file_id}")
            return result

        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise

    def analyze_data(self, analysis_type: str, params: Dict[str, Any], use_preprocessed: bool = False) -> Dict[str, Any]:
        """
        Analyze data according to specified analysis type.

        Args:
            analysis_type: Type of analysis to perform
            params: Parameters for the analysis
            use_preprocessed: Whether to use preprocessed data

        Returns:
            Dict with analysis results

        Raises:
            Exception: If analysis fails
        """
        try:
            if "file_id" not in st.session_state or not st.session_state.file_id:
                raise Exception("No file loaded. Please upload a file first.")

            # Use the API client to analyze the data
            result = self.api_client.analyze_data(
                st.session_state.file_id,
                analysis_type,
                params,
                use_preprocessed
            )

            # Update session state
            st.session_state.analysis_completed = True
            st.session_state.analysis_type = analysis_type
            st.session_state.analysis_results = result["results"]
            st.session_state.analysis_visualizations = result["visualizations"]

            logger.info(f"Analysis completed successfully for file: {st.session_state.file_id}")
            return result

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise

    def get_available_preprocessing_operations(self) -> Dict[str, Any]:
        """
        Get available preprocessing operations for the current engine.

        Returns:
            Dict with available operations

        Raises:
            Exception: If operation retrieval fails
        """
        try:
            # Use the API client to get available preprocessing operations
            return self.api_client.get_preprocessing_operations(self.get_current_engine())

        except Exception as e:
            logger.error(f"Error getting preprocessing operations: {str(e)}")
            raise

    def get_data_preview(self, use_preprocessed: bool = False) -> Dict[str, Any]:
        """
        Get a preview of the data.

        Args:
            use_preprocessed: Whether to use preprocessed data

        Returns:
            Dict with data preview and summary

        Raises:
            Exception: If data preview retrieval fails
        """
        try:
            if "file_id" not in st.session_state or not st.session_state.file_id:
                raise Exception("No file loaded. Please upload a file first.")

            # Use the API client to get data preview
            result = self.api_client.get_data_preview(
                st.session_state.file_id,
                use_preprocessed
            )

            # Update session state
            st.session_state.data_preview = result.get("data", [])
            st.session_state.data_summary = result.get("summary", {})
            st.session_state.use_preprocessed = use_preprocessed

            logger.info(f"Data preview updated to {'preprocessed' if use_preprocessed else 'original'} data")
            return result

        except Exception as e:
            logger.error(f"Error getting data preview: {str(e)}")
            raise

    def transform_data(self, transformation_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform data according to specified transformation type.

        Args:
            transformation_type: Type of transformation to perform
            params: Parameters for the transformation

        Returns:
            Dict with transformed data information

        Raises:
            Exception: If transformation fails
        """
        try:
            if "file_id" not in st.session_state or not st.session_state.file_id:
                raise Exception("No file loaded. Please upload a file first.")

            # Use the API client to transform the data
            result = self.api_client.transform_data(
                st.session_state.file_id,
                transformation_type,
                params
            )

            # Update session state with new file ID
            st.session_state.transformed_file_id = result.get("file_id")

            logger.info(f"Data transformed successfully, new file ID: {result.get('file_id')}")
            return result

        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise


    def undo_operation(self, operation_index: int) -> Dict[str, Any]:
        """
        Undo a specific operation in the history.

        Args:
            operation_index: Index of the operation to undo

        Returns:
            Dict with updated data information
        """
        try:
            if "file_id" not in st.session_state or not st.session_state.file_id:
                raise Exception("No file loaded. Please upload a file first.")

            if "operation_history" not in st.session_state or operation_index >= len(st.session_state.operation_history):
                raise Exception("Invalid operation index.")

            # Use the API client to undo the operation
            result = self.api_client.undo_operation(
                st.session_state.file_id,
                operation_index
            )

            # Update session state
            st.session_state.data_preview = result.get("preview", [])
            st.session_state.data_summary = result.get("processed_summary", {})

            # Remove the operation from history
            st.session_state.operation_history.pop(operation_index)

            logger.info(f"Operation at index {operation_index} undone for file: {st.session_state.file_id}")
            return result

        except Exception as e:
            logger.error(f"Undo operation error: {str(e)}")
            raise

    def clear_all_operations(self) -> Dict[str, Any]:
        """
        Clear all operations and revert to original data.

        Returns:
            Dict with original data information
        """
        try:
            if "file_id" not in st.session_state or not st.session_state.file_id:
                raise Exception("No file loaded. Please upload a file first.")

            # Use the API client to clear all operations
            result = self.api_client.clear_all_operations(
                st.session_state.file_id
            )

            # Update session state
            st.session_state.data_preview = result.get("preview", [])
            st.session_state.data_summary = result.get("original_summary", {})

            # Clear operation history
            st.session_state.operation_history = []

            logger.info(f"All operations cleared for file: {st.session_state.file_id}")
            return result

        except Exception as e:
            logger.error(f"Clear operations error: {str(e)}")
            raise