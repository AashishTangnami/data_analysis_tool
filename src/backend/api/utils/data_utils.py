"""
Shared utilities for data operations.
"""
import time
import traceback
from typing import Dict, Any, List, Union
from fastapi import HTTPException
from core.context import EngineContext
from src.shared.logging_utils import setup_logger
from src.shared.constants import EngineType

# Set up logger
logger = setup_logger("data_utils", console_output=False)

# Note: File storage operations have been moved to api.services.data_service
# This module now only contains utility functions for data processing

def extract_engine_type(file_id: str) -> str:
    """
    Extract engine type from file ID.

    Args:
        file_id: The file ID

    Returns:
        The engine type

    Raises:
        HTTPException: If the file ID format is invalid
    """
    try:
        logger.debug(f"Extracting engine type from file_id: {file_id}")
        parts = file_id.split("_")
        if not parts or len(parts) < 1:
            logger.error(f"Invalid file_id format: {file_id}")
            raise ValueError(f"Invalid file_id format: {file_id}")

        engine_type = parts[0]
        logger.debug(f"Extracted engine type: {engine_type}")
        return engine_type
    except Exception as e:
        logger.error(f"Failed to extract engine type: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid file ID format: {str(e)}")

def initialize_engine_context(engine_type: Union[str, EngineType]) -> EngineContext:
    """
    Initialize the engine context.

    Args:
        engine_type: Type of engine to initialize (string or EngineType enum)

    Returns:
        Initialized EngineContext

    Raises:
        HTTPException: If engine type is not supported or initialization fails
    """
    start_time = time.time()

    # Extract value if engine_type is an enum
    if hasattr(engine_type, 'value'):
        engine_type_value = engine_type.value
    else:
        engine_type_value = engine_type

    logger.debug(f"Initializing engine context for type: {engine_type_value}")

    try:
        context = EngineContext(engine_type_value)
        logger.debug(f"Engine context initialized successfully in {time.time() - start_time:.2f}s")
        return context
    except ValueError as e:
        logger.error(f"Invalid engine type: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid engine type: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to initialize engine context: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Engine initialization error: {str(e)}")

# Note: validate_file_exists has been moved to api.services.data_service
# Use data_service.check_file_exists() instead

def generate_data_preview(engine_context: EngineContext, data: Any) -> List[Dict[str, Any]]:
    """
    Generate a preview of the data.

    Args:
        engine_context: The engine context
        data: The data to preview

    Returns:
        A list of dictionaries representing the preview

    Raises:
        HTTPException: If there's an error generating the preview
    """
    start_time = time.time()
    logger.debug("Generating data preview")

    try:
        pandas_data = engine_context.to_pandas(data)
        preview = pandas_data.head(10).to_dict(orient="records")

        logger.debug(f"Preview generated in {time.time() - start_time:.2f}s")
        return preview
    except Exception as e:
        logger.error(f"Error generating preview: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating preview: {str(e)}")
