from typing import Dict, Any, List, Tuple, Optional
import time
import traceback
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
from core.context import EngineContext
from api.models.responses import PreprocessingResponse
from api.services import data_service
from src.shared.logging_utils import setup_logger
from api.utils.data_utils import extract_engine_type as shared_extract_engine_type
from api.utils.data_utils import initialize_engine_context as shared_initialize_engine_context
from api.utils.data_utils import generate_data_preview

# Set up logger - disable console output to only log to file
logger = setup_logger("preprocessing", console_output=False)

router = APIRouter()

# In-memory storage for preprocessed data
preprocessed_data_storage = {}

class PreprocessingRequest(BaseModel):
    """Request model for preprocessing operations"""
    file_id: str = Field(..., description="ID of the file to preprocess")
    operations: List[Dict[str, Any]] = Field(..., description="List of preprocessing operations to apply")

async def validate_file_exists(file_id: str) -> None:
    """
    Validate that the file exists in data storage.

    Args:
        file_id: ID of the file to validate

    Raises:
        HTTPException: If file is not found
    """
    # Use the data service
    exists = await data_service.check_file_exists(file_id)
    if not exists:
        logger.error(f"File not found in storage: {file_id}")
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

def extract_engine_type(file_id: str) -> str:
    """
    Extract engine type from file_id.

    Args:
        file_id: ID of the file

    Returns:
        Engine type string

    Raises:
        HTTPException: If engine type cannot be extracted
    """
    # Use the shared utility function
    return shared_extract_engine_type(file_id)

def initialize_engine_context(engine_type: str) -> EngineContext:
    """
    Initialize the engine context.

    Args:
        engine_type: Type of engine to initialize

    Returns:
        Initialized EngineContext

    Raises:
        HTTPException: If engine type is not supported or initialization fails
    """
    # Use the shared utility function
    return shared_initialize_engine_context(engine_type)

async def process_data(
    engine_context: EngineContext,
    file_id: str,
    operations: List[Dict[str, Any]]
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Process data with the given operations.

    Args:
        engine_context: Initialized engine context
        file_id: ID of the file to process
        operations: List of preprocessing operations to apply

    Returns:
        Tuple containing:
        - Processed data
        - Original data summary
        - Processed data summary
        - Preview of processed data

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    logger.info(f"Starting data processing for file: {file_id}")
    logger.debug(f"Operations to apply: {operations}")

    try:
        # Get original data using data service
        logger.debug(f"Retrieving original data for file: {file_id}")
        original_data = await data_service.get_file_data(file_id)

        # Generate summary for original data
        logger.debug("Generating summary for original data")
        original_summary = engine_context.get_data_summary(original_data)
        logger.debug(f"Original data shape: {original_summary.get('shape', 'unknown')}")

        # Validate operations
        if not operations:
            logger.warning("No preprocessing operations provided")
            raise ValueError("No preprocessing operations provided")

        # Apply preprocessing operations
        logger.info(f"Applying {len(operations)} preprocessing operations")
        operation_start_time = time.time()
        processed_data = engine_context.preprocess_data(original_data, operations)
        logger.info(f"Preprocessing completed in {time.time() - operation_start_time:.2f}s")

        # Generate summary for processed data
        logger.debug("Generating summary for processed data")
        processed_summary = engine_context.get_data_summary(processed_data)
        logger.debug(f"Processed data shape: {processed_summary.get('shape', 'unknown')}")

        # Generate preview using shared utility
        logger.debug("Generating data preview")
        preview = generate_data_preview(engine_context, processed_data)

        logger.info(f"Data processing completed successfully in {time.time() - start_time:.2f}s")
        return processed_data, original_summary, processed_summary, preview

    except ValueError as e:
        logger.error(f"Invalid operation: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid operation: {str(e)}")
    except TypeError as e:
        logger.error(f"Type error in preprocessing: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Type error in preprocessing: {str(e)}")
    except KeyError as e:
        logger.error(f"Missing key in operation: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Missing key in operation: {str(e)}")
    except MemoryError as e:
        logger.critical(f"Memory error during processing: {str(e)}")
        raise HTTPException(status_code=507, detail="Insufficient memory to process data")
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@router.post("/process", response_model=PreprocessingResponse)
async def preprocess_data(request: PreprocessingRequest, req: Request):
    """
    Apply preprocessing operations to data.

    Args:
        request: PreprocessingRequest with file_id and operations
        req: FastAPI Request object for request information

    Returns:
        PreprocessingResponse with processed data information
    """
    request_id = f"{time.time():.0f}"
    client_ip = req.client.host if req.client else "unknown"
    logger.info(f"Request {request_id} from {client_ip}: Processing data for file {request.file_id}")

    start_time = time.time()

    try:
        # Validate file exists using async function
        await validate_file_exists(request.file_id)

        # Extract engine type and initialize context
        engine_type = extract_engine_type(request.file_id)
        engine_context = initialize_engine_context(engine_type)

        # Process the data
        processed_data, original_summary, processed_summary, preview = await process_data(
            engine_context, request.file_id, request.operations
        )

        # Store processed data for later use
        logger.debug(f"Storing processed data for file: {request.file_id}")
        preprocessed_data_storage[request.file_id] = processed_data

        # Create response
        response = PreprocessingResponse(
            file_id=request.file_id,
            original_summary=original_summary,
            processed_summary=processed_summary,
            preview=preview,
            operations_applied=request.operations,
            message="Data preprocessed successfully"
        )

        # Log success
        processing_time = time.time() - start_time
        logger.info(f"Request {request_id} completed successfully in {processing_time:.2f}s")

        return response

    except HTTPException as e:
        # Log HTTP exceptions and re-raise
        logger.error(f"Request {request_id} failed with HTTP error: {e.status_code} - {e.detail}")
        raise

    except Exception as e:
        # Log and convert unexpected errors
        logger.error(f"Request {request_id} failed with unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

def get_preprocessing_operations(engine_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Get available preprocessing operations for the specified engine.

    Args:
        engine_type: Type of engine (pandas, polars, pyspark)

    Returns:
        Dictionary of available operations

    Raises:
        HTTPException: If engine type is not supported or operations cannot be retrieved
    """
    logger.debug(f"Getting preprocessing operations for engine: {engine_type}")

    try:
        # Import preprocessing classes based on engine type
        if engine_type == "pandas":
            from core.preprocessing.pandas_preprocessing import PandasPreprocessing
            operations = PandasPreprocessing().get_available_operations()
        elif engine_type == "polars":
            from core.preprocessing.polars_preprocessing import PolarsPreprocessing
            operations = PolarsPreprocessing().get_available_operations()
        elif engine_type == "pyspark":
            from core.preprocessing.pyspark_preprocessing import PySparkPreprocessing
            operations = PySparkPreprocessing().get_available_operations()
        else:
            logger.error(f"Unsupported engine type: {engine_type}")
            raise ValueError(f"Unsupported engine type: {engine_type}")

        logger.debug(f"Retrieved {len(operations)} operations for engine: {engine_type}")
        return operations

    except ImportError as e:
        logger.error(f"Failed to import preprocessing module: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to import preprocessing module: {str(e)}")
    except Exception as e:
        logger.error(f"Error retrieving operations: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Error retrieving operations: {str(e)}")

@router.get("/operations/{engine_type}", response_model=Dict[str, Any])
async def get_available_operations(engine_type: str, req: Request):
    """
    Get available preprocessing operations for the specified engine.

    Args:
        engine_type: Type of engine (pandas, polars, pyspark)
        req: FastAPI Request object for request information

    Returns:
        Dictionary of available operations
    """
    request_id = f"{time.time():.0f}"
    client_ip = req.client.host if req.client else "unknown"
    logger.info(f"Request {request_id} from {client_ip}: Getting operations for engine {engine_type}")

    start_time = time.time()

    try:
        operations = get_preprocessing_operations(engine_type)

        # Log success
        processing_time = time.time() - start_time
        logger.info(f"Request {request_id} completed successfully in {processing_time:.2f}s")

        return {"operations": operations}

    except HTTPException as e:
        # Log HTTP exceptions and re-raise
        logger.error(f"Request {request_id} failed with HTTP error: {e.status_code} - {e.detail}")
        raise

    except Exception as e:
        # Log and convert unexpected errors
        logger.error(f"Request {request_id} failed with unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")