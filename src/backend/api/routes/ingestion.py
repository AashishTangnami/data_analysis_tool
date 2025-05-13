
import os
import time
import tempfile
import shutil
import asyncio
import traceback
from typing import Dict, Any, Optional, List, Tuple
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Request
from core.context import EngineContext
from api.models.responses import DataResponse
from src.shared.constants import EngineType, FileType
from src.shared.logging_utils import setup_logger
from api.utils.data_utils import extract_engine_type as shared_extract_engine_type
from api.utils.data_utils import initialize_engine_context as shared_initialize_engine_context
from api.utils.data_utils import generate_data_preview
from api.services import data_service

# Set up logger - disable console output to only log to file
logger = setup_logger("ingestion", console_output=False)

router = APIRouter()

# For backward compatibility, expose the storage dictionaries
# Other modules should migrate to using the data_service instead
file_storage = data_service.file_storage
data_storage = data_service.data_storage
storage_lock = data_service.storage_lock

def validate_engine_type(engine_type: str = Form("pandas")):
    """
    Validate that the engine type is supported.

    Args:
        engine_type: The engine type to validate (from form data)

    Returns:
        The validated engine type as an EngineType enum

    Raises:
        HTTPException: If the engine type is not supported
    """
    logger.debug(f"Validating engine type: {engine_type}")

    try:
        validated_engine = EngineType(engine_type)
        logger.debug(f"Engine type validated: {validated_engine.value}")
        return validated_engine
    except ValueError:
        supported_types = ', '.join([e.value for e in EngineType])
        error_msg = f"Unsupported engine type: {engine_type}. Supported types: {supported_types}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

def validate_file_type(file: UploadFile = File(...)):
    """
    Validate that the file type is supported.

    Args:
        file: The uploaded file to validate

    Returns:
        Tuple containing the file and its type as a FileType enum

    Raises:
        HTTPException: If the file type is not supported or filename is invalid
    """
    if not file or not file.filename:
        logger.error("Invalid file: No filename provided")
        raise HTTPException(status_code=400, detail="Invalid file: No filename provided")

    logger.debug(f"Validating file type for: {file.filename}")

    try:
        # Extract file extension
        _, ext = os.path.splitext(file.filename)
        ext = ext.lower()

        # Check if extension is supported
        supported_extensions = {
            '.csv': FileType.CSV,
            '.xlsx': FileType.EXCEL,
            '.xls': FileType.EXCEL,
            '.json': FileType.JSON,
            '.parquet': FileType.PARQUET
        }

        if ext not in supported_extensions:
            supported_types = ', '.join(supported_extensions.keys())
            error_msg = f"Unsupported file type: {ext}. Supported types: {supported_types}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        file_type = supported_extensions[ext]
        logger.debug(f"File type validated: {file_type.value}")
        return file, file_type

    except Exception as e:
        logger.error(f"Error validating file type: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error validating file: {str(e)}")

async def save_upload_to_temp_file(file: UploadFile) -> Tuple[str, str]:
    """
    Save an uploaded file to a temporary location.

    Args:
        file: The uploaded file

    Returns:
        Tuple containing the temporary directory path and file path

    Raises:
        HTTPException: If there's an error saving the file
    """
    start_time = time.time()
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    logger.debug(f"Saving uploaded file to temporary location: {temp_file_path}")

    try:
        # Use larger chunks for file writing
        CHUNK_SIZE = 1024 * 1024  # 1MB chunks
        total_size = 0

        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(CHUNK_SIZE):
                chunk_size = len(chunk)
                total_size += chunk_size
                buffer.write(chunk)

        elapsed_time = time.time() - start_time
        logger.info(f"File saved successfully: {file.filename} ({total_size/1024/1024:.2f} MB) in {elapsed_time:.2f}s")
        return temp_dir, temp_file_path

    except Exception as e:
        # Clean up temp directory in case of error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        logger.error(f"Error saving file: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

async def load_data_with_engine(engine_type: EngineType, file_path: str, file_type: FileType):
    """
    Load data using the specified engine.

    Args:
        engine_type: The engine to use for loading
        file_path: Path to the file to load
        file_type: Type of the file

    Returns:
        Tuple containing the loaded data, engine context, and data summary

    Raises:
        HTTPException: If there's an error loading the data
    """
    start_time = time.time()
    logger.debug(f"Loading data with engine: {engine_type.value}, file type: {file_type.value}")

    try:
        # Initialize engine context with selected engine using shared utility
        engine_context = shared_initialize_engine_context(engine_type)

        # Load data using the selected engine with explicit file type
        data = engine_context.load_data(file_path, file_type=file_type.value)

        # Generate summary
        summary = engine_context.get_data_summary(data)

        elapsed_time = time.time() - start_time
        data_shape = summary.get('shape', 'unknown')
        logger.info(f"Data loaded successfully: shape={data_shape} in {elapsed_time:.2f}s")

        return data, engine_context, summary

    except MemoryError:
        logger.error(f"Memory error loading file: {file_path}")
        raise HTTPException(status_code=413, detail="File too large to process")

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(traceback.format_exc())

        # Provide specific error messages for common issues
        if "memory" in str(e).lower():
            raise HTTPException(status_code=413, detail="File too large to process")
        elif "parse" in str(e).lower() or "format" in str(e).lower():
            raise HTTPException(status_code=422, detail=f"File format error: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail=f"Error loading data: {str(e)}")

async def store_data_and_generate_preview(
    engine_context: EngineContext,
    data: Any,
    engine_type: EngineType,
    filename: str,
    temp_file_path: str
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Store data and generate a preview.

    Args:
        engine_context: The engine context
        data: The loaded data
        engine_type: The engine type
        filename: The original filename
        temp_file_path: Path to the temporary file

    Returns:
        Tuple containing the file ID and data preview

    Raises:
        HTTPException: If there's an error generating the preview
    """
    start_time = time.time()
    logger.debug(f"Generating preview and storing data for: {filename}")

    try:
        # Generate a unique file ID
        file_id = f"{engine_type.value.lower()}_{filename}"

        # Store file path and data using data service
        await data_service.store_file_data(file_id, data, temp_file_path)

        # Generate preview using shared utility
        preview = generate_data_preview(engine_context, data)

        elapsed_time = time.time() - start_time
        logger.info(f"Preview generated and data stored with ID: {file_id} in {elapsed_time:.2f}s")

        return file_id, preview

    except Exception as e:
        logger.error(f"Error generating preview: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating preview: {str(e)}")

@router.post("/upload", response_model=DataResponse)
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    engine_file_validation: tuple = Depends(validate_file_type),
    engine_type: EngineType = Depends(validate_engine_type)
):
    """
    Upload a file and process it with the selected engine.

    Args:
        request: The HTTP request
        background_tasks: FastAPI background tasks
        engine_file_validation: Tuple containing the validated file and its type
        engine_type: The validated engine type

    Returns:
        DataResponse with file ID, summary, and preview
    """
    request_id = f"{time.time():.0f}"
    client_ip = request.client.host if request.client else "unknown"

    file, file_type = engine_file_validation

    logger.info(f"Request {request_id} from {client_ip}: Uploading file {file.filename} with engine {engine_type.value}")

    start_time = time.time()
    temp_dir = None

    try:
        # Save uploaded file to temporary location
        temp_dir, temp_file_path = await save_upload_to_temp_file(file)

        # Load data with the selected engine
        data, engine_context, summary = await load_data_with_engine(engine_type, temp_file_path, file_type)

        # Store data and generate preview
        file_id, preview = await store_data_and_generate_preview(
            engine_context, data, engine_type, file.filename, temp_file_path
        )

        # Add background task to clean up temp file after some time
        background_tasks.add_task(cleanup_temp_file, temp_dir, file_id)

        # Create response
        response = DataResponse(
            file_id=file_id,
            summary=summary,
            preview=preview,
            message="File uploaded and processed successfully"
        )

        # Log success
        elapsed_time = time.time() - start_time
        logger.info(f"Request {request_id} completed successfully in {elapsed_time:.2f}s")

        return response

    except HTTPException as e:
        # Log HTTP exceptions and re-raise
        logger.error(f"Request {request_id} failed with HTTP error: {e.status_code} - {e.detail}")

        # Clean up temp directory in case of error
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        raise

    except Exception as e:
        # Log and convert unexpected errors
        logger.error(f"Request {request_id} failed with unexpected error: {str(e)}")
        logger.error(traceback.format_exc())

        # Clean up temp directory in case of error
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    finally:
        # Ensure file is closed
        await file.close()

def extract_engine_type_from_file_id(file_id: str) -> str:
    """
    Extract engine type from file ID.

    Args:
        file_id: The file ID

    Returns:
        The engine type

    Raises:
        HTTPException: If the file ID format is invalid
    """
    # Use the shared utility function
    return shared_extract_engine_type(file_id)

async def validate_file_exists_in_storage(file_id: str) -> None:
    """
    Validate that a file exists in data storage.

    Args:
        file_id: The file ID to check

    Raises:
        HTTPException: If the file is not found
    """
    # Use the data service
    exists = await data_service.check_file_exists(file_id)
    if not exists:
        logger.error(f"File not found in storage: {file_id}")
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

async def get_data_and_generate_preview(file_id: str) -> Tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Get data and generate preview for a file.

    Args:
        file_id: The file ID

    Returns:
        Tuple containing the data, summary, and preview

    Raises:
        HTTPException: If there's an error retrieving the data
    """
    start_time = time.time()
    logger.debug(f"Retrieving data for file: {file_id}")

    try:
        # Get data from data service
        data = await data_service.get_file_data(file_id)

        # Get summary and preview from data service
        summary, preview = await data_service.get_data_preview(file_id)

        elapsed_time = time.time() - start_time
        logger.info(f"Data retrieved successfully for file: {file_id} in {elapsed_time:.2f}s")

        return data, summary, preview

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        logger.error(f"Error retrieving data: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")

@router.get("/check/{file_id}")
async def check_file_exists(file_id: str, request: Request):
    """
    Check if a file exists in storage without retrieving its data.

    Args:
        file_id: The ID of the uploaded file
        request: The HTTP request

    Returns:
        JSON response indicating if the file exists
    """
    request_id = f"{time.time():.0f}"
    client_ip = request.client.host if request.client else "unknown"

    logger.info(f"Request {request_id} from {client_ip}: Checking existence of file {file_id}")

    try:
        # Validate file exists
        await validate_file_exists_in_storage(file_id)

        # Get engine type from file_id
        engine_type = shared_extract_engine_type(file_id)

        # Return success response
        return {
            "exists": True,
            "file_id": file_id,
            "engine_type": engine_type,
            "message": "File exists in storage"
        }

    except HTTPException as e:
        # For 404 errors, return exists=False instead of raising an exception
        if e.status_code == 404:
            return {
                "exists": False,
                "file_id": file_id,
                "message": "File not found in storage"
            }

        # Log other HTTP exceptions and re-raise
        logger.error(f"Request {request_id} failed with HTTP error: {e.status_code} - {e.detail}")
        raise

    except Exception as e:
        # Log and convert unexpected errors
        logger.error(f"Request {request_id} failed with unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/data/{file_id}", response_model=DataResponse)
async def get_data(file_id: str, request: Request):
    """
    Get data preview and summary for a previously uploaded file.

    Args:
        file_id: The ID of the uploaded file
        request: The HTTP request

    Returns:
        DataResponse with summary and preview information
    """
    request_id = f"{time.time():.0f}"
    client_ip = request.client.host if request.client else "unknown"

    logger.info(f"Request {request_id} from {client_ip}: Retrieving data for file {file_id}")

    start_time = time.time()

    try:
        # Validate file exists
        await validate_file_exists_in_storage(file_id)

        # Get data and generate preview
        _, summary, preview = await get_data_and_generate_preview(file_id)

        # Create response
        response = DataResponse(
            file_id=file_id,
            summary=summary,
            preview=preview,
            message="Data retrieved successfully"
        )

        # Log success
        elapsed_time = time.time() - start_time
        logger.info(f"Request {request_id} completed successfully in {elapsed_time:.2f}s")

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

async def cleanup_temp_file(temp_dir: str, file_id: str, delay_seconds: int = 3600):
    """
    Background task to clean up temporary files after a delay.

    Args:
        temp_dir: Directory to remove
        file_id: ID of the file to remove from storage
        delay_seconds: Delay before cleanup (default: 1 hour)
    """
    logger.debug(f"Scheduled cleanup for file {file_id} in {delay_seconds} seconds")

    try:
        await asyncio.sleep(delay_seconds)
        await delete_file_data(file_id, temp_dir)
    except Exception as e:
        logger.error(f"Error during cleanup for file {file_id}: {str(e)}")
        logger.error(traceback.format_exc())

async def delete_file_data(file_id: str, temp_dir: Optional[str] = None) -> bool:
    """
    Delete file data from storage and optionally remove temporary directory.

    Args:
        file_id: ID of the file to remove from storage
        temp_dir: Optional directory to remove

    Returns:
        True if deletion was successful, False otherwise
    """
    logger.info(f"Cleaning up data for file {file_id}")

    # Use the data service to delete the file data
    return await data_service.delete_file_data(file_id, delete_temp_dir=temp_dir is not None)

@router.delete("/data/{file_id}")
async def delete_data(file_id: str, request: Request):
    """
    Delete data for a previously uploaded file.

    Args:
        file_id: The ID of the uploaded file
        request: The HTTP request

    Returns:
        JSON response indicating success or failure
    """
    request_id = f"{time.time():.0f}"
    client_ip = request.client.host if request.client else "unknown"

    logger.info(f"Request {request_id} from {client_ip}: Deleting data for file {file_id}")

    try:
        # Check if file exists first
        try:
            await validate_file_exists_in_storage(file_id)
        except HTTPException as e:
            if e.status_code == 404:
                return {
                    "success": False,
                    "file_id": file_id,
                    "message": "File not found in storage"
                }
            raise

        # Delete the file data
        success = await delete_file_data(file_id)

        # Return response
        return {
            "success": success,
            "file_id": file_id,
            "message": "Data deleted successfully" if success else "Partial deletion occurred"
        }

    except HTTPException as e:
        # Log HTTP exceptions and re-raise
        logger.error(f"Request {request_id} failed with HTTP error: {e.status_code} - {e.detail}")
        raise

    except Exception as e:
        # Log and convert unexpected errors
        logger.error(f"Request {request_id} failed with unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
