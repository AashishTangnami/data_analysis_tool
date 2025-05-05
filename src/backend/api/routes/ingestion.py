
import os
import tempfile
import shutil
import asyncio
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from core.context import EngineContext
from api.models.responses import DataResponse
from src.shared.constants import EngineType, FileType
from src.shared.session import SessionManager
from src.shared.logging_config import get_context_logger

# Get context logger for this module
logger = get_context_logger(__name__)

router = APIRouter()

# Create a session manager for data persistence
session_manager = SessionManager()

# Mutex for protecting shared storage operations
storage_lock = asyncio.Lock()

def validate_engine_type(engine_type: str = Form("pandas")):
    """Validate that the engine type is supported."""
    try:
        return EngineType(engine_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported engine type: {engine_type}. Supported types: {', '.join([e.value for e in EngineType])}"
        )

def validate_file_type(file: UploadFile = File(...)):
    """Validate that the file type is supported."""
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
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported types: .csv, .xlsx, .xls, .json, .parquet"
        )

    return file, supported_extensions[ext]

@router.post("/upload", response_model=DataResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    engine_file_validation: tuple = Depends(validate_file_type),
    engine_type: str = Depends(validate_engine_type)
):
    """
    Upload a file and process it with the selected engine.

    Args:
        background_tasks: Background tasks runner
        engine_file_validation: Tuple of (file, file_type)
        engine_type: Type of engine to use for processing

    Returns:
        DataResponse with file_id, summary, and preview
    """
    file, file_type = engine_file_validation

    # Create a temporary file with increased buffer size
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Use larger chunks for file writing
        CHUNK_SIZE = 1024 * 1024  # 1MB chunks
        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(CHUNK_SIZE):
                buffer.write(chunk)

        # Get file size for optimized loading strategy
        file_size = os.path.getsize(temp_file_path)
        file_size_mb = file_size / (1024 * 1024)

        # Add context for logging
        file_context = {
            'file_name': file.filename,
            'file_size': file_size,
            'file_path': temp_file_path,
            'file_type': file_type.value if hasattr(file_type, 'value') else str(file_type),
            'engine_type': engine_type.value if hasattr(engine_type, 'value') else str(engine_type)
        }

        logger.add_context(**file_context).info(f"File {file.filename} uploaded to {temp_file_path}")

        # Initialize engine context with selected engine
        engine_context = EngineContext(engine_type)

        # Determine if chunking should be used based on file size and engine type
        use_chunking = False
        chunk_size = 100000  # Default chunk size

        # Get engine type as string
        engine_type_value = engine_type.value if hasattr(engine_type, 'value') else str(engine_type)

        # For large files, enable chunking with engine-specific thresholds
        if engine_type_value == "pandas" and file_size_mb > 100:
            use_chunking = True
            chunk_size = 50000  # Smaller chunks for pandas
        elif engine_type_value == "polars" and file_size_mb > 300:
            use_chunking = True
            chunk_size = 100000  # Larger chunks for polars
        elif engine_type_value == "pyspark" and file_size_mb > 500:
            use_chunking = True
            chunk_size = 200000  # Even larger chunks for pyspark

        # Load data using the selected engine with explicit file type and chunking if needed
        start_time = asyncio.get_event_loop().time()
        data = engine_context.load_data(
            temp_file_path,
            file_type=file_type.value,
            use_chunking=use_chunking,
            chunk_size=chunk_size,
            optimize_memory=True
        )
        load_time = asyncio.get_event_loop().time() - start_time

        logger.add_context(**file_context,
                          load_time_ms=int(load_time * 1000),
                          chunking_enabled=use_chunking,
                          chunk_size=chunk_size if use_chunking else None).info(
            f"Data loaded in {int(load_time * 1000)}ms using {engine_type} engine"
            f"{' with chunking' if use_chunking else ''}"
        )

        # Generate summary
        summary = engine_context.get_data_summary(data)

        # Use the engine_type_value we already extracted earlier
        # Ensure it's a valid engine type by checking against the enum
        try:
            # Validate engine type against the enum
            engine_type_enum = EngineType(engine_type_value)
            # Use the validated value to ensure consistency
            file_id = f"{engine_type_enum.value.lower()}_{file.filename}"
        except ValueError:
            # Fallback to the original value if it's not in the enum
            # This should not happen if validate_engine_type is working correctly
            logger.warning(f"Unexpected engine type: {engine_type_value}, using as-is")
            file_id = f"{engine_type_value.lower()}_{file.filename}"

        # Update context with file_id
        file_context['file_id'] = file_id

        # Store file path and data using the session manager
        await session_manager.store_file(file_id, temp_file_path)
        await session_manager.store_data(file_id, data)

        # Add background task to clean up temp file after some time
        # Pass file size in MB to adjust cleanup delay based on file size
        background_tasks.add_task(cleanup_temp_file, temp_dir, file_id, file_size_mb)

        # Convert data to pandas (if not already) and take a small sample for preview
        pandas_data = engine_context.to_pandas(data)
        preview = pandas_data.head(10).to_dict(orient="records")

        # Add summary info to context
        summary_context = {
            'rows': summary.get('rows', 0),
            'columns': len(summary.get('columns', [])),
            'memory_usage': summary.get('memory_usage', 0)
        }

        logger.add_context(**file_context, **summary_context).info(
            f"File {file_id} processed successfully: {summary_context['rows']} rows, "
            f"{summary_context['columns']} columns"
        )
        logger.clear_context()

        return DataResponse(
            file_id=file_id,
            summary=summary,
            preview=preview,
            message="File uploaded and processed successfully"
        )

    except Exception as e:
        # Clean up temp file in case of error
        shutil.rmtree(temp_dir)

        # Log the error with context
        error_context = {
            'file_name': file.filename if hasattr(file, 'filename') else 'unknown',
            'error_type': type(e).__name__,
            'engine_type': engine_type.value if hasattr(engine_type, 'value') else str(engine_type),
            'file_type': file_type.value if hasattr(file_type, 'value') else str(file_type)
        }

        logger.add_context(**error_context).exception(f"Error processing file: {str(e)}")
        logger.clear_context()

        # Provide specific error messages for common issues
        if "memory" in str(e).lower():
            raise HTTPException(status_code=413, detail="File too large to process")
        elif "parse" in str(e).lower() or "format" in str(e).lower():
            raise HTTPException(status_code=422, detail=f"File format error: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail=str(e))

    finally:
        # Ensure file is closed
        await file.close()

@router.get("/data/{file_id}", response_model=DataResponse)
async def get_data(file_id: str, use_preprocessed: bool = False):
    """
    Get data preview and summary for a previously uploaded file.

    Args:
        file_id: The ID of the uploaded file
        use_preprocessed: Whether to use preprocessed data if available

    Returns:
        DataResponse with summary and preview information
    """
    # Determine which data to retrieve based on the use_preprocessed flag
    data = None
    data_type = "original"

    if use_preprocessed:
        # Try to get preprocessed data first
        data = await session_manager.get_preprocessed_data(file_id)
        if data is not None:
            data_type = "preprocessed"
            logger.info(f"Using preprocessed data for file_id: {file_id}")

    # Fall back to original data if preprocessed data is not available or not requested
    if data is None:
        data = await session_manager.get_data(file_id)
        if data is None:
            raise HTTPException(status_code=404, detail="File not found")

    try:
        # Get engine type from file_id
        engine_type = file_id.split("_")[0]

        # Initialize engine context
        engine_context = EngineContext(engine_type)

        # Generate summary
        summary = engine_context.get_data_summary(data)

        # Convert to pandas and get preview
        pandas_data = engine_context.to_pandas(data)
        preview = pandas_data.head(10).to_dict(orient="records")

        logger.info(f"Retrieved {data_type} data for file_id: {file_id}")

        return DataResponse(
            file_id=file_id,
            summary=summary,
            preview=preview,
            message=f"{data_type.capitalize()} data retrieved successfully"
        )

    except Exception as e:
        logger.error(f"Error retrieving data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

async def cleanup_temp_file(temp_dir: str, file_id: str, file_size_mb: float = None, delay_seconds: int = None):
    """
    Background task to clean up temporary files after a delay.
    The delay is adjusted based on file size to optimize disk space usage.

    Args:
        temp_dir: Directory to remove
        file_id: ID of the file to remove from storage
        file_size_mb: Size of the file in MB (used to adjust delay)
        delay_seconds: Delay before cleanup (if None, calculated based on file size)
    """
    # Calculate delay based on file size if not provided
    if delay_seconds is None:
        if file_size_mb is None:
            # Default delay if file size is unknown
            delay_seconds = 3600  # 1 hour
        else:
            # Adjust delay based on file size:
            # - Small files (<10MB): 2 hours
            # - Medium files (10-100MB): 1 hour
            # - Large files (100-500MB): 30 minutes
            # - Very large files (>500MB): 15 minutes
            if file_size_mb < 10:
                delay_seconds = 7200  # 2 hours
            elif file_size_mb < 100:
                delay_seconds = 3600  # 1 hour
            elif file_size_mb < 500:
                delay_seconds = 1800  # 30 minutes
            else:
                delay_seconds = 900  # 15 minutes

    # Log the scheduled cleanup
    logger.add_context(
        file_id=file_id,
        temp_dir=temp_dir,
        file_size_mb=file_size_mb,
        delay_seconds=delay_seconds
    ).info(f"Scheduled cleanup for file {file_id} in {delay_seconds} seconds")
    logger.clear_context()

    await asyncio.sleep(delay_seconds)

    try:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

            logger.add_context(
                file_id=file_id,
                temp_dir=temp_dir
            ).info(f"Removed temporary directory for file {file_id}")
            logger.clear_context()

        # Note: We're not removing the data from session_manager here
        # as it might still be in use. The session_manager has its own
        # cleanup mechanism for old sessions.

        # Schedule a cleanup of old sessions
        start_time = asyncio.get_event_loop().time()
        await session_manager.cleanup_old_sessions()
        cleanup_time = asyncio.get_event_loop().time() - start_time

        logger.add_context(
            operation="session_cleanup",
            duration_ms=int(cleanup_time * 1000)
        ).info(f"Cleaned up old sessions in {int(cleanup_time * 1000)}ms")
        logger.clear_context()

    except Exception as e:
        logger.add_context(
            file_id=file_id,
            temp_dir=temp_dir,
            error_type=type(e).__name__
        ).exception(f"Error cleaning up temp file: {e}")
        logger.clear_context()
