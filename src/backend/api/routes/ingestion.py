import os
import tempfile
import shutil
import asyncio
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, validator
from core.context import EngineContext
from api.models.responses import DataResponse, ErrorResponse
from enum import Enum
# from ....shared.constants import EngineType, FileType

router = APIRouter()

# In-memory storage for uploaded files and processed data
# In a production app, use a proper database or file storage
file_storage = {}
data_storage = {}

# Mutex for protecting shared storage
storage_lock = asyncio.Lock()

class FileType(str, Enum):
    """Enum for supported file types."""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"

class EngineType(str, Enum):
    """Enum for supported data processing engines."""
    PANDAS = "pandas"
    POLARS = "polars"
    PYSPARK = "pyspark"
    
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
        '.json': FileType.JSON
    }
    
    if ext not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported types: .csv, .xlsx, .xls, .json"
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
        file: The file to upload
        engine_type: The engine to use for processing (pandas, polars, pyspark)
        
    Returns:
        DataResponse with file_id and summary information
    """
    file, file_type = engine_file_validation
    
    # Create a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded file to temp location
        with open(temp_file_path, "wb") as buffer:
            # Read in chunks to handle large files
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await file.read(chunk_size):
                buffer.write(chunk)
        
        # Initialize engine context with selected engine
        engine_context = EngineContext(engine_type)
        
        # Load data using the selected engine with explicit file type
        data = engine_context.load_data(temp_file_path, file_type=file_type.value)
        
        # Generate summary
        summary = engine_context.get_data_summary(data)
        
        # If engine_type is an enum, extract its value
        if hasattr(engine_type, 'value'):
            engine_type_value = engine_type.value
        else:
            engine_type_value = engine_type
        file_id = f"{engine_type_value.lower()}_{file.filename}"
        # # Generate a unique file ID
        # file_id = f"{engine_type}_{file.filename}"
        
        # Store file path and data for later use
        async with storage_lock:
            file_storage[file_id] = temp_file_path
            data_storage[file_id] = data
        
        # Add background task to clean up temp file after some time
        background_tasks.add_task(cleanup_temp_file, temp_dir, file_id)
        
        # Convert data to pandas (if not already) and take a small sample for preview
        pandas_data = engine_context.to_pandas(data)
        preview = pandas_data.head(10).to_dict(orient="records")
        
        return DataResponse(
            file_id=file_id,
            summary=summary,
            preview=preview,
            message="File uploaded and processed successfully"
        )
    
    except Exception as e:
        # Clean up temp file in case of error
        shutil.rmtree(temp_dir)
        
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
async def get_data(file_id: str):
    """
    Get data preview and summary for a previously uploaded file.
    
    Args:
        file_id: The ID of the uploaded file
        
    Returns:
        DataResponse with summary and preview information
    """
    async with storage_lock:
        if file_id not in data_storage:
            raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Get engine type from file_id
        engine_type = file_id.split("_")[0]
        
        # Initialize engine context
        engine_context = EngineContext(engine_type)
        
        # Get data and generate summary
        async with storage_lock:
            data = data_storage[file_id]
            
        summary = engine_context.get_data_summary(data)
        
        # Convert to pandas and get preview
        pandas_data = engine_context.to_pandas(data)
        preview = pandas_data.head(10).to_dict(orient="records")
        
        return DataResponse(
            file_id=file_id,
            summary=summary,
            preview=preview,
            message="Data retrieved successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

async def cleanup_temp_file(temp_dir: str, file_id: str, delay_seconds: int = 3600):
    """
    Background task to clean up temporary files after a delay.
    
    Args:
        temp_dir: Directory to remove
        file_id: ID of the file to remove from storage
        delay_seconds: Delay before cleanup (default: 1 hour)
    """
    # Use async sleep instead of blocking time.sleep
    await asyncio.sleep(delay_seconds)
    
    # Clean up temp directory and storage entries
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    async with storage_lock:
        if file_id in file_storage:
            del file_storage[file_id]
        
        if file_id in data_storage:
            del data_storage[file_id]
