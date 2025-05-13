"""
Data service for managing data storage and retrieval.
This service provides a layer of abstraction over the data storage mechanisms.
"""
import os
import time
import asyncio
import shutil
import traceback
from typing import Dict, Any, List, Tuple, Optional, Union

from fastapi import HTTPException
from core.context import EngineContext
from src.shared.constants import EngineType, FileType
from src.shared.logging_utils import setup_logger
from api.utils.data_utils import extract_engine_type, initialize_engine_context, generate_data_preview

# Set up logger
logger = setup_logger("data_service", console_output=False)

# In-memory storage for uploaded files and processed data
# In a production app, use a proper database or file storage
file_storage = {}
data_storage = {}

# Mutex for protecting shared storage
storage_lock = asyncio.Lock()

async def store_file_data(
    file_id: str, 
    data: Any, 
    file_path: str
) -> None:
    """
    Store file data in memory.

    Args:
        file_id: Unique identifier for the file
        data: The data to store
        file_path: Path to the file on disk

    Raises:
        HTTPException: If there's an error storing the data
    """
    logger.debug(f"Storing data for file: {file_id}")
    
    try:
        async with storage_lock:
            file_storage[file_id] = file_path
            data_storage[file_id] = data
            
        logger.info(f"Data stored successfully for file: {file_id}")
    except Exception as e:
        logger.error(f"Error storing data: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error storing data: {str(e)}")

async def get_file_data(file_id: str) -> Any:
    """
    Get file data from memory.

    Args:
        file_id: Unique identifier for the file

    Returns:
        The stored data

    Raises:
        HTTPException: If the file is not found or there's an error retrieving the data
    """
    logger.debug(f"Retrieving data for file: {file_id}")
    
    try:
        async with storage_lock:
            if file_id not in data_storage:
                logger.error(f"File not found in storage: {file_id}")
                raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
                
            data = data_storage[file_id]
            
        logger.debug(f"Data retrieved successfully for file: {file_id}")
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving data: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")

async def get_file_path(file_id: str) -> str:
    """
    Get file path from memory.

    Args:
        file_id: Unique identifier for the file

    Returns:
        The stored file path

    Raises:
        HTTPException: If the file is not found or there's an error retrieving the path
    """
    logger.debug(f"Retrieving file path for file: {file_id}")
    
    try:
        async with storage_lock:
            if file_id not in file_storage:
                logger.error(f"File not found in storage: {file_id}")
                raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
                
            file_path = file_storage[file_id]
            
        logger.debug(f"File path retrieved successfully for file: {file_id}")
        return file_path
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving file path: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving file path: {str(e)}")

async def check_file_exists(file_id: str) -> bool:
    """
    Check if a file exists in storage.

    Args:
        file_id: Unique identifier for the file

    Returns:
        True if the file exists, False otherwise
    """
    logger.debug(f"Checking if file exists in storage: {file_id}")
    
    async with storage_lock:
        exists = file_id in data_storage
        
    logger.debug(f"File exists check for {file_id}: {exists}")
    return exists

async def delete_file_data(file_id: str, delete_temp_dir: bool = True) -> bool:
    """
    Delete file data from storage.

    Args:
        file_id: Unique identifier for the file
        delete_temp_dir: Whether to delete the temporary directory

    Returns:
        True if deletion was successful, False otherwise
    """
    logger.info(f"Deleting data for file: {file_id}")
    success = True
    
    try:
        # Get file path before deleting from storage
        temp_dir = None
        if delete_temp_dir:
            try:
                temp_dir = await get_file_path(file_id)
            except HTTPException:
                success = False
        
        # Delete from storage
        async with storage_lock:
            if file_id in file_storage:
                del file_storage[file_id]
                logger.debug(f"Removed file from file_storage: {file_id}")
            else:
                success = False
                
            if file_id in data_storage:
                del data_storage[file_id]
                logger.debug(f"Removed data from data_storage: {file_id}")
            else:
                success = False
        
        # Delete temporary directory if it exists
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.debug(f"Removed temporary directory: {temp_dir}")
        
        logger.info(f"Deletion completed for file: {file_id}, success: {success}")
        return success
    except Exception as e:
        logger.error(f"Error deleting data: {str(e)}")
        logger.error(traceback.format_exc())
        return False

async def get_data_preview(file_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Get data preview and summary for a file.

    Args:
        file_id: Unique identifier for the file

    Returns:
        Tuple containing the summary and preview

    Raises:
        HTTPException: If there's an error generating the preview
    """
    logger.debug(f"Generating preview for file: {file_id}")
    
    try:
        # Get engine type from file_id
        engine_type = extract_engine_type(file_id)
        
        # Initialize engine context
        engine_context = initialize_engine_context(engine_type)
        
        # Get data
        data = await get_file_data(file_id)
        
        # Generate summary
        summary = engine_context.get_data_summary(data)
        
        # Generate preview
        preview = generate_data_preview(engine_context, data)
        
        logger.debug(f"Preview generated successfully for file: {file_id}")
        return summary, preview
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating preview: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating preview: {str(e)}")
