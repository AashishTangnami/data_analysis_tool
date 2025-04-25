from typing import Dict, Any, List
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from core.context import EngineContext
from api.models.responses import PreprocessingResponse
from src.shared.session import SessionManager

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Get the session manager instance from ingestion
from api.routes.ingestion import session_manager

class PreprocessingRequest(BaseModel):
    """Request model for preprocessing operations"""
    file_id: str
    operations: List[Dict[str, Any]]

@router.post("/process", response_model=PreprocessingResponse)
async def preprocess_data(request: PreprocessingRequest):
    """
    Apply preprocessing operations to data.

    Args:
        request: PreprocessingRequest with file_id and operations

    Returns:
        PreprocessingResponse with processed data information
    """
    # Check if data exists in session
    original_data = await session_manager.get_data(request.file_id)
    if original_data is None:
        raise HTTPException(status_code=404, detail="File not found")

    try:
        logger.info(f"Preprocessing data for file_id: {request.file_id}")
        logger.info(f"Operations to apply: {request.operations}")

        # Get engine type from file_id
        engine_type = request.file_id.split("_")[0]

        # Initialize engine context
        engine_context = EngineContext(engine_type)

        # Get summary of original data
        original_summary = engine_context.get_data_summary(original_data)

        # Apply preprocessing operations
        processed_data = engine_context.preprocess_data(original_data, request.operations)

        # Generate summary for processed data
        processed_summary = engine_context.get_data_summary(processed_data)

        # Store processed data in session manager
        await session_manager.store_preprocessed_data(request.file_id, processed_data)

        # Convert to pandas and get preview
        pandas_data = engine_context.to_pandas(processed_data)
        preview = pandas_data.head(10).to_dict(orient="records")

        logger.info(f"Preprocessing completed successfully for file_id: {request.file_id}")

        return PreprocessingResponse(
            file_id=request.file_id,
            original_summary=original_summary,
            processed_summary=processed_summary,
            preview=preview,
            operations_applied=request.operations,
            message="Data preprocessed successfully"
        )

    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/operations/{engine_type}", response_model=Dict[str, Any])
async def get_available_operations(engine_type: str):
    """
    Get available preprocessing operations for the specified engine.

    Args:
        engine_type: Type of engine (pandas, polars, pyspark)

    Returns:
        Dictionary of available operations
    """
    try:
        # Initialize engine context
        engine_context = EngineContext(engine_type)

        # Get available operations from the preprocessing component
        # We need to access the engine's preprocessing component directly
        operations = {}

        if engine_type == "pandas":
            from core.preprocessing.pandas_preprocessing import PandasPreprocessing
            operations = PandasPreprocessing().get_available_operations()
        elif engine_type == "polars":
            from core.preprocessing.polars_preprocessing import PolarsPreprocessing
            operations = PolarsPreprocessing().get_available_operations()
        elif engine_type == "pyspark":
            from core.preprocessing.pyspark_preprocessing import PySparkPreprocessing
            operations = PySparkPreprocessing().get_available_operations()

        return {"operations": operations}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


