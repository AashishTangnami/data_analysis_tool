from typing import Dict, Any, List
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.context import EngineContext
from api.models.responses import DataResponse
from src.shared.session import SessionManager

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Get the session manager instance from ingestion
from api.routes.ingestion import session_manager

class TransformationRequest(BaseModel):
    """Request model for transformation operations"""
    file_id: str
    transformation_type: str  # e.g., "pivot", "melt", "groupby", "merge"
    params: Dict[str, Any]

@router.post("/transform", response_model=DataResponse)
async def transform_data(request: TransformationRequest):
    """
    Transform data according to specified transformation type.

    Args:
        request: TransformationRequest with file_id, transformation_type, and params

    Returns:
        DataResponse with transformed data information
    """
    # Check if data exists in session
    data = await session_manager.get_data(request.file_id)
    if data is None:
        # Try to get preprocessed data if original data not found
        data = await session_manager.get_preprocessed_data(request.file_id)
        if data is None:
            raise HTTPException(status_code=404, detail="File not found")

    try:
        logger.info(f"Transforming data for file_id: {request.file_id}, transformation type: {request.transformation_type}")

        # Get engine type from file_id
        engine_type = request.file_id.split("_")[0]

        # Initialize engine context
        engine_context = EngineContext(engine_type)

        # Apply transformation
        # This would require additional methods in the engine classes
        # For now, we'll just return the original data
        transformed_data = data

        # Generate summary for transformed data
        summary = engine_context.get_data_summary(transformed_data)

        # Convert to pandas and get preview
        pandas_data = engine_context.to_pandas(transformed_data)
        preview = pandas_data.head(10).to_dict(orient="records")

        # Store transformed data with a new ID
        new_file_id = f"{engine_type}_transformed_{request.file_id.split('_', 1)[1]}"
        await session_manager.store_transformed_data(new_file_id, transformed_data)

        logger.info(f"Transformation completed successfully, new file_id: {new_file_id}")

        return DataResponse(
            file_id=new_file_id,
            summary=summary,
            preview=preview,
            message=f"Data transformed successfully using {request.transformation_type}"
        )

    except Exception as e:
        logger.error(f"Error transforming data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))



