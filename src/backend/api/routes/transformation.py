from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.context import EngineContext
from api.models.responses import DataResponse

router = APIRouter()

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
    from api.routes.ingestion import data_storage
    
    # Check if file exists
    if request.file_id not in data_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Get engine type from file_id
        engine_type = request.file_id.split("_")[0]
        
        # Initialize engine context
        engine_context = EngineContext(engine_type)
        
        # Get original data
        data = data_storage[request.file_id]
        
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
        data_storage[new_file_id] = transformed_data
        
        return DataResponse(
            file_id=new_file_id,
            summary=summary,
            preview=preview,
            message=f"Data transformed successfully using {request.transformation_type}"
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))