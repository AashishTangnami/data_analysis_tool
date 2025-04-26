from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from core.context import EngineContext
from api.models.responses import PreprocessingResponse
from src.shared.session import SessionManager
from src.shared.logging_config import get_context_logger

# Configure logging
logger = get_context_logger(__name__)

router = APIRouter()

# Get the session manager instance from ingestion
from api.routes.ingestion import session_manager


class UndoOperationRequest(BaseModel):
    """Request model for undoing an operation"""
    file_id: str
    operation_index: int

class ClearOperationsRequest(BaseModel):
    """Request model for clearing all operations"""
    file_id: str

class PreprocessingRequest(BaseModel):
    """Request model for preprocessing operations"""
    file_id: str
    operations: List[Dict[str, Any]]

class SingleOperationRequest(BaseModel):
    """Request model for a single preprocessing operation"""
    file_id: str
    operation: Dict[str, Any]


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

    # Set file context for logging
    logger.set_file_context(
        file_id=request.file_id,
        file_type=getattr(original_data, 'dtypes', {}).get('__file_type__', 'unknown')
    )

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


@router.post("/apply_operation", response_model=PreprocessingResponse)
async def apply_single_operation(request: SingleOperationRequest):
    """
    Apply a single preprocessing operation to data.
    
    Args:
        request: SingleOperationRequest with file_id and operation
        
    Returns:
        PreprocessingResponse with processed data information
    """
    # Check if data exists in session
    original_data = await session_manager.get_data(request.file_id)
    if original_data is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Set file context for logging
    logger.set_file_context(
        file_id=request.file_id,
        file_type=getattr(original_data, 'dtypes', {}).get('__file_type__', 'unknown')
    )
    
    try:
        logger.info(f"Applying operation {request.operation['type']} to file_id: {request.file_id}")
        
        # Get engine type from file_id
        engine_type = request.file_id.split("_")[0]
        
        # Initialize engine context
        engine_context = EngineContext(engine_type)
        
        # Get summary of original data
        original_summary = engine_context.get_data_summary(original_data)
        
        # Apply the single preprocessing operation
        processed_data = engine_context.apply_single_operation(original_data, request.operation)
        
        # Generate summary for processed data
        processed_summary = engine_context.get_data_summary(processed_data)
        
        # Replace the data in the session manager to avoid memory duplication
        await session_manager.replace_data(request.file_id, processed_data)
        
        # Convert to pandas and get preview
        pandas_data = engine_context.to_pandas(processed_data)
        preview = pandas_data.head(10).to_dict(orient="records")
        
        logger.info(f"Operation {request.operation['type']} completed successfully for file_id: {request.file_id}")
        
        return PreprocessingResponse(
            file_id=request.file_id,
            original_summary=original_summary,
            processed_summary=processed_summary,
            preview=preview,
            operations_applied=[request.operation],
            message=f"Operation {request.operation['type']} applied successfully"
        )
        
    except Exception as e:
        logger.error(f"Error applying operation: {str(e)}")
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


@router.post("/preview_operation", response_model=PreprocessingResponse)
async def preview_operation(request: SingleOperationRequest):
    """
    Preview the effect of a single preprocessing operation.
    
    Args:
        request: SingleOperationRequest with file_id and operation
        
    Returns:
        PreprocessingResponse with preview of operation effect
    """
    # Check if data exists in session
    original_data = await session_manager.get_data(request.file_id)
    if original_data is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        logger.info(f"Previewing operation {request.operation['type']} for file_id: {request.file_id}")
        
        # Get engine type from file_id
        engine_type = request.file_id.split("_")[0]
        
        # Initialize engine context
        engine_context = EngineContext(engine_type)
        
        # Get operation history
        operation_history = await session_manager.get_operation_history(request.file_id)
        
        # Check if operation is valid
        from core.preprocessing.pandas_preprocessing import PandasPreprocessing
        preprocessor = PandasPreprocessing()
        if not preprocessor.is_operation_valid(original_data, request.operation, operation_history):
            raise HTTPException(
                status_code=400, 
                detail=f"Operation {request.operation['type']} is not valid for the current data state"
            )
        
        # Get original summary
        original_summary = engine_context.get_data_summary(original_data)
        
        # Apply the operation temporarily for preview
        processed_data = engine_context.apply_single_operation(original_data, request.operation)
        
        # Generate summary for processed data
        processed_summary = engine_context.get_data_summary(processed_data)
        
        # Calculate impact metrics
        impact = {
            "rows_before": original_summary.get("row_count", 0),
            "rows_after": processed_summary.get("row_count", 0),
            "columns_before": original_summary.get("column_count", 0),
            "columns_after": processed_summary.get("column_count", 0),
            "missing_values_before": original_summary.get("missing_count", 0),
            "missing_values_after": processed_summary.get("missing_count", 0)
        }
        
        # Convert to pandas and get preview
        pandas_data = engine_context.to_pandas(processed_data)
        preview = pandas_data.head(10).to_dict(orient="records")
        
        return PreprocessingResponse(
            file_id=request.file_id,
            original_summary=original_summary,
            processed_summary=processed_summary,
            preview=preview,
            operations_applied=[request.operation],
            message=f"Preview of operation {request.operation['type']}",
            impact=impact
        )
        
    except Exception as e:
        logger.error(f"Error previewing operation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/undo_operation", response_model=PreprocessingResponse)
async def undo_operation(request: UndoOperationRequest):
    """
    Undo a specific preprocessing operation.
    
    Args:
        request: UndoOperationRequest with file_id and operation_index
        
    Returns:
        PreprocessingResponse with updated data information
    """
    # Check if data exists in session
    original_data = await session_manager.get_data(request.file_id)
    if original_data is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Get operation history
    operation_history = await session_manager.get_operation_history(request.file_id)
    
    # Check if operation index is valid
    if request.operation_index >= len(operation_history):
        raise HTTPException(status_code=400, detail="Invalid operation index")
    
    try:
        logger.info(f"Undoing operation at index {request.operation_index} for file_id: {request.file_id}")
        
        # Get engine type from file_id
        engine_type = request.file_id.split("_")[0]
        
        # Initialize engine context
        engine_context = EngineContext(engine_type)
        
        # Apply all operations up to the specified index
        processed_data = original_data.copy()
        applied_operations = []
        
        for i, operation in enumerate(operation_history):
            if i != request.operation_index:
                processed_data = engine_context.apply_single_operation(processed_data, operation)
                applied_operations.append(operation)
        
        # Remove the operation from history
        new_history = [op for i, op in enumerate(operation_history) if i != request.operation_index]
        await session_manager.set_operation_history(request.file_id, new_history)
        
        # Replace data in session
        await session_manager.replace_data(request.file_id, processed_data)
        
        # Get summary and preview
        processed_summary = engine_context.get_data_summary(processed_data)
        pandas_data = engine_context.to_pandas(processed_data)
        preview = pandas_data.head(10).to_dict(orient="records")
        
        logger.info(f"Operation undone successfully for file_id: {request.file_id}")
        
        return PreprocessingResponse(
            file_id=request.file_id,
            original_summary=engine_context.get_data_summary(original_data),
            processed_summary=processed_summary,
            preview=preview,
            operations_applied=applied_operations,
            message="Operation undone successfully"
        )
        
    except Exception as e:
        logger.error(f"Error undoing operation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/clear_operations", response_model=PreprocessingResponse)
async def clear_operations(request: ClearOperationsRequest):
    """
    Clear all preprocessing operations and revert to original data.
    
    Args:
        request: ClearOperationsRequest with file_id
        
    Returns:
        PreprocessingResponse with original data information
    """
    # Check if data exists in session
    original_data = await session_manager.get_data(request.file_id)
    if original_data is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        logger.info(f"Clearing all operations for file_id: {request.file_id}")
        
        # Get engine type from file_id
        engine_type = request.file_id.split("_")[0]
        
        # Initialize engine context
        engine_context = EngineContext(engine_type)
        
        # Reset operation history
        await session_manager.set_operation_history(request.file_id, [])
        
        # Get summary and preview
        original_summary = engine_context.get_data_summary(original_data)
        pandas_data = engine_context.to_pandas(original_data)
        preview = pandas_data.head(10).to_dict(orient="records")
        
        logger.info(f"All operations cleared for file_id: {request.file_id}")
        
        return PreprocessingResponse(
            file_id=request.file_id,
            original_summary=original_summary,
            processed_summary=original_summary,  # Same as original since all operations are cleared
            preview=preview,
            operations_applied=[],
            message="All operations cleared successfully"
        )
        
    except Exception as e:
        logger.error(f"Error clearing operations: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))