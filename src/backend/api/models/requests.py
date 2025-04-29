from typing import Dict, Any, List
from pydantic import BaseModel

class EngineSelectionRequest(BaseModel):
    """Request model for selecting an engine"""
    engine_type: str

class PreprocessingRequest(BaseModel):
    """Request model for preprocessing operations"""
    file_id: str
    operations: List[Dict[str, Any]]

class AnalysisRequest(BaseModel):
    """Request model for analysis operations"""
    file_id: str
    analysis_type: str  # descriptive, diagnostic, predictive, prescriptive
    params: Dict[str, Any]
    use_preprocessed: bool = False

class TransformationRequest(BaseModel):
    """Request model for transformation operations"""
    file_id: str
    transformation_type: str  # pivot, melt, groupby, merge
    params: Dict[str, Any]

class UndoOperationRequest(BaseModel):
    """Request model for undoing an operation"""
    file_id: str
    operation_index: int

class ClearOperationsRequest(BaseModel):
    """Request model for clearing all operations"""
    file_id: str

class SingleOperationRequest(BaseModel):
    """Request model for a single preprocessing operation"""
    file_id: str
    operation: Dict[str, Any]

class FileUploadForm(BaseModel):
    """Form data for file upload"""
    engine_type: str

    class Config:
        from_attributes = True  # Updated from orm_mode for Pydantic V2 compatibility