from typing import Dict, Any, List, Optional
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

class FileUploadForm(BaseModel):
    """Form data for file upload"""
    engine_type: str
    
    class Config:
        orm_mode = True