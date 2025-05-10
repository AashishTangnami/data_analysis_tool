from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class DataResponse(BaseModel):
    """Response model for data operations"""
    file_id: str
    summary: Dict[str, Any]
    preview: List[Dict[str, Any]]
    message: str

class PreprocessingResponse(BaseModel):
    """Response model for preprocessing operations"""
    file_id: str
    original_summary: Dict[str, Any]
    processed_summary: Dict[str, Any]
    preview: List[Dict[str, Any]]
    operations_applied: List[Dict[str, Any]]
    message: str

class AnalysisResponse(BaseModel):
    """Response model for analysis operations"""
    file_id: str
    analysis_type: str
    results: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    message: str

class ErrorResponse(BaseModel):
    """Response model for errors"""
    detail: str