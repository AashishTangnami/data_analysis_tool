"""
Standardized return types for consistent interfaces.
"""
from typing import Dict, List, Any, Optional, TypedDict, Union

class DataSummary(TypedDict):
    """Standardized return type for data summary."""
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    numeric_summary: Dict[str, Dict[str, Any]]
    error: Optional[str]

class AnalysisResult(TypedDict):
    """Standardized return type for analysis results."""
    dataset_info: DataSummary
    results: Dict[str, Any]
    error: Optional[str]

class DescriptiveAnalysisResult(AnalysisResult):
    """Standardized return type for descriptive analysis."""
    numeric_analysis: Dict[str, Any]
    categorical_analysis: Dict[str, Any]
    correlations: Dict[str, Any]

class DiagnosticAnalysisResult(AnalysisResult):
    """Standardized return type for diagnostic analysis."""
    feature_importance: Dict[str, Any]
    outlier_detection: Dict[str, Any]
    correlation_analysis: Dict[str, Any]

class PredictiveAnalysisResult(AnalysisResult):
    """Standardized return type for predictive analysis."""
    model_performance: Dict[str, Any]
    feature_importance: Dict[str, Any]
    predictions: List[Dict[str, Any]]

class PrescriptiveAnalysisResult(AnalysisResult):
    """Standardized return type for prescriptive analysis."""
    optimization_results: Dict[str, Any]
    scenario_comparison: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]

class PreprocessingResult(TypedDict):
    """Standardized return type for preprocessing operations."""
    original_data_summary: DataSummary
    processed_data_summary: DataSummary
    operations_applied: List[Dict[str, Any]]
    error: Optional[str]
