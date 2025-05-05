"""
Common analysis operations that can be used across different engine implementations.
This module helps reduce code duplication and ensures consistent behavior.
"""
from typing import Dict, Any, List, Optional, Union, Callable

def get_dataset_info_template() -> Dict[str, Any]:
    """
    Get a template for dataset information.
    
    Returns:
        Dictionary with dataset information structure
    """
    return {
        "shape": None,
        "columns": [],
        "dtypes": {},
        "missing_values": {},
    }

def get_descriptive_analysis_template() -> Dict[str, Any]:
    """
    Get a template for descriptive analysis results.
    
    Returns:
        Dictionary with descriptive analysis structure
    """
    return {
        "dataset_info": get_dataset_info_template(),
        "numeric_analysis": {
            "statistics": {},
            "skewness": {},
            "kurtosis": {}
        },
        "categorical_analysis": {
            "value_counts": {},
            "unique_counts": {}
        },
        "correlations": {}
    }

def get_diagnostic_analysis_template() -> Dict[str, Any]:
    """
    Get a template for diagnostic analysis results.
    
    Returns:
        Dictionary with diagnostic analysis structure
    """
    return {
        "feature_importance": {},
        "outlier_detection": {},
        "correlation_analysis": {}
    }

def extract_common_params(params: Dict[str, Any], default_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract common parameters with defaults.
    
    Args:
        params: Parameters provided by the user
        default_params: Default parameters
        
    Returns:
        Dictionary with extracted parameters
    """
    result = {}
    for key, default_value in default_params.items():
        result[key] = params.get(key, default_value)
    return result

def get_descriptive_params(params: Dict[str, Any], all_columns: List[str]) -> Dict[str, Any]:
    """
    Extract and validate descriptive analysis parameters.
    
    Args:
        params: Parameters provided by the user
        all_columns: List of all column names
        
    Returns:
        Dictionary with extracted parameters
    """
    default_params = {
        "columns": all_columns,
        "include_numeric": True,
        "include_categorical": True,
        "include_correlations": True
    }
    
    extracted_params = extract_common_params(params, default_params)
    
    # Ensure columns is a list
    if isinstance(extracted_params["columns"], str):
        extracted_params["columns"] = [extracted_params["columns"]]
    
    # Filter to valid columns
    extracted_params["columns"] = [col for col in extracted_params["columns"] if col in all_columns]
    
    return extracted_params

def get_diagnostic_params(params: Dict[str, Any], all_columns: List[str]) -> Dict[str, Any]:
    """
    Extract and validate diagnostic analysis parameters.
    
    Args:
        params: Parameters provided by the user
        all_columns: List of all column names
        
    Returns:
        Dictionary with extracted parameters
    """
    # Default parameters
    default_params = {
        "target_column": None,
        "feature_columns": all_columns,
        "run_feature_importance": True,
        "run_outlier_detection": True,
        "outlier_method": "zscore",
        "outlier_threshold": 3.0
    }
    
    extracted_params = extract_common_params(params, default_params)
    
    # Ensure feature_columns is a list
    if isinstance(extracted_params["feature_columns"], str):
        extracted_params["feature_columns"] = [extracted_params["feature_columns"]]
    
    # Filter to valid columns
    extracted_params["feature_columns"] = [
        col for col in extracted_params["feature_columns"] 
        if col in all_columns and col != extracted_params["target_column"]
    ]
    
    # Validate target column
    if extracted_params["target_column"] not in all_columns:
        raise ValueError(f"Target column '{extracted_params['target_column']}' not found in data")
    
    return extracted_params

def validate_analysis_params(params: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate analysis parameters against a schema.
    
    Args:
        params: Parameters to validate
        schema: Parameter schema
        
    Returns:
        Validated parameters
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Check for required parameters
    required_params = schema.get("required", [])
    for param in required_params:
        if param not in params:
            raise ValueError(f"Required parameter '{param}' is missing")
    
    # Check parameter types
    properties = schema.get("properties", {})
    for param, value in params.items():
        if param in properties:
            expected_type = properties[param].get("type")
            if expected_type:
                if expected_type == "array" and not isinstance(value, list):
                    raise ValueError(f"Parameter '{param}' must be a list")
                elif expected_type == "object" and not isinstance(value, dict):
                    raise ValueError(f"Parameter '{param}' must be a dictionary")
                elif expected_type == "string" and not isinstance(value, str):
                    raise ValueError(f"Parameter '{param}' must be a string")
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    raise ValueError(f"Parameter '{param}' must be a number")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    raise ValueError(f"Parameter '{param}' must be a boolean")
    
    return params
