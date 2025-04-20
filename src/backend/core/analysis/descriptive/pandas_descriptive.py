import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from core.analysis.descriptive.base import DescriptiveAnalysisBase

class PandasDescriptiveAnalysis(DescriptiveAnalysisBase):
    """
    Pandas implementation of descriptive analysis strategy.
    """
    
    def analyze(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform descriptive analysis using pandas.
        
        Args:
            data: pandas DataFrame
            params: Parameters for the analysis
                - columns: List of columns to analyze (default: all)
                - include_numeric: Whether to include numeric analysis (default: True)
                - include_categorical: Whether to include categorical analysis (default: True)
                - include_correlations: Whether to include correlation analysis (default: True)
            
        Returns:
            Dictionary containing analysis results
        """
        # Extract parameters
        columns = params.get("columns", data.columns.tolist())
        include_numeric = params.get("include_numeric", True)
        include_categorical = params.get("include_categorical", True)
        include_correlations = params.get("include_correlations", True)
        
        # Filter to selected columns
        selected_data = data[columns]
        
        # Initialize results
        results = {
            "dataset_info": {
                "shape": selected_data.shape,
                "columns": selected_data.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in selected_data.dtypes.items()},
                "missing_values": selected_data.isnull().sum().to_dict(),
            },
            "numeric_analysis": {},
            "categorical_analysis": {},
            "correlations": {}
        }
        
        # Numeric analysis
        if include_numeric:
            numeric_cols = selected_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Basic statistics
                results["numeric_analysis"]["statistics"] = selected_data[numeric_cols].describe().to_dict()
                
                # Distribution information
                results["numeric_analysis"]["skewness"] = {
                    col: float(selected_data[col].skew())
                    for col in numeric_cols
                    if not selected_data[col].isna().all()
                }
                
                results["numeric_analysis"]["kurtosis"] = {
                    col: float(selected_data[col].kurtosis())
                    for col in numeric_cols
                    if not selected_data[col].isna().all()
                }
        
        # Categorical analysis
        if include_categorical:
            categorical_cols = selected_data.select_dtypes(include=["object", "category"]).columns
            if len(categorical_cols) > 0:
                results["categorical_analysis"]["value_counts"] = {
                    col: selected_data[col].value_counts().head(10).to_dict()
                    for col in categorical_cols
                }
                
                results["categorical_analysis"]["unique_counts"] = {
                    col: int(selected_data[col].nunique())
                    for col in categorical_cols
                }
        
        # Correlation analysis
        if include_correlations:
            numeric_cols = selected_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                corr_matrix = selected_data[numeric_cols].corr().to_dict()
                results["correlations"] = corr_matrix
        
        return results