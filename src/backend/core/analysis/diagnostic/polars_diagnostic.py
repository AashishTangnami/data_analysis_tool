# core/analysis/diagnostic/polars_diagnostic.py
import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional
from core.analysis.diagnostic.base import DiagnosticAnalysisBase


def _is_numeric_dtype(dtype) -> bool:
    """Helper function to check if a polars dtype is numeric."""
    return any(
        isinstance(dtype, t)
        for t in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                 pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    )

class PolarsDiagnosticAnalysis(DiagnosticAnalysisBase):
    """
    Polars implementation of diagnostic analysis strategy.
    """
    
    def analyze(self, data: pl.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform diagnostic analysis using polars.
        
        Args:
            data: polars DataFrame
            params: Parameters for the analysis
                - target_column: Target variable for analysis
                - feature_columns: List of features to use
                - run_feature_importance: Whether to run feature importance analysis
                - run_outlier_detection: Whether to run outlier detection
            
        Returns:
            Dictionary containing analysis results
        """
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        # Extract parameters
        target_column = params.get("target_column")
        feature_columns = params.get("feature_columns", [])
        run_feature_importance = params.get("run_feature_importance", True)
        run_outlier_detection = params.get("run_outlier_detection", True)
        
        # Validate inputs
        if not target_column or target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        feature_columns = [col for col in feature_columns if col in data.columns]
        if not feature_columns:
            raise ValueError("No valid feature columns provided")
        
        # Initialize results
        results = {
            "feature_importance": {},
            "outlier_detection": {},
            "correlation_analysis": {}
        }
        
        # Select relevant columns
        selected_data = data.select([target_column] + feature_columns)
        
        # Handle missing values for analysis
        selected_data = selected_data.drop_nulls()
        
        # Feature importance analysis
        if run_feature_importance:
            # For feature importance, convert to pandas and use sklearn
            # as polars doesn't have native ML algorithms
            pandas_df = selected_data.to_pandas()
            
            # Determine if classification or regression
            is_categorical = False
            if not _is_numeric_dtype(selected_data[target_column].dtype) or selected_data[target_column].n_unique() < 10:
                is_categorical = True
            
            # Prepare features and target
            X = pandas_df[feature_columns]
            y = pandas_df[target_column]
            
            # Convert categorical features to numeric
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = X[col].astype('category').cat.codes
            
            # Train random forest for feature importance
            if is_categorical:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            try:
                model.fit(X, y)
                
                # Get feature importance
                importances = model.feature_importances_
                
                # Create a dictionary of feature importance
                results["feature_importance"] = {
                    feature: float(importance)
                    for feature, importance in zip(feature_columns, importances)
                }
                
                # Sort by importance
                results["feature_importance"] = dict(sorted(
                    results["feature_importance"].items(),
                    key=lambda x: x[1],
                    reverse=True
                ))
            
            except Exception as e:
                results["feature_importance"] = {"error": str(e)}
        
        # Outlier detection
        if run_outlier_detection:
            outlier_results = {}
            
            for col in feature_columns:
                if _is_numeric_dtype(selected_data[col].dtype):
                    # Calculate z-score
                    mean = selected_data[col].mean()
                    std = selected_data[col].std()
                    
                    if std > 0:
                        # Create z-score
                        z_scores = ((selected_data[col] - mean) / std).alias("z_score")
                        
                        # Find outliers (|z| > 3)
                        outliers = selected_data.with_column(z_scores).filter(pl.abs(pl.col("z_score")) > 3)
                        
                        # Save results
                        outlier_results[col] = {
                            "mean": float(mean),
                            "std": float(std),
                            "outlier_count": outliers.height,
                            "outlier_percentage": outliers.height / selected_data.height * 100
                        }
            
            results["outlier_detection"] = outlier_results
        
        # Correlation analysis with target
        corr_results = {}
        
        for col in feature_columns:
            if _is_numeric_dtype(selected_data[col].dtype) and _is_numeric_dtype(selected_data[target_column].dtype):
                # Calculate Pearson correlation
                corr = selected_data.select(pl.corr(col, target_column)).item()
                
                # p-value calculation would require scipy, so we'll use a placeholder
                p_value = 0.0  # Placeholder
                
                corr_results[col] = {
                    "correlation": float(corr) if corr is not None else 0.0,
                    "p_value": p_value
                }
        
        # Sort by absolute correlation
        results["correlation_analysis"] = dict(sorted(
            corr_results.items(),
            key=lambda x: abs(x[1]["correlation"]),
            reverse=True
        ))
        
        return results