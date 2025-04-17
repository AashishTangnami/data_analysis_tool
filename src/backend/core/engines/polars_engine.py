# core/engines/polars_engine.py
from typing import Dict, Any, Optional, Union, List
import polars as pl
import pandas as pd
import numpy as np
from core.engines.base import EngineBase
from core.ingestion.polars_ingestion import PolarsIngestion
from core.preprocessing.polars_preprocessing import PolarsPreprocessing
from core.analysis.descriptive.polars_descriptive import PolarsDescriptiveAnalysis
from core.analysis.diagnostic.polars_diagnostic import PolarsDiagnosticAnalysis
from core.analysis.predictive.polars_predictive import PolarsPredictiveAnalysis
from core.analysis.prescriptive.polars_prescriptive import PolarsPrescriptiveAnalysis


def _is_numeric_dtype(dtype) -> bool:
    """Helper function to check if a polars dtype is numeric."""
    return any(
        isinstance(dtype, t)
        for t in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                 pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    )

class PolarsEngine(EngineBase):
    """
    Polars implementation of the Engine interface.
    This is a Concrete Strategy in the Strategy Pattern.
    """
    
    def __init__(self):
        """Initialize components for each operation type"""
        self.ingestion = PolarsIngestion()
        self.preprocessing = PolarsPreprocessing()
        self.analysis_strategies = {
            "descriptive": PolarsDescriptiveAnalysis(),
            "diagnostic": PolarsDiagnosticAnalysis(),
            "predictive": PolarsPredictiveAnalysis(),
            "prescriptive": PolarsPrescriptiveAnalysis()
        }
    
    def load_data(self, file_path: str, file_type: Optional[str] = None, **kwargs) -> pl.DataFrame:
        """
        Load data using Polars engine.
        
        Args:
            file_path: Path to the file to load
            file_type: Optional file type override
            **kwargs: Additional arguments to pass to the loader
            
        Returns:
            polars DataFrame
        """
        if file_type is None:
            file_type = self.get_file_type(file_path)
        
        return self.ingestion.load_data(file_path, file_type, **kwargs)
    
    # def get_data_summary(self, data: pl.DataFrame) -> Dict[str, Any]:
    #     """
    #     Generate a summary of the data.
        
    #     Args:
    #         data: polars DataFrame
            
    #     Returns:
    #         Dictionary containing data summary information
    #     """
    #     # Convert column data types to strings for JSON serialization
    #     dtypes = {col: str(dtype) for col, dtype in zip(data.columns, data.dtypes)}
        
    #     # Count missing values
    #     missing_values = {col: int(data.select(pl.col(col).is_null().sum())[0, 0]) for col in data.columns}
        
    #     summary = {
    #         "shape": (data.height, data.width),
    #         "columns": data.columns,
    #         "dtypes": dtypes,
    #         "missing_values": missing_values,
    #         "numeric_summary": {}
    #     }
        
    #     # Add numeric summary for numeric columns
    #     numeric_cols = [col for col, dtype in zip(data.columns, data.dtypes) 
    #                     if _is_numeric_dtype(dtype)]
        
    #     if numeric_cols:
    #         # Calculate descriptive statistics
    #         desc_stats = data.select([
    #             pl.col(col).mean().alias(f"{col}_mean"),
    #             pl.col(col).std().alias(f"{col}_std"),
    #             pl.col(col).min().alias(f"{col}_min"),
    #             pl.col(col).quantile(0.25).alias(f"{col}_25%"),
    #             pl.col(col).median().alias(f"{col}_50%"),
    #             pl.col(col).quantile(0.75).alias(f"{col}_75%"),
    #             pl.col(col).max().alias(f"{col}_max")
    #         ] for col in numeric_cols).to_dict(as_series=False)
            
    #         # Reorganize the structure to match pandas format
    #         numeric_summary = {}
    #         for col in numeric_cols:
    #             numeric_summary[col] = {
    #                 "mean": desc_stats.get(f"{col}_mean", [None])[0],
    #                 "std": desc_stats.get(f"{col}_std", [None])[0],
    #                 "min": desc_stats.get(f"{col}_min", [None])[0],
    #                 "25%": desc_stats.get(f"{col}_25%", [None])[0],
    #                 "50%": desc_stats.get(f"{col}_50%", [None])[0],
    #                 "75%": desc_stats.get(f"{col}_75%", [None])[0],
    #                 "max": desc_stats.get(f"{col}_max", [None])[0]
    #             }
            
    #         summary["numeric_summary"] = numeric_summary
        
    #     return summary
    def get_data_summary(self, data: pl.DataFrame) -> Dict[str, Any]:
        """"""
        try:
            # Schema-based operations (O(1) per column access)
            schema = data.schema
            dtypes = {col: str(dtype) for col, dtype in schema.items()}
            missing_values = data.null_count().to_dicts()[0]  # Single O(n) pass
            
            # Numeric identification using schema (more reliable than string parsing)
            numeric_cols = [col for col, dtype in schema.items() if dtype.is_numeric()]
            
            # Initialize summary structure
            summary = {
                "shape": (data.height, len(schema)),
                "columns": list(schema.keys()),
                "dtypes": dtypes,
                "missing_values": missing_values,
                "numeric_summary": {}
            }

            # Batch compute statistics for numeric columns
            if numeric_cols:
                try:
                    # Single pass for all statistics using describe()
                    desc = data.select(numeric_cols).describe()
                    desc_dict = desc.to_dicts()[0]
                    
                    # Map statistics to columns
                    for col in numeric_cols:
                        col_stats = {
                            "mean": desc_dict.get(f"{col}_mean"),
                            "std": desc_dict.get(f"{col}_std"),
                            "min": desc_dict.get(f"{col}_min"),
                            "25%": desc_dict.get(f"{col}_25%"),
                            "50%": desc_dict.get(f"{col}_50%"),
                            "75%": desc_dict.get(f"{col}_75%"),
                            "max": desc_dict.get(f"{col}_max")
                        }
                        summary["numeric_summary"][col] = col_stats
                        
                except Exception as e:
                    # Fallback to column-by-column if describe() fails
                    for col in numeric_cols:
                        try:
                            col_stats = data.select(
                                pl.col(col).mean().alias("mean"),
                                pl.col(col).std().alias("std"),
                                pl.col(col).min().alias("min"),
                                pl.col(col).quantile(0.25).alias("25%"),
                                pl.col(col).median().alias("50%"),
                                pl.col(col).quantile(0.75).alias("75%"),
                                pl.col(col).max().alias("max")
                            ).to_dicts()[0]
                            
                            summary["numeric_summary"][col] = col_stats
                        except:
                            continue

            return summary

        except Exception as e:
            return {
                "shape": (data.height, len(data.schema)),
                "columns": data.columns,
                "dtypes": {col: str(dtype) for col, dtype in data.schema.items()},
                "error": f"Summary generation failed: {str(e)}"
            }

    def preprocess_data(self, data: pl.DataFrame, operations: List[Dict[str, Any]]) -> pl.DataFrame:
        """
        Preprocess data according to specified operations.
        
        Args:
            data: polars DataFrame
            operations: List of preprocessing operations to apply
            
        Returns:
            Preprocessed polars DataFrame
        """
        return self.preprocessing.process(data, operations)
    
    def analyze_data(self, data: pl.DataFrame, analysis_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data according to specified analysis type.
        
        Args:
            data: polars DataFrame
            analysis_type: Type of analysis to perform
            params: Parameters for the analysis
            
        Returns:
            Dictionary containing analysis results
        """
        if analysis_type not in self.analysis_strategies:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        return self.analysis_strategies[analysis_type].analyze(data, params)
    
    def to_pandas(self, data: pl.DataFrame) -> pd.DataFrame:
        """
        Convert polars DataFrame to pandas DataFrame.
        
        Args:
            data: polars DataFrame
            
        Returns:
            pandas DataFrame
        """
        return data.to_pandas()