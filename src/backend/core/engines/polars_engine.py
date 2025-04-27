
from typing import Dict, Any, Optional, Union, List
import polars as pl
import pandas as pd
import numpy as np
from core.engines.base import EngineBase
from core.ingestion.polars_ingestion import PolarsIngestion
from core.preprocessing.polars_preprocessing import PolarsPreprocessing
from core.analysis.descriptive.base import DescriptiveAnalysisBase
from core.analysis.diagnostic.base import DiagnosticAnalysisBase
from core.analysis.predictive.base import PredictiveAnalysisBase
from core.analysis.prescriptive.base import PrescriptiveAnalysisBase

class PolarsEngine(EngineBase):
    """
    Polars implementation of the Engine interface.
    This is a Concrete Strategy in the Strategy Pattern.
    """

    def __init__(self):
        """Initialize components for each operation type"""
        self.ingestion = PolarsIngestion()
        self.preprocessing = PolarsPreprocessing()
        self.engine_type = "polars"
        # Create analysis strategies using base class factory methods
        self.analysis_strategies = {
            "descriptive": DescriptiveAnalysisBase.create(self.engine_type),
            "diagnostic": DiagnosticAnalysisBase.create(self.engine_type),
            "predictive": PredictiveAnalysisBase.create(self.engine_type),
            "prescriptive": PrescriptiveAnalysisBase.create(self.engine_type)
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

    def get_data_summary(self, data: pl.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of the data.

        Args:
            data: polars DataFrame

        Returns:
            Dictionary containing data summary information
        """
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

        # Get the appropriate analysis strategy
        strategy = self.analysis_strategies[analysis_type]

        # Perform the analysis with error handling
        return strategy.analyze(data, params)

    def apply_single_operation(self, data: pl.DataFrame, operation: Dict[str, Any]) -> pl.DataFrame:
        """
        Apply a single preprocessing operation to data.

        Args:
            data: polars DataFrame
            operation: Single preprocessing operation to apply

        Returns:
            Processed polars DataFrame
        """
        try:
            # Delegate to the preprocessing component
            op_type = operation.get("type")
            params = operation.get("params", {})

            if op_type == "drop_columns":
                return self.preprocessing._drop_columns(data, **params)
            elif op_type == "fill_missing":
                return self.preprocessing._fill_missing(data, **params)
            elif op_type == "drop_missing":
                return self.preprocessing._drop_missing(data, **params)
            elif op_type == "encode_categorical":
                return self.preprocessing._encode_categorical(data, **params)
            elif op_type == "scale_numeric":
                return self.preprocessing._scale_numeric(data, **params)
            elif op_type == "apply_function":
                return self.preprocessing._apply_function(data, **params)
            else:
                raise ValueError(f"Unsupported operation type: {op_type}")
        except Exception as e:
            # Add better error handling
            raise ValueError(f"Error applying operation {op_type}: {str(e)}")

    def to_pandas(self, data: pl.DataFrame) -> pd.DataFrame:
        """
        Convert polars DataFrame to pandas DataFrame.

        Args:
            data: polars DataFrame

        Returns:
            pandas DataFrame
        """
        return data.to_pandas()