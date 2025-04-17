# core/preprocessing/polars_preprocessing.py
import polars as pl
import numpy as np
from typing import Any, Dict, List, Optional, Union
from core.preprocessing.base import PreprocessingBase


def _is_numeric_dtype(dtype) -> bool:
    """Helper function to check if a polars dtype is numeric."""
    return any(
        isinstance(dtype, t)
        for t in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                 pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    )


class PolarsPreprocessing(PreprocessingBase):
    """
    Polars implementation of preprocessing strategy.
    """
    
    def process(self, data: pl.DataFrame, operations: List[Dict[str, Any]]) -> pl.DataFrame:
        """
        Apply preprocessing operations to polars DataFrame.
        
        Args:
            data: polars DataFrame
            operations: List of preprocessing operations to apply
            
        Returns:
            Preprocessed polars DataFrame
            
        Raises:
            ValueError: If operation type is not supported
        """
        # Create a copy of the data to avoid modifying the original
        result = data.clone()
        
        # Apply each operation in sequence
        for operation in operations:
            op_type = operation.get("type")
            params = operation.get("params", {})
            
            if op_type == "drop_columns":
                result = self._drop_columns(result, **params)
            elif op_type == "fill_missing":
                result = self._fill_missing(result, **params)
            elif op_type == "drop_missing":
                result = self._drop_missing(result, **params)
            elif op_type == "encode_categorical":
                result = self._encode_categorical(result, **params)
            elif op_type == "scale_numeric":
                result = self._scale_numeric(result, **params)
            elif op_type == "apply_function":
                result = self._apply_function(result, **params)
            else:
                raise ValueError(f"Unsupported operation type: {op_type}")
        
        return result
    
    def get_available_operations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of available preprocessing operations.
        
        Returns:
            Dictionary mapping operation names to their metadata
        """
        return {
            "drop_columns": {
                "description": "Drop specified columns from the DataFrame",
                "params": {
                    "columns": "List of column names to drop"
                }
            },
            "fill_missing": {
                "description": "Fill missing values in specified columns",
                "params": {
                    "columns": "List of column names or 'all'",
                    "method": "Method to use: mean, median, mode, constant",
                    "value": "Value to use if method is constant"
                }
            },
            "drop_missing": {
                "description": "Drop rows with missing values",
                "params": {
                    "how": "How to drop: any (drop if any value is missing) or all (drop if all values are missing)"
                }
            },
            "encode_categorical": {
                "description": "Encode categorical variables",
                "params": {
                    "columns": "List of column names to encode",
                    "method": "Method to use: one_hot, label"
                }
            },
            "scale_numeric": {
                "description": "Scale numeric variables",
                "params": {
                    "columns": "List of column names to scale",
                    "method": "Method to use: standard, minmax"
                }
            },
            "apply_function": {
                "description": "Apply a function to specified columns",
                "params": {
                    "columns": "List of column names to apply function to",
                    "function": "Function to apply: log, sqrt, square, absolute"
                }
            }
        }
    
    def _drop_columns(self, data: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """Drop specified columns from the DataFrame."""
        return data.drop(columns)
    
    def _fill_missing(self, data: pl.DataFrame, columns: Union[List[str], str], 
                     method: str = "mean", value: Optional[Any] = None) -> pl.DataFrame:
        """Fill missing values in specified columns."""
        # Handle "all" columns option
        if columns == "all":
            target_columns = data.columns
        else:
            target_columns = columns
        
        # Create expressions for each column
        exprs = []
        for col in data.columns:
            if col in target_columns:
                # Apply the specified method
                if method == "mean" and _is_numeric_dtype(data[col].dtype):
                    expr = pl.col(col).fill_null(pl.col(col).mean())
                elif method == "median" and _is_numeric_dtype(data[col].dtype):
                    expr = pl.col(col).fill_null(pl.col(col).median())
                elif method == "mode":
                    # Mode in polars (most frequent value)
                    mode_val = data.select(
                        pl.col(col).value_counts().sort("count", descending=True).head(1)
                    )[col][0]
                    expr = pl.col(col).fill_null(mode_val)
                elif method == "constant":
                    expr = pl.col(col).fill_null(value)
                else:
                    # If method not applicable, keep the column as is
                    expr = pl.col(col)
            else:
                # Keep columns not in target_columns unchanged
                expr = pl.col(col)
            
            exprs.append(expr)
        
        # Apply all expressions
        return data.select(exprs)
    
    def _drop_missing(self, data: pl.DataFrame, how: str = "any") -> pl.DataFrame:
        """Drop rows with missing values."""
        if how == "any":
            return data.drop_nulls()
        elif how == "all":
            # Keep row if at least one value is not null
            return data.filter(~pl.all_horizontal(pl.all().is_null()))
        else:
            raise ValueError(f"Unsupported 'how' parameter: {how}. Use 'any' or 'all'.")
    
    def _encode_categorical(self, data: pl.DataFrame, columns: List[str], 
                           method: str = "one_hot") -> pl.DataFrame:
        """Encode categorical variables."""
        result = data.clone()
        
        for col in columns:
            if col not in data.columns:
                continue
            
            if method == "one_hot":
                # Get unique values
                unique_vals = data[col].unique().to_list()
                
                # Create one-hot columns
                for val in unique_vals:
                    if val is not None:
                        # Create column name for the dummy
                        dummy_col = f"{col}_{val}"
                        
                        # Create dummy column (1 if equal to val, 0 otherwise)
                        result = result.with_column(
                            pl.when(pl.col(col) == val).then(1).otherwise(0).alias(dummy_col)
                        )
                
                # Drop original column
                result = result.drop(col)
                
            elif method == "label":
                # Create a mapping of unique values to integers
                unique_vals = data[col].unique().to_list()
                mapping = {val: i for i, val in enumerate(unique_vals) if val is not None}
                
                # Replace values with their label
                result = result.with_column(
                    pl.col(col).replace(mapping, default=None).alias(col)
                )
            
            else:
                raise ValueError(f"Unsupported encoding method: {method}")
        
        return result
    
    def _scale_numeric(self, data: pl.DataFrame, columns: List[str], 
                      method: str = "standard") -> pl.DataFrame:
        """Scale numeric variables."""
        result = data.clone()
        
        for col in columns:
            if col not in data.columns:
                continue
            
            # Check if column is numeric
            if not _is_numeric_dtype(data[col].dtype):
                continue
            
            if method == "standard":
                # Standardize to mean=0, std=1
                mean = data[col].mean()
                std = data[col].std()
                
                if std > 0:  # Avoid division by zero
                    result = result.with_column(
                        ((pl.col(col) - mean) / std).alias(col)
                    )
            
            elif method == "minmax":
                # Scale to range [0, 1]
                min_val = data[col].min()
                max_val = data[col].max()
                
                if max_val > min_val:  # Avoid division by zero
                    result = result.with_column(
                        ((pl.col(col) - min_val) / (max_val - min_val)).alias(col)
                    )
            
            else:
                raise ValueError(f"Unsupported scaling method: {method}")
        
        return result
    
    def _apply_function(self, data: pl.DataFrame, columns: List[str], 
                       function: str = "log") -> pl.DataFrame:
        """Apply a function to specified columns."""
        result = data.clone()
        
        for col in columns:
            if col not in data.columns:
                continue
            
            # Check if column is numeric
            if not _is_numeric_dtype(data[col].dtype):
                continue
            
            if function == "log":
                # Apply log transformation (handle negative values)
                min_val = data[col].min()
                offset = 0 if min_val >= 0 else abs(min_val) + 1
                
                result = result.with_column(
                    pl.col(col).map_elements(lambda x: np.log(x + offset) if x is not None else None).alias(col)
                )
            
            elif function == "sqrt":
                # Apply square root (handle negative values)
                min_val = data[col].min()
                offset = 0 if min_val >= 0 else abs(min_val)
                
                result = result.with_column(
                    pl.col(col).map_elements(lambda x: np.sqrt(x + offset) if x is not None else None).alias(col)
                )
            
            elif function == "square":
                result = result.with_column(
                    (pl.col(col) ** 2).alias(col)
                )
            
            elif function == "absolute":
                result = result.with_column(
                    pl.col(col).abs().alias(col)
                )
            
            else:
                raise ValueError(f"Unsupported function: {function}")
        
        return result