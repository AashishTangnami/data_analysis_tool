import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from core.preprocessing.base import PreprocessingBase

class PandasPreprocessing(PreprocessingBase):
    """
    Pandas implementation of preprocessing strategy.
    """
    
    def process(self, data: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply preprocessing operations to pandas DataFrame.
        
        Args:
            data: pandas DataFrame
            operations: List of preprocessing operations to apply
            
        Returns:
            Preprocessed pandas DataFrame
            
        Raises:
            ValueError: If operation type is not supported
        """
        # Create a copy of the data to avoid modifying the original
        result = data.copy()
        
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
                    "method": "Method to use: one_hot, label, ordinal"
                }
            },
            "scale_numeric": {
                "description": "Scale numeric variables",
                "params": {
                    "columns": "List of column names to scale",
                    "method": "Method to use: standard, minmax, robust"
                }
            },
            "apply_function": {
                "description": "Apply a function to specified columns",
                "params": {
                    "columns": "List of column names to apply function to",
                    "function": "Function to apply: log, sqrt, square, absolute, etc."
                }
            }
        }
    
    def _drop_columns(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Drop specified columns from the DataFrame."""
        if "all" in columns:
            columns = data.columns.tolist()
        return data.drop(columns=columns)
    
    def _fill_missing(self, data: pd.DataFrame, columns: Union[List[str], str], 
                      method: str = "mean", value: Optional[Any] = None) -> pd.DataFrame:
        """Fill missing values in specified columns."""
        result = data.copy()
        
        # Determine which columns to process
        if columns == "all":
            target_columns = data.columns
        else:
            target_columns = columns
        
        # Apply the specified method
        for col in target_columns:
            if col not in data.columns:
                continue
                
            if method == "mean" and pd.api.types.is_numeric_dtype(data[col]):
                result[col] = data[col].fillna(data[col].mean())
            elif method == "median" and pd.api.types.is_numeric_dtype(data[col]):
                result[col] = data[col].fillna(data[col].median())
            elif method == "mode":
                result[col] = data[col].fillna(data[col].mode()[0])
            elif method == "constant":
                result[col] = data[col].fillna(value)
        
        return result
    
    def _drop_missing(self, data: pd.DataFrame, how: str = "any") -> pd.DataFrame:
        """Drop rows with missing values."""
        return data.dropna(how=how)
    
    def _encode_categorical(self, data: pd.DataFrame, columns: List[str], 
                           method: str = "one_hot") -> pd.DataFrame:
        """Encode categorical variables."""
        result = data.copy()
        
        # Handle 'all' selection
        if "all" in columns:
            columns = [
                col for col in data.columns
                if (
                    (data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col]))
                    and not pd.api.types.is_datetime64_any_dtype(data[col])
                )
            ]

        if not columns:
            return result

        if method == "one_hot":
            # Use pandas get_dummies for one-hot encoding
            dummies = pd.get_dummies(data[columns], prefix=columns)
            result = pd.concat([data.drop(columns=columns), dummies], axis=1)
        elif method == "label":
            # Use pandas factorize for label encoding
            for col in columns:
                result[col], _ = pd.factorize(data[col])
                
        # More encoding methods can be added here
                
        return result
    
    def _scale_numeric(self, data: pd.DataFrame, columns: List[str], 
                      method: str = "standard") -> pd.DataFrame:
        """Scale numeric variables."""
        result = data.copy()
        
        for col in columns:
            if col not in data.columns or not pd.api.types.is_numeric_dtype(data[col]):
                continue
                
            if method == "standard":
                # Standardize to mean=0, std=1
                mean = data[col].mean()
                std = data[col].std()
                if std != 0:  # Avoid division by zero
                    result[col] = (data[col] - mean) / std
            elif method == "minmax":
                # Scale to range [0, 1]
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val > min_val:  # Avoid division by zero
                    result[col] = (data[col] - min_val) / (max_val - min_val)
                    
        # More scaling methods can be added here
                
        return result
    
    def _apply_function(self, data: pd.DataFrame, columns: List[str], 
                       function: str = "log") -> pd.DataFrame:
        """Apply a function to specified columns."""
        result = data.copy()
        
        for col in columns:
            if col not in data.columns or not pd.api.types.is_numeric_dtype(data[col]):
                continue
                
            if function == "log":
                # Apply log transformation (avoiding log of negative numbers)
                result[col] = np.log1p(data[col] - data[col].min() + 1 if data[col].min() < 0 else data[col])
            elif function == "sqrt":
                # Apply square root (avoiding sqrt of negative numbers)
                result[col] = np.sqrt(data[col] - data[col].min() if data[col].min() < 0 else data[col])
            elif function == "square":
                result[col] = data[col] ** 2
            elif function == "absolute":
                result[col] = data[col].abs()
                
        # More functions can be added here
                
        return result