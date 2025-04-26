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
    
    def is_operation_valid(self, data: pd.DataFrame, operation: Dict, operation_history: List[Dict]) -> bool:
        """Check if an operation is valid given the current data state and operation history."""
        op_type = operation.get("type")
        params = operation.get("params", {})
        
        if op_type == "drop_columns":
            # Check if columns exist
            columns = params.get("columns", [])
            if "all" in columns:
                return len(data.columns) > 0
            return all(col in data.columns for col in columns)
        
        elif op_type == "fill_missing":
            # Check if there are missing values to fill
            columns = params.get("columns", [])
            if "all" in columns:
                return data.isna().any().any()
            return any(data[col].isna().any() for col in columns if col in data.columns)
        
        elif op_type == "drop_missing":
            # Check if there are missing values to drop
            return data.isna().any().any()
        
        elif op_type == "encode_categorical":
            # Check if categorical columns exist
            columns = params.get("columns", [])
            categorical_cols = [
                col for col in data.columns
                if (data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col]))
            ]
            if "all" in columns:
                return len(categorical_cols) > 0
            return any(col in categorical_cols for col in columns)
        
        return True
        
    # Add a method to get available columns (excluding dropped ones)
    def get_available_columns(self, data: pd.DataFrame, operation_history: List[Dict]) -> List[str]:
        """Get available columns considering previous drop operations."""
        available_columns = data.columns.tolist()
        
        # Check for dropped columns in operation history
        for op in operation_history:
            if op.get("type") == "drop_columns" and "columns" in op.get("params", {}):
                dropped_cols = op["params"]["columns"]
                if "all" in dropped_cols:
                    return []  # All columns were dropped
                available_columns = [col for col in available_columns if col not in dropped_cols]
        
        return available_columns


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
                       method: str = "one_hot", target_column: str = None) -> pd.DataFrame:
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
        elif method == "ordinal":
            # For ordinal encoding, we need a mapping
            for col in columns:
                categories = data[col].dropna().unique()
                mapping = {cat: i for i, cat in enumerate(categories)}
                result[col] = data[col].map(mapping)
                # Handle NaN values
                if data[col].isna().any():
                    result[col] = result[col].fillna(-1)
        elif method == "count":
            # Count encoding
            for col in columns:
                count_map = data[col].value_counts().to_dict()
                result[col] = data[col].map(count_map)
        elif method == "target":
            # Target encoding requires a target column
            if target_column is None or target_column not in data.columns:
                raise ValueError("Target encoding requires a valid target column")
            
            for col in columns:
                # Calculate mean target value for each category
                target_means = data.groupby(col)[target_column].mean().to_dict()
                result[col] = data[col].map(target_means)
        elif method == "leave_one_out":
            # Simplified leave-one-out implementation
            if target_column is None or target_column not in data.columns:
                raise ValueError("Leave-one-out encoding requires a valid target column")
            
            for col in columns:
                # For each row, calculate mean excluding the current row
                result[col] = data.apply(
                    lambda x: data[(data[col] == x[col]) & (data.index != x.name)][target_column].mean()
                    if len(data[(data[col] == x[col]) & (data.index != x.name)]) > 0
                    else data[target_column].mean(),
                    axis=1
                )
        elif method == "catboost":
            # Simplified CatBoost encoding
            if target_column is None or target_column not in data.columns:
                raise ValueError("CatBoost encoding requires a valid target column")
            
            for col in columns:
                # Create a random permutation
                np.random.seed(42)
                random_order = np.random.permutation(len(data))
                ordered_data = data.iloc[random_order].copy()
                
                # Apply ordered target statistics
                ordered_data[f'{col}_encoded'] = 0.0
                prior = 0.5  # Prior parameter
                
                for i in range(len(ordered_data)):
                    current_category = ordered_data.iloc[i][col]
                    past_data = ordered_data.iloc[:i]
                    
                    if len(past_data) == 0:
                        # For first observation, use prior
                        encoded_value = prior
                    else:
                        category_data = past_data[past_data[col] == current_category]
                        if len(category_data) == 0:
                            # If category not seen before, use prior
                            encoded_value = prior
                        else:
                            # Calculate statistic
                            count_in_class = sum(category_data[target_column])
                            total_count = len(category_data)
                            encoded_value = (count_in_class + prior) / (total_count + 1)
                    
                    ordered_data.iloc[i, ordered_data.columns.get_loc(f'{col}_encoded')] = encoded_value
                
                # Restore original order and update result
                reverse_mapping = {new_idx: old_idx for old_idx, new_idx in enumerate(random_order)}
                ordered_indices = [reverse_mapping[i] for i in range(len(data))]
                result[col] = ordered_data.iloc[ordered_indices][f'{col}_encoded'].values
        
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