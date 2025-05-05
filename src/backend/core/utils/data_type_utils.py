"""
Utility functions for data type handling across different engines.
This module helps reduce code duplication and ensures consistent behavior.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

def is_numeric_dtype(dtype) -> bool:
    """
    Check if a dtype is numeric.
    This is a generic function that works with different engines.
    
    Args:
        dtype: Data type to check
        
    Returns:
        True if dtype is numeric, False otherwise
    """
    # Convert dtype to string for consistent checking
    dtype_str = str(dtype).lower()
    
    # Check for common numeric type names
    return any(num_type in dtype_str for num_type in 
              ['int', 'float', 'double', 'decimal', 'number', 'numeric'])

def is_categorical_dtype(dtype) -> bool:
    """
    Check if a dtype is categorical.
    This is a generic function that works with different engines.
    
    Args:
        dtype: Data type to check
        
    Returns:
        True if dtype is categorical, False otherwise
    """
    # Convert dtype to string for consistent checking
    dtype_str = str(dtype).lower()
    
    # Check for common categorical type names
    return any(cat_type in dtype_str for cat_type in 
              ['object', 'string', 'category', 'str', 'char'])

def is_datetime_dtype(dtype) -> bool:
    """
    Check if a dtype is datetime.
    This is a generic function that works with different engines.
    
    Args:
        dtype: Data type to check
        
    Returns:
        True if dtype is datetime, False otherwise
    """
    # Convert dtype to string for consistent checking
    dtype_str = str(dtype).lower()
    
    # Check for common datetime type names
    return any(dt_type in dtype_str for dt_type in 
              ['datetime', 'timestamp', 'date', 'time'])

def infer_optimal_dtypes_pandas(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Infer optimal dtypes for pandas DataFrame columns.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Dictionary mapping column names to optimal dtypes
    """
    dtypes = {}

    for col in df.columns:
        # Skip columns with mixed types
        if df[col].dtype == 'object':
            # Check if column might be categorical
            if df[col].nunique() < len(df) * 0.5:
                dtypes[col] = 'category'
            continue

        # For numeric columns, try to downcast
        if np.issubdtype(df[col].dtype, np.integer):
            dtypes[col] = pd.Int64Dtype()  # Use nullable integer type
        elif np.issubdtype(df[col].dtype, np.floating):
            dtypes[col] = pd.Float64Dtype()  # Use nullable float type

    return dtypes

def optimize_dtypes_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize pandas DataFrame memory usage by selecting appropriate data types.
    
    Args:
        df: Input DataFrame to optimize
        
    Returns:
        Memory-optimized DataFrame
    """
    result = df.copy()

    # Get optimal dtypes
    optimal_dtypes = infer_optimal_dtypes_pandas(result)

    # Apply optimizations
    for col, dtype in optimal_dtypes.items():
        result[col] = result[col].astype(dtype)

    return result

def get_column_types(df: Any, engine_type: str = "pandas") -> Dict[str, str]:
    """
    Get column types for a DataFrame.
    
    Args:
        df: DataFrame in the engine's native format
        engine_type: Type of engine (pandas, polars, pyspark)
        
    Returns:
        Dictionary mapping column names to type categories ('numeric', 'categorical', 'datetime', 'other')
    """
    column_types = {}
    
    if engine_type == "pandas":
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                column_types[col] = 'numeric'
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                column_types[col] = 'categorical'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                column_types[col] = 'datetime'
            else:
                column_types[col] = 'other'
    elif engine_type == "polars":
        for col in df.columns:
            dtype_str = str(df[col].dtype).lower()
            if any(num_type in dtype_str for num_type in ['int', 'float']):
                column_types[col] = 'numeric'
            elif any(cat_type in dtype_str for cat_type in ['str', 'cat']):
                column_types[col] = 'categorical'
            elif any(dt_type in dtype_str for dt_type in ['date', 'time']):
                column_types[col] = 'datetime'
            else:
                column_types[col] = 'other'
    
    return column_types

def get_numeric_columns(df: Any, engine_type: str = "pandas") -> List[str]:
    """
    Get numeric columns from a DataFrame.
    
    Args:
        df: DataFrame in the engine's native format
        engine_type: Type of engine (pandas, polars, pyspark)
        
    Returns:
        List of numeric column names
    """
    column_types = get_column_types(df, engine_type)
    return [col for col, type_cat in column_types.items() if type_cat == 'numeric']

def get_categorical_columns(df: Any, engine_type: str = "pandas") -> List[str]:
    """
    Get categorical columns from a DataFrame.
    
    Args:
        df: DataFrame in the engine's native format
        engine_type: Type of engine (pandas, polars, pyspark)
        
    Returns:
        List of categorical column names
    """
    column_types = get_column_types(df, engine_type)
    return [col for col, type_cat in column_types.items() if type_cat == 'categorical']

def get_datetime_columns(df: Any, engine_type: str = "pandas") -> List[str]:
    """
    Get datetime columns from a DataFrame.
    
    Args:
        df: DataFrame in the engine's native format
        engine_type: Type of engine (pandas, polars, pyspark)
        
    Returns:
        List of datetime column names
    """
    column_types = get_column_types(df, engine_type)
    return [col for col, type_cat in column_types.items() if type_cat == 'datetime']
