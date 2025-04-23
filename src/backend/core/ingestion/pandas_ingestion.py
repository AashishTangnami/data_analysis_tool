
import pandas as pd
from typing import Any, Dict, Optional
from core.ingestion.base import DataIngestionBase

def validate_and_cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and casts columns to appropriate data types:
    - Converts datetime-like object columns to datetime
    - Converts numeric strings to numbers
    - Converts 'yes'/'no' to booleans if applicable

    Args:
        df: pandas DataFrame to validate and cast

    Returns:
        DataFrame with corrected dtypes
    """
    for col in df.columns:
        col_data = df[col]

        # Skip if already proper dtype
        if pd.api.types.is_datetime64_any_dtype(col_data):
            continue

        # Try datetime
        if col_data.dtype == 'object':
            try:
                df[col] = pd.to_datetime(col_data, errors='raise')
                continue
            except Exception:
                pass

        # Try numeric
        try:
            df[col] = pd.to_numeric(col_data, errors='raise')
            continue
        except Exception:
            pass

        # Try boolean from yes/no or true/false strings
        if col_data.dtype == 'object':
            lower_values = col_data.dropna().astype(str).str.lower().unique()
            if set(lower_values).issubset({'yes', 'no', 'true', 'false'}):
                df[col] = col_data.astype(str).str.lower().map({'yes': True, 'no': False, 'true': True, 'false': False})

    return df

class PandasIngestion(DataIngestionBase):
    """
    Pandas implementation of data ingestion strategy.
    """
    
    def load_data(self, file_path: str, file_type: str, **kwargs) -> pd.DataFrame:
        """
        Load data using pandas.
        
        Args:
            file_path: Path to the file to load
            file_type: Type of file (csv, excel, json)
            **kwargs: Additional arguments for pandas reader
            
        Returns:
            pandas DataFrame
            
        Raises:
            ValueError: If file type is not supported
        """
        if file_type == 'csv':
            df = pd.read_csv(file_path, **kwargs)
        elif file_type == 'excel':
            df = pd.read_excel(file_path, **kwargs)
        elif file_type == 'json':
            df = pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type for pandas ingestion: {file_type}")
        return validate_and_cast_dtypes(df)
