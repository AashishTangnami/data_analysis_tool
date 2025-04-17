# core/ingestion/pandas_ingestion.py
import pandas as pd
from typing import Any, Dict, Optional
from core.ingestion.base import DataIngestionBase

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
            return pd.read_csv(file_path, **kwargs)
        elif file_type == 'excel':
            return pd.read_excel(file_path, **kwargs)
        elif file_type == 'json':
            return pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type for pandas ingestion: {file_type}")