
import pandas as pd
import numpy as np
import os
from typing import Any, Dict
import gc
from core.ingestion.base import DataIngestionBase

class PandasIngestion(DataIngestionBase):
    """
    Pandas implementation of data ingestion strategy with memory optimization.
    
    This class provides methods to efficiently load different file types
    into pandas DataFrames while optimizing memory usage and handling
    large datasets appropriately.
    """
    
    def __init__(self):
        """Initialize with optimized settings for string inference"""
        # Enable string inference with PyArrow for better memory efficiency
        pd.options.future.infer_string = True
    
    def load_data(self, file_path: str, file_type: str, **kwargs) -> pd.DataFrame:
        """
        Load data using pandas with memory optimization.
        
        Args:
            file_path: Path to the file to load
            file_type: Type of file (csv, excel, json, parquet)
            **kwargs: Additional arguments for pandas reader
            
        Returns:
            Memory-optimized pandas DataFrame
            
        Raises:
            ValueError: If file type is not supported
        """
        # Set default memory optimization parameters if not provided
        optimize_memory = kwargs.pop('optimize_memory', True)
        use_chunking = kwargs.pop('use_chunking', False)
        chunk_size = kwargs.pop('chunk_size', 100000)
        
        # Choose appropriate loading method based on file type
        if file_type == 'csv':
            result = self._load_csv(file_path, use_chunking, chunk_size, **kwargs)
        elif file_type == 'excel':
            result = self._load_excel(file_path, **kwargs)
        elif file_type == 'json':
            result = self._load_json(file_path, **kwargs)
        elif file_type == 'parquet':
            result = self._load_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type for pandas ingestion: {file_type}")
        
        # Apply memory optimization if requested
        if optimize_memory:
            result = self._optimize_dtypes(result)
            
        # Force garbage collection to free memory
        gc.collect()
        
        return result
    
    def _load_csv(self, file_path: str, use_chunking: bool, chunk_size: int, **kwargs) -> pd.DataFrame:
        """
        Load CSV file with appropriate optimizations.
        
        For large files, uses chunking and dtype inference to minimize memory usage.
        
        Args:
            file_path: Path to the CSV file
            use_chunking: Whether to use chunking for large files
            chunk_size: Size of chunks when using chunking
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded pandas DataFrame
        """
        # First determine file size to choose appropriate loading strategy
        file_size = os.path.getsize(file_path)
        
        # For small files or when chunking is disabled
        if file_size < 1000 * 1024 * 1024 or not use_chunking:  # Less than 1GB
            # Use PyArrow engine for better performance
            if 'engine' not in kwargs:
                kwargs['engine'] = 'pyarrow'
            
            # Enable low_memory mode for better memory usage
            if 'low_memory' not in kwargs:
                kwargs['low_memory'] = True
                
            return pd.read_csv(file_path, **kwargs)
        
        # For large files, use chunking with PyArrow
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, engine='pyarrow', **kwargs):
            if not chunks:  # First chunk
                dtypes = self._infer_optimal_dtypes(chunk)
            else:
                # Apply optimal dtypes to subsequent chunks
                for col, dtype in dtypes.items():
                    if col in chunk.columns:
                        try:
                            chunk[col] = chunk[col].astype(dtype)
                        except (ValueError, TypeError):
                            pass
            chunks.append(chunk)
                
        result = pd.concat(chunks, ignore_index=True)
        chunks.clear()
        return result
    
    def _load_excel(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load Excel file with appropriate optimizations."""
        if 'engine' not in kwargs:
            kwargs['engine'] = 'openpyxl' if file_path.endswith('.xlsx') else 'xlrd'
                
        return pd.read_excel(file_path, **kwargs)
    
    def _load_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load JSON file with appropriate optimizations."""
        file_size = os.path.getsize(file_path)
        
        # For large files, try to use lines=True if appropriate
        if file_size > 50 * 1024 * 1024:  # Over 50MB
            with open(file_path, 'r') as f:
                first_char = f.read(1)
                if first_char != '[':
                    kwargs['lines'] = True
        
        return pd.read_json(file_path, **kwargs)
    
    def _load_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load Parquet file with optimizations."""
        return pd.read_parquet(file_path, **kwargs)
    
    def _infer_optimal_dtypes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Infer optimal dtypes for DataFrame columns."""
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
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by selecting appropriate data types.
        
        This method:
        1. Converts integers to the smallest possible integer type
        2. Converts floats to float32 where possible
        3. Converts string columns to categorical when beneficial
        4. Uses nullable dtypes for better NA handling
        
        Args:
            df: Input DataFrame to optimize
        
        Returns:
            Memory-optimized DataFrame
        """
        result = df.copy()
        
        # Uncomment to check how much memory has been reduced.
        # start_mem = result.memory_usage(deep=True).sum() / 1024**2
        
        # Process numeric columns
        numeric_columns = result.select_dtypes(include=['int', 'float']).columns
        for col in numeric_columns:
            # Get column stats
            col_series = result[col]
            col_min = col_series.min()
            col_max = col_series.max()
            
            # Check for presence of null values
            has_nulls = col_series.isna().any()
            
            """
            This optimization can be migrated to preprocessing step after ingestion.
            This will save optimization time during ingestion.
            
            """
            # For integers
            if np.issubdtype(col_series.dtype, np.integer):
                if has_nulls:
                    # Use nullable integer type
                    if col_min >= 0:
                        if col_max <= 255:
                            result[col] = col_series.astype(pd.UInt8Dtype())
                        elif col_max <= 65535:
                            result[col] = col_series.astype(pd.UInt16Dtype())
                        elif col_max <= 4294967295:
                            result[col] = col_series.astype(pd.UInt32Dtype())
                        else:
                            result[col] = col_series.astype(pd.UInt64Dtype())
                    else:
                        if col_min >= -128 and col_max <= 127:
                            result[col] = col_series.astype(pd.Int8Dtype())
                        elif col_min >= -32768 and col_max <= 32767:
                            result[col] = col_series.astype(pd.Int16Dtype())
                        elif col_min >= -2147483648 and col_max <= 2147483647:
                            result[col] = col_series.astype(pd.Int32Dtype())
                        else:
                            result[col] = col_series.astype(pd.Int64Dtype())
                else:
                    # Use standard integer type with downcast
                    result[col] = pd.to_numeric(col_series, downcast='integer')
            
            # For floats
            elif np.issubdtype(col_series.dtype, np.floating):
                if has_nulls:
                    # Use nullable float type
                    result[col] = col_series.astype(pd.Float32Dtype())
                else:
                    # Check if float32 precision is sufficient
                    float32_series = col_series.astype(np.float32)
                    if (col_series == float32_series).all():
                        result[col] = float32_series
                    else:
                        result[col] = pd.to_numeric(col_series, downcast='float')
        
        # Process string/object columns
        object_columns = result.select_dtypes(include=['object', 'string']).columns
        for col in object_columns:
            col_series = result[col]
            num_unique = col_series.nunique()
            num_total = len(col_series)
            
            # Convert to categorical if beneficial
            # Use more conservative threshold for very large datasets
            if num_total > 1_000_000:
                threshold = 0.1  # 10% for large datasets
            else:
                threshold = 0.5  # 50% for smaller datasets
            
            if num_unique / num_total < threshold:
                result[col] = col_series.astype('category')
            elif pd.api.types.infer_dtype(col_series) == 'string':
                # Use string dtype for string columns (more efficient than object)
                result[col] = col_series.astype(pd.StringDtype())
        
        # Process datetime columns
        datetime_columns = result.select_dtypes(include=['datetime']).columns
        for col in datetime_columns:
            # Convert to datetime64[ns] for better memory usage
            result[col] = pd.to_datetime(result[col])
        
        # end_mem = result.memory_usage(deep=True).sum() / 1024**2
        # reduction = (start_mem - end_mem) / start_mem
        
        # print(f"Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.2%} reduction)")
        
        return result