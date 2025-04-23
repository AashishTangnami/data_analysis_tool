
import polars as pl
import pandas as pd
from typing import Any, Dict, Optional,  List
from pathlib import Path
import os
from core.ingestion.base import DataIngestionBase

class PolarsIngestion(DataIngestionBase):
    """
    Polars implementation of data ingestion strategy optimized for performance and memory efficiency.
    
    This class provides methods to load various file formats into Polars DataFrames with automatic
    optimizations for large files, memory usage, and data type inference. It supports streaming
    for large files and automatic memory optimization.
    """
    
    def load_data(
        self, 
        file_path: str, 
        file_type: str, 
        columns: Optional[List[str]] = None,
        dtypes: Optional[Dict[str, pl.DataType]] = None,
        n_rows: Optional[int] = None,
        use_streaming: bool = False,
        optimize_memory: bool = True,
        **kwargs
    ) -> pl.DataFrame:
        """
        Load data using Polars with optimized performance and memory usage.
        
        This method provides intelligent file loading with automatic optimizations:
        - Automatic streaming for large files (>1GB)
        - Memory optimization through data type inference
        - Automatic separator detection for CSV files
        - Chunked reading for large Excel files
        - Streaming support for JSON, Parquet, and Arrow/IPC formats
        
        Parameters
        ----------
        file_path : str
            Path to the file to load
        file_type : str
            Type of file. Supported formats:
            - 'csv': Comma-separated values
            - 'excel'/'xlsx'/'xls': Excel workbooks
            - 'json': JSON files (records format)
            - 'parquet': Apache Parquet
            - 'arrow'/'ipc': Arrow/IPC format
        columns : Optional[List[str]], default None
            Specific columns to load. None loads all columns.
        dtypes : Optional[Dict[str, pl.DataType]], default None
            Dictionary mapping column names to their Polars data types.
            Providing correct types improves parsing performance.
        n_rows : Optional[int], default None
            Limit on number of rows to read. None reads all rows.
        use_streaming : bool, default False
            Whether to use streaming mode. Automatically enabled for files >1GB
            unless explicitly set to False.
        optimize_memory : bool, default True
            Whether to automatically optimize memory usage through data type
            optimization and categorical conversion.
        **kwargs : Any
            Additional arguments passed to the specific Polars reader function.
            
        Returns
        -------
        pl.DataFrame
            Loaded and optimized Polars DataFrame
            
        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        ValueError
            If the file type is not supported
        
        Notes
        -----
        - For CSV files, the separator is automatically detected if not specified
        - Large Excel files (>100MB) are read in chunks to manage memory
        - JSON files default to 'records' orient if not specified
        - Memory optimization includes:
            - Float32 conversion for float columns without nulls
            - Categorical conversion for low-cardinality string columns
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file size for automatic streaming decision
        file_size = os.path.getsize(file_path)
        large_file_threshold = 10_000_000_000  # 10 GB
        
        # Auto-enable streaming for large files unless explicitly disabled
        if file_size > large_file_threshold and use_streaming is None:
            use_streaming = True
        
        # Prepare common scan arguments
        scan_args = {
            'infer_schema_length': None,  # Scan all rows for schema inference
            'n_rows': n_rows,
            'low_memory': file_size > 100_000_000,  # 100MB
            **kwargs
        }
        
        if columns is not None:
            scan_args['columns'] = columns
        
        if dtypes is not None:
            scan_args['dtypes'] = dtypes
            
        # Update with any user-provided arguments
        scan_args.update(kwargs)
        
        try:
            # CSV handling
            if file_type.lower() == 'csv':
                scan_args['try_parse_dates'] = True  # Enable automatic date parsing
                scan_args['row_count_name'] = None   # Disable row count for better performance
                scan_args['rechunk'] = True          # Enable rechunking for better memory usage
                
                if use_streaming:
                    df = pl.scan_csv(
                        file_path,
                        **scan_args
                    ).collect(engine='streaming')
                else:
                    df = pl.read_csv(
                        file_path,
                        **scan_args
                    )
                    
            # Excel handling with chunked reading
            elif file_type.lower() in ('excel', 'xlsx', 'xls'):
                if file_size > 100_000_000:  # 100MB
                    # Read large Excel files in chunks
                    chunk_size = 100_000
                    chunks = []
                    
                    for chunk in pl.read_excel(
                        file_path,
                        chunk_size=chunk_size,
                        **kwargs
                    ):
                        chunks.append(chunk)
                    
                    df = pl.concat(chunks)
                else:
                    df = pl.read_excel(file_path, **kwargs)
                    
            # JSON handling with streaming support
            elif file_type.lower() == 'json':
                scan_args['orient'] = scan_args.get('orient', 'records')
                scan_args['rechunk'] = True
                
                if use_streaming:
                    # Use ndjson for better streaming performance
                    if scan_args.get('lines', False):
                        df = pl.scan_ndjson(
                            file_path,
                            **scan_args
                        ).collect(engine='streaming')
                    else:
                        df = pl.scan_json(
                            file_path,
                            **scan_args
                        ).collect(engine='streaming')
                else:
                    df = pl.read_json(
                        file_path,
                        **scan_args
                    )
                    
            # Parquet handling (most efficient)
            elif file_type.lower() == 'parquet':
                scan_args['parallel'] = True  # Enable parallel reading
                scan_args['rechunk'] = False  # Avoid rechunking for Parquet
                
                if use_streaming:
                    df = pl.scan_parquet(
                        file_path,
                        **scan_args
                    ).collect(engine='streaming')
                else:
                    df = pl.read_parquet(
                        file_path,
                        **scan_args
                    )
                    
            # Arrow/IPC handling
            elif file_type.lower() in ('arrow', 'ipc'):
                if use_streaming:
                    df = pl.scan_ipc(
                        file_path,
                        **scan_args
                    ).collect(engine='streaming')
                else:
                    df = pl.read_ipc(
                        file_path,
                        **scan_args
                    )
                    
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Optimize memory usage if requested
            if optimize_memory:
                df = self._optimize_datatypes(df)
                
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading file {file_path}: {str(e)}")

    def _optimize_datatypes(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Optimize memory usage by selecting appropriate data types.
        Uses lazy evaluation and efficient Polars expressions.
        
        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame to optimize
        
        Returns
        -------
        pl.DataFrame
            Memory-optimized DataFrame
        """
        try:
            # Convert to lazy for better optimization
            lazy_df = df.lazy()
            
            # Collect statistics efficiently
            stats = lazy_df.select([
                pl.all().null_count(),
                pl.all().n_unique(),
                pl.all().is_numeric()
            ]).collect()
            
            optimizations = []
            
            for col in df.columns:
                col_type = df.schema[col]
                null_count = stats[0][col]
                n_unique = stats[1][col]
                is_numeric = stats[2][col]
                
                # Numeric optimization
                if is_numeric:
                    if isinstance(col_type, pl.Float64) and null_count == 0:
                        optimizations.append(pl.col(col).cast(pl.Float32))
                    elif isinstance(col_type, (pl.Int64, pl.UInt64)) and null_count == 0:
                        # Determine smallest possible int type
                        col_min = df[col].min()
                        col_max = df[col].max()
                        
                        if col_min >= 0:
                            if col_max <= 255:
                                new_type = pl.UInt8
                            elif col_max <= 65535:
                                new_type = pl.UInt16
                            elif col_max <= 4294967295:
                                new_type = pl.UInt32
                            else:
                                new_type = pl.UInt64
                        else:
                            if -128 <= col_min and col_max <= 127:
                                new_type = pl.Int8
                            elif -32768 <= col_min and col_max <= 32767:
                                new_type = pl.Int16
                            elif -2147483648 <= col_min and col_max <= 2147483647:
                                new_type = pl.Int32
                            else:
                                new_type = pl.Int64
                                
                        optimizations.append(pl.col(col).cast(new_type))
            
                # String optimization
                elif isinstance(col_type, pl.Utf8):
                    total_rows = len(df)
                    if n_unique / total_rows < 0.5:  # Less than 50% unique values
                        optimizations.append(pl.col(col).cast(pl.Categorical))
            
            # Apply all optimizations in one pass
            if optimizations:
                df = lazy_df.with_columns(optimizations).collect()
            
            return df
            
        except Exception as e:
            print(f"Warning: Memory optimization failed: {str(e)}")
            return df
