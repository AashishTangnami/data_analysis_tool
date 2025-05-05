
import polars as pl
import numpy as np
import os
from typing import Any, Dict, Optional, Union, Tuple
import gc
from core.ingestion.base import DataIngestionBase

class PolarsIngestion(DataIngestionBase):
    """
    Polars implementation of data ingestion strategy with memory optimization.

    This class provides methods to efficiently load different file types
    into polars DataFrames while optimizing memory usage and handling
    large datasets appropriately.
    """

    def __init__(self):
        """Initialize with optimized settings"""
        pass

    def load_data(self, file_path: str, file_type: str, **kwargs) -> pl.DataFrame:
        """
        Load data using polars with memory optimization.

        Args:
            file_path: Path to the file to load
            file_type: Type of file (csv, excel, json, parquet)
            **kwargs: Additional arguments for polars reader

        Returns:
            Memory-optimized polars DataFrame

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
            raise ValueError(f"Unsupported file type for polars ingestion: {file_type}")

        # Apply memory optimization if requested
        if optimize_memory:
            result = self._optimize_dtypes(result)

        # Force garbage collection to free memory
        gc.collect()

        return result

    def _load_csv(self, file_path: str, use_chunking: bool, chunk_size: int, **kwargs) -> pl.DataFrame:
        """
        Load CSV file with appropriate optimizations.

        For large files, uses chunking and dtype inference to minimize memory usage.

        Args:
            file_path: Path to the CSV file
            use_chunking: Whether to use chunking for large files
            chunk_size: Size of chunks when using chunking
            **kwargs: Additional arguments for pl.read_csv

        Returns:
            Loaded polars DataFrame
        """
        # First determine file size to choose appropriate loading strategy
        file_size = os.path.getsize(file_path)

        # For small files or when chunking is disabled
        if file_size < 1000 * 1024 * 1024 or not use_chunking:  # Less than 1GB
            read_args = {
                'infer_schema_length': 1000,
                'use_pyarrow': True,
                **kwargs
            }
            return pl.read_csv(file_path, **read_args)

        # For large files, use streaming with PyArrow
        chunks = []
        for chunk in pl.read_csv(file_path,
                                 rechunk=False,
                                 streaming=True,
                                 **kwargs):
            if not chunks:  # First chunk
                dtypes = self._infer_optimal_dtypes(chunk)
                # Apply optimized dtypes to first chunk
                if dtypes:
                    optimizations = [
                        pl.col(col_name).cast(dtype)
                        for col_name, dtype in dtypes.items()
                    ]
                    chunk = chunk.with_columns(optimizations)
            else:
                # Apply same optimized dtypes to subsequent chunks
                if dtypes:
                    optimizations = [
                        pl.col(col_name).cast(dtype)
                        for col_name, dtype in dtypes.items()
                    ]
                    chunk = chunk.with_columns(optimizations)
            chunks.append(chunk)

        # Combine chunks
        result = pl.concat(chunks)

        # Clear chunks list to free memory
        chunks.clear()

        return result

    def _load_excel(self, file_path: str, **kwargs) -> pl.DataFrame:
        """
        Load Excel file with appropriate optimizations.

        Args:
            file_path: Path to the Excel file
            **kwargs: Additional arguments for pl.read_excel

        Returns:
            Loaded polars DataFrame
        """
        read_args = {
            'infer_schema_length': 1000,
            **kwargs
        }
        df = pl.read_excel(file_path, **read_args)

        # Apply optimized dtypes
        dtypes = self._infer_optimal_dtypes(df)
        if dtypes:
            optimizations = [
                pl.col(col_name).cast(dtype)
                for col_name, dtype in dtypes.items()
            ]
            df = df.with_columns(optimizations)

        return df

    def _load_json(self, file_path: str, **kwargs) -> pl.DataFrame:
        """
        Load JSON file with appropriate optimizations.

        Args:
            file_path: Path to the JSON file
            **kwargs: Additional arguments for pl.read_json

        Returns:
            Loaded polars DataFrame
        """
        read_args = {
            'infer_schema_length': 1000,
            **kwargs
        }
        df = pl.read_json(file_path, **read_args)

        # Apply optimized dtypes
        dtypes = self._infer_optimal_dtypes(df)
        if dtypes:
            optimizations = [
                pl.col(col_name).cast(dtype)
                for col_name, dtype in dtypes.items()
            ]
            df = df.with_columns(optimizations)

        return df

    def _load_parquet(self, file_path: str, use_chunking: bool = False, chunk_size: int = 100000, **kwargs) -> pl.DataFrame:
        """
        Load Parquet file with appropriate optimizations.

        Args:
            file_path: Path to the Parquet file
            use_chunking: Whether to use chunking for large files
            chunk_size: Size of chunks when using chunking
            **kwargs: Additional arguments for pl.read_parquet

        Returns:
            Loaded polars DataFrame with optimized memory usage
        """
        # First determine file size to choose appropriate loading strategy
        file_size = os.path.getsize(file_path)

        # Set default read arguments
        read_args = {
            'use_pyarrow': True,
            **kwargs
        }

        # For small files or when chunking is disabled
        if file_size < 1000 * 1024 * 1024 or not use_chunking:  # Less than 1GB
            df = pl.read_parquet(file_path, **read_args)

            # Apply optimized dtypes
            dtypes = self._infer_optimal_dtypes(df)
            if dtypes:
                optimizations = [
                    pl.col(col_name).cast(dtype)
                    for col_name, dtype in dtypes.items()
                ]
                df = df.with_columns(optimizations)

            return df

        # For large files, use lazy execution and streaming
        try:
            # Use lazy execution for better memory efficiency
            lazy_df = pl.scan_parquet(file_path, **read_args)

            # Get schema to infer optimal dtypes
            schema_sample = lazy_df.fetch(1000)
            dtypes = self._infer_optimal_dtypes(schema_sample)

            # Apply optimizations if available
            if dtypes:
                optimizations = [
                    pl.col(col_name).cast(dtype)
                    for col_name, dtype in dtypes.items()
                ]
                lazy_df = lazy_df.with_columns(optimizations)

            # Collect the results
            df = lazy_df.collect()
            return df

        except Exception as e:
            # Fallback to standard reading if lazy execution fails
            print(f"Lazy execution failed, falling back to standard reading: {str(e)}")
            df = pl.read_parquet(file_path, **read_args)

            # Apply optimized dtypes
            dtypes = self._infer_optimal_dtypes(df)
            if dtypes:
                optimizations = [
                    pl.col(col_name).cast(dtype)
                    for col_name, dtype in dtypes.items()
                ]
                df = df.with_columns(optimizations)

            return df

    def _infer_optimal_dtypes(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Infer optimal dtypes for DataFrame columns to minimize memory usage.

        Args:
            df: Sample DataFrame to analyze

        Returns:
            Dictionary mapping column names to optimal dtypes
        """
        dtypes = {}

        for col_name, dtype in df.schema.items():
            # Skip columns with mixed types or complex objects
            if isinstance(dtype, pl.Utf8):
                # Check if column might be categorical
                if df[col_name].n_unique() < len(df) * 0.5:
                    dtypes[col_name] = pl.Categorical
                continue

            # Handle numeric columns
            if dtype.is_numeric():
                # # Get min and max values to determine optimal integer type
                col_min = df[col_name].min()
                col_max = df[col_name].max()

                # Choose smallest possible integer type
                if isinstance(dtype, pl.Int64):
                    # Ensure col_min and col_max are not None and are numeric
                    if col_min is not None and col_max is not None and isinstance(col_min, (int, float)) and isinstance(col_max, (int, float)):
                        if col_min >= 0:  # Unsigned
                            if col_max <= 255:
                                dtypes[col_name] = pl.UInt8
                            elif col_max <= 65535:
                                dtypes[col_name] = pl.UInt16
                            elif col_max <= 4294967295:
                                dtypes[col_name] = pl.UInt32
                        else:  # Signed
                            if col_min >= -128 and col_max <= 127:
                                dtypes[col_name] = pl.Int8
                            elif col_min >= -32768 and col_max <= 32767:
                                dtypes[col_name] = pl.Int16
                            elif col_min >= -2147483648 and col_max <= 2147483647:
                                dtypes[col_name] = pl.Int32

                # For float columns, try to use float32 if precision allows
                elif isinstance(dtype, pl.Float64):
                    if (col_min is not None and col_max is not None and
                        isinstance(col_min, (int, float)) and isinstance(col_max, (int, float)) and
                        abs(float(col_max)) < 3.4e38 and abs(float(col_min)) < 3.4e38):
                        dtypes[col_name] = pl.Float32

        return dtypes

    def _optimize_dtypes(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Optimize DataFrame memory usage by converting to appropriate dtypes.

        This method reduces memory usage of the DataFrame by:
        1. Converting integers to the smallest possible type
        2. Converting floats to float32 where possible
        3. Converting string columns to category for low-cardinality data

        Args:
            df: DataFrame to optimize

        Returns:
            Memory-optimized DataFrame
        """
        # result = df.clone()

        result = df.clone()
        # Get memory usage before optimization
        # start_mem_opt = result.estimated_size()

        # Get optimal dtypes
        optimal_dtypes = self._infer_optimal_dtypes(result)

        # Apply optimizations
        # result = df
        if optimal_dtypes:
            optimizations = [
                pl.col(col_name).cast(dtype)
                for col_name, dtype in optimal_dtypes.items()
            ]
            result = result.with_columns(optimizations)

        # Get memory usage after optimization
        # end_mem = result.estimated_size()
        # reduction = (start_mem_opt - end_mem) / start_mem_opt

        # Print memory reduction statistics
        # print(f"Memory usage decreased from {start_mem_opt/1024**2:.2f} MB to {end_mem/1024**2:.2f} MB ({reduction:.2%} reduction)")

        return result
