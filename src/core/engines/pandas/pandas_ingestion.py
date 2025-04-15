from typing import Dict, Callable
import pandas as pd
from ...interfaces.ingestion_base import DataIngestionBase
from .config import FORMAT_DEFAULTS
import logging

logger = logging.getLogger(__name__)

class PandasIngestion(DataIngestionBase):
    """Handles data ingestion operations using pandas."""
    
    def __init__(self) -> None:
        """Initialize the ingestion handler with reader mappings."""
        self._readers: Dict[str, Callable] = {
            'read_csv': self.read_csv,
            'read_parquet': self.read_parquet,
            'read_json': self.read_json,
            'read_excel': self.read_excel,
            'avro': self.avro,
            'orc': self.orc
        }

    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data using appropriate reader based on file extension."""
        try:
            # 1. Validate file
            self.validate_file(file_path)
            
            # 2. Get appropriate reader
            reader_method = self.get_reader_method(file_path)
            reader_func = self._readers.get(reader_method)
            
            if not reader_func:
                raise ValueError(f"Reader not implemented: {reader_method}")
            
            # 3. Apply engine-specific optimizations
            reader_kwargs = self._get_reader_defaults(reader_method)
            reader_kwargs.update(kwargs)
            
            # 4. Load data
            logger.debug(f"Reading file {file_path} using {reader_method}")
            return reader_func(file_path, **reader_kwargs)
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise ValueError(f"Error reading file: {str(e)}") from e

    def read_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read CSV file with optimized defaults."""
        reader_kwargs = self._get_reader_defaults('read_csv')
        reader_kwargs.update(kwargs)
        return pd.read_csv(file_path, **reader_kwargs)

    def read_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read Parquet file with optimized defaults."""
        reader_kwargs = self._get_reader_defaults('read_parquet')
        reader_kwargs.update(kwargs)
        return pd.read_parquet(file_path, **reader_kwargs)

    def read_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read JSON file with optimized defaults."""
        reader_kwargs = self._get_reader_defaults('read_json')
        reader_kwargs.update(kwargs)
        return pd.read_json(file_path, **reader_kwargs)

    def read_excel(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read Excel file with optimized defaults."""
        reader_kwargs = self._get_reader_defaults('read_excel')
        reader_kwargs.update(kwargs)
        return pd.read_excel(file_path, **reader_kwargs)

    def avro(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read Avro file - Not supported in Pandas."""
        raise NotImplementedError(
            "Avro format is not supported in Pandas engine. "
            "Use Polars or PySpark engine for Avro support."
        )

    def orc(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read ORC file - Not supported in Pandas."""
        raise NotImplementedError(
            "ORC format is not supported in Pandas engine. "
            "Use Polars or PySpark engine for ORC support."
        )

'''
reindexing the codebase is the workflow.

in main.py at base file dir. enginebase has its own data loading policies base on backend engine.

in engine_context it loads the data using load_data from engine_base in dependencies.py, but where is it being used ??

we also have ingestion_base.py that is being used in pandas_ingestion.py. but i see no connection between load_data in previous file dependencies.py and this in DataIngestionBase.py. 

the load_data should index any relevant function based on file type based on the extension so it maches the engine type for the error free data ingestion.

The workflow here doesnot match actual workflow required.
