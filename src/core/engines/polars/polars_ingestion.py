import polars as pl
from typing import Any, Dict
from ...interfaces.ingestion_base import DataIngestionBase

class PolarsIngestion(DataIngestionBase):
    SUPPORTED_FORMATS = {
        'csv': 'read_csv',
        'excel': 'read_excel',
        'parquet': 'read_parquet',
        'json': 'read_json',
        'avro': 'avro',
        'orc': 'orc'
    }

    def load_data(self, file_path: str, **kwargs) -> pl.DataFrame:
        file_extension = file_path.split('.')[-1].lower()
        reader_method = self.SUPPORTED_FORMATS.get(file_extension)
        
        if not reader_method:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        try:
            reader_func = getattr(self, reader_method)
            # Polars-specific optimization: use streaming for large files
            if kwargs.get('streaming', False):
                return reader_func(file_path, streaming=True, **kwargs)
            return reader_func(file_path, **kwargs)
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")

    def read_csv(self, file_path: str, **kwargs) -> pl.DataFrame:
        return pl.read_csv(file_path, **kwargs)
    
    def read_excel(self, file_path: str, **kwargs) -> pl.DataFrame:
        try:
            import polars.io.excel as pxl
            return pxl.read_excel(file_path, **kwargs)
        except ImportError:
            raise ImportError("polars excel support required. Install with: pip install polars[excel]")
    
    def read_parquet(self, file_path: str, **kwargs) -> pl.DataFrame:
        return pl.read_parquet(file_path, **kwargs)
    
    def read_json(self, file_path: str, **kwargs) -> pl.DataFrame:
        return pl.read_json(file_path, **kwargs)

    def avro(self, file_path: str, **kwargs) -> pl.DataFrame:
        try:
            return pl.read_avro(file_path, **kwargs)
        except AttributeError:
            # Fallback to using Apache Arrow
            try:
                import pyarrow.avro as avro
                table = avro.read_table(file_path)
                return pl.from_arrow(table)
            except ImportError:
                raise ImportError("pyarrow is required for reading Avro files. Install with: pip install pyarrow")

    def orc(self, file_path: str, **kwargs) -> pl.DataFrame:
        return pl.read_orc(file_path, **kwargs)
