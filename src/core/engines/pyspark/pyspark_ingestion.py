from pyspark.sql import DataFrame, SparkSession
from typing import Any, Dict
from ...interfaces.ingestion_base import DataIngestionBase

class PySparkIngestion(DataIngestionBase):
    def __init__(self, spark: SparkSession = None):
        self.spark = spark or SparkSession.builder.getOrCreate()
        self._readers = {
            'csv': lambda self, path, **kwargs: self.read_csv(path, **kwargs),
            'xlsx': lambda self, path, **kwargs: self.read_excel(path, **kwargs),
            'xls': lambda self, path, **kwargs: self.read_excel(path, **kwargs),
            'parquet': lambda self, path, **kwargs: self.read_parquet(path, **kwargs),
            'json': lambda self, path, **kwargs: self.read_json(path, **kwargs),
            'avro': lambda self, path, **kwargs: self.avro(path, **kwargs),
            'orc': lambda self, path, **kwargs: self.orc(path, **kwargs)
        }

    def read_excel(self, file_path: str, **kwargs) -> DataFrame:
        """Read Excel file (.xlsx or .xls)"""
        file_extension = file_path.split('.')[-1].lower()
        
        try:
            import pandas as pd
            
            if file_extension == 'xlsx':
                try:
                    import openpyxl
                    pdf = pd.read_excel(file_path, engine='openpyxl', **kwargs)
                except ImportError:
                    raise ImportError("openpyxl is required for .xlsx files. Install with: pip install openpyxl")
            elif file_extension == 'xls':
                try:
                    import xlrd
                    pdf = pd.read_excel(file_path, engine='xlrd', **kwargs)
                except ImportError:
                    raise ImportError("xlrd is required for .xls files. Install with: pip install xlrd")
                    
            return self.spark.createDataFrame(pdf)
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {str(e)}")

    def load_data(self, file_path: str, **kwargs) -> DataFrame:
        file_extension = file_path.split('.')[-1].lower()
        reader_method = self.SUPPORTED_FORMATS.get(file_extension)
        
        if not reader_method:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        try:
            reader_func = getattr(self, reader_method)
            # PySpark-specific: handle distributed files
            if '*' in file_path:
                return reader_func(file_path, **kwargs)
            # Handle single file
            return reader_func(file_path, **kwargs)
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
