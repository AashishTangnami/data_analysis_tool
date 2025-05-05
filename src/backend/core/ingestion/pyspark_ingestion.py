
from typing import Any, Dict, Optional
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from core.ingestion.base import DataIngestionBase

class PySparkIngestion(DataIngestionBase):
    """
    PySpark implementation of data ingestion strategy.
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize with SparkSession.

        Args:
            spark: Active SparkSession
        """
        self.spark = spark

    def load_data(self, file_path: str, file_type: str, **kwargs) -> SparkDataFrame:
        """
        Load data using PySpark.

        Args:
            file_path: Path to the file to load
            file_type: Type of file (csv, excel, json, parquet)
            **kwargs: Additional arguments for PySpark reader

        Returns:
            PySpark DataFrame

        Raises:
            ValueError: If file type is not supported
        """
        # Set header option to True by default for CSV files
        if file_type == 'csv' and 'header' not in kwargs:
            kwargs['header'] = True

        if file_type == 'csv':
            return self.spark.read.csv(file_path, **kwargs)
        elif file_type == 'json':
            return self.spark.read.json(file_path, **kwargs)
        elif file_type == 'parquet':
            return self.spark.read.parquet(file_path, **kwargs)
        elif file_type == 'excel':
            # PySpark doesn't have native Excel support, using 3rd party library
            try:
                # Using com.crealytics Spark Excel library
                return (self.spark.read
                        .format("com.crealytics.spark.excel")
                        .option("header", "true")
                        .option("inferSchema", "true")
                        .option("dataAddress", "'Sheet1'!A1")  # Adjust as needed
                        .load(file_path))
            except Exception as e:
                # Fallback: Read using pandas and convert to Spark
                import pandas as pd
                pandas_df = pd.read_excel(file_path, **kwargs)
                return self.spark.createDataFrame(pandas_df)
        else:
            raise ValueError(f"Unsupported file type for PySpark ingestion: {file_type}")