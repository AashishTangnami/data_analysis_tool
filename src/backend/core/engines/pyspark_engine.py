# core/engines/pyspark_engine.py
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from core.engines.base import EngineBase
from core.ingestion.pyspark_ingestion import PySparkIngestion
from core.preprocessing.pyspark_preprocessing import PySparkPreprocessing
from core.analysis.descriptive.pyspark_descriptive import PySparkDescriptiveAnalysis
from core.analysis.diagnostic.pyspark_diagnostic import PySparkDiagnosticAnalysis
from core.analysis.predictive.pyspark_predictive import PySparkPredictiveAnalysis
from core.analysis.prescriptive.pyspark_prescriptive import PySparkPrescriptiveAnalysis

class PySparkEngine(EngineBase):
    """
    PySpark implementation of the Engine interface.
    This is a Concrete Strategy in the Strategy Pattern.
    """
    
    def __init__(self):
        """Initialize Spark session and components for each operation type"""
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("DataAnalysisTool") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        
        self.ingestion = PySparkIngestion(self.spark)
        self.preprocessing = PySparkPreprocessing()
        self.analysis_strategies = {
            "descriptive": PySparkDescriptiveAnalysis(),
            "diagnostic": PySparkDiagnosticAnalysis(),
            "predictive": PySparkPredictiveAnalysis(),
            "prescriptive": PySparkPrescriptiveAnalysis()
        }
    
    def load_data(self, file_path: str, file_type: Optional[str] = None, **kwargs) -> SparkDataFrame:
        """
        Load data using PySpark engine.
        
        Args:
            file_path: Path to the file to load
            file_type: Optional file type override
            **kwargs: Additional arguments to pass to the loader
            
        Returns:
            PySpark DataFrame
        """
        if file_type is None:
            file_type = self.get_file_type(file_path)
        
        return self.ingestion.load_data(file_path, file_type, **kwargs)
    
    def get_data_summary(self, data: SparkDataFrame) -> Dict[str, Any]:
        """
        Generate a summary of the data.
        
        Args:
            data: PySpark DataFrame
            
        Returns:
            Dictionary containing data summary information
        """
        # Get basic information
        columns = data.columns
        
        # Convert schema to dictionary of data types
        dtypes = {field.name: str(field.dataType) for field in data.schema.fields}
        
        # Count null values for each column
        missing_values = {}
        for col in columns:
            missing_count = data.filter(data[col].isNull()).count()
            missing_values[col] = missing_count
        
        summary = {
            "shape": (data.count(), len(columns)),
            "columns": columns,
            "dtypes": dtypes,
            "missing_values": missing_values,
            "numeric_summary": {}
        }
        
        # Add numeric summary for numeric columns
        from pyspark.sql.types import NumericType
        numeric_cols = [field.name for field in data.schema.fields 
                        if isinstance(field.dataType, NumericType)]
        
        if numeric_cols:
            # Calculate descriptive statistics
            numeric_stats = data.select(numeric_cols).summary(
                "mean", "stddev", "min", "25%", "50%", "75%", "max"
            ).collect()
            
            # Convert to structured dictionary
            numeric_summary = {}
            stats_map = {
                "mean": "mean",
                "stddev": "std", 
                "min": "min", 
                "25%": "25%", 
                "50%": "50%", 
                "75%": "75%", 
                "max": "max"
            }
            
            for col in numeric_cols:
                numeric_summary[col] = {}
                for i, stat_name in enumerate(stats_map.keys()):
                    try:
                        # Convert to float, handling potential errors
                        value = float(numeric_stats[i][col])
                    except (ValueError, TypeError):
                        value = None
                    
                    numeric_summary[col][stats_map[stat_name]] = value
            
            summary["numeric_summary"] = numeric_summary
        
        return summary
    
    def preprocess_data(self, data: SparkDataFrame, operations: List[Dict[str, Any]]) -> SparkDataFrame:
        """
        Preprocess data according to specified operations.
        
        Args:
            data: PySpark DataFrame
            operations: List of preprocessing operations to apply
            
        Returns:
            Preprocessed PySpark DataFrame
        """
        return self.preprocessing.process(data, operations)
    
    def analyze_data(self, data: SparkDataFrame, analysis_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data according to specified analysis type.
        
        Args:
            data: PySpark DataFrame
            analysis_type: Type of analysis to perform
            params: Parameters for the analysis
            
        Returns:
            Dictionary containing analysis results
        """
        if analysis_type not in self.analysis_strategies:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        return self.analysis_strategies[analysis_type].analyze(data, params)
    
    def to_pandas(self, data: SparkDataFrame) -> pd.DataFrame:
        """
        Convert PySpark DataFrame to pandas DataFrame.
        
        Args:
            data: PySpark DataFrame
            
        Returns:
            pandas DataFrame
        """
        # Note: This can be memory-intensive for large datasets
        return data.toPandas()