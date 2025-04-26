from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np
from core.engines.base import EngineBase
from core.ingestion.pandas_ingestion import PandasIngestion
from core.preprocessing.pandas_preprocessing import PandasPreprocessing
from core.analysis.descriptive.pandas_descriptive import PandasDescriptiveAnalysis
from core.analysis.diagnostic.pandas_diagnostic import PandasDiagnosticAnalysis
from core.analysis.predictive.pandas_predictive import PandasPredictiveAnalysis
from core.analysis.prescriptive.pandas_prescriptive import PandasPrescriptiveAnalysis

class PandasEngine(EngineBase):
    """
    Pandas implementation of the Engine interface.
    This is a Concrete Strategy in the Strategy Pattern.
    """
    
    def __init__(self):
        """Initialize components for each operation type"""
        self.ingestion = PandasIngestion()
        self.preprocessing = PandasPreprocessing()
        self.analysis_strategies = {
            "descriptive": PandasDescriptiveAnalysis(),
            "diagnostic": PandasDiagnosticAnalysis(),
            "predictive": PandasPredictiveAnalysis(),
            "prescriptive": PandasPrescriptiveAnalysis()
        }
    
    def load_data(self, file_path: str, file_type: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data using Pandas engine.
        
        Args:
            file_path: Path to the file to load
            file_type: Optional file type override
            **kwargs: Additional arguments to pass to the loader
            
        Returns:
            pandas DataFrame
        """
        if file_type is None:
            file_type = self.get_file_type(file_path)
        
        return self.ingestion.load_data(file_path, file_type, **kwargs)
    
    def apply_single_operation(self, data: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply a single preprocessing operation to a pandas DataFrame.
        
        Args:
            data: pandas DataFrame to process
            operation: Single preprocessing operation to apply
            
        Returns:
            Processed pandas DataFrame
        """
        if not self.preprocessor:
            from core.preprocessing.pandas_preprocessing import PandasPreprocessing
            self.preprocessor = PandasPreprocessing()
        
        # Apply the single operation
        op_type = operation.get("type")
        params = operation.get("params", {})
        
        # Use the appropriate method based on operation type
        if op_type == "drop_columns":
            return self.preprocessor._drop_columns(data, **params)
        elif op_type == "fill_missing":
            return self.preprocessor._fill_missing(data, **params)
        elif op_type == "drop_missing":
            return self.preprocessor._drop_missing(data, **params)
        elif op_type == "encode_categorical":
            return self.preprocessor._encode_categorical(data, **params)
        elif op_type == "scale_numeric":
            return self.preprocessor._scale_numeric(data, **params)
        elif op_type == "apply_function":
            return self.preprocessor._apply_function(data, **params)
        else:
            raise ValueError(f"Unsupported operation type: {op_type}")

    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of the data.
        
        Args:
            data: pandas DataFrame
            
        Returns:
            Dictionary containing data summary information
        """
        summary = {
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "missing_values": data.isnull().sum().to_dict(),
            "numeric_summary": {}
        }
        
        # Add numeric summary for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            summary["numeric_summary"] = data[numeric_cols].describe().to_dict()
        
        return summary
    
    def preprocess_data(self, data: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Preprocess data according to specified operations.
        
        Args:
            data: pandas DataFrame
            operations: List of preprocessing operations to apply
            
        Returns:
            Preprocessed pandas DataFrame
        """
        return self.preprocessing.process(data, operations)
    
    def analyze_data(self, data: pd.DataFrame, analysis_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data according to specified analysis type.
        
        Args:
            data: pandas DataFrame
            analysis_type: Type of analysis to perform
            params: Parameters for the analysis
            
        Returns:
            Dictionary containing analysis results
        """
        if analysis_type not in self.analysis_strategies:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        return self.analysis_strategies[analysis_type].analyze(data, params)
    
    def to_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data to pandas DataFrame (already in pandas format).
        
        Args:
            data: pandas DataFrame
            
        Returns:
            Same pandas DataFrame
        """
        return data