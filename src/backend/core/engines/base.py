from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import os
import pandas as pd  # For type hints

class EngineBase(ABC):
    """
    Base class for all data processing engines.
    This is the Strategy interface in the Strategy Pattern.
    """
    
    @abstractmethod
    def load_data(self, file_path: str, file_type: Optional[str] = None, **kwargs) -> Any:
        """
        Load data from a file using the appropriate method based on file type.
        
        Args:
            file_path: Path to the file to load
            file_type: Optional file type override (csv, excel, json)
            **kwargs: Additional arguments to pass to the loader
            
        Returns:
            Loaded data in the engine's native format
        """
        pass
    
    @abstractmethod
    def get_data_summary(self, data: Any) -> Dict[str, Any]:
        """
        Generate a summary of the data.
        
        Args:
            data: Data in the engine's native format
            
        Returns:
            Dictionary containing data summary information
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, data: Any, operations: List[Dict[str, Any]]) -> Any:
        """
        Preprocess data according to specified operations.
        
        Args:
            data: Data in the engine's native format
            operations: List of preprocessing operations to apply
            
        Returns:
            Preprocessed data in the engine's native format
        """
        pass
    
    @abstractmethod
    def analyze_data(self, data: Any, analysis_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data according to specified analysis type.
        
        Args:
            data: Data in the engine's native format
            analysis_type: Type of analysis to perform (descriptive, diagnostic, predictive, prescriptive)
            params: Parameters for the analysis
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """
        Determine file type from file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type as string (csv, excel, json)
            
        Raises:
            ValueError: If file type is not supported
        """
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()
        
        if extension == '.csv':
            return 'csv'
        elif extension in ['.xls', '.xlsx']:
            return 'excel'
        elif extension == '.json':
            return 'json'
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    # to pandas to dataframe.ls
    @abstractmethod
    def to_pandas(self, data: Any) -> Any:
        """
        Convert the engine's native data format to pandas DataFrame for consistent visualization.
        
        Args:
            data: Data in the engine's native format
            
        Returns:
            pandas DataFrame
        """
        pass