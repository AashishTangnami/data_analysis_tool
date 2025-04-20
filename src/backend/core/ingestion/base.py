
from abc import ABC, abstractmethod
from typing import Any, Optional

class DataIngestionBase(ABC):
    """
    Base class for all data ingestion strategies.
    This is the Strategy interface for data ingestion.
    """
    
    @abstractmethod
    def load_data(self, file_path: str, file_type: str, **kwargs) -> Any:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the file to load
            file_type: Type of file (csv, excel, json)
            **kwargs: Additional arguments for the loader
            
        Returns:
            Loaded data in the engine's native format
        """
        pass