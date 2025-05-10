
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class PreprocessingBase(ABC):
    """
    Base class for all preprocessing strategies.
    This is the Strategy interface for preprocessing.
    """
    
    @abstractmethod
    def process(self, data: Any, operations: List[Dict[str, Any]]) -> Any:
        """
        Apply preprocessing operations to data.
        
        Args:
            data: Data in the engine's native format
            operations: List of preprocessing operations to apply
            
        Returns:
            Preprocessed data in the engine's native format
        """
        pass
    
    @abstractmethod
    def get_available_operations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of available preprocessing operations.
        
        Returns:
            Dictionary mapping operation names to their metadata
        """
        pass