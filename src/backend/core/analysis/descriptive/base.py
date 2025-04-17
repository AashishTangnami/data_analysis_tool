# core/analysis/descriptive/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class DescriptiveAnalysisBase(ABC):
    """
    Base class for all descriptive analysis strategies.
    This is the Strategy interface for descriptive analysis.
    """
    
    @abstractmethod
    def analyze(self, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform descriptive analysis on data.
        
        Args:
            data: Data in the engine's native format
            params: Parameters for the analysis
            
        Returns:
            Dictionary containing analysis results
        """
        pass