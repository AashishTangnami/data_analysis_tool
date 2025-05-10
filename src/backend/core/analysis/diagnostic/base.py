from abc import ABC, abstractmethod
from typing import Any, Dict

class DiagnosticAnalysisBase(ABC):
    """
    Base class for all diagnostic analysis strategies.
    This is the Strategy interface for diagnostic analysis.
    """
    
    @abstractmethod
    def analyze(self, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform diagnostic analysis on data.
        
        Args:
            data: Data in the engine's native format
            params: Parameters for the analysis
                - target_column: Target variable for analysis
                - feature_columns: List of features to use
                - run_feature_importance: Whether to run feature importance analysis
                - run_outlier_detection: Whether to run outlier detection
            
        Returns:
            Dictionary containing analysis results
                - feature_importance: Feature importance scores
                - outlier_detection: Outlier detection results
                - correlation_analysis: Correlation with target
        """
        pass