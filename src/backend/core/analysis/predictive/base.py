
from abc import ABC, abstractmethod
from typing import Any, Dict

class PredictiveAnalysisBase(ABC):
    """
    Base class for all predictive analysis strategies.
    This is the Strategy interface for predictive analysis.
    """
    
    @abstractmethod
    def analyze(self, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform predictive analysis on data.
        
        Args:
            data: Data in the engine's native format
            params: Parameters for the analysis
                - target_column: Target variable to predict
                - feature_columns: List of features to use
                - problem_type: 'regression' or 'classification'
                - model_type: Type of model to train
                - test_size: Size of test set
            
        Returns:
            Dictionary containing analysis results
                - model_performance: Model evaluation metrics
                - feature_importance: Feature importance scores
                - predictions: Sample predictions
        """
        pass