
from abc import ABC, abstractmethod
from typing import Any, Dict

class PredictiveAnalysisBase(ABC):
    """
    Base class for all predictive analysis strategies.
    This is the Strategy interface for predictive analysis.
    Also serves as a factory for creating predictive analysis instances.
    """

    @classmethod
    def create(cls, engine_type: str) -> 'PredictiveAnalysisBase':
        """
        Factory method to create a predictive analysis instance based on the engine type.

        Args:
            engine_type: Type of engine to use (pandas, polars)

        Returns:
            Instance of the appropriate predictive analysis strategy

        Raises:
            ValueError: If engine type is not supported
        """
        if engine_type == "pandas":
            from core.analysis.predictive.pandas_predictive import PandasPredictiveAnalysis
            return PandasPredictiveAnalysis()
        elif engine_type == "polars":
            from core.analysis.predictive.polars_predictive import PolarsPredictiveAnalysis
            return PolarsPredictiveAnalysis()
        else:
            raise ValueError(f"Unsupported engine type for predictive analysis: {engine_type}")

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