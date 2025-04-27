from abc import ABC, abstractmethod
from typing import Any, Dict

class DiagnosticAnalysisBase(ABC):
    """
    Base class for all diagnostic analysis strategies.
    This is the Strategy interface for diagnostic analysis.
    Also serves as a factory for creating diagnostic analysis instances.
    """

    @classmethod
    def create(cls, engine_type: str) -> 'DiagnosticAnalysisBase':
        """
        Factory method to create a diagnostic analysis instance based on the engine type.

        Args:
            engine_type: Type of engine to use (pandas, polars)

        Returns:
            Instance of the appropriate diagnostic analysis strategy

        Raises:
            ValueError: If engine type is not supported
        """
        if engine_type == "pandas":
            from core.analysis.diagnostic.pandas_diagnostic import PandasDiagnosticAnalysis
            return PandasDiagnosticAnalysis()
        elif engine_type == "polars":
            from core.analysis.diagnostic.polars_diagnostic import PolarsDiagnosticAnalysis
            return PolarsDiagnosticAnalysis()
        else:
            raise ValueError(f"Unsupported engine type for diagnostic analysis: {engine_type}")

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