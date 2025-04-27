from abc import ABC, abstractmethod
from typing import Any, Dict

class DescriptiveAnalysisBase(ABC):
    """
    Base class for all descriptive analysis strategies.
    This is the Strategy interface for descriptive analysis.
    Also serves as a factory for creating descriptive analysis instances.
    """

    @classmethod
    def create(cls, engine_type: str) -> 'DescriptiveAnalysisBase':
        """
        Factory method to create a descriptive analysis instance based on the engine type.

        Args:
            engine_type: Type of engine to use (pandas, polars)

        Returns:
            Instance of the appropriate descriptive analysis strategy

        Raises:
            ValueError: If engine type is not supported
        """
        if engine_type == "pandas":
            from core.analysis.descriptive.pandas_descriptive import PandasDescriptiveAnalysis
            return PandasDescriptiveAnalysis()
        elif engine_type == "polars":
            from core.analysis.descriptive.polars_descriptive import PolarsDescriptiveAnalysis
            return PolarsDescriptiveAnalysis()
        else:
            raise ValueError(f"Unsupported engine type for descriptive analysis: {engine_type}")

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