
from abc import ABC, abstractmethod
from typing import Any, Dict

class PrescriptiveAnalysisBase(ABC):
    """
    Base class for all prescriptive analysis strategies.
    This is the Strategy interface for prescriptive analysis.
    Also serves as a factory for creating prescriptive analysis instances.
    """

    @classmethod
    def create(cls, engine_type: str) -> 'PrescriptiveAnalysisBase':
        """
        Factory method to create a prescriptive analysis instance based on the engine type.

        Args:
            engine_type: Type of engine to use (pandas, polars)

        Returns:
            Instance of the appropriate prescriptive analysis strategy

        Raises:
            ValueError: If engine type is not supported
        """
        if engine_type == "pandas":
            from core.analysis.prescriptive.pandas_prescriptive import PandasPrescriptiveAnalysis
            return PandasPrescriptiveAnalysis()
        elif engine_type == "polars":
            from core.analysis.prescriptive.polars_prescriptive import PolarsPrescriptiveAnalysis
            return PolarsPrescriptiveAnalysis()
        else:
            raise ValueError(f"Unsupported engine type for prescriptive analysis: {engine_type}")

    @abstractmethod
    def analyze(self, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform prescriptive analysis on data.

        Args:
            data: Data in the engine's native format
            params: Parameters for the analysis
                - objective_column: Column to optimize
                - objective_type: 'maximize' or 'minimize'
                - decision_variables: List of variables to adjust
                - constraints: List of constraints on decision variables

        Returns:
            Dictionary containing analysis results
                - optimization_results: Optimal solution
                - scenario_comparison: Comparison of different scenarios
                - sensitivity_analysis: Sensitivity to parameter changes
        """
        pass