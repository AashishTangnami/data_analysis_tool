# core/analysis/prescriptive/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class PrescriptiveAnalysisBase(ABC):
    """
    Base class for all prescriptive analysis strategies.
    This is the Strategy interface for prescriptive analysis.
    """
    
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