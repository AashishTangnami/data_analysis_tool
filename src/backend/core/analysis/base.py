from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union

class AnalysisBase(ABC):
    """
    Base abstract class for all analysis operations.
    This is the highest-level Strategy interface for analysis.
    
    All analysis types (descriptive, diagnostic, predictive, prescriptive)
    inherit from this base class.
    """
    
    @abstractmethod
    def analyze(self, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform analysis on data.
        
        Args:
            data: Data in the engine's native format
            params: Parameters for the analysis
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """
        Get a schema describing the parameters accepted by this analysis.
        
        Returns:
            Dictionary containing parameter schema information
        """
        pass
    
    @classmethod
    def validate_parameters(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the parameters for this analysis.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Validated and potentially transformed parameters
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Default implementation performs basic validation using the schema
        schema = cls.get_parameter_schema()
        required_params = schema.get("required", [])
        
        # Check for required parameters
        for param in required_params:
            if param not in params:
                raise ValueError(f"Required parameter '{param}' is missing")
        
        # Check parameter types
        for param, value in params.items():
            if param in schema.get("properties", {}):
                expected_type = schema["properties"][param].get("type")
                if expected_type:
                    if expected_type == "array" and not isinstance(value, list):
                        raise ValueError(f"Parameter '{param}' must be a list")
                    elif expected_type == "object" and not isinstance(value, dict):
                        raise ValueError(f"Parameter '{param}' must be a dictionary")
                    elif expected_type == "string" and not isinstance(value, str):
                        raise ValueError(f"Parameter '{param}' must be a string")
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        raise ValueError(f"Parameter '{param}' must be a number")
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        raise ValueError(f"Parameter '{param}' must be a boolean")
        
        return params
    
    @classmethod
    def get_available_visualizations(cls) -> List[Dict[str, Any]]:
        """
        Get a list of available visualizations for this analysis type.
        
        Returns:
            List of dictionaries describing available visualizations
        """
        # Default implementation returns an empty list
        # Subclasses should override this method to provide visualization options
        return []
    
    @classmethod
    def create_visualization(cls, 
                            data: Any, 
                            analysis_results: Dict[str, Any], 
                            viz_type: str, 
                            viz_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a visualization from analysis results.
        
        Args:
            data: Original data
            analysis_results: Results from analysis
            viz_type: Type of visualization to create
            viz_params: Parameters for the visualization
            
        Returns:
            Dictionary with visualization data
            
        Raises:
            ValueError: If visualization type is not supported
        """
        # Default implementation raises an error
        # Subclasses should override this method to provide visualizations
        raise ValueError(f"Visualization type '{viz_type}' not supported by this analysis")