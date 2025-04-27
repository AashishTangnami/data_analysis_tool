
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class PreprocessingBase(ABC):
    """
    Base class for all preprocessing strategies.
    This is the Strategy interface for preprocessing.
    Also serves as a factory for creating preprocessing instances.
    """

    @classmethod
    def create(cls, engine_type: str) -> 'PreprocessingBase':
        """
        Factory method to create a preprocessing instance based on the engine type.

        Args:
            engine_type: Type of engine to use (pandas, polars)

        Returns:
            Instance of the appropriate preprocessing strategy

        Raises:
            ValueError: If engine type is not supported
        """
        if engine_type == "pandas":
            from core.preprocessing.pandas_preprocessing import PandasPreprocessing
            return PandasPreprocessing()
        elif engine_type == "polars":
            from core.preprocessing.polars_preprocessing import PolarsPreprocessing
            return PolarsPreprocessing()
        else:
            raise ValueError(f"Unsupported engine type for preprocessing: {engine_type}")

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