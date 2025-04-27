from typing import Dict, Any, Optional, Union, List
import pandas as pd
from core.engines.base import EngineBase

class EngineContext:
    """
    Context class for the Strategy Pattern.
    Manages the currently selected engine and delegates operations to it.
    """

    def __init__(self, engine_type: str = "pandas"):
        """
        Initialize the engine context with the specified engine type.

        Args:
            engine_type: Type of engine to use (pandas, polars, pyspark)
        """
        self._engine = self._get_engine(engine_type)
        self._engine_type = engine_type

    def _get_engine(self, engine_type: str) -> EngineBase:
        """
        Get the appropriate engine instance based on the engine type.

        Args:
            engine_type: Type of engine to use

        Returns:
            Instance of the appropriate engine

        Raises:
            ValueError: If engine type is not supported
        """
        return EngineBase.create(engine_type)

    def change_engine(self, engine_type: str) -> None:
        """
        Change the current engine to the specified type.

        Args:
            engine_type: Type of engine to use
        """
        self._engine = self._get_engine(engine_type)
        self._engine_type = engine_type

    def get_current_engine_type(self) -> str:
        """
        Get the type of the currently selected engine.

        Returns:
            Current engine type
        """
        return self._engine_type

    def load_data(self, file_path: str, file_type: Optional[str] = None, **kwargs) -> Any:
        """
        Load data using the current engine.

        Args:
            file_path: Path to the file to load
            file_type: Optional file type override
            **kwargs: Additional arguments to pass to the loader

        Returns:
            Loaded data in the engine's native format
        """
        return self._engine.load_data(file_path, file_type, **kwargs)

    def apply_single_operation(self, data: Any, operation: Dict[str, Any]) -> Any:
        """
        Apply a single preprocessing operation using the current engine.

        Args:
            data: Data in the engine's native format
            operation: Single preprocessing operation to apply

        Returns:
            Processed data in the engine's native format
        """
        return self._engine.apply_single_operation(data, operation)

    def get_data_summary(self, data: Any) -> Dict[str, Any]:
        """
        Generate a summary of the data using the current engine.

        Args:
            data: Data in the engine's native format

        Returns:
            Dictionary containing data summary information
        """
        return self._engine.get_data_summary(data)

    def preprocess_data(self, data: Any, operations: List[Dict[str, Any]]) -> Any:
        """
        Preprocess data according to specified operations using the current engine.

        Args:
            data: Data in the engine's native format
            operations: List of preprocessing operations to apply

        Returns:
            Preprocessed data in the engine's native format
        """
        return self._engine.preprocess_data(data, operations)

    def analyze_data(self, data: Any, analysis_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data according to specified analysis type using the current engine.

        Args:
            data: Data in the engine's native format
            analysis_type: Type of analysis to perform
            params: Parameters for the analysis

        Returns:
            Dictionary containing analysis results
        """
        return self._engine.analyze_data(data, analysis_type, params)

    def to_pandas(self, data: Any) -> pd.DataFrame:
        """
        Convert the engine's native data format to pandas DataFrame for visualization.

        Args:
            data: Data in the engine's native format

        Returns:
            pandas DataFrame
        """
        return self._engine.to_pandas(data)