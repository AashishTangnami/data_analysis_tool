from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Type
from pathlib import Path
import logging

class EngineBase(ABC):
    """Base class for all data processing engines."""
    
    def __init__(self):
        self.df = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._ingestion = self._create_ingestion_handler()
        self._preprocessing = None
    
    @classmethod
    def get_engine(cls, engine_type: str) -> "EngineBase":
        """
        Factory method to get the appropriate engine instance.
        
        Args:
            engine_type: String identifier for the engine ("pandas", "polars", "pyspark")
            
        Returns:
            An instance of the appropriate engine class
            
        Raises:
            ValueError: If engine_type is not supported
        """
        if engine_type not in cls._engines:
            raise ValueError(
                f"Unsupported engine type: {engine_type}. "
                f"Supported types are: {', '.join(cls._engines.keys())}"
            )
        
        # Import the appropriate engine class
        engine_class_name = cls._engines[engine_type]
        if engine_type == "pandas":
            from .engines.pandas.pandas_engine import PandasEngine
            return PandasEngine()
        elif engine_type == "polars":
            from .engines.polars.polars_engine import PolarsEngine
            return PolarsEngine()
        elif engine_type == "pyspark":
            from .engines.pyspark.pyspark_engine import PySparkEngine
            return PySparkEngine()
    @property
    def ingestion(self):
        """Get the ingestion handler for the engine."""
        return self._ingestion
    
    @abstractmethod
    def _create_ingestion_handler(self):
        """Create engine-specific ingestion handler."""
        pass
    
    def load_data(self, file_path: str, **kwargs) -> Any:
        """Load data using the appropriate ingestion handler."""
        try:
            # Delegate to the ingestion handler
            self.df = self.ingestion.load_data(file_path, **kwargs)
            return self.df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
