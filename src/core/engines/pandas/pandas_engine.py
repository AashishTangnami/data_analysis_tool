from ...engine_base import EngineBase
from ...interfaces.ingestion_base import DataIngestionBase
from .pandas_ingestion import PandasIngestion

class PandasEngine(EngineBase):
    """Pandas implementation of the engine."""
    
    def _create_ingestion_handler(self) -> DataIngestionBase:
        return PandasIngestion()