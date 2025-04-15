import polars as pl
from typing import Dict, Any

class PolarsTransformation:
    def filter_data(self, df: pl.DataFrame, conditions: Dict[str, Any]) -> pl.DataFrame:
        """Filter data based on conditions"""
        for column, condition in conditions.items():
            if isinstance(condition, (list, tuple)):
                df = df.filter(pl.col(column).is_in(condition))
            else:
                df = df.filter(pl.col(column) == condition)
        return df