import polars as pl
from typing import List, Any, Optional
from ...interfaces.preprocessing_base import DataPreprocessingBase

class PolarsPreprocessing(DataPreprocessingBase):
    def handle_missing_values(self, data: pl.DataFrame, strategy: str = 'mean') -> pl.DataFrame:
        df = data.clone()
        if strategy == 'mean':
            return df.fill_null(df.mean())
        elif strategy == 'median':
            return df.fill_null(df.median())
        elif strategy == 'drop':
            return df.drop_nulls()
        raise ValueError(f"Unsupported strategy: {strategy}")

    def remove_duplicates(self, data: pl.DataFrame, subset: Optional[List[str]] = None) -> pl.DataFrame:
        return data.unique(subset=subset) if subset else data.unique()

    def detect_outliers(self, data: pl.DataFrame, columns: List[str], method: str = 'iqr') -> pl.DataFrame:
        df = data.clone()
        if method == 'iqr':
            for col in columns:
                Q1 = df.select(pl.col(col).quantile(0.25)).item()
                Q3 = df.select(pl.col(col).quantile(0.75)).item()
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df.with_columns(
                    pl.col(col).is_between(lower_bound, upper_bound).alias(f"{col}_is_outlier")
                )
        return df

    def scale_column(self, data: pl.DataFrame, column: str, method: str = 'standard') -> pl.DataFrame:
        df = data.clone()
        if method == 'standard':
            mean = df[column].mean()
            std = df[column].std()
            df = df.with_columns(
                ((pl.col(column) - mean) / std).alias(column)
            )
        elif method == 'minmax':
            min_val = df[column].min()
            max_val = df[column].max()
            df = df.with_columns(
                ((pl.col(column) - min_val) / (max_val - min_val)).alias(column)
            )
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        return df
