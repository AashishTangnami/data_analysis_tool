import pandas as pd
from typing import Any, List, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ...interfaces.transformation_base import DataTransformationBase

class PandasTransformation(DataTransformationBase):
    def filter_data(self, data: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
        df = data.copy()
        for column, value in conditions.items():
            df = df[df[column] == value]
        return df

    def aggregate_data(self, data: pd.DataFrame, group_by: List[str], agg_functions: Dict[str, str]) -> pd.DataFrame:
        return data.groupby(group_by).agg(agg_functions)

    def normalize_column(self, data: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
        df = data.copy()
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        df[column] = scaler.fit_transform(df[[column]])
        return df

    def encode_categorical(self, data: pd.DataFrame, columns: List[str], method: str = 'onehot') -> pd.DataFrame:
        df = data.copy()
        if method == 'onehot':
            return pd.get_dummies(df, columns=columns)
        elif method == 'label':
            for col in columns:
                df[col] = df[col].astype('category').cat.codes
            return df
        raise ValueError(f"Unsupported encoding method: {method}")