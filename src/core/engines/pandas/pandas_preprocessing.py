import pandas as pd
import numpy as np
from typing import Any, List, Dict, Optional, Union
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from ...interfaces.preprocessing_base import DataPreprocessingBase

class PandasPreprocessing(DataPreprocessingBase):
    def __init__(self):
        self.label_encoders = {}

    def _encode_categorical(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Encode categorical column using LabelEncoder with handling for unseen labels"""
        if column not in self.label_encoders:
            self.label_encoders[column] = LabelEncoder()
            # Fit on non-null values
            non_null_values = df[column].dropna()
            self.label_encoders[column].fit(non_null_values)
        
        # Transform non-null values, handle unseen labels
        non_null_mask = df[column].notna()
        try:
            df.loc[non_null_mask, column] = self.label_encoders[column].transform(df.loc[non_null_mask, column])
        except ValueError as e:
            if "contains previously unseen labels" in str(e):
                # Get unique values and refit the encoder
                all_values = set(df.loc[non_null_mask, column].unique())
                known_values = set(self.label_encoders[column].classes_)
                new_values = all_values | known_values  # union of both sets
                
                # Refit the encoder with all values
                self.label_encoders[column].fit(list(new_values))
                # Transform again with updated encoder
                df.loc[non_null_mask, column] = self.label_encoders[column].transform(df.loc[non_null_mask, column])
            else:
                raise e
        
        return df

    def _handle_datetime(self, df: pd.DataFrame, column: str, extract_features: bool = True) -> pd.DataFrame:
        """
        Process datetime columns and optionally extract useful features.
        
        Args:
            df: Input DataFrame
            column: Column name containing datetime data
            extract_features: If True, extracts features like year, month, day, etc.
        
        Returns:
            DataFrame with processed datetime features
        """
        try:
            # Clean and standardize date format first
            def clean_date(date_str):
                if pd.isna(date_str):
                    return pd.NaT
                try:
                    # Remove any whitespace and handle potential concatenated dates
                    date_str = str(date_str).strip()
                    # Split if multiple dates are concatenated
                    if len(date_str) > 10:  # Standard date length is 10 (YYYY/MM/DD)
                        date_str = date_str[:10]  # Take first date only
                    return date_str
                except:
                    return pd.NaT

            # Clean the dates first
            df[column] = df[column].apply(clean_date)
            
            # Try to convert to datetime with multiple format attempts
            date_formats = [
                '%Y/%m/%d',  # 2023/6/25
                '%Y-%m-%d',  # 2023-06-25
                '%d/%m/%Y',  # 25/06/2023
                '%d-%m-%Y',  # 25-06-2023
                '%m/%d/%Y',  # 06/25/2023
                '%m-%d-%Y'   # 06-25-2023
            ]
            
            for date_format in date_formats:
                try:
                    df[column] = pd.to_datetime(df[column], format=date_format, errors='coerce')
                    if not df[column].isna().all():  # If we successfully converted some dates
                        break
                except:
                    continue
            
            if df[column].isna().all():
                # If all conversion attempts failed, try one last time with pandas' flexible parser
                df[column] = pd.to_datetime(df[column], errors='coerce')
            
            if extract_features and not df[column].isna().all():
                # Extract basic time components
                df[f'{column}_year'] = df[column].dt.year
                df[f'{column}_month'] = df[column].dt.month
                df[f'{column}_day'] = df[column].dt.day
                
                # Extract day of week (0 = Monday, 6 = Sunday)
                df[f'{column}_dayofweek'] = df[column].dt.dayofweek
                
                # Extract quarter
                df[f'{column}_quarter'] = df[column].dt.quarter
                
                # Create cyclical features for month and day of week
                df[f'{column}_sin_month'] = np.sin(2 * np.pi * df[column].dt.month/12)
                df[f'{column}_cos_month'] = np.cos(2 * np.pi * df[column].dt.month/12)
                df[f'{column}_sin_day'] = np.sin(2 * np.pi * df[column].dt.dayofweek/7)
                df[f'{column}_cos_day'] = np.cos(2 * np.pi * df[column].dt.dayofweek/7)
                
                # If time information exists
                if (df[column].dt.hour != 0).any():
                    df[f'{column}_hour'] = df[column].dt.hour
                    df[f'{column}_minute'] = df[column].dt.minute
                    df[f'{column}_sin_time'] = np.sin(2 * np.pi * df[column].dt.hour/24)
                    df[f'{column}_cos_time'] = np.cos(2 * np.pi * df[column].dt.hour/24)
                
                # Optional: Drop original datetime column
                df = df.drop(columns=[column])
            
            return df
            
        except Exception as e:
            print(f"Error processing datetime column {column}: {str(e)}")
            return df

    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        df = data.copy()
        
        # Identify column types
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        
        # First process datetime columns
        for col in df.columns:
            # Check if column might contain datetime values
            if df[col].dtype == 'object':
                try:
                    # Try to convert a sample to datetime
                    sample = df[col].dropna().iloc[0]
                    pd.to_datetime(sample)
                    # If successful, process the entire column
                    df = self._process_datetime(df, col)
                except:
                    pass
        
        # Then handle categorical columns
        for col in categorical_columns:
            unique_values = df[col].dropna().unique()
            if len(unique_values) < df.shape[0] / 2:  # Only encode if cardinality is reasonable
                df = self._encode_categorical(df, col)
        
        # Finally handle missing values based on strategy
        if strategy == 'mean':
            df = df.fillna(df.mean())
        elif strategy == 'median':
            df = df.fillna(df.median())
        elif strategy == 'mode':
            df = df.fillna(df.mode().iloc[0])
        elif strategy == 'drop':
            return df.dropna()
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        
        return df

    def remove_duplicates(self, data: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        return data.drop_duplicates(subset=subset)

    def detect_outliers(self, data: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        df = data.copy()
        if method == 'iqr':
            for col in columns:
                # Only process if column is numeric or has been encoded
                if df[col].dtype in ['int64', 'float64']:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df[f'{col}_is_outlier'] = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        return df

    def scale_column(self, data: pd.DataFrame, column: str, method: str = 'standard') -> pd.DataFrame:
        df = data.copy()
        
        # Check if column needs encoding first
        if df[column].dtype not in ['int64', 'float64']:
            unique_values = df[column].dropna().unique()
            if len(unique_values) < df.shape[0] / 2:  # Only encode if cardinality is reasonable
                df = self._encode_categorical(df, column)
        
        # Now scale the column
        if df[column].dtype in ['int64', 'float64']:
            if method == 'standard':
                mean = df[column].mean()
                std = df[column].std()
                if std != 0:
                    df[column] = (df[column] - mean) / std
            elif method == 'minmax':
                min_val = df[column].min()
                max_val = df[column].max()
                if max_val != min_val:
                    df[column] = (df[column] - min_val) / (max_val - min_val)
            else:
                raise ValueError(f"Unsupported scaling method: {method}")
        
        return df
