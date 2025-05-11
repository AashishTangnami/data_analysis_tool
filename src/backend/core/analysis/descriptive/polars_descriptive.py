import polars as pl
import numpy as np
from typing import Dict, Any
from core.analysis.descriptive.base import DescriptiveAnalysisBase

class PolarsDescriptiveAnalysis(DescriptiveAnalysisBase):
    """
    Polars implementation of descriptive analysis strategy.
    """

    def analyze(self, data: pl.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform descriptive analysis using polars.

        Args:
            data: polars DataFrame
            params: Parameters for the analysis
                - columns: List of columns to analyze (default: all)
                - include_numeric: Whether to include numeric analysis (default: True)
                - include_categorical: Whether to include categorical analysis (default: True)
                - include_correlations: Whether to include correlation analysis (default: True)

        Returns:
            Dictionary containing analysis results
        """
        # Extract parameters
        columns = params.get("columns", data.columns)
        include_numeric = params.get("include_numeric", True)
        include_categorical = params.get("include_categorical", True)
        include_correlations = params.get("include_correlations", True)

        # Filter to selected columns
        selected_data = data.select(columns)

        # Schema-based operations (more efficient)
        schema = selected_data.schema

        # Convert Polars dtypes to match Pandas dtype strings
        dtypes = {}
        for col, dtype in schema.items():
            dtype_str = str(dtype)
            if "Float" in dtype_str:
                dtypes[col] = "float64"
            elif "Int" in dtype_str:
                dtypes[col] = "int64"
            elif "Utf8" in dtype_str or "String" in dtype_str or "Categorical" in dtype_str:
                dtypes[col] = "object"
            else:
                dtypes[col] = dtype_str

        # Get missing values count in a single pass
        # Convert to dict with scalar values to match pandas format
        missing_values_df = selected_data.null_count()
        missing_values = {}
        for col in selected_data.columns:
            missing_values[col] = int(missing_values_df.get_column(col)[0])

        # Initialize results
        results = {
            "dataset_info": {
                "shape": (selected_data.height, selected_data.width),
                "columns": selected_data.columns,
                "dtypes": dtypes,
                "missing_values": missing_values,
            },
            "numeric_analysis": {},
            "categorical_analysis": {},
            "correlations": {}
        }

        # Numeric analysis
        if include_numeric:
            # Get numeric columns using schema
            numeric_cols = [col for col, dtype in schema.items()
                          if "Float" in str(dtype) or "Int" in str(dtype)]

            if numeric_cols:
                statistics = {}

                for col in numeric_cols:
                    try:
                        # Skip columns with all nulls
                        if selected_data[col].is_null().all():
                            # If all nulls, return zeros
                            col_stats = {
                                "count": 0,
                                "mean": 0.0,
                                "std": 0.0,
                                "min": 0.0,
                                "25%": 0.0,
                                "50%": 0.0,
                                "75%": 0.0,
                                "max": 0.0
                            }
                        else:
                            # Use Polars' native methods for statistics
                            # Get count of non-null values
                            count = selected_data[col].drop_nulls().len()

                            # Calculate basic statistics using Polars
                            mean = selected_data[col].mean()
                            std = selected_data[col].std(ddof=1)  # Use ddof=1 to match pandas
                            min_val = selected_data[col].min()
                            max_val = selected_data[col].max()

                            # For exact matching with pandas, we need to use numpy
                            # Polars and pandas have slightly different quantile implementations
                            col_values = selected_data[col].drop_nulls().to_numpy()
                            q25 = np.percentile(col_values, 25)
                            q50 = np.percentile(col_values, 50)
                            q75 = np.percentile(col_values, 75)

                            # Create statistics dictionary
                            col_stats = {
                                "count": int(count),
                                "mean": float(mean),
                                "std": float(std),
                                "min": float(min_val),
                                "25%": float(q25),
                                "50%": float(q50),
                                "75%": float(q75),
                                "max": float(max_val)
                            }
                    except Exception as e:
                        # If error, return zeros
                        print(f"Error calculating statistics for column {col}: {str(e)}")
                        col_stats = {
                            "count": 0,
                            "mean": 0.0,
                            "std": 0.0,
                            "min": 0.0,
                            "25%": 0.0,
                            "50%": 0.0,
                            "75%": 0.0,
                            "max": 0.0
                        }

                    statistics[col] = col_stats

                results["numeric_analysis"]["statistics"] = statistics

                # Calculate skewness and kurtosis
                skewness = {}
                kurtosis = {}

                for col in numeric_cols:
                    try:
                        # Skip columns with all nulls
                        if selected_data[col].is_null().all():
                            skewness[col] = 0.0
                            kurtosis[col] = 0.0
                            continue

                        # Use Polars' native skew and kurtosis methods
                        # Apply Fisher's adjustment to match pandas' kurtosis definition
                        skew = selected_data[col].skew()
                        kurt = selected_data[col].kurtosis()

                        # Handle None values
                        skewness[col] = float(skew) if skew is not None else 0.0
                        kurtosis[col] = float(kurt) if kurt is not None else 0.0
                    except Exception as e:
                        print(f"Error calculating skewness/kurtosis for column {col}: {str(e)}")
                        # Default values if calculation fails
                        skewness[col] = 0.0
                        kurtosis[col] = 0.0


                results["numeric_analysis"]["skewness"] = skewness
                results["numeric_analysis"]["kurtosis"] = kurtosis

        # Categorical analysis
        if include_categorical:
            # Get non-numeric columns using schema
            categorical_cols = [col for col, dtype in schema.items()
                              if not ("Float" in str(dtype) or "Int" in str(dtype))]

            if categorical_cols:
                value_counts = {}
                unique_counts = {}

                for col in categorical_cols:
                    # Get value counts (top 10)
                    try:
                        # Use Polars' native value_counts method
                        value_counts_df = selected_data[col].value_counts(sort=True).limit(10)

                        # Convert to dictionary with string keys
                        counts = {}
                        if value_counts_df.height > 0:
                            # Get column names for values and counts
                            value_col = value_counts_df.columns[0]
                            count_col = value_counts_df.columns[1]

                            # Extract values and counts
                            for i in range(value_counts_df.height):
                                value = value_counts_df[value_col][i]
                                count = value_counts_df[count_col][i]

                                # Convert key to string
                                str_key = str(value) if value is not None else "None"
                                counts[str_key] = int(count)
                    except Exception as e:
                        print(f"Error calculating value counts for column {col}: {str(e)}")
                        # If there's an error, just return an empty dictionary
                        counts = {}

                    value_counts[col] = counts

                    # Count unique values
                    unique_count = selected_data[col].n_unique()
                    unique_counts[col] = int(unique_count)

                results["categorical_analysis"]["value_counts"] = value_counts
                results["categorical_analysis"]["unique_counts"] = unique_counts

        # Correlation analysis
        if include_correlations:
            # Get numeric columns using schema
            numeric_cols = [col for col, dtype in schema.items()
                          if "Float" in str(dtype) or "Int" in str(dtype)]

            if len(numeric_cols) >= 2:
                try:
                    # Calculate correlation matrix using Polars
                    # Create a nested dictionary for the correlation matrix
                    corr_matrix = {}

                    for col1 in numeric_cols:
                        corr_matrix[col1] = {}

                        for col2 in numeric_cols:
                            # Calculate Pearson correlation using Polars
                            corr = selected_data.select(pl.corr(col1, col2).alias("corr"))[0, 0]

                            # Handle None values
                            corr_matrix[col1][col2] = float(corr) if corr is not None else 0.0
                except Exception as e:
                    print(f"Error calculating correlation matrix: {str(e)}")
                    # Create empty correlation matrix
                    corr_matrix = {col: {col2: 0.0 for col2 in numeric_cols} for col in numeric_cols}

                results["correlations"] = corr_matrix

        return results