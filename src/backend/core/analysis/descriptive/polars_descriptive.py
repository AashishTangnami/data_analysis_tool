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
        dtypes = {col: str(dtype) for col, dtype in schema.items()}
        missing_values = selected_data.null_count().to_dicts()[0]  # Single O(n) pass

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
            # Get numeric columns using schema (more reliable than string parsing)
            numeric_cols = [col for col, dtype in schema.items() if dtype.is_numeric()]

            if numeric_cols:
                statistics = {}

                # Create a structure similar to pandas describe()
                # First, calculate count for each column
                count_dict = {}
                for col in numeric_cols:
                    count_dict[col] = selected_data.select(pl.col(col).count()).item(0, 0)

                # Then calculate other statistics for each column
                for col in numeric_cols:
                    # Calculate basic statistics
                    stats_df = selected_data.select([
                        pl.col(col).mean().alias("mean"),
                        pl.col(col).std().alias("std"),
                        pl.col(col).min().alias("min"),
                        pl.col(col).quantile(0.25).alias("25%"),
                        pl.col(col).median().alias("50%"),
                        pl.col(col).quantile(0.75).alias("75%"),
                        pl.col(col).max().alias("max")
                    ])

                    # Convert to dictionary
                    stats = stats_df.to_dict(as_series=False)

                    # Store in the format pandas uses
                    if col not in statistics:
                        statistics[col] = {}

                    statistics[col]["count"] = float(count_dict[col])
                    statistics[col]["mean"] = float(stats["mean"][0])
                    statistics[col]["std"] = float(stats["std"][0])
                    statistics[col]["min"] = float(stats["min"][0])
                    statistics[col]["25%"] = float(stats["25%"][0])
                    statistics[col]["50%"] = float(stats["50%"][0])
                    statistics[col]["75%"] = float(stats["75%"][0])
                    statistics[col]["max"] = float(stats["max"][0])

                results["numeric_analysis"]["statistics"] = statistics

                # Calculate skewness and kurtosis
                skewness = {}
                kurtosis = {}

                for col in numeric_cols:
                    # Skip columns with all nulls
                    if selected_data[col].is_null().all():
                        continue

                    # Skip columns with all nulls and just use built-in methods
                    # Skewness: Use polars' built-in skew function
                    # Add null check before accessing
                    skew_df = selected_data.select(pl.col(col).skew().alias("skew"))
                    skew = skew_df.item(0, 0) if skew_df.height > 0 else None
                    skewness[col] = float(skew) if skew is not None else 0.0


                    # Kurtosis: Use polars' built-in kurtosis function
                    # Add null check before accessing
                    kurt_df = selected_data.select(pl.col(col).kurtosis().alias("kurt"))
                    kurt = kurt_df.item(0, 0) if kurt_df.height > 0 else None
                    kurtosis[col] = float(kurt) if kurt is not None else 0.0


                results["numeric_analysis"]["skewness"] = skewness
                results["numeric_analysis"]["kurtosis"] = kurtosis

        # Categorical analysis
        if include_categorical:
            # Get non-numeric columns using schema
            categorical_cols = [col for col, dtype in schema.items()
                              if not dtype.is_numeric()]

            if categorical_cols:
                value_counts = {}
                unique_counts = {}

                for col in categorical_cols:
                    # Get value counts (top 10)
                    # In newer versions of Polars, value_counts() returns a DataFrame with columns named after the original column and 'count'
                    try:
                        # First try the direct approach
                        counts_df = selected_data.select(
                            pl.col(col).value_counts(sort=True).limit(10)
                        )

                        # We don't need to use the column names, just access by index

                        # Convert to dictionary
                        counts = {}
                        if counts_df.height > 0:  # Critical check
                            for row in counts_df.rows():
                                key = str(row[0]) if row[0] is not None else "None"
                                # Use the second column regardless of its name
                                counts[key] = int(row[1])
                    except Exception as e:
                        # Fallback approach for compatibility
                        counts = {}
                        try:
                            # Try a different approach using groupby
                            counts_df = selected_data.group_by(col).agg(pl.count()).sort("count", descending=True).limit(10)

                            for row in counts_df.rows():
                                key = str(row[0]) if row[0] is not None else "None"
                                counts[key] = int(row[1])
                        except Exception as inner_e:
                            # If all else fails, return an empty dict
                            print(f"Error in value_counts for column {col}: {str(inner_e)}")
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
                          if dtype.is_numeric()]

            if len(numeric_cols) >= 2:
                # Calculate correlation matrix
                corr_matrix = {}

                for col1 in numeric_cols:
                    corr_matrix[col1] = {}

                    for col2 in numeric_cols:
                        # Calculate Pearson correlation
                        corr_df = selected_data.select(
                            pl.corr(col1, col2).alias("correlation")
                        )
                        corr = corr_df.item(0, 0)

                        # Handle NaN values the same way pandas does
                        if corr is None or np.isnan(corr):
                            corr_matrix[col1][col2] = None
                        else:
                            corr_matrix[col1][col2] = float(corr)

                results["correlations"] = corr_matrix

        return results