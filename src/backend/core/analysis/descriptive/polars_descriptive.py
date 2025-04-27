import polars as pl
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
        try:
            # Extract parameters
            columns = params.get("columns", data.columns)
            include_numeric = params.get("include_numeric", True)
            include_categorical = params.get("include_categorical", True)
            include_correlations = params.get("include_correlations", True)

            # Validate columns
            valid_columns = [col for col in columns if col in data.columns]
            if not valid_columns:
                return {
                    "error": "No valid columns found for analysis",
                    "dataset_info": {
                        "shape": (data.height, data.width),
                        "columns": data.columns,
                    }
                }

            # Filter to selected columns
            selected_data = data.select(valid_columns)

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
                try:
                    # Get numeric columns using schema (more reliable than string parsing)
                    numeric_cols = [col for col, dtype in schema.items() if dtype.is_numeric()]

                    if numeric_cols:
                        statistics = {}

                        for col in numeric_cols:
                            try:
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
                                col_stats = {
                                    "mean": stats["mean"][0],
                                    "std": stats["std"][0],
                                    "min": stats["min"][0],
                                    "25%": stats["25%"][0],
                                    "50%": stats["50%"][0],
                                    "75%": stats["75%"][0],
                                    "max": stats["max"][0]
                                }

                                statistics[col] = col_stats
                            except Exception as e:
                                statistics[col] = {"error": f"Could not calculate statistics: {str(e)}"}

                        results["numeric_analysis"]["statistics"] = statistics

                        # Calculate skewness and kurtosis
                        skewness = {}
                        kurtosis = {}

                        for col in numeric_cols:
                            try:
                                # Skip columns with all nulls
                                if selected_data[col].is_null().all():
                                    continue

                                # Skewness: Use polars' built-in skew function
                                skew_df = selected_data.select(pl.col(col).skew().alias("skew"))
                                skew = skew_df.item(0, 0) if skew_df.height > 0 else None
                                skewness[col] = float(skew) if skew is not None else 0.0

                                # Kurtosis: Use polars' built-in kurtosis function
                                kurt_df = selected_data.select(pl.col(col).kurtosis().alias("kurt"))
                                kurt = kurt_df.item(0, 0) if kurt_df.height > 0 else None
                                kurtosis[col] = float(kurt) if kurt is not None else 0.0
                            except Exception as e:
                                skewness[col] = 0.0
                                kurtosis[col] = 0.0

                        results["numeric_analysis"]["skewness"] = skewness
                        results["numeric_analysis"]["kurtosis"] = kurtosis
                except Exception as e:
                    results["numeric_analysis"] = {"error": f"Numeric analysis failed: {str(e)}"}

            # Categorical analysis
            if include_categorical:
                try:
                    # Get non-numeric columns using schema
                    categorical_cols = [col for col, dtype in schema.items()
                                      if not dtype.is_numeric()]

                    if categorical_cols:
                        value_counts = {}
                        unique_counts = {}

                        for col in categorical_cols:
                            try:
                                # Get value counts (top 10)
                                counts_df = selected_data.select(
                                    pl.col(col).value_counts(sort=True).limit(10)
                                )

                                # Convert to dictionary
                                counts = {}
                                if counts_df.height > 0:  # Critical check
                                    for row in counts_df.rows():
                                        key = str(row[0]) if row[0] is not None else "None"
                                        counts[key] = row[1]

                                value_counts[col] = counts

                                # Count unique values
                                unique_count = selected_data[col].n_unique()
                                unique_counts[col] = int(unique_count)
                            except Exception as e:
                                value_counts[col] = {"error": f"Could not calculate value counts: {str(e)}"}
                                unique_counts[col] = 0

                        results["categorical_analysis"]["value_counts"] = value_counts
                        results["categorical_analysis"]["unique_counts"] = unique_counts
                except Exception as e:
                    results["categorical_analysis"] = {"error": f"Categorical analysis failed: {str(e)}"}

            # Correlation analysis
            if include_correlations:
                try:
                    # Get numeric columns using schema
                    numeric_cols = [col for col, dtype in schema.items()
                                  if dtype.is_numeric()]

                    if len(numeric_cols) >= 2:
                        # Calculate correlation matrix
                        corr_matrix = {}

                        for col1 in numeric_cols:
                            corr_matrix[col1] = {}

                            for col2 in numeric_cols:
                                try:
                                    # Calculate Pearson correlation
                                    corr_df = selected_data.select(
                                        pl.corr(col1, col2).alias("correlation")
                                    )
                                    corr = corr_df.item(0, 0) if corr_df.height > 0 else None
                                    corr_matrix[col1][col2] = float(corr) if corr is not None else 0.0
                                except Exception as e:
                                    corr_matrix[col1][col2] = 0.0

                        results["correlations"] = corr_matrix
                except Exception as e:
                    results["correlations"] = {"error": f"Correlation analysis failed: {str(e)}"}

            return results

        except Exception as e:
            # Return a minimal result with error information
            return {
                "error": f"Descriptive analysis failed: {str(e)}",
                "dataset_info": {
                    "shape": (data.height, data.width) if hasattr(data, "height") else (0, 0),
                    "columns": data.columns if hasattr(data, "columns") else [],
                }
            }