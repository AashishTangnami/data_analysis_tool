import polars as pl
import numpy as np
from typing import Dict, Any, List
from core.analysis.diagnostic.base import DiagnosticAnalysisBase


def _is_numeric_dtype(dtype) -> bool:
    """Helper function to check if a polars dtype is numeric."""
    return any(
        isinstance(dtype, t)
        for t in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                 pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    )


class PolarsDiagnosticAnalysis(DiagnosticAnalysisBase):
    """
    Polars implementation of diagnostic analysis strategy.
    """

    def analyze(self, data: pl.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform diagnostic analysis using polars.

        Args:
            data: polars DataFrame
            params: Parameters for the analysis
                - target_column: Target variable for analysis
                - feature_columns: List of features to use
                - run_outlier_detection: Whether to run outlier detection
                - outlier_method: Method for outlier detection ('zscore' or 'iqr')
                - outlier_threshold: Threshold for outlier detection

        Returns:
            Dictionary containing analysis results
        """
        # Extract parameters
        target_column = params.get("target_column")
        feature_columns = params.get("feature_columns", [])
        # run_feature_importance = params.get("run_feature_importance", True)  # Commented out to match pandas implementation
        run_outlier_detection = params.get("run_outlier_detection", True)

        # Validate inputs
        if not target_column or target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        # Make sure feature_columns is a list even if it's passed as a single string
        if isinstance(feature_columns, str):
            feature_columns = [feature_columns]

        feature_columns = [col for col in feature_columns if col in data.columns]
        if not feature_columns:
            raise ValueError("No valid feature columns provided")

        # Initialize results
        results = {
            # "feature_importance": {},  # Commented out to match pandas implementation
            "outlier_detection": {},
            "correlation_analysis": {}
        }

        # Select relevant columns
        selected_data = data.select([target_column] + feature_columns)

        # Handle missing values for analysis - this creates a new DataFrame and doesn't modify the original
        selected_data_no_na = selected_data.drop_nulls()

        if selected_data_no_na.height == 0:
            raise ValueError("After dropping NA values, no data remains for analysis")

        # Outlier detection
        if run_outlier_detection:
            outlier_results = {}

            # Get outlier method and threshold from params, with defaults
            outlier_method = params.get("outlier_method", "zscore")
            outlier_threshold = params.get("outlier_threshold", 3.0)

            for col in feature_columns:
                try:
                    if _is_numeric_dtype(selected_data[col].dtype):
                        # Skip columns with all nulls
                        if selected_data[col].is_null().all():
                            continue

                        # Calculate outliers based on the specified method
                        col_data_series = selected_data[col].drop_nulls()

                        # Skip if no data
                        if col_data_series.len() == 0:
                            continue

                        if outlier_method == "zscore":
                            # Z-score method (default)
                            mean = col_data_series.mean()
                            std = col_data_series.std()

                            if std > 0:
                                # Create z-scores (only for non-null values)
                                z_scores = ((col_data_series - mean) / std).alias("z_score")

                                # Create a temporary dataframe with row numbers to track original positions
                                indices_df = pl.DataFrame({
                                    "original_index": pl.arange(0, col_data_series.len(), eager=True),
                                    col: col_data_series,
                                    "z_score": z_scores
                                })

                                # Find outliers (|z| > threshold)
                                outliers_df = indices_df.filter((pl.col("z_score")).abs() > outlier_threshold)

                                # Get first 20 indices and values for consistency with pandas implementation
                                outlier_indices = outliers_df.select("original_index").head(20).to_series().to_list()
                                outlier_values = outliers_df.select(pl.col(col)).head(20).to_series().to_list()

                                # Convert values to regular Python types to ensure serialization works
                                outlier_values = [float(val) for val in outlier_values]

                                # Save results
                                outlier_results[col] = {
                                    "mean": float(mean),
                                    "std": float(std),
                                    "outlier_count": outliers_df.height,
                                    "outlier_percentage": outliers_df.height / col_data_series.len() * 100,
                                    "outlier_indices": outlier_indices,
                                    "outlier_values": outlier_values,
                                    "method": "zscore",
                                    "threshold": outlier_threshold
                                }

                            else:
                                outlier_results[col] = {
                                    "error": "Standard deviation is zero, cannot calculate z-scores"
                                }

                        elif outlier_method == "iqr":
                            # IQR method
                            q1 = col_data_series.quantile(0.25)
                            q3 = col_data_series.quantile(0.75)
                            iqr = q3 - q1

                            if iqr > 0:
                                # Define bounds
                                lower_bound = q1 - outlier_threshold * iqr
                                upper_bound = q3 + outlier_threshold * iqr

                                # Create a temporary dataframe with row numbers to track original positions
                                indices_df = pl.DataFrame({
                                    "original_index": pl.arange(0, col_data_series.len(), eager=True),
                                    col: col_data_series
                                })

                                # Find outliers
                                outliers_df = indices_df.filter(
                                    (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
                                )

                                # Get first 20 indices and values for consistency with pandas implementation
                                outlier_indices = outliers_df.select("original_index").head(20).to_series().to_list()
                                outlier_values = outliers_df.select(pl.col(col)).head(20).to_series().to_list()

                                # Convert values to regular Python types to ensure serialization works
                                outlier_values = [float(val) for val in outlier_values]

                                # Save results
                                outlier_results[col] = {
                                    "q1": float(q1),
                                    "q3": float(q3),
                                    "iqr": float(iqr),
                                    "lower_bound": float(lower_bound),
                                    "upper_bound": float(upper_bound),
                                    "outlier_count": outliers_df.height,
                                    "outlier_percentage": outliers_df.height / col_data_series.len() * 100,
                                    "outlier_indices": outlier_indices,
                                    "outlier_values": outlier_values,
                                    "method": "iqr",
                                    "threshold": outlier_threshold
                                }
                            else:
                                outlier_results[col] = {
                                    "error": "IQR is zero, cannot detect outliers"
                                }
                        else:
                            # Unsupported method
                            outlier_results[col] = {
                                "error": f"Unsupported outlier detection method: {outlier_method}"
                            }
                except Exception as e:
                    outlier_results[col] = {"error": str(e)}

            results["outlier_detection"] = outlier_results

        # Correlation analysis with target
        corr_results = {}

        for col in feature_columns:
            try:
                if _is_numeric_dtype(selected_data[col].dtype):
                    # Get data without nulls for this column and target
                    valid_data = selected_data.filter(
                        ~pl.col(col).is_null() & ~pl.col(target_column).is_null()
                    )

                    # Skip if not enough data
                    if valid_data.height < 2:
                        corr_results[col] = {
                            "correlation": None,
                            "p_value": None,
                            "error": "Insufficient data for correlation analysis"
                        }
                        continue

                    if _is_numeric_dtype(selected_data[target_column].dtype):
                        # Calculate Pearson correlation using polars
                        corr_df = valid_data.select(pl.corr(col, target_column).alias("correlation"))
                        # Safely extract the correlation value
                        correlation = None
                        if corr_df.height > 0 and corr_df.width > 0:
                            correlation = corr_df[0, 0]

                        # For p-value calculation, we need to use scipy.stats
                        # We'll compute this with the to_numpy() method which is more efficient
                        try:
                            from scipy import stats

                            # Get the data as numpy arrays
                            x = valid_data[col].to_numpy()
                            y = valid_data[target_column].to_numpy()

                            # Calculate p-value
                            _, p_value = stats.pearsonr(x, y)

                            corr_results[col] = {
                                "correlation": float(correlation) if correlation is not None and not np.isnan(correlation) else None,
                                "p_value": float(p_value) if p_value is not None and not np.isnan(p_value) else None
                            }
                        except ImportError:
                            # If scipy is not available
                            corr_results[col] = {
                                "correlation": float(correlation) if correlation is not None and not np.isnan(correlation) else None,
                                "p_value": None,
                                "note": "P-value calculation requires scipy.stats module"
                            }
                    else:
                        # For categorical target, use a similar approach to ANOVA F-value
                        categories = valid_data[target_column].unique().drop_nulls()

                        # Skip if there's only one category
                        if categories.len() < 2:
                            corr_results[col] = {
                                "correlation": None,
                                "p_value": None,
                                "error": "Target has only one category"
                            }
                            continue

                        try:
                            from scipy import stats

                            # We'll convert to pandas temporarily for ANOVA calculation
                            # since polars doesn't have a built-in ANOVA function
                            pd_df = valid_data.select([target_column, col]).to_pandas()

                            # Group the data by category
                            groups = []
                            for cat in categories.to_list():
                                group_data = pd_df[pd_df[target_column] == cat][col].dropna()
                                if len(group_data) >= 2:
                                    groups.append(group_data)

                            # Check if we have enough groups
                            if len(groups) < 2:
                                corr_results[col] = {
                                    "correlation": None,
                                    "p_value": None,
                                    "error": "Some groups have insufficient samples for ANOVA"
                                }
                                continue

                            # Calculate ANOVA
                            f_stat, p_value = stats.f_oneway(*groups)

                            corr_results[col] = {
                                "correlation": float(f_stat) if not np.isnan(f_stat) else None,
                                "p_value": float(p_value) if not np.isnan(p_value) else None
                            }
                        except ImportError:
                            # Fall back to a simple measure of association (eta-squared)
                            # if scipy is not available
                            overall_mean = valid_data[col].mean()

                            between_ss = 0
                            within_ss = 0

                            # Calculate between-group and within-group sum of squares
                            for cat in categories.to_list():
                                group_data = valid_data.filter(pl.col(target_column) == cat)
                                if group_data.height >= 2:
                                    group_mean = group_data[col].mean()
                                    group_var = group_data[col].var()
                                    group_size = group_data.height

                                    between_ss += group_size * ((group_mean - overall_mean) ** 2)
                                    within_ss += (group_size - 1) * group_var

                            # Calculate eta-squared
                            total_ss = between_ss + within_ss
                            if total_ss > 0:
                                eta_squared = between_ss / total_ss
                                corr_results[col] = {
                                    "correlation": float(eta_squared),
                                    "p_value": None,
                                    "note": "Using eta-squared as measure of association; p-value unavailable without scipy"
                                }
                            else:
                                corr_results[col] = {
                                    "correlation": None,
                                    "p_value": None,
                                    "error": "Could not calculate association measure"
                                }
                else:
                    corr_results[col] = {
                        "correlation": None,
                        "p_value": None,
                        "error": "Feature is not numeric"
                    }
            except Exception as e:
                corr_results[col] = {
                    "correlation": None,
                    "p_value": None,
                    "error": str(e)
                }

        # Sort by absolute correlation (only for entries with valid correlations)
        corr_results_filtered = {k: v for k, v in corr_results.items()
                              if v.get("correlation") is not None}

        if corr_results_filtered:
            sorted_correlations = dict(sorted(
                corr_results_filtered.items(),
                key=lambda x: abs(x[1]["correlation"]) if x[1]["correlation"] is not None else 0,
                reverse=True
            ))
            # Merge with entries that have errors
            corr_results_with_errors = {k: v for k, v in corr_results.items()
                                     if v.get("correlation") is None}
            results["correlation_analysis"] = {**sorted_correlations, **corr_results_with_errors}
        else:
            results["correlation_analysis"] = corr_results

        return results