import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats
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
                - run_feature_importance: Whether to run feature importance analysis
                - run_outlier_detection: Whether to run outlier detection

        Returns:
            Dictionary containing analysis results
        """
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        # Extract parameters
        target_column = params.get("target_column")
        feature_columns = params.get("feature_columns", [])
        run_feature_importance = params.get("run_feature_importance", True)
        run_outlier_detection = params.get("run_outlier_detection", True)

        # Validate inputs
        if not target_column or target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        feature_columns = [col for col in feature_columns if col in data.columns]
        if not feature_columns:
            raise ValueError("No valid feature columns provided")

        # Initialize results
        results = {
            "feature_importance": {},
            "outlier_detection": {},
            "correlation_analysis": {}
        }

        # Select relevant columns
        selected_data = data.select([target_column] + feature_columns)

        # Handle missing values for analysis
        selected_data = selected_data.drop_nulls()

        # Feature importance analysis
        if run_feature_importance:
            # For feature importance, convert to pandas and use sklearn
            # as polars doesn't have native ML algorithms
            pandas_df = selected_data.to_pandas()

            # Determine if classification or regression
            is_categorical = False
            if not _is_numeric_dtype(selected_data[target_column].dtype) or selected_data[target_column].n_unique() < 10:
                is_categorical = True

            # Prepare features and target
            X = pandas_df[feature_columns]
            y = pandas_df[target_column]

            # Convert categorical features to numeric
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = X[col].astype('category').cat.codes

            # Train random forest for feature importance
            if is_categorical:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            try:
                model.fit(X, y)

                # Get feature importance
                importances = model.feature_importances_

                # Create a dictionary of feature importance
                results["feature_importance"] = {
                    feature: float(importance)
                    for feature, importance in zip(feature_columns, importances)
                }

                # Sort by importance
                results["feature_importance"] = dict(sorted(
                    results["feature_importance"].items(),
                    key=lambda x: x[1],
                    reverse=True
                ))

            except Exception as e:
                results["feature_importance"] = {"error": str(e)}

        # Outlier detection
        if run_outlier_detection:
            outlier_results = {}

            for col in feature_columns:
                try:
                    if _is_numeric_dtype(selected_data[col].dtype):
                        # Skip columns with all nulls
                        if selected_data[col].is_null().all():
                            continue

                        # Get non-null values - convert to numpy for exact matching with pandas
                        col_values = selected_data[col].drop_nulls().to_numpy()

                        if len(col_values) > 0:
                            # Calculate mean and std using numpy for consistency with pandas
                            mean = np.mean(col_values)
                            std = np.std(col_values, ddof=1)  # ddof=1 for sample standard deviation (pandas default)

                            if std > 0:
                                # Calculate z-scores using numpy
                                z_scores = (col_values - mean) / std

                                # Find outliers (|z| > 3)
                                outlier_mask = np.abs(z_scores) > 3
                                outlier_count = np.sum(outlier_mask)

                                # Save results
                                outlier_results[col] = {
                                    "mean": float(mean),
                                    "std": float(std),
                                    "outlier_count": int(outlier_count),
                                    "outlier_percentage": float(outlier_count) / len(col_values) * 100,
                                    "outlier_indices": []  # Polars doesn't have index like pandas, but include for compatibility
                                }
                except Exception as e:
                    outlier_results[col] = {
                        "mean": 0.0,
                        "std": 0.0,
                        "outlier_count": 0,
                        "outlier_percentage": 0.0,
                        "outlier_indices": [],
                        "error": str(e)
                    }

            results["outlier_detection"] = outlier_results

        # Correlation analysis with target
        corr_results = {}

        for col in feature_columns:
            try:
                if _is_numeric_dtype(selected_data[col].dtype):
                    # Get data for both column and target
                    # Convert to numpy arrays for exact matching with pandas
                    col_array = selected_data[col].to_numpy()
                    target_array = selected_data[target_column].to_numpy()

                    # Create mask for non-null values in both arrays
                    valid_mask = ~(np.isnan(col_array) | np.isnan(target_array))

                    # Filter arrays to only include rows where both values are non-null
                    col_values = col_array[valid_mask]
                    target_values = target_array[valid_mask]

                    # Check if we have enough data
                    if len(col_values) < 2:
                        corr_results[col] = {
                            "correlation": 0,
                            "p_value": 1.0,
                            "error": "Insufficient data for correlation analysis"
                        }
                        continue

                    if _is_numeric_dtype(selected_data[target_column].dtype):
                        # Calculate Pearson correlation
                        try:
                            # Calculate correlation and p-value using scipy
                            corr, p_value = stats.pearsonr(col_values, target_values)
                        except Exception as e:
                            corr_results[col] = {
                                "correlation": 0,
                                "p_value": 1.0,
                                "error": f"Error in correlation calculation: {str(e)}"
                            }
                            continue
                    else:
                        # For categorical target, use ANOVA F-value
                        # Get all rows where feature column is not null
                        valid_data = selected_data.filter(~selected_data[col].is_null())

                        # Get unique categories
                        categories = valid_data[target_column].unique().to_list()

                        # Skip ANOVA if there's only one category
                        if len(categories) < 2:
                            corr_results[col] = {
                                "correlation": 0,
                                "p_value": 1.0,
                                "error": "Target has only one category"
                            }
                            continue

                        # Check if we have enough samples in each category
                        groups_data = []
                        valid_groups = True

                        for cat in categories:
                            # Get values for this category
                            group_data = valid_data.filter(pl.col(target_column) == cat)[col].drop_nulls()

                            if group_data.len() < 2:
                                valid_groups = False
                                break

                            groups_data.append(group_data.to_numpy())

                        # If any group has insufficient samples, skip
                        if not valid_groups or len(groups_data) < 2:
                            corr_results[col] = {
                                "correlation": 0,
                                "p_value": 1.0,
                                "error": "Some groups have insufficient samples for ANOVA"
                            }
                            continue

                        # Calculate F-statistic
                        f_stat, p_value = stats.f_oneway(*groups_data)
                        corr = f_stat  # Using F-statistic as measure of association

                    # Store results with proper handling of None/NaN values
                    corr_results[col] = {
                        "correlation": float(corr) if corr is not None and not np.isnan(corr) else 0,
                        "p_value": float(p_value) if p_value is not None and not np.isnan(p_value) else 1.0
                    }
            except Exception as e:
                corr_results[col] = {
                    "correlation": 0,
                    "p_value": 1.0,
                    "error": str(e)
                }

        # Sort by absolute correlation
        sorted_correlations = dict(sorted(
            corr_results.items(),
            key=lambda x: abs(float(x[1]["correlation"])) if isinstance(x[1]["correlation"], (int, float)) else 0,
            reverse=True
        ))

        results["correlation_analysis"] = sorted_correlations

        return results