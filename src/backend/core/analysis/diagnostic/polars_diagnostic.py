import polars as pl
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any
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
            # "feature_importance": {},  # Comment out to match pandas implementation
            "outlier_detection": {},
            "correlation_analysis": {}
        }

        # Select relevant columns
        selected_data = data.select([target_column] + feature_columns)

        # Handle missing values for analysis
        selected_data = selected_data.drop_nulls()

        # Feature importance analysis
        if run_feature_importance:
            try:
                # For feature importance, we need to convert to pandas and use sklearn
                # Import here to avoid unnecessary dependencies if not used
                import pandas as pd
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

                # Convert to pandas
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

                # Make sure all data is numeric and handle any remaining NaN
                for col in X.columns:
                    X[col] = pl.Series(X[col]).to_numeric(X[col], errors='coerce')

                X = X.fillna(X.mean())

                # Train random forest for feature importance
                if is_categorical:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)

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

                        # Calculate z-score
                        mean = selected_data[col].mean()
                        std = selected_data[col].std()

                        if std > 0:
                            # Create z-score
                            z_scores = ((selected_data[col] - mean) / std).alias("z_score")

                            # Find outliers (|z| > 3)
                            outliers = selected_data.with_column(z_scores).filter(pl.abs(pl.col("z_score")) > 3)

                            # Get outlier indices (convert to list for JSON serialization)
                            # Limit to 20 indices like pandas implementation
                            outlier_indices = []
                            if outliers.height > 0:
                                # Convert to pandas to get indices (simpler than tracking in polars)
                                pd_outliers = outliers.to_pandas()
                                outlier_indices = pd_outliers.index.tolist()[:20]

                            # Save results
                            outlier_results[col] = {
                                "mean": float(mean),
                                "std": float(std),
                                "outlier_count": outliers.height,
                                "outlier_percentage": outliers.height / selected_data.height * 100,
                                "outlier_indices": outlier_indices
                            }
                except Exception as e:
                    outlier_results[col] = {"error": str(e)}

            results["outlier_detection"] = outlier_results

        # Correlation analysis with target
        corr_results = {}

        for col in feature_columns:
            try:
                if _is_numeric_dtype(selected_data[col].dtype) and _is_numeric_dtype(selected_data[target_column].dtype):
                    # Calculate Pearson correlation
                    corr_df = selected_data.select(pl.corr(col, target_column).alias("correlation"))

                    # Check if we got a valid result
                    if corr_df.height > 0:
                        corr = corr_df.item(0, 0)

                        # Convert to pandas for p-value calculation
                        pd_col = selected_data[col].to_pandas()
                        pd_target = selected_data[target_column].to_pandas()

                        # Calculate p-value
                        _, p_value = stats.pearsonr(pd_col, pd_target)

                        corr_results[col] = {
                            "correlation": float(corr) if corr is not None else 0.0,
                            "p_value": p_value
                        }
                    else:
                        corr_results[col] = {
                            "correlation": None,
                            "p_value": None,
                            "error": "Could not calculate correlation"
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