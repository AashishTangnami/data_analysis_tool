import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
from core.analysis.diagnostic.base import DiagnosticAnalysisBase

class PandasDiagnosticAnalysis(DiagnosticAnalysisBase):
    """
    Pandas implementation of diagnostic analysis strategy.
    """

    def analyze(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform diagnostic analysis using pandas.

        Args:
            data: pandas DataFrame
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
        # run_feature_importance = params.get("run_feature_importance", True)
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
            # "feature_importance": {},
            "outlier_detection": {},
            "correlation_analysis": {}
        }

        # Select relevant columns
        selected_data = data[[target_column] + feature_columns].copy()

        # Handle missing values for analysis - make a copy to avoid modifying original
        selected_data_no_na = selected_data.dropna().copy()
        if len(selected_data_no_na) == 0:
            raise ValueError("After dropping NA values, no data remains for analysis")

        # Feature importance analysis
        '''
        if run_feature_importance is :
            try:
                # Determine if classification or regression
                is_categorical = False
                if pd.api.types.is_object_dtype(selected_data[target_column]) or selected_data[target_column].nunique() < 10:
                    is_categorical = True

                # Prepare features and target
                X = selected_data_no_na[feature_columns].copy()
                y = selected_data_no_na[target_column].copy()

                # Convert categorical features to numeric
                for col in X.select_dtypes(include=['object', 'category']).columns:
                    X[col] = X[col].astype('category').cat.codes

                # Make sure all data is numeric and handle any remaining NaN
                X = X.apply(pd.to_numeric, errors='coerce')
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
        '''
        # Outlier detection
        if run_outlier_detection:
            outlier_results = {}

            for col in feature_columns:
                try:
                    if pd.api.types.is_numeric_dtype(selected_data[col]):
                        # Skip columns with all NaN
                        if selected_data[col].isna().all():
                            continue

                        # Calculate z-score safely
                        col_data = selected_data[col].dropna()
                        if len(col_data) > 0:
                            z_scores = stats.zscore(col_data, nan_policy='omit')

                            # Find outliers (|z| > 3)
                            outlier_indices = np.where(abs(z_scores) > 3)[0]
                            outliers = col_data.iloc[outlier_indices]

                            # Save results
                            outlier_results[col] = {
                                "mean": float(col_data.mean()),
                                "std": float(col_data.std()),
                                "outlier_count": len(outliers),
                                "outlier_percentage": len(outliers) / len(col_data) * 100,
                                "outlier_indices": outliers.index.tolist()[:20]  # Limit to 20 indices
                            }
                except Exception as e:
                    outlier_results[col] = {"error": str(e)}

            results["outlier_detection"] = outlier_results

        # Correlation analysis with target
        corr_results = {}
        for col in feature_columns:
            try:
                if pd.api.types.is_numeric_dtype(selected_data[col]):
                    # Skip columns with insufficient data
                    col_data = selected_data[col].dropna()
                    target_data = selected_data[target_column].dropna()

                    # Get common indices between feature and target to handle missing values
                    common_indices = col_data.index.intersection(target_data.index)
                    if len(common_indices) < 2:
                        corr_results[col] = {
                            "correlation": 0,
                            "p_value": 1.0,
                            "error": "Insufficient data for correlation analysis"
                        }
                        continue

                    col_data = col_data[common_indices]
                    target_data = target_data[common_indices]

                    if pd.api.types.is_numeric_dtype(selected_data[target_column]):
                        # Calculate Pearson correlation
                        correlation = col_data.corr(target_data)
                        p_value = stats.pearsonr(col_data, target_data)[1]
                    else:
                        # For categorical target, use ANOVA F-value if there are enough samples
                        categories = target_data.unique()

                        # Skip ANOVA if there's only one category
                        if len(categories) < 2:
                            corr_results[col] = {
                                "correlation": 0,
                                "p_value": 1.0,
                                "error": "Target has only one category"
                            }
                            continue

                        # Check if we have enough samples in each category
                        groups = [col_data[target_data == cat].dropna() for cat in categories]
                        if any(len(group) < 2 for group in groups):
                            corr_results[col] = {
                                "correlation": 0,
                                "p_value": 1.0,
                                "error": "Some groups have insufficient samples for ANOVA"
                            }
                            continue

                        f_stat, p_value = stats.f_oneway(*groups)
                        correlation = f_stat  # Using F-statistic as measure of association

                    corr_results[col] = {
                        "correlation": float(correlation) if not pd.isna(correlation) else 0,
                        "p_value": float(p_value) if not pd.isna(p_value) else 1.0
                    }
            except Exception as e:
                corr_results[col] = {
                    "correlation": 0,
                    "p_value": 1.0,
                    "error": str(e)
                }

        # Sort by absolute correlation
        # All entries should have correlation values now (0 for errors/invalid)
        sorted_correlations = dict(sorted(
            corr_results.items(),
            key=lambda x: abs(float(x[1]["correlation"])) if isinstance(x[1]["correlation"], (int, float)) else 0,
            reverse=True
        ))

        results["correlation_analysis"] = sorted_correlations

        return results