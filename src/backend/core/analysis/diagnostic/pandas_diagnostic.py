# import pandas as pd
# import numpy as np
# from typing import Dict, Any, List, Optional
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from scipy import stats
# from core.analysis.diagnostic.base import DiagnosticAnalysisBase

# class PandasDiagnosticAnalysis(DiagnosticAnalysisBase):
#     """
#     Pandas implementation of diagnostic analysis strategy.
#     """
    
#     def analyze(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Perform diagnostic analysis using pandas.
        
#         Args:
#             data: pandas DataFrame
#             params: Parameters for the analysis
#                 - target_column: Target variable for analysis
#                 - feature_columns: List of features to use
#                 - run_feature_importance: Whether to run feature importance analysis
#                 - run_outlier_detection: Whether to run outlier detection
            
#         Returns:
#             Dictionary containing analysis results
#         """
#         # Extract parameters
#         target_column = params.get("target_column")
#         feature_columns = params.get("feature_columns", [])
#         run_feature_importance = params.get("run_feature_importance", True)
#         run_outlier_detection = params.get("run_outlier_detection", True)
        
#         # Validate inputs
#         if not target_column or target_column not in data.columns:
#             raise ValueError(f"Target column '{target_column}' not found in data")
        
#         feature_columns = [col for col in feature_columns if col in data.columns]
#         if not feature_columns:
#             raise ValueError("No valid feature columns provided")
        
#         # Initialize results
#         results = {
#             "feature_importance": {},
#             "outlier_detection": {},
#             "correlation_analysis": {}
#         }
        
#         # Select relevant columns
#         selected_data = data[[target_column] + feature_columns].copy()
        
#         # Handle missing values for analysis
#         selected_data = selected_data.dropna()
        
#         # Feature importance analysis
#         if run_feature_importance:
#             # Determine if classification or regression
#             is_categorical = False
#             if pd.api.types.is_object_dtype(selected_data[target_column]) or selected_data[target_column].nunique() < 10:
#                 is_categorical = True
            
#             # Prepare features and target
#             X = selected_data[feature_columns]
#             y = selected_data[target_column]
            
#             # Convert categorical features to numeric
#             for col in X.select_dtypes(include=['object', 'category']).columns:
#                 X[col] = X[col].astype('category').cat.codes
            
#             # Train random forest for feature importance
#             if is_categorical:
#                 model = RandomForestClassifier(n_estimators=100, random_state=42)
#             else:
#                 model = RandomForestRegressor(n_estimators=100, random_state=42)
            
#             try:
#                 model.fit(X, y)
                
#                 # Get feature importance
#                 importances = model.feature_importances_
                
#                 # Create a dictionary of feature importance
#                 results["feature_importance"] = {
#                     feature: float(importance)
#                     for feature, importance in zip(feature_columns, importances)
#                 }
                
#                 # Sort by importance
#                 results["feature_importance"] = dict(sorted(
#                     results["feature_importance"].items(),
#                     key=lambda x: x[1],
#                     reverse=True
#                 ))
            
#             except Exception as e:
#                 results["feature_importance"] = {"error": str(e)}
        
#         # Outlier detection
#         if run_outlier_detection:
#             outlier_results = {}
            
#             for col in feature_columns:
#                 if pd.api.types.is_numeric_dtype(selected_data[col]):
#                     # Calculate z-score
#                     z_scores = stats.zscore(selected_data[col], nan_policy='omit')
                    
#                     # Find outliers (|z| > 3)
#                     outliers = selected_data[abs(z_scores) > 3]
                    
#                     # Save results
#                     outlier_results[col] = {
#                         "mean": float(selected_data[col].mean()),
#                         "std": float(selected_data[col].std()),
#                         "outlier_count": len(outliers),
#                         "outlier_percentage": len(outliers) / len(selected_data) * 100,
#                         "outlier_indices": outliers.index.tolist()[:20]  # Limit to 20 indices
#                     }
            
#             results["outlier_detection"] = outlier_results
        
#         # Correlation analysis with target
#         corr_results = {}
#         for col in feature_columns:
#             if pd.api.types.is_numeric_dtype(selected_data[col]):
#                 if pd.api.types.is_numeric_dtype(selected_data[target_column]):
#                     # Calculate Pearson correlation
#                     correlation = selected_data[col].corr(selected_data[target_column])
#                     p_value = stats.pearsonr(selected_data[col].dropna(), 
#                                             selected_data[target_column].dropna())[1]
#                 else:
#                     # For categorical target, use ANOVA F-value
#                     categories = selected_data[target_column].unique()
#                     f_stat, p_value = stats.f_oneway(
#                         *[selected_data[col][selected_data[target_column] == cat].dropna() 
#                           for cat in categories]
#                     )
#                     correlation = f_stat  # Using F-statistic as measure of association
                
#                 corr_results[col] = {
#                     "correlation": float(correlation),
#                     "p_value": float(p_value)
#                 }
        
#         # Sort by absolute correlation
#         results["correlation_analysis"] = dict(sorted(
#             corr_results.items(),
#             key=lambda x: abs(x[1]["correlation"]),
#             reverse=True
#         ))
        
#         return results



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
        run_feature_importance = params.get("run_feature_importance", True)
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
            "feature_importance": {},
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
        if run_feature_importance:
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
                            "correlation": None,
                            "p_value": None,
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
                                "correlation": None,
                                "p_value": None,
                                "error": "Target has only one category"
                            }
                            continue
                            
                        # Check if we have enough samples in each category
                        groups = [col_data[target_data == cat].dropna() for cat in categories]
                        if any(len(group) < 2 for group in groups):
                            corr_results[col] = {
                                "correlation": None,
                                "p_value": None,
                                "error": "Some groups have insufficient samples for ANOVA"
                            }
                            continue
                            
                        f_stat, p_value = stats.f_oneway(*groups)
                        correlation = f_stat  # Using F-statistic as measure of association
                    
                    corr_results[col] = {
                        "correlation": float(correlation) if not pd.isna(correlation) else None,
                        "p_value": float(p_value) if not pd.isna(p_value) else None
                    }
            except Exception as e:
                corr_results[col] = {
                    "correlation": None, 
                    "p_value": None,
                    "error": str(e)
                }
        
        # Sort by absolute correlation
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