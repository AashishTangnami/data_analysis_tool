# core/analysis/predictive/polars_predictive.py
import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional
from core.analysis.predictive.base import PredictiveAnalysisBase

class PolarsPredictiveAnalysis(PredictiveAnalysisBase):
    """
    Polars implementation of predictive analysis strategy.
    """
    
    def analyze(self, data: pl.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform predictive analysis using polars.
        
        Args:
            data: polars DataFrame
            params: Parameters for the analysis
                - target_column: Target variable to predict
                - feature_columns: List of features to use
                - problem_type: 'regression' or 'classification'
                - model_type: Type of model to train
                - test_size: Size of test set
            
        Returns:
            Dictionary containing analysis results
        """
        # Since Polars doesn't have native ML capabilities,
        # we'll convert to pandas and use sklearn
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.linear_model import LinearRegression, LogisticRegression
        
        # Extract parameters
        target_column = params.get("target_column")
        feature_columns = params.get("feature_columns", [])
        problem_type = params.get("problem_type", "regression")
        model_type = params.get("model_type", "random_forest")
        test_size = params.get("test_size", 0.2)
        
        # Validate inputs
        if not target_column or target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        feature_columns = [col for col in feature_columns if col in data.columns]
        if not feature_columns:
            raise ValueError("No valid feature columns provided")
        
        # Initialize results
        results = {
            "model_performance": {},
            "feature_importance": {},
            "predictions": []
        }
        
        # Select relevant columns
        selected_data = data.select([target_column] + feature_columns)
        
        # Handle missing values for analysis
        selected_data = selected_data.drop_nulls()
        
        # Convert to pandas for ML
        pandas_df = selected_data.to_pandas()
        
        # Prepare features and target
        X = pandas_df[feature_columns]
        y = pandas_df[target_column]
        
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing for numeric and categorical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Select model based on problem type and model type
        if problem_type == "regression":
            if model_type == "random_forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == "gradient_boosting":
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif model_type == "linear_model":
                model = LinearRegression()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Create and train pipeline
            pipe = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            pipe.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipe.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results["model_performance"] = {
                "mean_squared_error": float(mse),
                "root_mean_squared_error": float(rmse),
                "r2_score": float(r2)
            }
            
        elif problem_type == "classification":
            if model_type == "random_forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == "gradient_boosting":
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            elif model_type == "linear_model":
                model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Create and train pipeline
            pipe = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            pipe.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipe.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Handle multi-class classification
            if len(np.unique(y)) > 2:
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
            else:
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
            
            results["model_performance"] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            feature_names = X.columns.tolist()
            importances = model.feature_importances_
            
            # Create a dictionary of feature importance
            feature_importance = {
                feature: float(importance)
                for feature, importance in zip(feature_names, importances)
            }
            
            # Sort by importance
            results["feature_importance"] = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        elif hasattr(model, 'coef_'):
            # For linear models
            feature_names = X.columns.tolist()
            
            # Get coefficients
            coefficients = model.coef_
            if len(coefficients.shape) > 1:
                # For multi-class, use the mean absolute coefficient
                coefficients = np.mean(np.abs(coefficients), axis=0)
            
            # Create a dictionary of feature importance
            feature_importance = {
                feature: float(importance)
                for feature, importance in zip(feature_names, coefficients)
            }
            
            # Sort by absolute importance
            results["feature_importance"] = dict(sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            ))
        
        # Sample predictions
        sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
        sample_data = X_test.iloc[sample_indices].copy()
        sample_predictions = y_pred[sample_indices]
        sample_actual = y_test.iloc[sample_indices]
        
        # Format sample predictions
        for i, idx in enumerate(sample_indices):
            prediction_entry = {
                "row_index": int(X_test.index[idx]),
                "features": X_test.iloc[idx].to_dict(),
                "predicted": float(sample_predictions[i]) if problem_type == "regression" else str(sample_predictions[i]),
                "actual": float(sample_actual.iloc[i]) if problem_type == "regression" else str(sample_actual.iloc[i])
            }
            results["predictions"].append(prediction_entry)
        
        return results