from typing import Dict, Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator
from core.analysis.diagnostic.base import DiagnosticAnalysisBase

class PySparkDiagnosticAnalysis(DiagnosticAnalysisBase):
    """
    PySpark implementation of diagnostic analysis strategy.
    """
    
    def analyze(self, data: SparkDataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform diagnostic analysis using PySpark.
        
        Args:
            data: PySpark DataFrame
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
        
        # Initialize results
        results = {
            "feature_importance": {},
            "outlier_detection": {},
        }
        
        # Feature importance analysis
        if run_feature_importance and feature_columns and target_column:
            # Create feature vector
            assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
            vector_df = assembler.transform(data)
            
            # Determine if classification or regression
            is_categorical = False
            target_type = [field.dataType for field in data.schema.fields if field.name == target_column][0]
            if "StringType" in str(target_type) or data.select(target_column).distinct().count() < 10:
                is_categorical = True
            
            # Train a model for feature importance
            if is_categorical:
                model = RandomForestClassifier(
                    featuresCol="features",
                    labelCol=target_column,
                    numTrees=10
                )
            else:
                model = RandomForestRegressor(
                    featuresCol="features",
                    labelCol=target_column,
                    numTrees=10
                )
            
            # Fit the model
            model_fitted = model.fit(vector_df)
            
            # Extract feature importance
            importances = model_fitted.featureImportances.toArray()
            
            # Create a dictionary of feature importance
            results["feature_importance"] = {
                feature: float(importance)
                for feature, importance in zip(feature_columns, importances)
            }
        
        # Outlier detection
        if run_outlier_detection and feature_columns:
            # Calculate Z-scores for numeric columns
            outlier_results = {}
            
            for col in feature_columns:
                # Check if column is numeric
                col_type = [field.dataType for field in data.schema.fields if field.name == col][0]
                is_numeric = "DoubleType" in str(col_type) or "IntegerType" in str(col_type) or "FloatType" in str(col_type)
                
                if is_numeric:
                    # Calculate mean and std
                    mean = data.select(F.mean(col)).collect()[0][0]
                    std = data.select(F.stddev(col)).collect()[0][0]
                    
                    if std > 0:
                        # Calculate z-score
                        z_score_col = f"{col}_zscore"
                        z_score_df = data.withColumn(z_score_col, (F.col(col) - mean) / std)
                        
                        # Find outliers (|z| > 3)
                        outliers = z_score_df.filter(F.abs(F.col(z_score_col)) > 3).count()
                        
                        # Save results
                        outlier_results[col] = {
                            "mean": float(mean),
                            "std": float(std),
                            "outlier_count": outliers,
                            "outlier_percentage": float(outliers) / data.count() * 100
                        }
            
            results["outlier_detection"] = outlier_results
        
        return results