from typing import Dict, Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
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
            # Get outlier method and threshold from params, with defaults
            outlier_method = params.get("outlier_method", "zscore")
            outlier_threshold = params.get("outlier_threshold", 3.0)

            # Calculate outliers based on the specified method
            outlier_results = {}

            for col in feature_columns:
                try:
                    # Check if column is numeric
                    col_type = [field.dataType for field in data.schema.fields if field.name == col][0]
                    is_numeric = "DoubleType" in str(col_type) or "IntegerType" in str(col_type) or "FloatType" in str(col_type)

                    if is_numeric:
                        if outlier_method == "zscore":
                            # Z-score method (default)
                            # Calculate mean and std
                            mean = data.select(F.mean(col)).collect()[0][0]
                            std = data.select(F.stddev(col)).collect()[0][0]

                            if std > 0:
                                # Calculate z-score
                                z_score_col = f"{col}_zscore"
                                z_score_df = data.withColumn(z_score_col, (F.col(col) - mean) / std)

                                # Find outliers (|z| > threshold)
                                outlier_df = z_score_df.filter(F.abs(F.col(z_score_col)) > outlier_threshold)
                                outlier_count = outlier_df.count()

                                # Get sample of outlier values (limited to 20)
                                outlier_values = []
                                if outlier_count > 0:
                                    # Collect a sample of outlier values
                                    outlier_sample = outlier_df.select(col).limit(20).collect()
                                    outlier_values = [float(row[0]) for row in outlier_sample]

                                # Save results
                                outlier_results[col] = {
                                    "mean": float(mean),
                                    "std": float(std),
                                    "outlier_count": outlier_count,
                                    "outlier_percentage": float(outlier_count) / data.count() * 100,
                                    "outlier_values": outlier_values,
                                    "method": "zscore",
                                    "threshold": outlier_threshold
                                }

                        elif outlier_method == "iqr":
                            # IQR method
                            # Calculate quartiles
                            q1 = data.selectExpr(f"percentile_approx({col}, 0.25)").collect()[0][0]
                            q3 = data.selectExpr(f"percentile_approx({col}, 0.75)").collect()[0][0]
                            iqr = q3 - q1

                            if iqr > 0:
                                # Define bounds
                                lower_bound = q1 - outlier_threshold * iqr
                                upper_bound = q3 + outlier_threshold * iqr

                                # Find outliers
                                outlier_df = data.filter((F.col(col) < lower_bound) | (F.col(col) > upper_bound))
                                outlier_count = outlier_df.count()

                                # Get sample of outlier values (limited to 20)
                                outlier_values = []
                                if outlier_count > 0:
                                    # Collect a sample of outlier values
                                    outlier_sample = outlier_df.select(col).limit(20).collect()
                                    outlier_values = [float(row[0]) for row in outlier_sample]

                                # Save results
                                outlier_results[col] = {
                                    "q1": float(q1),
                                    "q3": float(q3),
                                    "iqr": float(iqr),
                                    "lower_bound": float(lower_bound),
                                    "upper_bound": float(upper_bound),
                                    "outlier_count": outlier_count,
                                    "outlier_percentage": float(outlier_count) / data.count() * 100,
                                    "outlier_values": outlier_values,
                                    "method": "iqr",
                                    "threshold": outlier_threshold
                                }
                        else:
                            # Unsupported method
                            outlier_results[col] = {
                                "error": f"Unsupported outlier detection method: {outlier_method}"
                            }
                except Exception as e:
                    outlier_results[col] = {"error": str(e)}

            results["outlier_detection"] = outlier_results

        return results