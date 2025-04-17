from typing import Dict, Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from core.analysis.predictive.base import PredictiveAnalysisBase

class PySparkPredictiveAnalysis(PredictiveAnalysisBase):
    """
    PySpark implementation of predictive analysis strategy.
    """
    
    def analyze(self, data: SparkDataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform predictive analysis using PySpark ML.
        
        Args:
            data: PySpark DataFrame
            params: Parameters for the analysis
                - target_column: Target variable to predict
                - feature_columns: List of features to use
                - problem_type: 'regression' or 'classification'
                - model_type: Type of model to train
                - test_size: Size of test set
            
        Returns:
            Dictionary containing analysis results
        """
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
        selected_data = selected_data.dropna()
        
        # Identify numeric and categorical columns
        numeric_cols = []
        categorical_cols = []
        
        for col in feature_columns:
            col_type = [field.dataType for field in selected_data.schema.fields if field.name == col][0]
            if "DoubleType" in str(col_type) or "IntegerType" in str(col_type) or "FloatType" in str(col_type):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Create a preprocessing pipeline
        stages = []
        
        # Process categorical columns
        indexed_categorical_cols = []
        encoded_categorical_cols = []
        
        for cat_col in categorical_cols:
            # String indexing (convert to numeric category)
            indexer = StringIndexer(
                inputCol=cat_col,
                outputCol=f"{cat_col}_indexed",
                handleInvalid="keep"
            )
            stages.append(indexer)
            indexed_categorical_cols.append(f"{cat_col}_indexed")
            
            # One-hot encoding
            encoder = OneHotEncoder(
                inputCols=[f"{cat_col}_indexed"],
                outputCols=[f"{cat_col}_encoded"],
                dropLast=True
            )
            stages.append(encoder)
            encoded_categorical_cols.append(f"{cat_col}_encoded")
        
        # Combine all preprocessed features
        feature_assembler_input = numeric_cols + encoded_categorical_cols
        
        # Handle target column for classification
        if problem_type == "classification":
            target_indexer = StringIndexer(
                inputCol=target_column,
                outputCol=f"{target_column}_indexed",
                handleInvalid="keep"
            )
            stages.append(target_indexer)
            labeled_target = f"{target_column}_indexed"
        else:
            labeled_target = target_column
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_assembler_input,
            outputCol="features",
            handleInvalid="keep"
        )
        stages.append(assembler)
        
        # Scale features
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        stages.append(scaler)
        
        # Select model based on problem type and model type
        if problem_type == "regression":
            if model_type == "random_forest":
                model = RandomForestRegressor(
                    featuresCol="scaled_features",
                    labelCol=labeled_target,
                    numTrees=100
                )
            elif model_type == "gradient_boosting":
                model = GBTRegressor(
                    featuresCol="scaled_features",
                    labelCol=labeled_target,
                    maxIter=100
                )
            elif model_type == "linear_model":
                model = LinearRegression(
                    featuresCol="scaled_features",
                    labelCol=labeled_target,
                    maxIter=100,
                    regParam=0.1,
                    elasticNetParam=0.5
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        elif problem_type == "classification":
            if model_type == "random_forest":
                model = RandomForestClassifier(
                    featuresCol="scaled_features",
                    labelCol=labeled_target,
                    numTrees=100
                )
            elif model_type == "gradient_boosting":
                model = GBTClassifier(
                    featuresCol="scaled_features",
                    labelCol=labeled_target,
                    maxIter=100
                )
            elif model_type == "linear_model":
                model = LogisticRegression(
                    featuresCol="scaled_features",
                    labelCol=labeled_target,
                    maxIter=100,
                    regParam=0.1,
                    elasticNetParam=0.5
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")
        
        # Add model to pipeline
        stages.append(model)
        pipeline = Pipeline(stages=stages)
        
        # Split data into training and test sets
        train_data, test_data = selected_data.randomSplit([1 - test_size, test_size], seed=42)
        
        # Train the model
        model_pipeline = pipeline.fit(train_data)
        
        # Make predictions
        predictions = model_pipeline.transform(test_data)
        
        # Evaluate model performance
        if problem_type == "regression":
            evaluator = RegressionEvaluator(
                labelCol=labeled_target,
                predictionCol="prediction",
                metricName="rmse"
            )
            
            rmse = evaluator.evaluate(predictions)
            
            # Also calculate MSE and R2
            evaluator.setMetricName("mse")
            mse = evaluator.evaluate(predictions)
            
            evaluator.setMetricName("r2")
            r2 = evaluator.evaluate(predictions)
            
            results["model_performance"] = {
                "root_mean_squared_error": float(rmse),
                "mean_squared_error": float(mse),
                "r2_score": float(r2)
            }
            
        elif problem_type == "classification":
            # Calculate accuracy
            evaluator = MulticlassClassificationEvaluator(
                labelCol=labeled_target,
                predictionCol="prediction",
                metricName="accuracy"
            )
            accuracy = evaluator.evaluate(predictions)
            
            # Calculate precision
            evaluator.setMetricName("weightedPrecision")
            precision = evaluator.evaluate(predictions)
            
            # Calculate recall
            evaluator.setMetricName("weightedRecall")
            recall = evaluator.evaluate(predictions)
            
            # Calculate F1 score
            evaluator.setMetricName("f1")
            f1 = evaluator.evaluate(predictions)
            
            results["model_performance"] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }
        
        # Extract feature importance if available
        if hasattr(model_pipeline.stages[-1], "featureImportances"):
            # For tree-based models
            importances = model_pipeline.stages[-1].featureImportances.toArray()
            
            # Map back to original feature names (challenging in PySpark)
            # We'll use a simplified approach
            feature_names = feature_assembler_input
            
            if len(feature_names) == len(importances):
                # Direct mapping
                feature_importance = {
                    feature: float(importance)
                    for feature, importance in zip(feature_names, importances)
                }
            else:
                # Just use indices if lengths don't match
                feature_importance = {
                    f"feature_{i}": float(importance)
                    for i, importance in enumerate(importances)
                }
            
            # Sort by importance
            results["feature_importance"] = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        
        elif hasattr(model_pipeline.stages[-1], "coefficients"):
            # For linear models
            coefficients = model_pipeline.stages[-1].coefficients.toArray()
            
            # Map back to feature names
            feature_names = feature_assembler_input
            
            if len(feature_names) == len(coefficients):
                # Direct mapping
                feature_importance = {
                    feature: float(abs(coef))  # Use absolute values for linear models
                    for feature, coef in zip(feature_names, coefficients)
                }
            else:
                # Just use indices if lengths don't match
                feature_importance = {
                    f"feature_{i}": float(abs(coef))
                    for i, coef in enumerate(coefficients)
                }
            
            # Sort by importance
            results["feature_importance"] = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        
        # Sample predictions (top 10 rows)
        prediction_samples = predictions.select(
            "*", 
            F.monotonically_increasing_id().alias("row_id")
        ).limit(10)
        
        prediction_rows = prediction_samples.collect()
        
        # Format sample predictions
        for row in prediction_rows:
            features = {col: row[col] for col in feature_columns}
            
            prediction_entry = {
                "row_index": int(row["row_id"]),
                "features": features,
                "predicted": float(row["prediction"]) if problem_type == "regression" else int(row["prediction"]),
                "actual": float(row[labeled_target]) if problem_type == "regression" else int(row[labeled_target])
            }
            
            results["predictions"].append(prediction_entry)
        
        return results