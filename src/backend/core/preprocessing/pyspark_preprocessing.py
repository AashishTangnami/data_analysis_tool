
from typing import Any, Dict, List
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, StandardScaler, MinMaxScaler
from pyspark.ml import Pipeline
from core.preprocessing.base import PreprocessingBase

class PySparkPreprocessing(PreprocessingBase):
    """
    PySpark implementation of preprocessing strategy.
    """
    
    def process(self, data: SparkDataFrame, operations: List[Dict[str, Any]]) -> SparkDataFrame:
        """
        Apply preprocessing operations to PySpark DataFrame.
        
        Args:
            data: PySpark DataFrame
            operations: List of preprocessing operations to apply
            
        Returns:
            Preprocessed PySpark DataFrame
            
        Raises:
            ValueError: If operation type is not supported
        """
        # Create a copy of the data to avoid modifying the original
        result = data
        
        # Apply each operation in sequence
        for operation in operations:
            op_type = operation.get("type")
            params = operation.get("params", {})
            
            if op_type == "drop_columns":
                result = self._drop_columns(result, **params)
            elif op_type == "fill_missing":
                result = self._fill_missing(result, **params)
            elif op_type == "drop_missing":
                result = self._drop_missing(result, **params)
            elif op_type == "encode_categorical":
                result = self._encode_categorical(result, **params)
            elif op_type == "scale_numeric":
                result = self._scale_numeric(result, **params)
            elif op_type == "apply_function":
                result = self._apply_function(result, **params)
            else:
                raise ValueError(f"Unsupported operation type: {op_type}")
        
        return result
    
    def get_available_operations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of available preprocessing operations.
        
        Returns:
            Dictionary mapping operation names to their metadata
        """
        return {
            "drop_columns": {
                "description": "Drop specified columns from the DataFrame",
                "params": {
                    "columns": "List of column names to drop"
                }
            },
            "fill_missing": {
                "description": "Fill missing values in specified columns",
                "params": {
                    "columns": "List of column names or 'all'",
                    "method": "Method to use: mean, median, mode, constant",
                    "value": "Value to use if method is constant"
                }
            },
            "drop_missing": {
                "description": "Drop rows with missing values",
                "params": {
                    "how": "How to drop: any (drop if any value is missing) or all (drop if all values are missing)"
                }
            },
            "encode_categorical": {
                "description": "Encode categorical variables",
                "params": {
                    "columns": "List of column names to encode",
                    "method": "Method to use: one_hot, label"
                }
            },
            "scale_numeric": {
                "description": "Scale numeric variables",
                "params": {
                    "columns": "List of column names to scale",
                    "method": "Method to use: standard, minmax"
                }
            },
            "apply_function": {
                "description": "Apply a function to specified columns",
                "params": {
                    "columns": "List of column names to apply function to",
                    "function": "Function to apply: log, sqrt, square, absolute, etc."
                }
            }
        }
    
    def _drop_columns(self, data: SparkDataFrame, columns: List[str]) -> SparkDataFrame:
        """Drop specified columns from the DataFrame."""
        return data.drop(*columns)
    
    def _fill_missing(self, data: SparkDataFrame, columns: List[str], 
                     method: str = "mean", value: Optional[Any] = None) -> SparkDataFrame:
        """Fill missing values in specified columns."""
        # Handle the case where columns is "all"
        if columns == "all":
            columns = data.columns
        
        result = data
        
        # Apply different methods based on column type and specified method
        for col in columns:
            if col not in data.columns:
                continue
            
            # Determine column type
            col_type = [field.dataType for field in data.schema.fields if field.name == col][0]
            is_numeric = "DoubleType" in str(col_type) or "IntegerType" in str(col_type) or "FloatType" in str(col_type)
            
            if method == "mean" and is_numeric:
                # Calculate mean and fill
                mean_value = data.select(F.avg(F.col(col))).collect()[0][0]
                result = result.withColumn(col, F.when(F.col(col).isNull(), mean_value).otherwise(F.col(col)))
            
            elif method == "median" and is_numeric:
                # Calculate approximate median (using percentile_approx)
                median_value = data.select(F.expr(f"percentile_approx({col}, 0.5)")).collect()[0][0]
                result = result.withColumn(col, F.when(F.col(col).isNull(), median_value).otherwise(F.col(col)))
            
            elif method == "mode":
                # Find the most frequent value
                mode_value = (data.groupBy(col)
                              .count()
                              .orderBy(F.desc("count"))
                              .select(col)
                              .first())
                
                if mode_value is not None:
                    mode_value = mode_value[0]
                    result = result.withColumn(col, F.when(F.col(col).isNull(), mode_value).otherwise(F.col(col)))
            
            elif method == "constant":
                # Fill with constant value
                result = result.withColumn(col, F.when(F.col(col).isNull(), value).otherwise(F.col(col)))
        
        return result
    
    def _drop_missing(self, data: SparkDataFrame, how: str = "any") -> SparkDataFrame:
        """Drop rows with missing values."""
        if how == "any":
            return data.dropna()
        elif how == "all":
            return data.dropna(how="all")
        else:
            raise ValueError(f"Unsupported 'how' parameter: {how}. Use 'any' or 'all'.")
    
    def _encode_categorical(self, data: SparkDataFrame, columns: List[str], 
                           method: str = "one_hot") -> SparkDataFrame:
        """Encode categorical variables."""
        if method == "one_hot":
            # Create pipeline stages
            stages = []
            for col in columns:
                # String indexer for categorical values
                indexer = StringIndexer(inputCol=col, outputCol=f"{col}_indexed")
                # One-hot encoder
                encoder = OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded")
                stages.extend([indexer, encoder])
            
            # Create and apply pipeline
            pipeline = Pipeline(stages=stages)
            model = pipeline.fit(data)
            encoded_data = model.transform(data)
            
            # Drop intermediate columns
            for col in columns:
                encoded_data = encoded_data.drop(f"{col}_indexed")
            
            return encoded_data
        
        elif method == "label":
            # Create pipeline stages
            stages = []
            for col in columns:
                # String indexer for categorical values
                indexer = StringIndexer(inputCol=col, outputCol=f"{col}_indexed")
                stages.append(indexer)
            
            # Create and apply pipeline
            pipeline = Pipeline(stages=stages)
            model = pipeline.fit(data)
            encoded_data = model.transform(data)
            
            # Rename columns if needed
            for col in columns:
                encoded_data = encoded_data.withColumnRenamed(f"{col}_indexed", col)
            
            return encoded_data
        
        else:
            raise ValueError(f"Unsupported encoding method: {method}. Use 'one_hot' or 'label'.")
    
    def _scale_numeric(self, data: SparkDataFrame, columns: List[str], 
                      method: str = "standard") -> SparkDataFrame:
        """Scale numeric variables."""
        if method == "standard":
            # Create pipeline stages
            stages = []
            for col in columns:
                # Standard scaler
                scaler = StandardScaler(inputCol=col, outputCol=f"{col}_scaled",
                                       withMean=True, withStd=True)
                stages.append(scaler)
            
            # Create and apply pipeline
            pipeline = Pipeline(stages=stages)
            model = pipeline.fit(data)
            scaled_data = model.transform(data)
            
            # Replace original columns with scaled versions
            for col in columns:
                scaled_data = scaled_data.withColumn(col, F.col(f"{col}_scaled")).drop(f"{col}_scaled")
            
            return scaled_data
        
        elif method == "minmax":
            # Create pipeline stages
            stages = []
            for col in columns:
                # MinMax scaler
                scaler = MinMaxScaler(inputCol=col, outputCol=f"{col}_scaled")
                stages.append(scaler)
            
            # Create and apply pipeline
            pipeline = Pipeline(stages=stages)
            model = pipeline.fit(data)
            scaled_data = model.transform(data)
            
            # Replace original columns with scaled versions
            for col in columns:
                scaled_data = scaled_data.withColumn(col, F.col(f"{col}_scaled")).drop(f"{col}_scaled")
            
            return scaled_data
        
        else:
            raise ValueError(f"Unsupported scaling method: {method}. Use 'standard' or 'minmax'.")
    
    def _apply_function(self, data: SparkDataFrame, columns: List[str], 
                       function: str = "log") -> SparkDataFrame:
        """Apply a function to specified columns."""
        result = data
        
        for col in columns:
            if col not in data.columns:
                continue
            
            # Determine column type
            col_type = [field.dataType for field in data.schema.fields if field.name == col][0]
            is_numeric = "DoubleType" in str(col_type) or "IntegerType" in str(col_type) or "FloatType" in str(col_type)
            
            if not is_numeric:
                continue
            
            if function == "log":
                # Log transformation (ensure values are positive)
                min_val = data.select(F.min(col)).collect()[0][0]
                offset = 1.0 if min_val >= 0 else -min_val + 1.0
                result = result.withColumn(col, F.log(F.col(col) + offset))
            
            elif function == "sqrt":
                # Square root (ensure values are positive)
                min_val = data.select(F.min(col)).collect()[0][0]
                offset = 0.0 if min_val >= 0 else -min_val
                result = result.withColumn(col, F.sqrt(F.col(col) + offset))
            
            elif function == "square":
                # Square
                result = result.withColumn(col, F.pow(F.col(col), 2))
            
            elif function == "absolute":
                # Absolute value
                result = result.withColumn(col, F.abs(F.col(col)))
            
            else:
                raise ValueError(f"Unsupported function: {function}. Use 'log', 'sqrt', 'square', or 'absolute'.")
        
        return result