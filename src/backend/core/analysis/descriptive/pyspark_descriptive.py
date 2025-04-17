# core/analysis/descriptive/pyspark_descriptive.py
from typing import Any, Dict, List, Optional
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.ml.stat import Correlation
from core.analysis.descriptive.base import DescriptiveAnalysisBase

class PySparkDescriptiveAnalysis(DescriptiveAnalysisBase):
    """
    PySpark implementation of descriptive analysis strategy.
    """
    
    def analyze(self, data: SparkDataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform descriptive analysis using PySpark.
        
        Args:
            data: PySpark DataFrame
            params: Parameters for the analysis
                - columns: List of columns to analyze (default: all)
                - include_numeric: Whether to include numeric analysis (default: True)
                - include_categorical: Whether to include categorical analysis (default: True)
                - include_correlations: Whether to include correlation analysis (default: True)
            
        Returns:
            Dictionary containing analysis results
        """
        # Extract parameters
        columns = params.get("columns", data.columns)
        include_numeric = params.get("include_numeric", True)
        include_categorical = params.get("include_categorical", True)
        include_correlations = params.get("include_correlations", True)
        
        # Filter to selected columns
        selected_data = data.select(columns)
        
        # Initialize results
        results = {
            "dataset_info": {
                "shape": (selected_data.count(), len(selected_data.columns)),
                "columns": selected_data.columns,
                "dtypes": {field.name: str(field.dataType) for field in selected_data.schema.fields},
                "missing_values": {col: selected_data.filter(F.col(col).isNull()).count() for col in selected_data.columns},
            },
            "numeric_analysis": {},
            "categorical_analysis": {},
            "correlations": {}
        }
        
        # Numeric analysis
        if include_numeric:
            numeric_cols = [field.name for field in selected_data.schema.fields 
                          if "DoubleType" in str(field.dataType) or 
                             "IntegerType" in str(field.dataType) or 
                             "FloatType" in str(field.dataType)]
            
            if numeric_cols:
                # Get statistics using Spark's summary method
                stats_df = selected_data.select(numeric_cols).summary(
                    "count", "mean", "stddev", "min", "25%", "50%", "75%", "max"
                )
                
                # Convert to dictionary format
                stats_rows = stats_df.collect()
                stats_dict = {}
                
                for col in numeric_cols:
                    stats_dict[col] = {}
                    # core/analysis/descriptive/pyspark_descriptive.py (continued)
                    for i, stat_name in enumerate(["count", "mean", "std", "min", "25%", "50%", "75%", "max"]):
                        try:
                            # Try to convert to float for numeric values
                            value = float(stats_rows[i][col])
                        except (ValueError, TypeError):
                            value = stats_rows[i][col]
                        
                        stats_dict[col][stat_name] = value
                
                results["numeric_analysis"]["statistics"] = stats_dict
                
                # Calculate additional metrics (skewness, kurtosis)
                skewness = {}
                kurtosis = {}
                
                for col in numeric_cols:
                    # Calculate skewness: E[((X-μ)/σ)^3]
                    mean_val = selected_data.select(F.mean(F.col(col))).collect()[0][0]
                    std_val = selected_data.select(F.stddev(F.col(col))).collect()[0][0]
                    
                    if std_val and std_val > 0:
                        skew_df = selected_data.select(
                            F.mean(F.pow((F.col(col) - mean_val) / std_val, 3)).alias("skewness")
                        ).collect()
                        skewness[col] = float(skew_df[0]["skewness"])
                        
                        # Calculate kurtosis: E[((X-μ)/σ)^4] - 3
                        kurt_df = selected_data.select(
                            (F.mean(F.pow((F.col(col) - mean_val) / std_val, 4)) - 3).alias("kurtosis")
                        ).collect()
                        kurtosis[col] = float(kurt_df[0]["kurtosis"])
                
                results["numeric_analysis"]["skewness"] = skewness
                results["numeric_analysis"]["kurtosis"] = kurtosis
        
        # Categorical analysis
        if include_categorical:
            categorical_cols = [field.name for field in selected_data.schema.fields 
                              if "StringType" in str(field.dataType)]
            
            if categorical_cols:
                value_counts = {}
                unique_counts = {}
                
                for col in categorical_cols:
                    # Get value counts (top 10)
                    counts = (selected_data.groupBy(col)
                              .count()
                              .orderBy(F.desc("count"))
                              .limit(10)
                              .collect())
                    
                    # Convert to dictionary
                    value_counts[col] = {row[col]: row["count"] for row in counts if row[col] is not None}
                    
                    # Count unique values
                    unique_count = selected_data.select(col).distinct().count()
                    unique_counts[col] = unique_count
                
                results["categorical_analysis"]["value_counts"] = value_counts
                results["categorical_analysis"]["unique_counts"] = unique_counts
        
        # Correlation analysis
        if include_correlations and len(numeric_cols) >= 2:
            # PySpark requires a vector column for correlation
            from pyspark.ml.feature import VectorAssembler
            
            # Create vector assembler
            assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
            vector_df = assembler.transform(selected_data)
            
            # Calculate correlation matrix
            corr_matrix = Correlation.corr(vector_df, "features").collect()[0][0]
            
            # Convert to dictionary format
            corr_dict = {}
            for i, col1 in enumerate(numeric_cols):
                corr_dict[col1] = {}
                for j, col2 in enumerate(numeric_cols):
                    # Extract the correlation value from the matrix
                    corr_dict[col1][col2] = float(corr_matrix[i, j])
            
            results["correlations"] = corr_dict
        
        return results