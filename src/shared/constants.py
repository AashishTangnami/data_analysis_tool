from enum import Enum

class EngineType(str, Enum):
    """Enum for supported engine types."""
    PANDAS = "pandas"
    POLARS = "polars"
    PYSPARK = "pyspark"

class FileType(str, Enum):
    """Enum for supported file types."""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PARQUET = "parquet"

class AnalysisType(str, Enum):
    """Enum for supported analysis types."""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"

class PreprocessingOperation(str, Enum):
    """Enum for preprocessing operation types."""
    DROP_COLUMNS = "drop_columns"
    FILL_MISSING = "fill_missing"
    DROP_MISSING = "drop_missing"
    ENCODE_CATEGORICAL = "encode_categorical"
    SCALE_NUMERIC = "scale_numeric"
    APPLY_FUNCTION = "apply_function"

class FillMethod(str, Enum):
    """Enum for missing value fill methods."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"

class EncodingMethod(str, Enum):
    """Enum for categorical encoding methods."""
    ONE_HOT = "one_hot"
    LABEL = "label"
    ORDINAL = "ordinal"

class ScalingMethod(str, Enum):
    """Enum for numeric scaling methods."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"

class FunctionType(str, Enum):
    """Enum for function types to apply."""
    LOG = "log"
    SQRT = "sqrt"
    SQUARE = "square"
    ABSOLUTE = "absolute"

class VisualizationType(str, Enum):
    """Enum for visualization types."""
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    BAR = "bar"
    LINE = "line"
    HEATMAP = "heatmap"
    BOX = "box"
    PIE = "pie"