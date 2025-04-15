from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Optional
from pathlib import Path
import os
from src.api.dependencies import get_engine
from src.core.engine_context import EngineContext
from pydantic import BaseModel, Field
import polars as pl

router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])

# Define constants for file paths
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

class PreprocessingParams(BaseModel):
    file_id: str = Field(..., description="The unique identifier of the file to process")
    handle_missing: bool = Field(default=True, description="Whether to handle missing values")
    missing_strategy: Optional[str] = Field(default=None, description="Strategy for handling missing values")
    remove_duplicates: bool = Field(default=True, description="Whether to remove duplicate rows")
    duplicate_subset: Optional[List[str]] = Field(default=None, description="Columns to consider for duplicate detection")
    handle_outliers: bool = Field(default=False, description="Whether to handle outliers")
    outlier_method: Optional[str] = Field(default=None, description="Method for handling outliers")
    scale_features: bool = Field(default=False, description="Whether to scale features")
    scaler_method: Optional[str] = Field(default=None, description="Method for scaling features")

    model_config = {
        "json_schema_extra": {
            "example": {
                "file_id": "example.csv",
                "handle_missing": True,
                "missing_strategy": "mean",
                "remove_duplicates": True,
                "duplicate_subset": ["column1", "column2"]
            }
        }
    }

@router.post("/")
async def preprocess_data(
    params: PreprocessingParams,
    engine: EngineContext = Depends(get_engine)
):
    """Preprocess the data with specified configuration"""
    try:
        # Validate file_id
        if not params.file_id or params.file_id.strip() == "":
            raise HTTPException(
                status_code=400,
                detail="file_id is required and cannot be empty"
            )

        # Look for the file with the exact file_id prefix
        matching_files = list(RAW_DATA_DIR.glob(f"{params.file_id}*"))
        if not matching_files:
            raise HTTPException(
                status_code=404,
                detail=f"File not found with ID: {params.file_id}"
            )
        
        # Use the first matching file
        raw_file_path = matching_files[0]
        if not raw_file_path.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file path for ID: {params.file_id}"
            )

        # ---------------- Preprocessing ----------------
        # Load and process the data
        df = engine.load_data(str(raw_file_path))
        original_rows = len(df) if engine.get_engine_name() == "PandasEngine" else df.height
        
        # Handle missing values if strategy is provided
        if params.handle_missing and params.missing_strategy:
            df = engine.preprocessing.handle_missing_values(
                df, 
                strategy=params.missing_strategy
            )
        
        # Remove duplicates if requested
        if params.remove_duplicates:
            df = engine.preprocessing.remove_duplicates(
                df, 
                subset=params.duplicate_subset
            )
        
        # Handle outliers
        if params.handle_outliers and params.outlier_method:
            numeric_columns = (
                df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if engine.get_engine_name() == "PandasEngine"
                else [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Int64]]
            )
            
            if numeric_columns:  # Only process if numeric columns exist
                df = engine.preprocessing.detect_outliers(
                    df,
                    columns=numeric_columns,
                    method=params.outlier_method
                )
        
        # Scale features
        if params.scale_features and params.scaler_method:
            if engine.get_engine_name() == "PandasEngine":
                # Get numeric and categorical columns that can be converted
                df_types = df.dtypes
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # First handle categorical columns that can be converted
                for col in categorical_columns:
                    unique_values = df[col].dropna().unique()
                    if len(unique_values) < len(df) / 2:  # Only process if cardinality is reasonable
                        df = engine.preprocessing.scale_column(
                            df,
                            column=col,
                            method=params.scaler_method
                        )
                
                # Then handle numeric columns
                for col in numeric_columns:
                    df = engine.preprocessing.scale_column(
                        df,
                        column=col,
                        method=params.scaler_method
                    )
            else:  # polars
                numeric_columns = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Int64]]
                for col in numeric_columns:
                    df = engine.preprocessing.scale_column(
                        df,
                        column=col,
                        method=params.scaler_method
                    )
        
        # Save processed data using file_id
        processed_path = PROCESSED_DATA_DIR / f"{params.file_id}_processed.parquet"
        
        if engine.get_engine_name() == "PandasEngine":
            df.to_parquet(processed_path)
        else:  # polars
            df.write_parquet(processed_path)
        
        # Get basic statistics
        stats = {
            "original_rows": original_rows,
            "processed_rows": len(df) if engine.get_engine_name() == "PandasEngine" else df.height,
            "columns": list(df.columns),
            "numeric_columns": (
                df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if engine.get_engine_name() == "PandasEngine"
                else [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Int64]]
            ),
            "missing_values": (
                df.isna().sum().to_dict()
                if engine.get_engine_name() == "PandasEngine"
                else {col: df[col].null_count() for col in df.columns}
            )
        }
        
        return {
            "status": "success",
            "file_id": params.file_id,
            "statistics": stats,
            "processed_path": str(processed_path),
            "engine_used": engine.get_engine_name()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
