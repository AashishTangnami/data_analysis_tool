"""
Data ingestion module for handling file uploads and parsing.
"""
import pandas as pd
from pathlib import Path
import json
import uuid
import os
from typing import Dict, Any, Optional, Tuple

from ..config import RAW_DATA_DIR, ALLOWED_EXTENSIONS

def validate_file_format(file_path: str) -> bool:
    """
    Validate if the file format is supported.
    
    Args:
        file_path: Path to the uploaded file
        
    Returns:
        bool: True if format is supported, False otherwise
    """
    file_extension = Path(file_path).suffix.lower()
    return file_extension in ALLOWED_EXTENSIONS

def save_uploaded_file(file_content: bytes, original_filename: str) -> str:
    """
    Save an uploaded file to the raw data directory.
    
    Args:
        file_content: The binary content of the uploaded file
        original_filename: Original filename from the upload
        
    Returns:
        str: Path where the file was saved
    """
    # Generate a unique filename
    unique_id = str(uuid.uuid4())
    ext = Path(original_filename).suffix
    filename = f"{unique_id}{ext}"
    
    # Create the save path
    save_path = os.path.join(RAW_DATA_DIR, filename)
    
    # Save the file
    with open(save_path, "wb") as f:
        f.write(file_content)
    
    return save_path

def load_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load data from a file into a pandas DataFrame.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Tuple containing:
            - DataFrame: The loaded data
            - Dict: Metadata about the loaded file
    """
    file_extension = Path(file_path).suffix.lower()
    metadata = {
        "original_file_path": file_path,
        "file_format": file_extension[1:],  # Remove the dot
    }
    
    try:
        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension == ".json":
            df = pd.read_json(file_path)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Add metadata
        metadata["rows"] = len(df)
        metadata["columns"] = len(df.columns)
        metadata["column_names"] = df.columns.tolist()
        
        return df, metadata
    
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")
