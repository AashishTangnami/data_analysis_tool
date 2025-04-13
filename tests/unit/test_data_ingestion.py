"""
Unit tests for the data ingestion module.
"""
import pytest
import os
from pathlib import Path

from src.core.data_ingestion import validate_file_format, load_data

def test_validate_file_format_csv(sample_csv_file):
    """Test validation of CSV file format."""
    assert validate_file_format(sample_csv_file) is True

def test_validate_file_format_json(sample_json_file):
    """Test validation of JSON file format."""
    assert validate_file_format(sample_json_file) is True

def test_validate_file_format_excel(sample_excel_file):
    """Test validation of Excel file format."""
    assert validate_file_format(sample_excel_file) is True

def test_validate_file_format_invalid():
    """Test validation of an invalid file format."""
    # Create a temporary text file
    fd, path = pytest.importorskip("tempfile").mkstemp(suffix=".txt")
    os.close(fd)
    
    try:
        assert validate_file_format(path) is False
    finally:
        # Clean up
        if os.path.exists(path):
            os.remove(path)

def test_load_data_csv(sample_csv_file):
    """Test loading data from a CSV file."""
    df, metadata = load_data(sample_csv_file)
    
    assert df is not None
    assert len(df) == 4  # Based on our sample data
    assert len(df.columns) == 4  # Based on our sample data
    assert "name" in df.columns
    assert "file_format" in metadata
    assert metadata["file_format"] == "csv"

def test_load_data_json(sample_json_file):
    """Test loading data from a JSON file."""
    df, metadata = load_data(sample_json_file)
    
    assert df is not None
    assert len(df) == 4  # Based on our sample data
    assert "name" in df.columns
    assert "file_format" in metadata
    assert metadata["file_format"] == "json"

def test_load_data_excel(sample_excel_file):
    """Test loading data from an Excel file."""
    df, metadata = load_data(sample_excel_file)
    
    assert df is not None
    assert len(df) == 4  # Based on our sample data
    assert "name" in df.columns
    assert "file_format" in metadata
    assert metadata["file_format"] == "xlsx"
