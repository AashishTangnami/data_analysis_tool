"""
Pytest configuration file for Dynamic Data Analysis Platform tests.
"""
import pytest
import pandas as pd
import os
import tempfile
from pathlib import Path

@pytest.fixture
def sample_csv_data():
    """
    Create a sample CSV data fixture.
    """
    data = {
        "name": ["Alice", "Bob", "Charlie", "David"],
        "age": [25, 30, 35, 40],
        "income": [50000, 60000, 70000, 80000],
        "expenses": [30000, 40000, 50000, 60000]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_csv_file(sample_csv_data):
    """
    Create a temporary CSV file with sample data.
    """
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".csv")
    
    # Close the file descriptor
    os.close(fd)
    
    # Write the data to the file
    sample_csv_data.to_csv(path, index=False)
    
    yield path
    
    # Clean up
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture
def sample_json_file(sample_csv_data):
    """
    Create a temporary JSON file with sample data.
    """
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".json")
    
    # Close the file descriptor
    os.close(fd)
    
    # Write the data to the file
    sample_csv_data.to_json(path, orient="records")
    
    yield path
    
    # Clean up
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture
def sample_excel_file(sample_csv_data):
    """
    Create a temporary Excel file with sample data.
    """
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".xlsx")
    
    # Close the file descriptor
    os.close(fd)
    
    # Write the data to the file
    sample_csv_data.to_excel(path, index=False)
    
    yield path
    
    # Clean up
    if os.path.exists(path):
        os.remove(path)
