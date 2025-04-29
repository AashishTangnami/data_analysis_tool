"""
Utility functions for Streamlit components and UI elements.
"""
import time
import random
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional

def generate_unique_key(prefix: str) -> str:
    """
    Generate a truly unique key for Streamlit elements to avoid duplicate ID errors.
    
    Args:
        prefix: Prefix for the key to make it more readable
        
    Returns:
        A unique string key
    """
    timestamp = int(time.time() * 1000)
    random_component = random.randint(0, 1000000)
    return f"{prefix}_{timestamp}_{random_component}"

def format_bytes(size: int) -> str:
    """
    Format bytes to human readable string.
    
    Args:
        size: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def render_data_preview(preview_data: List[Dict[str, Any]], summary_data: Dict[str, Any]) -> None:
    """
    Render a preview of the data and its summary.
    
    Args:
        preview_data: A list of dictionaries containing the data preview
        summary_data: A dictionary containing the data summary
    """
    if not preview_data or not summary_data:
        st.warning("No data available to preview.")
        return
        
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Data Preview", "Data Summary"])
    
    with tab1:
        # Convert preview data to DataFrame and display
        df_preview = pd.DataFrame(preview_data)
        st.dataframe(df_preview, use_container_width=True)
        st.caption(f"Showing first {len(preview_data)} rows of the data")
        
        # Display basic info
        if "shape" in summary_data:
            rows, cols = summary_data["shape"]
            st.info(f"Dataset has {rows:,} rows and {cols:,} columns")
    
    with tab2:
        # Display summary information
        if "dtypes" in summary_data:
            st.subheader("Column Data Types")
            dtypes_df = pd.DataFrame({
                "Column": list(summary_data["dtypes"].keys()),
                "Type": list(summary_data["dtypes"].values())
            })
            st.dataframe(dtypes_df, use_container_width=True)
        
        if "missing_values" in summary_data:
            st.subheader("Missing Values")
            missing_df = pd.DataFrame({
                "Column": list(summary_data["missing_values"].keys()),
                "Missing Count": list(summary_data["missing_values"].values())
            })
            missing_df["Missing %"] = missing_df["Missing Count"] / summary_data["shape"][0] * 100
            st.dataframe(missing_df, use_container_width=True)
        
        # Show full summary as JSON for advanced users
        with st.expander("Full Data Summary (JSON)"):
            st.json(summary_data)
