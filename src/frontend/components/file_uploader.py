import streamlit as st
import math

def format_bytes(size):
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def render_file_uploader():
    """Render file uploader with size information"""
    st.subheader("Upload Data File")
    
    # Show max upload size
    # max_size_mb = 500  # Match server configuration
    # st.warning(f"Maximum file size less than : {max_size_mb} MB ")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help="Upload a CSV, Excel, JSON, or Parquet file to analyze"
    )
    
    if uploaded_file is not None:
        file_size = len(uploaded_file.getvalue())
        st.write(f"File size: {format_bytes(file_size)}")
        
        
        using_pandas_engine = st.session_state.engine_type == "pandas"
    
        # Show warning about large files only when using pandas engine
        # and before a file is uploaded
        if using_pandas_engine and not st.session_state.get("file_id"):
            st.warning(
                "Large files may be slow to process with the Pandas engine. "
                "Consider using Polars engine for better performance with large datasets."
            )
    
     
    else:
        st.info("""
        Supported file formats:
        - CSV (.csv): Comma-separated values files
        - Excel (.xlsx, .xls): Microsoft Excel workbooks
        - JSON (.json): JavaScript Object Notation files
        - Parquet (.parquet): Apache Parquet files (recommended for large datasets)
        """)
    
    return uploaded_file
