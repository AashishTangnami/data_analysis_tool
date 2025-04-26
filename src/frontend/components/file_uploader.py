import streamlit as st
import math

def format_bytes(size):
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

# def render_file_uploader():
#     """Render file uploader with size information"""
#     st.subheader("Upload Data File")
    
#     # Show max upload size
#     # max_size_mb = 500  # Match server configuration
#     # st.warning(f"Maximum file size less than : {max_size_mb} MB ")
    
#     uploaded_file = st.file_uploader(
#         "Choose a file",
#         type=["csv", "xlsx", "xls", "json", "parquet"],
#         help="Upload a CSV, Excel, JSON, or Parquet file to analyze"
#     )
    
#     if uploaded_file is not None:
#         file_size = len(uploaded_file.getvalue())
#         st.write(f"File size: {format_bytes(file_size)}")
        
        
#         using_pandas_engine = st.session_state.engine_type == "pandas"
    
#         # Show warning about large files only when using pandas engine
#         # and before a file is uploaded
#         if using_pandas_engine and not st.session_state.get("file_id"):
#             st.warning(
#                 "Large files may be slow to process with the Pandas engine. "
#                 "Consider using Polars engine for better performance with large datasets."
#             )
    
     
#     else:
#         st.info("""
#         Supported file formats:
#         - CSV (.csv): Comma-separated values files
#         - Excel (.xlsx, .xls): Microsoft Excel workbooks
#         - JSON (.json): JavaScript Object Notation files
#         - Parquet (.parquet): Apache Parquet files (recommended for large datasets)
#         """)
    
#     return uploaded_file

def render_file_uploader():
    """Render file uploader with size information and restrictions"""
    st.subheader("Upload Data File")
    
    # Define max size based on engine
    max_size_mb = 500  # Default
    if st.session_state.engine_type == "pandas":
        max_size_mb = 200  # Lower for pandas
    elif st.session_state.engine_type == "polars":
        max_size_mb = 800  # Higher for polars
    
    st.info(f"Maximum recommended file size: {max_size_mb} MB")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help="Upload a CSV, Excel, JSON, or Parquet file to analyze"
    )
    
    if uploaded_file is not None:
        file_size = len(uploaded_file.getvalue())
        file_size_mb = file_size / (1024 * 1024)
        
        st.write(f"File size: {format_bytes(file_size)}")
        
        # Warn if file is too large for selected engine
        if file_size_mb > max_size_mb:
            st.warning(
                f"This file ({format_bytes(file_size)}) exceeds the recommended size "
                f"for the {st.session_state.engine_type} engine. Processing may be slow. "
                f"Consider switching to a more efficient engine or using a smaller dataset."
            )
            
            # Suggest alternative engine
            if st.session_state.engine_type == "pandas":
                if st.button("Switch to Polars Engine"):
                    st.session_state.frontend_context.set_engine("polars")
                    st.rerun()
    
    return uploaded_file