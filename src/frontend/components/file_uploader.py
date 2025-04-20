import streamlit as st

def render_file_uploader():
    """
    Render a component for uploading data files.
    
    Returns:
        UploadedFile: The uploaded file object or None
    """
    st.subheader("Upload Data File")
    
    # Create file uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "json"],
        help="Upload a CSV, Excel, or JSON file to analyze"
    )
    
    # Display upload instructions
    if uploaded_file is None:
        st.info("""
        Supported file formats:
        - CSV (.csv): Comma-separated values files
        - Excel (.xlsx, .xls): Microsoft Excel workbooks
        - JSON (.json): JavaScript Object Notation files
        """)
    
    return uploaded_file