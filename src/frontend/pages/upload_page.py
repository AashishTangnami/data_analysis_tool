import streamlit as st
from src.frontend.components.file_uploader import render_upload_section
from src.frontend.components.sidebar import render_sidebar

def upload_page():
    """Page for uploading data files."""
    # Render sidebar with upload-specific options
    render_sidebar()

    st.title("Data Upload")
    
    # Description or instructions
    st.write("""
    Welcome to the data upload page. Here you can:
    - Upload your data files
    - Preview the data
    - Select processing engine
    """)

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_upload_section()  # This already handles file upload and data storage
    
    with col2:
        if "uploaded_file_info" in st.session_state and st.session_state.uploaded_file_info is not None:
            st.subheader("File Information")
            st.json(st.session_state.uploaded_file_info)
            
            # Navigation buttons
            st.divider()
            if st.button("Start Over"):
                # Clear session state and restart
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.session_state.current_step = "upload"
                st.rerun()
    # Sidebar information
    with st.sidebar:
        st.info("""
        Supported file formats:
        - CSV
        - Excel (xlsx, xls)
        - Parquet
        - JSON
        - Avro
        - ORC
        - XML
        """)
        
        if "uploaded_file_info" in st.session_state and st.session_state.uploaded_file_info is not None:
            info = st.session_state.uploaded_file_info
            if "filename" in info:
                st.success(f"Current file: {info['filename']}")
            if "engine" in info:
                st.success(f"Selected engine: {info['engine']}")

__all__ = ['upload_page']
