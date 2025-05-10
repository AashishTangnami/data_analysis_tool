import streamlit as st
import requests
from components.file_uploader import render_file_uploader
from components.data_preview import render_data_preview

API_BASE_URL = "http://localhost:8000/api"

def render_upload_page():
    """Render the file upload page."""
    st.title("Data Upload")
    
    # Engine selector with current value from session state
    engine_options = ["pandas", "polars", "pyspark"]
    st.session_state.engine_type = st.selectbox(
        "Select Processing Engine",
        options=engine_options,
        index=engine_options.index(st.session_state.engine_type) if st.session_state.engine_type in engine_options else 0
    )

    # File upload component
    uploaded_file = render_file_uploader()
    
    # Process the uploaded file when the upload button is clicked
    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            # Create a multipart form request
            files = {"file": uploaded_file}
            data = {"engine_type": st.session_state.engine_type}
            
            try:
                response = requests.post(
                    f"{API_BASE_URL}/ingestion/upload",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Save results to session state
                    st.session_state.file_id = result["file_id"]
                    st.session_state.data_preview = result["preview"]
                    st.session_state.data_summary = result["summary"]
                    st.session_state.engine_type = data["engine_type"]
                    
                    st.success("File uploaded successfully!")
                    
                    # Reset any previous analysis or preprocessing
                    if "preprocessing_applied" in st.session_state:
                        del st.session_state.preprocessing_applied
                    if "analysis_completed" in st.session_state:
                        del st.session_state.analysis_completed
                    if "preprocessing_operations" in st.session_state:
                        del st.session_state.preprocessing_operations
                    
                    # Display data preview
                    render_data_preview(
                        result["preview"],
                        result["summary"]
                    )
                    
                    # Add buttons to navigate to preprocessing or analysis
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Proceed to Preprocessing"):
                            st.session_state.page = "Preprocessing"
                            st.rerun()
                    
                    with col2:
                        if st.button("Proceed to Analysis"):
                            st.session_state.page = "Analysis"
                            st.rerun()
                else:
                    st.error(f"Error: {response.json()['detail']}")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")