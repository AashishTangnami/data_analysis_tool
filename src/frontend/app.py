import streamlit as st
import requests
import json
from components.engine_selector import render_engine_selector
from components.file_uploader import render_file_uploader
from components.data_preview import render_data_preview
from pages.preprocessing import render_preprocessing_page
from pages.analysis import render_analysis_page

# API endpoint base URL
API_BASE_URL = "http://localhost:8000/api"

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Dynamic Data Analysis Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Add a sidebar for navigation
    st.sidebar.title("Navigation")
    
    # Initialize session state for page if not present
    if "page" not in st.session_state:
        st.session_state.page = "Upload"
    
    # Page selection with radio buttons
    page = st.sidebar.radio(
        "Select a page",
        ["Upload", "Preprocessing", "Analysis"],
        index=0 if st.session_state.page == "Upload" else 
              1 if st.session_state.page == "Preprocessing" else 2
    )
    
    # Update current page in session state
    # frontend/app.py (continued)
    # Update current page in session state
    st.session_state.page = page
    
    # Initialize session state
    if "file_id" not in st.session_state:
        st.session_state.file_id = None
    if "engine_type" not in st.session_state:
        st.session_state.engine_type = "pandas"
    if "data_preview" not in st.session_state:
        st.session_state.data_preview = None
    if "data_summary" not in st.session_state:
        st.session_state.data_summary = None
    if "preprocessed_data" not in st.session_state:
        st.session_state.preprocessed_data = None
    
    # Add engine selector in the sidebar (always visible)
    st.sidebar.subheader("Engine Selection")
    current_engine = render_engine_selector()
    
    # Display status information in the sidebar
    st.sidebar.subheader("Status")
    if st.session_state.file_id:
        st.sidebar.success(f"File loaded: {st.session_state.file_id.split('_', 1)[1]}")
        st.sidebar.info(f"Engine: {st.session_state.engine_type}")
        
        if "preprocessing_applied" in st.session_state and st.session_state.preprocessing_applied:
            st.sidebar.success("✅ Preprocessing applied")
        
        if "analysis_completed" in st.session_state and st.session_state.analysis_completed:
            st.sidebar.success(f"✅ {st.session_state.analysis_type.capitalize()} analysis completed")
    else:
        st.sidebar.warning("No file loaded")
    
    # Render the selected page
    if page == "Upload":
        render_upload_page()
    elif page == "Preprocessing":
        if "file_id" not in st.session_state or st.session_state.file_id is None:
            st.warning("Please upload a file first.")
            render_upload_page()
        # if st.session_state.file_id is None:
        #     st.warning("Please upload a file first.")
        #     render_upload_page()
        else:
            render_preprocessing_page()
    elif page == "Analysis":
        if st.session_state.file_id is None:
            st.warning("Please upload a file first.")
            render_upload_page()
        else:
            render_analysis_page()

def render_upload_page():
    """Render the file upload page."""
    st.title("Data Upload")
    
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

if __name__ == "__main__":
    main()




