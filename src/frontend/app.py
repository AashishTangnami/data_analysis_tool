import streamlit as st
import requests
import json
from context import FrontendContext
from components.engine_selector import render_engine_selector
from components.file_uploader import render_file_uploader
from components.data_preview import render_data_preview
from pages.preprocessing import render_preprocessing_page
from pages.analysis import render_analysis_page
from src.shared.logging_config import get_context_logger

# Configure logging
logger = get_context_logger(__name__)

# Initialize the frontend context (Strategy pattern)
if "frontend_context" not in st.session_state:
    st.session_state.frontend_context = FrontendContext()

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

    # Get the frontend context from session state
    frontend_context = st.session_state.frontend_context

    # File upload component
    uploaded_file = render_file_uploader()

    # Process the uploaded file when the upload button is clicked
    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            try:
                # Use the frontend context to upload the file
                result = frontend_context.upload_file(uploaded_file)

                st.success("File uploaded successfully!")

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

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Log application startup
    logger.add_context(
        environment="development",
        component="frontend"
    ).info("Starting Dynamic Data Analysis Tool Frontend")
    logger.clear_context()

    # Run the main application
    main()




