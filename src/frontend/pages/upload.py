import streamlit as st
from components.file_uploader import render_file_uploader
from components.data_preview import render_data_preview
from src.shared.logging_config import get_context_logger

# Get logger for this module
logger = get_context_logger(__name__)

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
            try:
                # Get the frontend context from session state
                frontend_context = st.session_state.frontend_context

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
                logger.exception(f"Error uploading file: {str(e)}")
                st.error(f"An error occurred: {str(e)}")