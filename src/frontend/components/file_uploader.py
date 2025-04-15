import streamlit as st
import requests


def render_upload_section():
    """Render the file upload section."""
    
    # Engine selection
    engine = st.selectbox(
        "Select Processing Engine",
        options=["pandas", "polars"],
        help="Choose the data processing engine"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "parquet", "json", "avro", "orc", "xml"],
        help="Upload your data file"
    )

    if uploaded_file is not None:
        try:
            # Create form data
            files = {"file": uploaded_file}
            
            # Upload to backend
            response = requests.post(
                "http://localhost:8000/api/v1/ingestion/upload",
                files=files,
                headers={"X-Engine-Type": engine}  # Pass engine type in header
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Verify we have a file_id in the response
                if "file_id" not in result:
                    st.error("Server response missing file_id")
                    return
                    
                st.success("File uploaded successfully!")
                
                # Store complete file info in session state
                st.session_state.uploaded_file_info = {
                    "file_id": result["file_id"],
                    "filename": uploaded_file.name,
                    "rows": result.get("rows", 0),
                    "columns": result.get("columns", []),
                    "engine": engine
                }

                # Also store the original data for preview
                try:
                    import pandas as pd
                    df = pd.read_csv(uploaded_file)  # Adjust based on file type
                    st.session_state["original_data"] = df
                except Exception as e:
                    st.error(f"Error loading preview data: {str(e)}")
                
                # Debug information
                st.write("Debug - File Info:")
                st.json(st.session_state.uploaded_file_info)
                
                st.session_state.current_step = "preprocess"
                
                # Display file info
                st.write("File Summary:")
                st.json({
                    "file_id": result["file_id"],  # Add file_id to summary
                    "rows": result["rows"],
                    "columns": result["columns"]
                })

                # Show continue button
                if st.button("Continue to Preprocessing", type="primary"):
                    st.rerun()
                
            else:
                st.error(f"Upload failed: {response.text}")
                
        except Exception as e:
            st.error(f"Error during upload: {str(e)}")
