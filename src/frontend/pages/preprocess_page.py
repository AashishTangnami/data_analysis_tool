import streamlit as st
import requests
from typing import Dict, Any

def preprocess_page():
    st.title("Data Preprocessing")
    
    if "uploaded_file_info" not in st.session_state:
        st.warning("Please upload a file first")
        return
    
    file_info = st.session_state.uploaded_file_info
    
    if "original_data" not in st.session_state:
        st.warning("Please upload a file first.")
        return
    
    # Display the original data preview
    st.subheader("Original Data Preview")
    st.dataframe(st.session_state["original_data"].head(10))  # Show the first 10 rows
    
    # Preprocessing Options
    with st.sidebar:
        st.header("Preprocessing Options")
        
        # Missing Values
        st.subheader("Handle Missing Values")
        handle_missing = st.checkbox("Handle missing values", value=True)
        if handle_missing:
            missing_strategy = st.selectbox(
                "Strategy",
                ["drop", "mean", "median", "mode", "constant"],
                help="How to handle missing values"
            )
            if missing_strategy == "constant":
                fill_value = st.text_input("Fill value", "0")
        
        # Duplicates
        st.subheader("Handle Duplicates")
        remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
        if remove_duplicates:
            subset_cols = st.multiselect(
                "Consider columns",
                file_info.get("columns", []),
                help="Select columns to consider for duplicate detection"
            )
        
        # Outliers
        st.subheader("Handle Outliers")
        handle_outliers = st.checkbox("Handle outliers")
        if handle_outliers:
            outlier_method = st.selectbox(
                "Method",
                ["IQR", "Z-Score", "Isolation Forest"]
            )
        
        # Feature Scaling
        st.subheader("Feature Scaling")
        scale_features = st.checkbox("Scale numerical features")
        if scale_features:
            scaler_method = st.selectbox(
                "Scaling Method",
                ["StandardScaler", "MinMaxScaler", "RobustScaler"]
            )
    
    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Apply Preprocessing", type="primary"):
            with st.spinner("Preprocessing data..."):
                try:
                    # Get selected engine from session state (set during file upload)
                    selected_engine = st.session_state.get('selected_engine', 'pandas')
                    
                    # Prepare preprocessing parameters
                    params = {
                        "file_id": file_info["file_id"],
                        "handle_missing": handle_missing,
                        "missing_strategy": missing_strategy if handle_missing else None,
                        "remove_duplicates": remove_duplicates,
                        "duplicate_subset": subset_cols if remove_duplicates else None,
                        "handle_outliers": handle_outliers,
                        "outlier_method": outlier_method if handle_outliers else None,
                        "scale_features": scale_features,
                        "scaler_method": scaler_method if scale_features else None
                    }
                    
                    # Call preprocessing API with engine selection in headers
                    headers = {
                        "X-Engine-Type": selected_engine
                    }
                    
                    response = requests.post(
                        "http://localhost:8000/api/v1/preprocessing",
                        json=params,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.preprocessed_data = result
                        st.success(f"Preprocessing completed successfully using {selected_engine} engine!")
                        display_preprocessing_summary(result)
                    else:
                        st.error(f"Preprocessing failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error during preprocessing: {str(e)}")

def display_preprocessing_summary(result: Dict[str, Any]):
    """Display summary of preprocessing operations."""
    st.subheader("Preprocessing Summary")
    
    # Display changes in data shape
    st.write("Data Shape Changes:")
    cols = st.columns(2)
    
    # Get statistics from the result
    stats = result.get("statistics", {})
    
    with cols[0]:
        st.metric("Original Rows", stats.get("original_rows", 0))
    with cols[1]:
        st.metric("Processed Rows", stats.get("processed_rows", 0))
    
    # Display column information
    st.write("Column Information:")
    st.write(f"Total Columns: {len(stats.get('columns', []))}")
    
    # Display missing values information
    with st.expander("Missing Values Summary"):
        missing_values = stats.get("missing_values", {})
        for col, count in missing_values.items():
            st.write(f"{col}: {count} missing values")
    
    # Display processed data path
    st.write("Processed Data Path:", result.get("processed_path", ""))
    st.write("Engine Used:", result.get("engine_used", ""))
    
    # Display operations summary
    st.write("Operations Performed:")
    for operation, details in result["operations"].items():
        with st.expander(operation):
            st.write(details)
    
    # Display sample of processed data
    if "sample_data" in result:
        st.write("Sample of Processed Data:")
        st.dataframe(result["sample_data"])
    
    # Navigation buttons
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ Back to Upload"):
            st.session_state.current_step = "upload"
            st.rerun()
    
    with col3:
        if "preprocessed_data" in st.session_state:
            if st.button("Continue to Analysis ➡️", type="primary"):
                st.session_state.current_step = "analyze"
                st.rerun()
