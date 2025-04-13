"""
Streamlit application for the Dynamic Data Analysis Platform.
"""
import streamlit as st
import pandas as pd
import os
import requests
import json
from io import StringIO, BytesIO

# Set page configuration
st.set_page_config(
    page_title="Dynamic Data Analysis Platform",
    page_icon="📊",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"  # Update based on your API configuration

def main():
    """Main Streamlit application."""
    st.title("Dynamic Data Analysis Platform")
    st.write("Upload your data for comprehensive analysis")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Select a page",
        ["Upload Data", "Configure Analysis", "View Results"]
    )
    
    if page == "Upload Data":
        upload_page()
    elif page == "Configure Analysis":
        configure_analysis_page()
    elif page == "View Results":
        view_results_page()

def upload_page():
    """Page for uploading data files."""
    st.header("Upload Data")
    
    uploaded_file = st.file_uploader(
        "Choose a file (CSV, JSON, or Excel)",
        type=["csv", "json", "xlsx", "xls"]
    )
    
    if uploaded_file is not None:
        # Display file details
        st.write(f"Filename: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size} bytes")
        
        # Upload to API
        if st.button("Process File"):
            with st.spinner("Uploading file..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(f"{API_URL}/upload", files=files)
                
                if response.status_code == 200:
                    st.success("File uploaded successfully!")
                    st.session_state["uploaded_file_info"] = response.json()
                    st.session_state["current_page"] = "Configure Analysis"
                    st.rerun()  # Changed from st.experimental_rerun()
                else:
                    st.error(f"Error uploading file: {response.text}")

def configure_analysis_page():
    """Page for configuring analysis options."""
    st.header("Configure Analysis")
    
    if "uploaded_file_info" not in st.session_state:
        st.warning("Please upload a file first.")
        if st.button("Go to Upload Page"):
            st.session_state["current_page"] = "Upload Data"
            st.rerun()  # Changed from st.experimental_rerun()
        return
    
    # Show file information
    file_info = st.session_state["uploaded_file_info"]
    st.write(f"Analyzing file: {file_info['filename']}")
    
    # Analysis options
    st.subheader("Select Analysis Types")
    
    descriptive = st.checkbox("Descriptive Analysis", value=True)
    diagnostic = st.checkbox("Diagnostic Analysis", value=True)
    predictive = st.checkbox("Predictive Analysis")
    prescriptive = st.checkbox("Prescriptive Analysis")
    
    if not any([descriptive, diagnostic, predictive, prescriptive]):
        st.warning("Please select at least one analysis type.")
    
    # Advanced options (placeholder)
    with st.expander("Advanced Options"):
        st.write("Advanced configuration options will be available here.")
    
    # Submit for analysis
    if st.button("Run Analysis"):
        if any([descriptive, diagnostic, predictive, prescriptive]):
            st.session_state["analysis_config"] = {
                "descriptive": descriptive,
                "diagnostic": diagnostic,
                "predictive": predictive,
                "prescriptive": prescriptive,
                "file_info": file_info
            }
            st.success("Analysis started!")
            st.session_state["current_page"] = "View Results"
            st.rerun()  # Changed from st.experimental_rerun()
        else:
            st.error("Please select at least one analysis type.")

def view_results_page():
    """Page for viewing analysis results."""
    st.header("Analysis Results")
    
    if "analysis_config" not in st.session_state:
        st.warning("No analysis has been run yet.")
        if st.button("Go to Upload Page"):
            st.session_state["current_page"] = "Upload Data"
            st.rerun()  # Changed from st.experimental_rerun()
        return
    
    # Display analysis configuration
    config = st.session_state["analysis_config"]
    st.write(f"Analysis for file: {config['file_info']['filename']}")
    
    # Placeholder for results
    st.info("In a full implementation, analysis results would be displayed here.")
    
    # Tabs for different analysis types
    tabs = []
    if config["descriptive"]:
        tabs.append("Descriptive Analysis")
    if config["diagnostic"]:
        tabs.append("Diagnostic Analysis")
    if config["predictive"]:
        tabs.append("Predictive Analysis")
    if config["prescriptive"]:
        tabs.append("Prescriptive Analysis")
    
    current_tab = st.tabs(tabs)
    
    # Mock results for demonstration
    with st.container():
        st.subheader("Sample Visualization")
        st.write("Placeholder for analysis results and visualizations.")

if __name__ == "__main__":
    main()
