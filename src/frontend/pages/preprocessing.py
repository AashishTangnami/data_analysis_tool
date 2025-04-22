import streamlit as st
import pandas as pd
import requests
import json

def render_preprocessing_interface():
    """Render the preprocessing interface with available operations."""
    st.subheader("Preprocessing Operations")
    
    # Get available operations for the current engine
    try:
        
        response = requests.get(
            f"http://localhost:8000/api/preprocessing/operations/{st.session_state.engine_type}"
        )
        
        if response.status_code == 200:
            operations = response.json().get("operations", {})
            
            # Create an expandable section for each operation type
            for op_name, op_details in operations.items():
                with st.expander(f"{op_name.replace('_', ' ').title()} - {op_details['description']}"):
                    # Display parameter inputs based on operation type
                    params = {}
                    
                    for param_name, param_desc in op_details.get("params", {}).items():
                        if "columns" in param_name:
                            # Multi-select for columns
                            all_columns = st.session_state.data_summary.get("columns", [])
                            params[param_name] = st.multiselect(
                                f"{param_name.replace('_', ' ').title()}: {param_desc}",
                                options=["all"] + all_columns,
                                default=["all"] if "all" in param_desc else []
                            )
                        
                        elif "method" in param_name:
                            # Select for method options
                            method_options = []
                            if "mean" in param_desc:
                                method_options.extend(["mean", "median", "mode", "constant"])
                            elif "one_hot" in param_desc:
                                method_options.extend(["one_hot", "label", "ordinal"])
                            elif "standard" in param_desc:
                                method_options.extend(["standard", "minmax", "robust"])
                            elif "log" in param_desc:
                                method_options.extend(["log", "sqrt", "square", "absolute"])
                            
                            params[param_name] = st.selectbox(
                                f"{param_name.replace('_', ' ').title()}: {param_desc}",
                                options=method_options
                            )
                        
                        elif "value" in param_name:
                            # Text input for value
                            params[param_name] = st.text_input(
                                f"{param_name.replace('_', ' ').title()}: {param_desc}",
                                value="0"
                            )
                        
                        elif "how" in param_name:
                            # Select for how options
                            params[param_name] = st.selectbox(
                                f"{param_name.replace('_', ' ').title()}: {param_desc}",
                                options=["any", "all"]
                            )
                        
                        else:
                            # Default text input
                            params[param_name] = st.text_input(
                                f"{param_name.replace('_', ' ').title()}: {param_desc}"
                            )
                    
                    # Add button to apply this operation
                    if st.button(f"Apply {op_name.replace('_', ' ').title()}", key=f"btn_{op_name}"):
                        # Create operation dictionary
                        operation = {
                            "type": op_name,
                            "params": params
                        }
                        
                        # Add to operations list in session state
                        if "preprocessing_operations" not in st.session_state:
                            st.session_state.preprocessing_operations = []
                        
                        st.session_state.preprocessing_operations.append(operation)
                        st.success(f"{op_name.replace('_', ' ').title()} operation added!")
    
        else:
            st.error(f"Error loading operations: {response.json().get('detail', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def render_operations_summary():
    """Render a summary of the operations to be applied."""
    if "preprocessing_operations" in st.session_state and st.session_state.preprocessing_operations:
        st.subheader("Operations to Apply")
        
        # Display operations in order
        for i, operation in enumerate(st.session_state.preprocessing_operations):
            op_type = operation["type"].replace("_", " ").title()
            params_str = ", ".join([f"{k}={v}" for k, v in operation["params"].items()])
            
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.write(f"{i+1}. **{op_type}** ({params_str})")
            
            with col2:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.preprocessing_operations.pop(i)
                    st.rerun()
        
        # Add apply all button
        if st.button("Apply All Operations", key="apply_all"):
            # Send preprocessing request to API
            try:
                response = requests.post(
                    "http://localhost:8000/api/preprocessing/process",
                    json={
                        "file_id": st.session_state.file_id,
                        "engine_type": st.session_state.engine_type,
                        "operations": st.session_state.preprocessing_operations
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Save results to session state
                    st.session_state.preprocessed_data_preview = result["preview"]
                    st.session_state.original_summary = result["original_summary"]
                    st.session_state.processed_summary = result["processed_summary"]
                    st.session_state.preprocessing_applied = True
                    
                    st.success("Preprocessing operations applied successfully!")
                    st.rerun()
                
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
        # Add clear all button
        if st.button("Clear All Operations", key="clear_all"):
            st.session_state.preprocessing_operations = []
            st.rerun()
    
    else:
        st.info("No preprocessing operations selected yet. Add operations from the list above.")

def render_preprocessing_results():
    """Render before and after preprocessing results."""
    if "preprocessing_applied" in st.session_state and st.session_state.preprocessing_applied:
        st.subheader("Preprocessing Results")
        
        # Create tabs for before/after
        tab1, tab2 = st.tabs(["Before Preprocessing", "After Preprocessing"])
        
        with tab1:
            # Display original data preview
            st.write("**Original Data Preview**")
            st.dataframe(pd.DataFrame(st.session_state.data_preview))
            
            # Display original summary
            with st.expander("Original Data Summary"):
                st.json(st.session_state.original_summary)
        
        with tab2:
            # Display processed data preview
            st.write("**Processed Data Preview**")
            st.dataframe(pd.DataFrame(st.session_state.preprocessed_data_preview))
            
            # Display processed summary
            with st.expander("Processed Data Summary"):
                st.json(st.session_state.processed_summary)
        
        # Add button to use preprocessed data for analysis
        if st.button("Proceed to Analysis with Preprocessed Data"):
            st.session_state.use_preprocessed = True
            st.session_state.page = "Analysis"
            st.rerun()

def render_preprocessing_page():
    """Render the complete preprocessing page."""
    st.title("Data Preprocessing")
    
    engine_type = st.session_state.file_id.split("_")[0]
    st.write(f"**Current Engine:** {engine_type}")

    # Display current data info
    st.write(f"**Current Engine:** {st.session_state.engine_type}")
    st.write(f"**File:** {st.session_state.file_id.split('_', 1)[1]}")
    
    # Create columns for original data preview
    st.subheader("Original Data Preview")
    st.dataframe(pd.DataFrame(st.session_state.data_preview))
    
    # Create two columns for operations and summary
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        render_preprocessing_interface()
    
    with col2:
        render_operations_summary()
    
    # Render preprocessing results if applied
    render_preprocessing_results()