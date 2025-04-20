import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from components.visualization import (
    render_distribution_plot, 
    render_correlation_heatmap,
    render_scatter_plot,
    render_categorical_plot
)

def render_descriptive_analysis_options():
    """Render options for descriptive analysis."""
    st.subheader("Descriptive Analysis Options")
    
    # Get columns from data summary
    columns = st.session_state.data_summary.get("columns", [])
    
    # Select columns to analyze
    selected_columns = st.multiselect(
        "Select columns to analyze",
        options=columns,
        default=columns[:5] if len(columns) > 5 else columns
    )
    
    # Analysis options
    include_numeric = st.checkbox("Include numeric analysis", value=True)
    include_categorical = st.checkbox("Include categorical analysis", value=True)
    include_correlations = st.checkbox("Include correlation analysis", value=True)
    
    # Create params dictionary
    params = {
        "columns": selected_columns,
        "include_numeric": include_numeric,
        "include_categorical": include_categorical,
        "include_correlations": include_correlations
    }
    
    # Add analyze button
    if st.button("Run Descriptive Analysis"):
        run_analysis("descriptive", params)

def render_diagnostic_analysis_options():
    """Render options for diagnostic analysis."""
    st.subheader("Diagnostic Analysis Options")
    
    # Get columns from data summary
    columns = st.session_state.data_summary.get("columns", [])
    numeric_cols = [col for col, dtype in st.session_state.data_summary.get("dtypes", {}).items() 
                   if "float" in dtype.lower() or "int" in dtype.lower()]
    
    # Select target column
    target_column = st.selectbox(
        "Select target column",
        options=columns
    )
    
    # First, create the options list
    available_options = [col for col in columns if col != target_column]

    # Then filter numeric_cols to only include available options
    available_numeric_cols = [col for col in numeric_cols if col in available_options]

    # Now use the filtered list for defaults
    feature_columns = st.multiselect(
        "Select feature columns",
        options=available_options,
        default=available_numeric_cols[:4] if len(available_numeric_cols) > 4 else available_numeric_cols
    )
    
    # Analysis options
    run_feature_importance = st.checkbox("Run feature importance analysis", value=True)
    run_outlier_detection = st.checkbox("Run outlier detection", value=True)
    
    # Create params dictionary
    params = {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "run_feature_importance": run_feature_importance,
        "run_outlier_detection": run_outlier_detection
    }
    
    # Add analyze button
    if st.button("Run Diagnostic Analysis"):
        run_analysis("diagnostic", params)

def render_predictive_analysis_options():
    """Render options for predictive analysis."""
    st.subheader("Predictive Analysis Options")
    
    # Get columns from data summary
    columns = st.session_state.data_summary.get("columns", [])
    numeric_cols = [col for col, dtype in st.session_state.data_summary.get("dtypes", {}).items() 
                   if "float" in dtype.lower() or "int" in dtype.lower()]
    
    # Select target column
    target_column = st.selectbox(
        "Select target column to predict",
        options=columns
    )
    
    # Determine if classification or regression
    problem_type = st.radio(
        "Problem type",
        options=["Regression", "Classification"]
    )
    
    # Select feature columns
    feature_columns = st.multiselect(
        "Select feature columns",
        options=[col for col in columns if col != target_column],
        default=numeric_cols[:4] if len(numeric_cols) > 4 else numeric_cols
    )
    
    # Model selection
    model_type = st.selectbox(
        "Select model type",
        options=["Random Forest", "Gradient Boosting", "Linear Model"]
    )
    
    # Train/test split
    test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=20, step=5) / 100
    
    # Create params dictionary
    params = {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "problem_type": problem_type.lower(),
        "model_type": model_type.lower().replace(" ", "_"),
        "test_size": test_size
    }
    
    # Add analyze button
    if st.button("Run Predictive Analysis"):
        run_analysis("predictive", params)

def render_prescriptive_analysis_options():
    """Render options for prescriptive analysis."""
    st.subheader("Prescriptive Analysis Options")
    
    # Get columns from data summary
    columns = st.session_state.data_summary.get("columns", [])
    numeric_cols = [col for col, dtype in st.session_state.data_summary.get("dtypes", {}).items() 
                   if "float" in dtype.lower() or "int" in dtype.lower()]
    
    # Select objective column
    objective_column = st.selectbox(
        "Select objective column to optimize",
        options=numeric_cols
    )
    
    # Objective type
    objective_type = st.radio(
        "Optimization objective",
        options=["Maximize", "Minimize"]
    )
    
    # Select decision variables
    decision_vars = st.multiselect(
        "Select decision variables",
        options=[col for col in numeric_cols if col != objective_column],
        default=numeric_cols[:3] if len(numeric_cols) > 3 else numeric_cols
    )
    
    # Constraints
    st.subheader("Constraints")
    constraints = []
    
    for var in decision_vars:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_val = st.number_input(f"Min {var}", value=0.0, step=0.1)
        
        with col2:
            max_val = st.number_input(f"Max {var}", value=10.0, step=0.1)
        
        with col3:
            weight = st.number_input(f"Weight for {var}", value=1.0, step=0.1)
        
        constraints.append({
            "variable": var,
            "min": min_val,
            "max": max_val,
            "weight": weight
        })
    
    # Create params dictionary
    params = {
        "objective_column": objective_column,
        "objective_type": objective_type.lower(),
        "decision_variables": decision_vars,
        "constraints": constraints
    }
    
    # Add analyze button
    if st.button("Run Prescriptive Analysis"):
        run_analysis("prescriptive", params)

def run_analysis(analysis_type, params):
    """
    Run the specified analysis type with the given parameters.
    
    Args:
        analysis_type: Type of analysis to run
        params: Parameters for the analysis
    """
    try:
        # Determine whether to use preprocessed data
        use_preprocessed = "use_preprocessed" in st.session_state and st.session_state.use_preprocessed
        
        # Make API request
        response = requests.post(
            "http://localhost:8000/api/analysis/analyze",
            json={
                "file_id": st.session_state.file_id,
                "analysis_type": analysis_type,
                "params": params,
                "use_preprocessed": use_preprocessed
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Save results to session state
            st.session_state.analysis_results = result["results"]
            st.session_state.analysis_visualizations = result["visualizations"]
            st.session_state.analysis_type = analysis_type
            st.session_state.analysis_completed = True
            
            st.success(f"{analysis_type.capitalize()} analysis completed successfully!")
            st.rerun()
        
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def render_analysis_results():
    """Render analysis results and visualizations."""
    if "analysis_completed" in st.session_state and st.session_state.analysis_completed:
        st.subheader("Analysis Results")
        
        # Display different results based on analysis type
        if st.session_state.analysis_type == "descriptive":
            render_descriptive_results()
        elif st.session_state.analysis_type == "diagnostic":
            render_diagnostic_results()
        elif st.session_state.analysis_type == "predictive":
            render_predictive_results()
        elif st.session_state.analysis_type == "prescriptive":
            render_prescriptive_results()

def render_descriptive_results():
    """Render descriptive analysis results."""
    results = st.session_state.analysis_results
    
    # Create tabs for different result sections
    tabs = []
    tab_titles = []
    
    if "numeric_analysis" in results and results["numeric_analysis"]:
        tabs.append("numeric")
        tab_titles.append("Numeric Analysis")
    
    if "categorical_analysis" in results and results["categorical_analysis"]:
        tabs.append("categorical")
        tab_titles.append("Categorical Analysis")
    
    if "correlations" in results and results["correlations"]:
        tabs.append("correlations")
        tab_titles.append("Correlations")
    
    # Create the tabs
    if tabs:
        selected_tabs = st.tabs(tab_titles)
        
        for i, tab_type in enumerate(tabs):
            with selected_tabs[i]:
                if tab_type == "numeric":
                    # Display numeric statistics
                    if "statistics" in results["numeric_analysis"]:
                        st.subheader("Descriptive Statistics")
                        
                        # Convert to DataFrame for display
                        stats_dict = results["numeric_analysis"]["statistics"]
                        stats_df = pd.DataFrame(stats_dict)
                        
                        # Display transpose for better readability
                        st.dataframe(stats_df.T)
                    
                    # Display distributions
                    st.subheader("Distributions")
                    
                    # Create a multi-select for columns to visualize
                    if "statistics" in results["numeric_analysis"]:
                        numeric_cols = list(results["numeric_analysis"]["statistics"].keys())
                        selected_cols = st.multiselect(
                            "Select columns to visualize",
                            options=numeric_cols,
                            default=numeric_cols[:3] if len(numeric_cols) > 3 else numeric_cols
                        )
                        
                        # Create a DataFrame from preview data for visualization
                        if "data_preview" in st.session_state:
                            df_preview = pd.DataFrame(st.session_state.data_preview)
                            
                            for col in selected_cols:
                                if col in df_preview.columns:
                                    st.write(f"**Distribution of {col}**")
                                    try:
                                        render_distribution_plot(df_preview, col)
                                    except Exception as e:
                                        st.error(f"Could not render plot for {col}: {str(e)}")
                
                elif tab_type == "categorical":
                    # Display categorical analysis
                    if "value_counts" in results["categorical_analysis"]:
                        st.subheader("Value Counts")
                        
                        # Select a column to display
                        categorical_cols = list(results["categorical_analysis"]["value_counts"].keys())
                        
                        if categorical_cols:
                            selected_col = st.selectbox(
                                "Select column to visualize",
                                options=categorical_cols
                            )
                            
                            # Display value counts
                            counts = results["categorical_analysis"]["value_counts"][selected_col]
                            counts_df = pd.DataFrame(
                                {"Count": list(counts.values())},
                                index=list(counts.keys())
                            )
                            
                            st.dataframe(counts_df)
                            
                            # Create bar chart
                            st.subheader(f"Distribution of {selected_col}")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            counts_df.plot.bar(ax=ax)
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Display unique counts
                    if "unique_counts" in results["categorical_analysis"]:
                        st.subheader("Unique Value Counts")
                        
                        unique_counts = results["categorical_analysis"]["unique_counts"]
                        unique_df = pd.DataFrame(
                            {"Unique Values": list(unique_counts.values())},
                            index=list(unique_counts.keys())
                        )
                        
                        st.dataframe(unique_df)
                
                elif tab_type == "correlations":
                    # Display correlation matrix
                    st.subheader("Correlation Matrix")
                    
                    # Create DataFrame from correlation dictionary
                    corr_dict = results["correlations"]
                    corr_df = pd.DataFrame(corr_dict)
                    
                    # Display the correlation matrix
                    st.dataframe(corr_df)
                    
                    # Create heatmap visualization
                    st.subheader("Correlation Heatmap")
                    
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Add scatter plot for highest correlations
                    st.subheader("Scatter Plots for Highest Correlations")
                    
                    # Find highest correlations (excluding self-correlations)
                    high_corrs = []
                    for col1 in corr_df.columns:
                        for col2 in corr_df.columns:
                            if col1 != col2:
                                corr_val = abs(corr_df.loc[col1, col2])
                                high_corrs.append((col1, col2, corr_val))
                    
                    # Sort by correlation value
                    high_corrs.sort(key=lambda x: x[2], reverse=True)
                    
                    # Display top correlations
                    if high_corrs:
                        top_n = min(3, len(high_corrs))
                        
                        for i in range(top_n):
                            col1, col2, corr_val = high_corrs[i]
                            st.write(f"**{col1} vs {col2}** (correlation: {corr_val:.2f})")
                            
                            # Create scatter plot
                            if "data_preview" in st.session_state:
                                df_preview = pd.DataFrame(st.session_state.data_preview)
                                try:
                                    render_scatter_plot(df_preview, col1, col2)
                                except Exception as e:
                                    st.error(f"Could not render scatter plot: {str(e)}")

def render_diagnostic_results():
    """Render diagnostic analysis results (placeholder)."""
    st.info("Diagnostic analysis results rendering will be implemented in the next phase.")
    
    # Display raw results in JSON format
    with st.expander("Raw Results"):
        st.json(st.session_state.analysis_results)

def render_predictive_results():
    """Render predictive analysis results (placeholder)."""
    st.info("Predictive analysis results rendering will be implemented in the next phase.")
    
    # Display raw results in JSON format
    with st.expander("Raw Results"):
        st.json(st.session_state.analysis_results)

def render_prescriptive_results():
    """Render prescriptive analysis results (placeholder)."""
    st.info("Prescriptive analysis results rendering will be implemented in the next phase.")
    
    # Display raw results in JSON format
    with st.expander("Raw Results"):
        st.json(st.session_state.analysis_results)

def render_analysis_page():
    """Render the complete analysis page."""
    st.title("Data Analysis")
    
    # Display current data info
    st.write(f"**Current Engine:** {st.session_state.engine_type}")
    st.write(f"**File:** {st.session_state.file_id.split('_', 1)[1]}")
    
    if "use_preprocessed" in st.session_state and st.session_state.use_preprocessed:
        st.write("**Using:** Preprocessed Data")
    else:
        st.write("**Using:** Original Data")
    
    # Analysis type selector
    analysis_type = st.selectbox(
        "Select Analysis Type",
        options=["Descriptive Analysis", "Diagnostic Analysis", "Predictive Analysis", "Prescriptive Analysis"]
    )
    
    # Display different analysis options based on type
    if analysis_type == "Descriptive Analysis":
        render_descriptive_analysis_options()
    
    elif analysis_type == "Diagnostic Analysis":
        render_diagnostic_analysis_options()
    
    elif analysis_type == "Predictive Analysis":
        render_predictive_analysis_options()
    
    elif analysis_type == "Prescriptive Analysis":
        render_prescriptive_analysis_options()
    
    # Render results if analysis is completed
    render_analysis_results()