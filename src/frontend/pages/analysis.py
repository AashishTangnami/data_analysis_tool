import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from src.shared.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

from components.visualization import (
    render_distribution_plot,
    render_correlation_heatmap,
    render_scatter_plot,
    render_categorical_plot
)

# Import the analysis renderers
from components.analysis_renderers import (
    render_diagnostic_results,
    render_feature_importance,
    render_correlation_analysis,
    render_outlier_detection,
    create_tabs_for_results
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

def render_preprocessing_options():
    """Render options for data preprocessing."""
    st.subheader("Data Preprocessing Options")

    # Get columns from data summary
    columns = st.session_state.data_summary.get("columns", [])
    numeric_cols = [col for col, dtype in st.session_state.data_summary.get("dtypes", {}).items()
                   if "float" in dtype.lower() or "int" in dtype.lower()]
    categorical_cols = [col for col in columns if col not in numeric_cols]
    print(categorical_cols)

    # Create tabs for different preprocessing operations
    preprocessing_tabs = st.tabs([
        "Missing Values",
        "Scaling",
        "Encoding",
        "Feature Engineering",
        "Outlier Treatment"
    ])

    preprocessing_operations = []

    # Missing Values tab
    with preprocessing_tabs[0]:
        st.write("**Handle Missing Values**")

        # Choose strategy for missing values
        missing_strategy = st.radio(
            "Missing Values Strategy",
            options=["Fill Missing Values", "Drop Missing Values"]
        )

        if missing_strategy == "Fill Missing Values":
            # Fill missing values options
            fill_columns = st.multiselect(
                "Select columns to fill missing values",
                options=columns,
                default=[],
                help="Leave empty to apply to all columns"
            )

            fill_method = st.selectbox(
                "Fill method",
                options=["mean", "median", "mode", "constant", "knn", "forward_fill", "backward_fill"],
                help="Method to use for filling missing values"
            )

            fill_params = {"columns": fill_columns if fill_columns else "all", "method": fill_method}

            # Add constant value if selected
            if fill_method == "constant":
                fill_value = st.text_input("Fill value", "0")
                fill_params["value"] = fill_value

            # Add KNN params if selected
            if fill_method == "knn":
                knn_neighbors = st.slider("Number of neighbors (k)", min_value=1, max_value=10, value=5)
                knn_weights = st.selectbox("Weights", options=["uniform", "distance"])
                fill_params["knn_params"] = {"n_neighbors": knn_neighbors, "weights": knn_weights}

            # Add operation to list
            if st.checkbox("Include missing value handling"):
                preprocessing_operations.append({
                    "type": "handle_missing_values",
                    "params": {
                        "strategy": "fill",
                        "fill_params": fill_params
                    }
                })

        else:  # Drop Missing Values
            # Drop missing values options
            drop_axis = st.radio(
                "Drop axis",
                options=["Rows", "Columns"],
                format_func=lambda x: x
            )

            drop_how = st.selectbox(
                "How to drop",
                options=["any", "all"],
                help="'any': drop if any value is missing, 'all': drop if all values are missing"
            )

            # Optional minimum threshold for non-NA values
            use_thresh = st.checkbox("Use threshold for non-NA values")
            drop_thresh = None
            if use_thresh:
                drop_thresh = st.slider(
                    "Minimum non-NA values required",
                    min_value=1,
                    max_value=len(columns),
                    value=len(columns) // 2
                )

            # Add operation to list
            if st.checkbox("Include missing value handling"):
                preprocessing_operations.append({
                    "type": "handle_missing_values",
                    "params": {
                        "strategy": "drop",
                        "drop_params": {
                            "axis": 0 if drop_axis == "Rows" else 1,
                            "how": drop_how,
                            "thresh": drop_thresh
                        }
                    }
                })

    # Scaling tab
    with preprocessing_tabs[1]:
        st.write("**Scale Numeric Features**")

        scale_columns = st.multiselect(
            "Select columns to scale",
            options=numeric_cols,
            default=[]
        )

        scale_method = st.selectbox(
            "Scaling method",
            options=["standard", "minmax", "robust", "quantile"]
        )

        # Add specific parameters based on the method
        scale_params = {"columns": scale_columns, "method": scale_method}

        if scale_method == "standard":
            scale_params["with_mean"] = st.checkbox("Center data (subtract mean)", value=True)
            scale_params["with_std"] = st.checkbox("Scale to unit variance", value=True)
        elif scale_method == "minmax":
            min_val = st.number_input("Min value", value=0.0)
            max_val = st.number_input("Max value", value=1.0)
            scale_params["feature_range"] = (min_val, max_val)

        # Add operation to list
        if st.checkbox("Include scaling"):
            preprocessing_operations.append({
                "type": "scale_numeric",
                "params": scale_params
            })

    # Encoding tab
    with preprocessing_tabs[2]:
        st.write("**Encode Categorical Variables**")

        encode_columns = st.multiselect(
            "Select categorical columns to encode",
            options=categorical_cols,
            default=[]
        )

        encode_method = st.selectbox(
            "Encoding method",
            options=["one_hot", "label", "ordinal", "target"]
        )

        # Add specific parameters based on the method
        encode_params = {"columns": encode_columns, "method": encode_method}

        if encode_method == "one_hot":
            encode_params["drop_first"] = st.checkbox("Drop first category (dummy encoding)", value=False)
        elif encode_method == "target":
            encode_params["target_column"] = st.selectbox(
                "Target column for encoding",
                options=numeric_cols
            )

        # Add operation to list
        if st.checkbox("Include encoding"):
            preprocessing_operations.append({
                "type": "encode_categorical",
                "params": encode_params
            })

    # Feature Engineering tab
    with preprocessing_tabs[3]:
        st.write("**Feature Engineering**")

        # Feature creation
        st.subheader("Create New Features")

        # Dynamic feature creation
        feature_expressions = {}

        st.write("Define new features using expressions (e.g., `column_a + column_b` or `np.log(column_a)`)")

        # Allow adding multiple feature expressions
        num_features = st.number_input("Number of features to create", min_value=0, max_value=10, value=1)

        for i in range(int(num_features)):
            col1, col2 = st.columns(2)
            with col1:
                feature_name = st.text_input(f"Feature {i+1} name", f"new_feature_{i+1}")
            with col2:
                feature_expr = st.text_input(f"Feature {i+1} expression", "")

            if feature_name and feature_expr:
                feature_expressions[feature_name] = feature_expr

        # Add operation to list if there are expressions
        if feature_expressions and st.checkbox("Include feature creation"):
            preprocessing_operations.append({
                "type": "create_features",
                "params": {
                    "features": feature_expressions
                }
            })

        # Feature transformation
        st.subheader("Transform Existing Features")

        transform_columns = st.multiselect(
            "Select columns to transform",
            options=numeric_cols,
            default=[]
        )

        transform_function = st.selectbox(
            "Transformation function",
            options=["log", "log1p", "sqrt", "square", "exp", "reciprocal", "abs"]
        )

        # Add operation to list
        if transform_columns and st.checkbox("Include feature transformation"):
            preprocessing_operations.append({
                "type": "apply_function",
                "params": {
                    "columns": transform_columns,
                    "function": transform_function
                }
            })

    # Outlier Treatment tab
    with preprocessing_tabs[4]:
        st.write("**Outlier Treatment**")

        outlier_columns = st.multiselect(
            "Select columns for outlier treatment",
            options=numeric_cols,
            default=[]
        )

        outlier_method = st.selectbox(
            "Outlier detection method",
            options=["zscore", "iqr"]
        )

        outlier_threshold = st.slider(
            "Threshold" + (" (standard deviations)" if outlier_method == "zscore" else " (IQR multiplier)"),
            min_value=1.0,
            max_value=5.0 if outlier_method == "zscore" else 3.0,
            value=3.0 if outlier_method == "zscore" else 1.5,
            step=0.1
        )

        outlier_strategy = st.selectbox(
            "How to handle outliers",
            options=["clip", "remove", "winsorize"]
        )

        # Add operation to list
        if outlier_columns and st.checkbox("Include outlier treatment"):
            preprocessing_operations.append({
                "type": "handle_outliers",
                "params": {
                    "columns": outlier_columns,
                    "method": outlier_method,
                    "threshold": outlier_threshold,
                    "strategy": outlier_strategy
                }
            })

    # Submit preprocessing operations
    if preprocessing_operations:
        st.write("**Selected Preprocessing Operations:**")

        # Display selected operations in a readable format
        for i, op in enumerate(preprocessing_operations):
            op_type = op["type"].replace("_", " ").title()
            st.write(f"{i+1}. {op_type}")

        if st.button("Run Preprocessing"):
            run_preprocessing(preprocessing_operations)
    else:
        st.warning("No preprocessing operations selected.")

def run_preprocessing(operations):
    """
    Run the specified preprocessing operations.
    Uses the Strategy pattern through the FrontendContext.

    Args:
        operations: List of preprocessing operations to apply
    """
    try:
        # Get the frontend context from session state
        frontend_context = st.session_state.frontend_context

        # Use the frontend context to preprocess the data
        result = frontend_context.preprocess_data(operations)

        # Save preprocessing info to session state
        st.session_state.preprocessed = True
        st.session_state.use_preprocessed = True
        st.session_state.preprocessing_summary = result.get("summary", {})

        st.success("Preprocessing completed successfully!")

        # Show summary
        st.subheader("Preprocessing Summary")

        # Display operation results
        if "operations_summary" in result:
            for op_summary in result["operations_summary"]:
                op_type = op_summary.get("operation", "").replace("_", " ").title()
                status = op_summary.get("status", "")
                details = op_summary.get("details", "")

                if status == "success":
                    st.write(f"✅ {op_type}: {details}")
                else:
                    st.write(f"❌ {op_type}: {details}")

        # Display shape changes
        if "shape_before" in result and "shape_after" in result:
            shape_before = result["shape_before"]
            shape_after = result["shape_after"]

            st.write(f"Original data shape: {shape_before[0]} rows × {shape_before[1]} columns")
            st.write(f"Preprocessed data shape: {shape_after[0]} rows × {shape_after[1]} columns")

        # Offer to update data preview
        if st.button("Update Data Preview with Preprocessed Data"):
            fetch_data_preview(use_preprocessed=True)

    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def fetch_data_preview(use_preprocessed=False):
    """
    Fetch a preview of the data.
    Uses the Strategy pattern through the FrontendContext.

    Args:
        use_preprocessed: Whether to use preprocessed data
    """
    try:
        # Get the frontend context from session state
        frontend_context = st.session_state.frontend_context

        # Use the frontend context to get data preview
        result = frontend_context.get_data_preview(use_preprocessed)

        # Success message and refresh
        st.success(f"Data preview updated to {'preprocessed' if use_preprocessed else 'original'} data.")
        st.rerun()

    except Exception as e:
        logger.error(f"Error fetching data preview: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

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
    Uses the Strategy pattern through the FrontendContext.

    Args:
        analysis_type: Type of analysis to run
        params: Parameters for the analysis
    """
    try:
        # Get the frontend context from session state
        frontend_context = st.session_state.frontend_context

        # Determine whether to use preprocessed data
        use_preprocessed = "use_preprocessed" in st.session_state and st.session_state.use_preprocessed

        # Use the frontend context to run the analysis
        result = frontend_context.analyze_data(
            analysis_type=analysis_type,
            params=params,
            use_preprocessed=use_preprocessed
        )

        # Success message and refresh
        st.success(f"{analysis_type.capitalize()} analysis completed successfully!")
        st.rerun()

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def render_analysis_results():
    """Render analysis results and visualizations."""
    if "analysis_completed" in st.session_state and st.session_state.analysis_completed:
        st.subheader("Analysis Results")

        # Display different results based on analysis type
        if st.session_state.analysis_type == "descriptive":
            render_descriptive_results()
        elif st.session_state.analysis_type == "diagnostic":
            # Use the imported render_diagnostic_results function
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

    # Add toggle for preprocessed data if available
    if "preprocessed" in st.session_state and st.session_state.preprocessed:
        use_preprocessed = st.toggle(
            "Use preprocessed data",
            value=st.session_state.use_preprocessed,
            help="Toggle between original and preprocessed data"
        )

        if use_preprocessed != st.session_state.use_preprocessed:
            fetch_data_preview(use_preprocessed=use_preprocessed)

    # Main page tabs
    main_tabs = st.tabs(["Analysis", "Preprocessing", "Data Preview"])

    # Analysis tab
    with main_tabs[0]:
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

