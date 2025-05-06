"""
Enhanced preprocessing page with advanced preprocessing techniques.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from src.shared.logging_config import get_context_logger
from src.frontend.context import FrontendContext
# Import workflow UI component when available
# from src.frontend.components.preprocessing.workflow_ui import render_workflow_ui

# Temporary placeholder for workflow UI
def render_workflow_ui():
    """Temporary placeholder for workflow UI component."""
    st.subheader("Preprocessing Workflow")

    if "preprocessing_operations" not in st.session_state or not st.session_state.preprocessing_operations:
        st.info("No preprocessing operations have been applied yet. Add operations from the tabs above.")
        return

    # Display operations in a more readable format
    st.write(f"**Applied Operations ({len(st.session_state.preprocessing_operations)}):**")

    for i, op in enumerate(st.session_state.preprocessing_operations):
        op_type = op["type"].replace("_", " ").title()
        params = op.get("params", {})

        with st.expander(f"{i+1}. {op_type}", expanded=i==0):
            st.write("**Parameters:**")
            for param_name, param_value in params.items():
                if isinstance(param_value, list):
                    if param_name == "columns" and "all" in param_value:
                        st.write(f"- **{param_name.replace('_', ' ').title()}**: All columns")
                    else:
                        st.write(f"- **{param_name.replace('_', ' ').title()}**: {', '.join(map(str, param_value))}")
                else:
                    if param_value is not None:
                        st.write(f"- **{param_name.replace('_', ' ').title()}**: {param_value}")

    # Add buttons for workflow management
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Apply All Operations", key="apply_all_workflow"):
            # Get the frontend context
            frontend_context = st.session_state.frontend_context

            # Apply all operations
            try:
                result = frontend_context.preprocess_data(st.session_state.preprocessing_operations)
                st.success("All operations applied successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error applying operations: {str(e)}")

    with col2:
        if st.button("Clear All Operations", key="clear_all_workflow"):
            st.session_state.preprocessing_operations = []
            st.success("All operations cleared")
            st.rerun()

# Get logger for this module
logger = get_context_logger(__name__)

def render_preprocessing_page():
    """Render the complete preprocessing page with advanced techniques."""
    st.title("Data Preprocessing")

    # Get file information
    engine_type = st.session_state.file_id.split("_")[0]
    file_name = st.session_state.file_id.split('_', 1)[1]

    # Set file context for logging
    logger.set_file_context(
        file_id=st.session_state.file_id,
        file_name=file_name
    )

    # Generate a unique key for this rendering to prevent duplicate logs
    page_render_key = f"preprocessing_page_render_{st.session_state.file_id}"

    # Only log the first time we render this page with this file
    if page_render_key not in st.session_state:
        logger.info(f"Rendering preprocessing page for file: {st.session_state.file_id}")
        st.session_state[page_render_key] = True

    # Check if preprocessed data is available
    has_preprocessed = "preprocessed_data_preview" in st.session_state

    # =====================================================================
    # DATA PREVIEW SECTION
    # =====================================================================
    st.header("Data Preview")

    # Show data preview
    if has_preprocessed:
        # Create tabs for original and processed data
        preview_tabs = st.tabs(["Original Data", "Processed Data"])

        with preview_tabs[0]:
            st.dataframe(pd.DataFrame(st.session_state.data_preview), height=300)
            with st.expander("Data Summary"):
                st.json(st.session_state.data_summary)

        with preview_tabs[1]:
            st.dataframe(pd.DataFrame(st.session_state.preprocessed_data_preview), height=300)
            with st.expander("Data Summary"):
                st.json(st.session_state.processed_summary)

            # Show impact metrics
            if "impact" in st.session_state:
                st.write("**Impact of Preprocessing:**")

                # Create columns for metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Rows",
                        value=st.session_state.impact.get("rows_after", 0),
                        delta=st.session_state.impact.get("rows_after", 0) - st.session_state.impact.get("rows_before", 0)
                    )

                with col2:
                    st.metric(
                        "Columns",
                        value=st.session_state.impact.get("columns_after", 0),
                        delta=st.session_state.impact.get("columns_after", 0) - st.session_state.impact.get("columns_before", 0)
                    )

                with col3:
                    st.metric(
                        "Missing Values",
                        value=st.session_state.impact.get("missing_values_after", 0),
                        delta=st.session_state.impact.get("missing_values_after", 0) - st.session_state.impact.get("missing_values_before", 0),
                        delta_color="inverse"  # Fewer missing values is better
                    )
    else:
        # Just show original data
        st.dataframe(pd.DataFrame(st.session_state.data_preview), height=300)
        with st.expander("Data Summary"):
            st.json(st.session_state.data_summary)

        st.info("Apply preprocessing operations below to see the impact on your data.")

    # =====================================================================
    # PREPROCESSING TECHNIQUES SECTION
    # =====================================================================
    st.header("Preprocessing Techniques")

    # Get the frontend context from session state
    if "frontend_context" not in st.session_state:
        st.session_state.frontend_context = FrontendContext()
    frontend_context = st.session_state.frontend_context

    # Create a visual workflow with tabs, arrows, and associations
    st.write("**Preprocessing Workflow:**")

    # Define tab names and their indices
    tab_names = [
        "Data Cleaning",
        "Data Transformation",
        "Feature Engineering",
        "Data Reduction",
        "Data Balancing",
        "Data Discretization",
        "Data Partitioning"
    ]

    # Define icons for each tab
    tab_icons = [
        "🧹", # Cleaning
        "🔄", # Transformation
        "⚙️", # Feature Engineering
        "📊", # Data Reduction
        "⚖️", # Data Balancing
        "📏", # Data Discretization
        "✂️"  # Data Partitioning
    ]

    # Define descriptions for each step
    tab_descriptions = [
        "Clean data by handling missing values, duplicates, outliers, and data types",
        "Transform data through encoding, scaling, normalization, and conversions",
        "Create new features through functions, polynomials, interactions, and time features",
        "Reduce dimensionality through feature selection, PCA, and correlation analysis",
        "Balance class distributions through oversampling, undersampling, and SMOTE",
        "Discretize continuous variables into categorical bins",
        "Split data into training, validation, and test sets"
    ]

    # Define dependencies between steps
    dependencies = {
        0: [],                  # Data Cleaning has no prerequisites
        1: [0],                 # Data Transformation depends on Data Cleaning
        2: [0, 1],              # Feature Engineering depends on Cleaning and Transformation
        3: [0, 1, 2],           # Data Reduction depends on previous steps
        4: [0, 1, 2, 3],        # Data Balancing depends on previous steps
        5: [0, 1, 2, 3],        # Data Discretization depends on previous steps (not on Balancing)
        6: [0, 1, 2, 3, 4, 5]   # Data Partitioning depends on all previous steps
    }

    # Track which tab is selected
    if "active_preprocessing_tab" not in st.session_state:
        st.session_state.active_preprocessing_tab = 0

    # Track completed steps
    if "completed_preprocessing_steps" not in st.session_state:
        st.session_state.completed_preprocessing_steps = set()

    # Create a progress indicator
    progress_value = (st.session_state.active_preprocessing_tab / (len(tab_names) - 1)) * 100
    st.progress(progress_value / 100)

    # Create a workflow diagram
    st.write("**Workflow Diagram:**")

    # Create a row of tabs with arrows in between
    cols = st.columns([1, 0.2, 1, 0.2, 1, 0.2, 1, 0.2, 1, 0.2, 1, 0.2, 1])

    # Create the tabs with arrows
    for i, (tab_name, icon) in enumerate(zip(tab_names, tab_icons)):
        # Check if this step is available (all dependencies completed)
        deps = dependencies[i]
        deps_completed = all(dep in st.session_state.completed_preprocessing_steps for dep in deps)
        is_available = i == 0 or deps_completed or i == st.session_state.active_preprocessing_tab

        # Tab column
        with cols[i*2]:
            # Create a button that looks like a tab
            is_active = st.session_state.active_preprocessing_tab == i

            # Determine button type based on state
            if is_active:
                button_type = "primary"
            elif i in st.session_state.completed_preprocessing_steps:
                button_type = "secondary"
            elif is_available:
                button_type = "secondary"
            else:
                button_type = "secondary"

            # Add a visual indicator for step status
            if i in st.session_state.completed_preprocessing_steps:
                status_icon = "✅ "
            elif i == st.session_state.active_preprocessing_tab:
                status_icon = "🔍 "
            elif not is_available:
                status_icon = "🔒 "
            else:
                status_icon = "⏳ "

            # Create the button with icon and status
            button_disabled = not is_available and i != st.session_state.active_preprocessing_tab

            # Create a container for the button and tooltip
            btn_container = st.container()

            # Show tooltip with dependencies
            if deps and not is_available:
                prereq_names = [tab_names[dep] for dep in deps]
                tooltip = f"Prerequisites: {', '.join(prereq_names)}"
                btn_container.caption(f"🔒 {tooltip}")

            # Create the button
            if st.button(f"{status_icon}{icon} {tab_name}",
                        key=f"tab_{i}",
                        type=button_type,
                        disabled=button_disabled,
                        use_container_width=True):
                st.session_state.active_preprocessing_tab = i
                st.rerun()

            # Show description below the button
            st.caption(tab_descriptions[i])

        # Arrow column (except after the last tab)
        if i < len(tab_names) - 1:
            with cols[i*2 + 1]:
                # Determine arrow style based on completion status
                if i in st.session_state.completed_preprocessing_steps:
                    arrow_style = "color: #0068C9; font-size: 24px;" # Blue for completed
                elif i == st.session_state.active_preprocessing_tab:
                    arrow_style = "color: #FF9900; font-size: 24px;" # Orange for active
                else:
                    arrow_style = "color: #CCCCCC; font-size: 24px;" # Gray for inactive

                # Show different arrow types based on dependency relationships
                if i+1 in dependencies and i in dependencies[i+1]:
                    # Direct dependency - solid arrow
                    arrow = "➡️"
                else:
                    # Indirect dependency - dashed arrow
                    arrow = "⇢"

                st.markdown(f'<div style="display: flex; justify-content: center; align-items: center; height: 100%;"><span style="{arrow_style}">{arrow}</span></div>', unsafe_allow_html=True)

    # Add a "Mark as Complete" button for the current tab
    st.write("")
    col1, col2 = st.columns([3, 1])
    with col2:
        current_tab = st.session_state.active_preprocessing_tab
        if current_tab not in st.session_state.completed_preprocessing_steps:
            if st.button("✅ Mark Step as Complete", type="primary"):
                st.session_state.completed_preprocessing_steps.add(current_tab)
                # Move to next tab if available
                if current_tab < len(tab_names) - 1:
                    st.session_state.active_preprocessing_tab = current_tab + 1
                st.rerun()
        else:
            if st.button("⛔ Mark as Incomplete"):
                st.session_state.completed_preprocessing_steps.remove(current_tab)
                st.rerun()

    # Create the actual tabs (hidden from the user)
    preprocessing_tabs = st.tabs(tab_names)

    # Show only the active tab content
    active_tab = st.session_state.active_preprocessing_tab

    # Tab 1: Data Cleaning
    with preprocessing_tabs[0]:
        if active_tab == 0:
            render_data_cleaning_tab(frontend_context)

    # Tab 2: Data Transformation
    with preprocessing_tabs[1]:
        if active_tab == 1:
            render_data_transformation_tab(frontend_context)

    # Tab 3: Feature Engineering
    with preprocessing_tabs[2]:
        if active_tab == 2:
            render_feature_engineering_tab(frontend_context)

    # Tab 4: Data Reduction
    with preprocessing_tabs[3]:
        if active_tab == 3:
            render_data_reduction_tab(frontend_context)

    # Tab 5: Data Balancing
    with preprocessing_tabs[4]:
        if active_tab == 4:
            render_data_balancing_tab(frontend_context)

    # Tab 6: Data Discretization
    with preprocessing_tabs[5]:
        if active_tab == 5:
            st.write("**Data Discretization**")
            st.info("This feature will be implemented in a future update.")

    # Tab 7: Data Partitioning
    with preprocessing_tabs[6]:
        if active_tab == 6:
            st.write("**Data Partitioning**")
            st.info("This feature will be implemented in a future update.")

    # =====================================================================
    # WORKFLOW VISUALIZATION SECTION
    # =====================================================================
    st.header("Applied Operations")

    # Track operations in session state
    if "preprocessing_operations" not in st.session_state:
        st.session_state.preprocessing_operations = []

    # Show the operations that have been applied
    if not st.session_state.preprocessing_operations:
        st.info("No preprocessing operations have been applied yet. Use the tabs above to add operations.")
    else:
        # Create a flowchart of operations
        st.write("**Operation Flow:**")

        # Create a visual representation of the operations
        operations = st.session_state.preprocessing_operations

        # Group operations by category
        operation_categories = {
            "data_cleaning": [],
            "data_transformation": [],
            "feature_engineering": [],
            "data_reduction": [],
            "data_balancing": [],
            "data_discretization": [],
            "data_partitioning": []
        }

        # Map operation types to categories
        operation_type_to_category = {
            "fill_missing": "data_cleaning",
            "drop_missing": "data_cleaning",
            "drop_duplicates": "data_cleaning",
            "drop_columns": "data_cleaning",
            "convert_dtype": "data_cleaning",
            "remove_outliers": "data_cleaning",

            "encode_categorical": "data_transformation",
            "scale_numeric": "data_transformation",
            "normalize": "data_transformation",
            "convert_datetime": "data_transformation",
            "process_text": "data_transformation",

            "apply_function": "feature_engineering",
            "create_polynomial": "feature_engineering",
            "create_interaction": "feature_engineering",
            "bin_numeric": "feature_engineering",
            "extract_time_features": "feature_engineering",

            "select_features": "data_reduction",
            "pca_transform": "data_reduction",
            "select_by_importance": "data_reduction",
            "select_by_correlation": "data_reduction",

            "oversample": "data_balancing",
            "undersample": "data_balancing",
            "smote": "data_balancing",
            "calculate_class_weights": "data_balancing",

            "discretize": "data_discretization",

            "split_data": "data_partitioning"
        }

        # Categorize operations
        for op in operations:
            op_type = op["type"]
            category = operation_type_to_category.get(op_type, "other")
            operation_categories[category].append(op)

        # Create a visual flow of operations
        col1, col2 = st.columns([1, 3])

        with col1:
            st.write("**Original Data**")
            st.markdown("```\nRaw Data\n```")

        # Show operations by category
        for i, (category, ops) in enumerate(operation_categories.items()):
            if not ops:
                continue

            # Get category name for display
            category_name = category.replace("_", " ").title()
            category_index = list(tab_names).index(category_name) if category_name in tab_names else -1

            # Get category icon
            icon = tab_icons[category_index] if category_index >= 0 else "🔄"

            with col2:
                st.write(f"**{icon} {category_name} Operations:**")

                for j, op in enumerate(ops):
                    op_type = op["type"].replace("_", " ").title()
                    params = op.get("params", {})

                    # Create a more descriptive title for the operation
                    operation_title = f"{j+1}. {op_type}"

                    # Add specific details based on operation type
                    if "columns" in params:
                        columns = params["columns"]
                        if isinstance(columns, list):
                            if len(columns) <= 3:
                                operation_title += f": {', '.join(str(c) for c in columns)}"
                            else:
                                operation_title += f": {len(columns)} columns"

                    # Add method if available
                    if "method" in params:
                        operation_title += f" ({params['method']})"

                    st.markdown(f"- {operation_title}")

        with col1:
            st.write("**Processed Data**")
            st.markdown("```\nTransformed Data\n```")

        # Add buttons to manage operations
        st.write("")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Apply All Operations", key="apply_all_workflow"):
                # Get the frontend context
                frontend_context = st.session_state.frontend_context

                # Apply all operations
                try:
                    # Apply operations and store results in session state
                    result = frontend_context.preprocess_data(st.session_state.preprocessing_operations)

                    # Save results to session state for later use
                    if result:
                        st.session_state.preprocessed_data_preview = result.get("preview", [])
                        st.session_state.original_summary = result.get("original_summary", {})
                        st.session_state.processed_summary = result.get("processed_summary", {})
                        st.session_state.preprocessing_applied = True

                    st.success("All operations applied successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error applying operations: {str(e)}")

        with col2:
            if st.button("Clear All Operations", key="clear_all_workflow"):
                st.session_state.preprocessing_operations = []
                st.success("All operations cleared")
                st.rerun()

    # =====================================================================
    # PROCEED TO ANALYSIS SECTION
    # =====================================================================
    st.header("Proceed to Analysis")

    # Only show data choice if preprocessing has been applied
    if "preprocessed_data_preview" in st.session_state:
        st.write("Choose which data to use for analysis:")

        # Create columns for the data choice options
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Use Original Data", key="use_original", type="secondary"):
                st.session_state.use_preprocessed = False
                st.session_state.page = "Analysis"
                st.rerun()

        with col2:
            if st.button("Use Preprocessed Data", key="use_preprocessed", type="primary"):
                st.session_state.use_preprocessed = True
                st.session_state.page = "Analysis"
                st.rerun()

        # Add a note about the data choice
        st.info("💡 You can always switch between original and preprocessed data on the Analysis page.")
    else:
        # If no preprocessing has been applied, just show a single button
        if st.button("Proceed to Analysis", type="primary"):
            st.session_state.use_preprocessed = False
            st.session_state.page = "Analysis"
            st.rerun()

def render_data_cleaning_tab(frontend_context: FrontendContext):
    """Render the data cleaning tab with operations for cleaning data."""
    st.subheader("Data Cleaning")

    # Create subtabs for different cleaning operations
    cleaning_tabs = st.tabs([
        "Drop Columns",
        "Missing Values",
        "Duplicates",
        "Outliers",
    ])

    # Subtab 1: Missing Values
    with cleaning_tabs[0]:
        st.write("**Handle Missing Values**")

        missing_strategy = st.radio(
            "Missing Values Strategy",
            ["Fill Missing Values", "Drop Rows with Missing Values"]
        )

        if missing_strategy == "Fill Missing Values":
            # Fill missing values parameters
            columns = st.multiselect(
                "Select columns to fill",
                options=["all"] + st.session_state.data_summary.get("columns", []),
                default=["all"]
            )

            method = st.selectbox(
                "Fill method",
                options=["mean", "median", "mode", "constant", "forward_fill", "backward_fill"],
                help="""
                mean: Fill with column mean (numeric only)
                median: Fill with column median (numeric only)
                mode: Fill with most frequent value
                constant: Fill with a specified value
                forward_fill: Fill with the previous valid value
                backward_fill: Fill with the next valid value
                """
            )

            value = None
            if method == "constant":
                value = st.text_input("Constant value", "0")

            if st.button("Preview Fill Effect"):
                operation = {
                    "type": "fill_missing",
                    "params": {
                        "columns": columns,
                        "method": method,
                        "value": value
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply Fill Operation"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("Fill operation applied successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

        else:
            # Drop missing values parameters
            how = st.radio(
                "How to drop",
                ["any", "all"],
                help="'any': Drop rows where any value is missing. 'all': Drop rows where all values are missing."
            )

            threshold = st.slider(
                "Minimum missing values threshold (%)",
                min_value=0,
                max_value=100,
                value=50,
                help="Only drop rows with at least this percentage of missing values."
            )

            if st.button("Preview Drop Effect"):
                operation = {
                    "type": "drop_missing",
                    "params": {
                        "how": how,
                        "threshold": threshold / 100.0  # Convert to fraction
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply Drop Operation"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("Drop operation applied successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

    # Subtab 2: Duplicates

def display_side_by_side_preview(preview: Dict[str, Any]):
    """
    Display side-by-side comparison of original and processed data with impact metrics.

    Args:
        preview: Dictionary containing preview information
    """
    if not preview:
        st.info("No preview information available.")
        return

    # Calculate impact metrics
    impact = preview.get("impact", {})

    # Show impact metrics
    st.write("**Impact Metrics:**")

    # Create columns for metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            value=impact.get("rows_after", 0),
            delta=impact.get("rows_after", 0) - impact.get("rows_before", 0)
        )

    with col2:
        st.metric(
            "Columns",
            value=impact.get("columns_after", 0),
            delta=impact.get("columns_after", 0) - impact.get("columns_before", 0)
        )

    with col3:
        st.metric(
            "Missing Values",
            value=impact.get("missing_values_after", 0),
            delta=impact.get("missing_values_after", 0) - impact.get("missing_values_before", 0),
            delta_color="inverse"  # Fewer missing values is better
        )

    # Show column changes
    if "original_summary" in preview and "processed_summary" in preview:
        original_cols = set(preview["original_summary"].get("columns", []))
        processed_cols = set(preview["processed_summary"].get("columns", []))

        # Find added and removed columns
        added_cols = processed_cols - original_cols
        removed_cols = original_cols - processed_cols

        if added_cols or removed_cols:
            st.write("**Column Changes**")

            # Create columns for better visual separation
            col1, col2 = st.columns(2)

            with col1:
                if removed_cols:
                    st.warning(f"**Removed Columns ({len(removed_cols)}):**")
                    for col in sorted(removed_cols):
                        st.write(f"- {col}")

            with col2:
                if added_cols:
                    st.success(f"**Added Columns ({len(added_cols)}):**")
                    for col in sorted(added_cols):
                        st.write(f"- {col}")

    # Create columns for original and processed data
    col1, col2 = st.columns(2)

    # Get original and processed data
    original_data = pd.DataFrame(preview.get("original_preview", []))
    if original_data.empty and "original_summary" in preview:
        # Try to reconstruct from summary
        original_data = pd.DataFrame(preview.get("preview", []))

    processed_data = pd.DataFrame(preview.get("preview", []))

    with col1:
        st.write("**Original Data**")
        st.dataframe(original_data, height=200)

        # Show column count
        if "original_summary" in preview:
            col_count = len(preview["original_summary"].get("columns", []))
            st.caption(f"Columns: {col_count}")

    with col2:
        st.write("**Processed Data (Preview)**")
        st.dataframe(processed_data, height=200)

        # Show column count
        if "processed_summary" in preview:
            col_count = len(preview["processed_summary"].get("columns", []))
            st.caption(f"Columns: {col_count}")

def render_data_transformation_tab(frontend_context: FrontendContext):
    """Render the data transformation tab with operations for transforming data."""
    st.subheader("Data Transformation")

    # Create subtabs for different transformation operations
    transformation_tabs = st.tabs([
        "Encoding",
        "Scaling",
        "Normalization",
        "Datetime Conversion",
        "Text Processing"
    ])

    # Subtab 1: Encoding
    with transformation_tabs[0]:
        st.write("**Categorical Encoding**")

        # Get categorical columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        categorical_cols = [
            col for col, dtype in col_dtypes.items()
            if "object" in dtype or "category" in dtype
        ]

        if not categorical_cols:
            st.info("No categorical columns detected in the dataset.")
        else:
            method = st.selectbox(
                "Encoding method",
                options=["one_hot", "label", "ordinal", "count", "target", "leave_one_out", "catboost"],
                help="""
                one_hot: Convert to binary columns (one per category)
                label: Convert to integer labels
                ordinal: Convert to ordered integers
                count: Replace with frequency counts
                target: Replace with target mean (requires target column)
                leave_one_out: Target encoding excluding current row
                catboost: Robust target encoding with random permutation
                """
            )

            columns = st.multiselect(
                "Select categorical columns to encode",
                options=["all"] + categorical_cols,
                default=["all"]
            )

            # For target-based encoding methods
            target_column = None
            if method in ["target", "leave_one_out", "catboost"]:
                target_column = st.selectbox(
                    "Select target column for encoding",
                    options=st.session_state.data_summary.get("columns", [])
                )

            if st.button("Preview Encoding Effect"):
                operation = {
                    "type": "encode_categorical",
                    "params": {
                        "columns": columns,
                        "method": method,
                        "target_column": target_column
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply Encoding Operation"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("Encoding applied successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

    # Subtab 2: Scaling
    with transformation_tabs[1]:
        st.write("**Numeric Scaling**")

        # Get numeric columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        numeric_cols = [
            col for col, dtype in col_dtypes.items()
            if "int" in dtype or "float" in dtype
        ]

        if not numeric_cols:
            st.info("No numeric columns detected in the dataset.")
        else:
            method = st.selectbox(
                "Scaling method",
                options=["standard", "minmax", "robust", "maxabs"],
                help="""
                standard: Standardize to mean=0, std=1
                minmax: Scale to range [0, 1]
                robust: Scale using median and quartiles
                maxabs: Scale by maximum absolute value
                """
            )

            columns = st.multiselect(
                "Select numeric columns to scale",
                options=["all"] + numeric_cols,
                default=["all"]
            )

            if st.button("Preview Scaling Effect"):
                operation = {
                    "type": "scale_numeric",
                    "params": {
                        "columns": columns,
                        "method": method
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply Scaling Operation"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("Scaling applied successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

    # Subtab 3: Normalization
    with transformation_tabs[2]:
        st.write("**Data Normalization**")

        # Get numeric columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        numeric_cols = [
            col for col, dtype in col_dtypes.items()
            if "int" in dtype or "float" in dtype
        ]

        if not numeric_cols:
            st.info("No numeric columns detected in the dataset.")
        else:
            norm_type = st.selectbox(
                "Normalization type",
                options=["l1", "l2", "max"],
                help="""
                l1: Normalize by L1 norm (sum of absolute values)
                l2: Normalize by L2 norm (Euclidean norm)
                max: Normalize by maximum value
                """
            )

            columns = st.multiselect(
                "Select numeric columns to normalize",
                options=["all"] + numeric_cols,
                default=["all"]
            )

            if st.button("Preview Normalization Effect"):
                operation = {
                    "type": "normalize",
                    "params": {
                        "columns": columns,
                        "norm": norm_type
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply Normalization Operation"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("Normalization applied successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

    # Subtab 4: Datetime Conversion
    with transformation_tabs[3]:
        st.write("**Datetime Conversion**")

        # Get string columns that might be dates
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        string_cols = [
            col for col, dtype in col_dtypes.items()
            if "object" in dtype
        ]

        if not string_cols:
            st.info("No string columns detected for datetime conversion.")
        else:
            columns = st.multiselect(
                "Select columns to convert to datetime",
                options=string_cols
            )

            date_format = st.text_input(
                "Date format (leave empty for auto-detection)",
                value="",
                help="Example: '%Y-%m-%d' for YYYY-MM-DD format"
            )

            if st.button("Preview Datetime Conversion"):
                operation = {
                    "type": "convert_datetime",
                    "params": {
                        "columns": columns,
                        "format": date_format if date_format else None
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply Datetime Conversion"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("Datetime conversion applied successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

    # Subtab 5: Text Processing
    with transformation_tabs[4]:
        st.write("**Text Processing**")

        # Get string columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        string_cols = [
            col for col, dtype in col_dtypes.items()
            if "object" in dtype
        ]

        if not string_cols:
            st.info("No string columns detected for text processing.")
        else:
            columns = st.multiselect(
                "Select text columns to process",
                options=string_cols
            )

            operation_type = st.selectbox(
                "Text operation",
                options=["lowercase", "uppercase", "strip", "remove_punctuation", "remove_numbers", "remove_stopwords"]
            )

            if st.button("Preview Text Processing"):
                operation = {
                    "type": "process_text",
                    "params": {
                        "columns": columns,
                        "operation": operation_type
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply Text Processing"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("Text processing applied successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

def render_feature_engineering_tab(frontend_context: FrontendContext):
    """Render the feature engineering tab with operations for creating new features."""
    st.subheader("Feature Engineering")

    # Create subtabs for different feature engineering operations
    feature_tabs = st.tabs([
        "Function Application",
        "Polynomial Features",
        "Interaction Features",
        "Binning",
        "Time Features"
    ])

    # Subtab 1: Function Application
    with feature_tabs[0]:
        st.write("**Apply Functions to Create New Features**")

        # Get numeric columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        numeric_cols = [
            col for col, dtype in col_dtypes.items()
            if "int" in dtype or "float" in dtype
        ]

        if not numeric_cols:
            st.info("No numeric columns detected for feature engineering.")
        else:
            function = st.selectbox(
                "Function to apply",
                options=["log", "sqrt", "square", "cube", "absolute", "sin", "cos", "tan", "exp"],
                help="""
                log: Natural logarithm (log(x+1) for negative values)
                sqrt: Square root (of absolute value for negative inputs)
                square: Square (x^2)
                cube: Cube (x^3)
                absolute: Absolute value (|x|)
                sin/cos/tan: Trigonometric functions
                exp: Exponential (e^x)
                """
            )

            columns = st.multiselect(
                "Select columns to transform",
                options=numeric_cols
            )

            create_new = st.checkbox("Create new columns instead of replacing", value=True)

            if st.button("Preview Function Application"):
                operation = {
                    "type": "apply_function",
                    "params": {
                        "columns": columns,
                        "function": function,
                        "create_new": create_new
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply Function"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("Function applied successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

    # Subtab 2: Polynomial Features
    with feature_tabs[1]:
        st.write("**Create Polynomial Features**")

        # Get numeric columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        numeric_cols = [
            col for col, dtype in col_dtypes.items()
            if "int" in dtype or "float" in dtype
        ]

        if not numeric_cols:
            st.info("No numeric columns detected for polynomial features.")
        else:
            columns = st.multiselect(
                "Select columns for polynomial features",
                options=numeric_cols
            )

            degree = st.slider("Polynomial degree", min_value=2, max_value=5, value=2)

            include_bias = st.checkbox("Include bias term (constant feature)", value=False)

            if st.button("Preview Polynomial Features"):
                operation = {
                    "type": "create_polynomial",
                    "params": {
                        "columns": columns,
                        "degree": degree,
                        "include_bias": include_bias
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply Polynomial Features"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("Polynomial features created successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

    # Subtab 3: Interaction Features
    with feature_tabs[2]:
        st.write("**Create Interaction Features**")

        # Get numeric columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        numeric_cols = [
            col for col, dtype in col_dtypes.items()
            if "int" in dtype or "float" in dtype
        ]

        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns to create interaction features.")
        else:
            columns = st.multiselect(
                "Select columns for interactions",
                options=numeric_cols
            )

            interaction_type = st.selectbox(
                "Interaction type",
                options=["multiplication", "division", "addition", "subtraction"],
                help="""
                multiplication: a * b
                division: a / b (with safeguards for division by zero)
                addition: a + b
                subtraction: a - b
                """
            )

            if len(columns) < 2:
                st.warning("Select at least 2 columns to create interactions.")
            else:
                if st.button("Preview Interaction Features"):
                    operation = {
                        "type": "create_interaction",
                        "params": {
                            "columns": columns,
                            "interaction_type": interaction_type
                        }
                    }

                    try:
                        # Get preview of operation effect
                        preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                        # Show side-by-side preview
                        st.write("**Preview of effect:**")
                        display_side_by_side_preview(preview)

                        # Add button to apply
                        if st.button("Apply Interaction Features"):
                            # Apply the operation and ignore the result since we'll rerun the page
                            frontend_context.apply_single_operation(st.session_state.file_id, operation)
                            st.success("Interaction features created successfully!")
                            st.rerun()

                    except Exception as e:
                        st.error(f"Preview failed: {str(e)}")

    # Subtab 4: Binning
    with feature_tabs[3]:
        st.write("**Bin Numeric Features**")

        # Get numeric columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        numeric_cols = [
            col for col, dtype in col_dtypes.items()
            if "int" in dtype or "float" in dtype
        ]

        if not numeric_cols:
            st.info("No numeric columns detected for binning.")
        else:
            columns = st.multiselect(
                "Select columns to bin",
                options=numeric_cols
            )

            binning_method = st.selectbox(
                "Binning method",
                options=["equal_width", "equal_frequency", "kmeans", "custom"],
                help="""
                equal_width: Divide range into equal-width bins
                equal_frequency: Create bins with equal number of samples
                kmeans: Use k-means clustering to find bin edges
                custom: Specify custom bin edges
                """
            )

            n_bins = st.slider("Number of bins", min_value=2, max_value=20, value=5)

            labels = st.text_input(
                "Custom bin labels (comma-separated, leave empty for default)",
                value=""
            )

            if st.button("Preview Binning"):
                operation = {
                    "type": "bin_numeric",
                    "params": {
                        "columns": columns,
                        "method": binning_method,
                        "n_bins": n_bins,
                        "labels": labels.split(",") if labels else None
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply Binning"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("Binning applied successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

    # Subtab 5: Time Features
    with feature_tabs[4]:
        st.write("**Extract Time Features**")

        # Get datetime columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        datetime_cols = [
            col for col, dtype in col_dtypes.items()
            if "datetime" in dtype or "date" in dtype
        ]

        if not datetime_cols:
            st.info("No datetime columns detected. Convert string columns to datetime first.")
        else:
            columns = st.multiselect(
                "Select datetime columns",
                options=datetime_cols
            )

            features = st.multiselect(
                "Select time features to extract",
                options=["year", "month", "day", "hour", "minute", "second", "dayofweek", "quarter", "is_weekend", "is_month_end", "is_month_start"],
                default=["year", "month", "day"]
            )

            if st.button("Preview Time Features"):
                operation = {
                    "type": "extract_time_features",
                    "params": {
                        "columns": columns,
                        "features": features
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply Time Features"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("Time features extracted successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

def render_data_reduction_tab(frontend_context: FrontendContext):
    """Render the data reduction tab with operations for dimensionality reduction."""
    st.subheader("Data Reduction")

    # Create subtabs for different reduction operations
    reduction_tabs = st.tabs([
        "Feature Selection",
        "PCA",
        "Feature Importance",
        "Correlation Analysis"
    ])

    # Subtab 1: Feature Selection
    with reduction_tabs[0]:
        st.write("**Select Important Features**")

        # Get numeric columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        numeric_cols = [
            col for col, dtype in col_dtypes.items()
            if "int" in dtype or "float" in dtype
        ]

        if not numeric_cols:
            st.info("No numeric columns detected for feature selection.")
        else:
            columns = st.multiselect(
                "Select columns to consider",
                options=numeric_cols,
                default=numeric_cols
            )

            method = st.selectbox(
                "Selection method",
                options=["variance", "k_best", "percentile", "manual"],
                help="""
                variance: Remove features with low variance
                k_best: Select k best features based on statistical tests
                percentile: Select top percentile features
                manual: Manually select features to keep
                """
            )

            if method == "variance":
                threshold = st.slider("Variance threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
                params = {"threshold": threshold}
            elif method == "k_best":
                k = st.slider("Number of features to select", min_value=1, max_value=len(columns), value=min(5, len(columns)))
                params = {"k": k}
            elif method == "percentile":
                percentile = st.slider("Percentile of features to select", min_value=1, max_value=100, value=25)
                params = {"percentile": percentile}
            elif method == "manual":
                keep_columns = st.multiselect(
                    "Select columns to keep",
                    options=columns
                )
                params = {"keep_columns": keep_columns}

            if st.button("Preview Feature Selection"):
                operation = {
                    "type": "select_features",
                    "params": {
                        "columns": columns,
                        "method": method,
                        **params
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply Feature Selection"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("Feature selection applied successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

    # Subtab 2: PCA
    with reduction_tabs[1]:
        st.write("**Principal Component Analysis**")

        # Get numeric columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        numeric_cols = [
            col for col, dtype in col_dtypes.items()
            if "int" in dtype or "float" in dtype
        ]

        if not numeric_cols:
            st.info("No numeric columns detected for PCA.")
        else:
            columns = st.multiselect(
                "Select columns for PCA",
                options=numeric_cols,
                default=numeric_cols
            )

            n_components = st.slider(
                "Number of components",
                min_value=1,
                max_value=min(len(columns), 10),
                value=min(2, len(columns))
            )

            whiten = st.checkbox("Whiten (normalize components to unit variance)", value=False)

            if st.button("Preview PCA"):
                operation = {
                    "type": "pca_transform",
                    "params": {
                        "columns": columns,
                        "n_components": n_components,
                        "whiten": whiten
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply PCA"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("PCA applied successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

    # Subtab 3: Feature Importance
    with reduction_tabs[2]:
        st.write("**Feature Importance Analysis**")

        # Get numeric columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        numeric_cols = [
            col for col, dtype in col_dtypes.items()
            if "int" in dtype or "float" in dtype
        ]

        if not numeric_cols:
            st.info("No numeric columns detected for feature importance analysis.")
        else:
            columns = st.multiselect(
                "Select feature columns",
                options=numeric_cols,
                default=numeric_cols
            )

            target_column = st.selectbox(
                "Select target column",
                options=st.session_state.data_summary.get("columns", [])
            )

            method = st.selectbox(
                "Importance method",
                options=["random_forest", "mutual_info", "chi2", "f_value"],
                help="""
                random_forest: Use Random Forest feature importance
                mutual_info: Mutual information between features and target
                chi2: Chi-squared test (for classification)
                f_value: F-test (for regression)
                """
            )

            if st.button("Preview Feature Importance"):
                operation = {
                    "type": "select_by_importance",
                    "params": {
                        "columns": columns,
                        "target_column": target_column,
                        "method": method,
                        "top_k": len(columns)  # Just analyze all features
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Show feature importance chart
                    if "feature_importance" in preview:
                        st.write("**Feature Importance:**")
                        importance_df = pd.DataFrame(preview["feature_importance"])
                        st.bar_chart(importance_df.set_index("feature")["importance"])

                    # Add button to select top features
                    top_k = st.slider(
                        "Select top K features to keep",
                        min_value=1,
                        max_value=len(columns),
                        value=min(5, len(columns))
                    )

                    if st.button("Apply Feature Selection"):
                        # Create a new operation with top_k parameter
                        selection_operation = {
                            "type": "select_by_importance",
                            "params": {
                                "columns": columns,
                                "target_column": target_column,
                                "method": method,
                                "top_k": top_k
                            }
                        }

                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, selection_operation)
                        st.success(f"Selected top {top_k} features successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

    # Subtab 4: Correlation Analysis
    with reduction_tabs[3]:
        st.write("**Correlation Analysis**")

        # Get numeric columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        numeric_cols = [
            col for col, dtype in col_dtypes.items()
            if "int" in dtype or "float" in dtype
        ]

        if not numeric_cols:
            st.info("No numeric columns detected for correlation analysis.")
        else:
            columns = st.multiselect(
                "Select columns for correlation analysis",
                options=numeric_cols,
                default=numeric_cols
            )

            method = st.selectbox(
                "Correlation method",
                options=["pearson", "spearman", "kendall"],
                help="""
                pearson: Standard correlation coefficient
                spearman: Rank correlation
                kendall: Rank correlation based on concordant/discordant pairs
                """
            )

            if st.button("Analyze Correlations"):
                operation = {
                    "type": "select_by_correlation",
                    "params": {
                        "columns": columns,
                        "method": method,
                        "threshold": 0.0  # Just analyze all correlations
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show correlation matrix
                    if "correlation_matrix" in preview:
                        st.write("**Correlation Matrix:**")
                        corr_df = pd.DataFrame(preview["correlation_matrix"])
                        st.dataframe(corr_df, height=400)

                        # Show heatmap
                        st.write("**Correlation Heatmap:**")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr_df, annot=True, cmap="coolwarm", ax=ax)
                        st.pyplot(fig)

                    # Add button to remove highly correlated features
                    threshold = st.slider(
                        "Correlation threshold for removal",
                        min_value=0.5,
                        max_value=1.0,
                        value=0.9,
                        step=0.05,
                        help="Features with correlation above this threshold will be considered redundant"
                    )

                    if st.button("Remove Correlated Features"):
                        # Create a new operation with threshold parameter
                        selection_operation = {
                            "type": "select_by_correlation",
                            "params": {
                                "columns": columns,
                                "method": method,
                                "threshold": threshold
                            }
                        }

                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, selection_operation)
                        st.success(f"Removed highly correlated features (threshold: {threshold})!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

def render_data_balancing_tab(frontend_context: FrontendContext):
    """Render the data balancing tab with operations for handling imbalanced data."""
    st.subheader("Data Balancing")

    # Create subtabs for different balancing operations
    balancing_tabs = st.tabs([
        "Class Distribution",
        "Oversampling",
        "Undersampling",
        "SMOTE",
        "Class Weights"
    ])

    # Subtab 1: Class Distribution
    with balancing_tabs[0]:
        st.write("**Analyze Class Distribution**")

        # Get categorical columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        categorical_cols = [
            col for col, dtype in col_dtypes.items()
            if "object" in dtype or "category" in dtype or "int" in dtype
        ]

        if not categorical_cols:
            st.info("No categorical columns detected for class distribution analysis.")
        else:
            target_column = st.selectbox(
                "Select target column (class label)",
                options=categorical_cols
            )

            if st.button("Analyze Class Distribution"):
                try:
                    # Get the data
                    if "data_preview" in st.session_state:
                        df = pd.DataFrame(st.session_state.data_preview)

                        # Calculate class distribution
                        class_counts = df[target_column].value_counts()
                        class_percentages = df[target_column].value_counts(normalize=True) * 100

                        # Display results
                        st.write("**Class Distribution:**")

                        # Create a DataFrame for display
                        distribution_df = pd.DataFrame({
                            "Count": class_counts,
                            "Percentage (%)": class_percentages
                        })

                        st.dataframe(distribution_df)

                        # Create a bar chart
                        st.write("**Class Distribution Chart:**")
                        st.bar_chart(class_counts)

                        # Calculate imbalance metrics
                        max_class = class_counts.max()
                        min_class = class_counts.min()
                        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')

                        st.write("**Imbalance Metrics:**")
                        st.write(f"- Majority class count: {max_class}")
                        st.write(f"- Minority class count: {min_class}")
                        st.write(f"- Imbalance ratio: {imbalance_ratio:.2f}")

                        # Provide recommendations
                        st.write("**Recommendations:**")
                        if imbalance_ratio > 10:
                            st.warning("Severe class imbalance detected. Consider using SMOTE, class weights, or other balancing techniques.")
                        elif imbalance_ratio > 3:
                            st.info("Moderate class imbalance detected. Consider using class weights or sampling techniques.")
                        else:
                            st.success("Class distribution is relatively balanced.")

                except Exception as e:
                    st.error(f"Error analyzing class distribution: {str(e)}")

    # Subtab 2: Oversampling
    with balancing_tabs[1]:
        st.write("**Oversample Minority Classes**")

        # Get categorical columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        categorical_cols = [
            col for col, dtype in col_dtypes.items()
            if "object" in dtype or "category" in dtype or "int" in dtype
        ]

        if not categorical_cols:
            st.info("No categorical columns detected for oversampling.")
        else:
            target_column = st.selectbox(
                "Select target column (class label)",
                options=categorical_cols,
                key="oversample_target"
            )

            method = st.selectbox(
                "Oversampling method",
                options=["random", "minority", "not_minority", "all"],
                help="""
                random: Randomly oversample minority classes
                minority: Oversample only the minority class
                not_minority: Oversample all classes except the majority class
                all: Oversample all classes to match the majority class
                """
            )

            if st.button("Preview Oversampling"):
                operation = {
                    "type": "oversample",
                    "params": {
                        "target_column": target_column,
                        "method": method
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply Oversampling"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("Oversampling applied successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

    # Subtab 3: Undersampling
    with balancing_tabs[2]:
        st.write("**Undersample Majority Classes**")

        # Get categorical columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        categorical_cols = [
            col for col, dtype in col_dtypes.items()
            if "object" in dtype or "category" in dtype or "int" in dtype
        ]

        if not categorical_cols:
            st.info("No categorical columns detected for undersampling.")
        else:
            target_column = st.selectbox(
                "Select target column (class label)",
                options=categorical_cols,
                key="undersample_target"
            )

            method = st.selectbox(
                "Undersampling method",
                options=["random", "majority", "not_minority", "all"],
                help="""
                random: Randomly undersample majority classes
                majority: Undersample only the majority class
                not_minority: Undersample all classes except the minority class
                all: Undersample all classes to match the minority class
                """
            )

            if st.button("Preview Undersampling"):
                operation = {
                    "type": "undersample",
                    "params": {
                        "target_column": target_column,
                        "method": method
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply Undersampling"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("Undersampling applied successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

    # Subtab 4: SMOTE
    with balancing_tabs[3]:
        st.write("**Synthetic Minority Over-sampling Technique (SMOTE)**")

        # Get categorical columns for target
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        categorical_cols = [
            col for col, dtype in col_dtypes.items()
            if "object" in dtype or "category" in dtype or "int" in dtype
        ]

        # Get numeric columns for features
        numeric_cols = [
            col for col, dtype in col_dtypes.items()
            if "int" in dtype or "float" in dtype
        ]

        if not categorical_cols or len(numeric_cols) < 2:
            st.info("SMOTE requires at least 2 numeric feature columns and a categorical target column.")
        else:
            target_column = st.selectbox(
                "Select target column (class label)",
                options=categorical_cols,
                key="smote_target"
            )

            feature_columns = st.multiselect(
                "Select feature columns for SMOTE",
                options=numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )

            k_neighbors = st.slider(
                "Number of nearest neighbors",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of nearest neighbors to use for synthetic sample generation"
            )

            if st.button("Preview SMOTE"):
                operation = {
                    "type": "smote",
                    "params": {
                        "target_column": target_column,
                        "feature_columns": feature_columns,
                        "k_neighbors": k_neighbors
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

                    # Add button to apply
                    if st.button("Apply SMOTE"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("SMOTE applied successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

    # Subtab 5: Class Weights
    with balancing_tabs[4]:
        st.write("**Calculate Class Weights**")

        # Get categorical columns
        col_dtypes = st.session_state.data_summary.get("dtypes", {})
        categorical_cols = [
            col for col, dtype in col_dtypes.items()
            if "object" in dtype or "category" in dtype or "int" in dtype
        ]

        if not categorical_cols:
            st.info("No categorical columns detected for class weight calculation.")
        else:
            target_column = st.selectbox(
                "Select target column (class label)",
                options=categorical_cols,
                key="weights_target"
            )

            method = st.selectbox(
                "Weight calculation method",
                options=["balanced", "inverse", "sqrt_inverse"],
                help="""
                balanced: Weights inversely proportional to class frequencies
                inverse: Weights exactly inverse to class frequencies
                sqrt_inverse: Weights inverse to square root of class frequencies
                """
            )

            if st.button("Calculate Class Weights"):
                operation = {
                    "type": "calculate_class_weights",
                    "params": {
                        "target_column": target_column,
                        "method": method
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show class weights
                    if "class_weights" in preview:
                        st.write("**Class Weights:**")
                        weights_df = pd.DataFrame(preview["class_weights"].items(), columns=["Class", "Weight"])
                        st.dataframe(weights_df)

                        # Create a bar chart
                        st.write("**Class Weights Chart:**")
                        st.bar_chart(weights_df.set_index("Class"))

                        # Add to session state for later use
                        st.session_state.class_weights = preview["class_weights"]

                        st.success("Class weights calculated successfully!")

                        # Show usage instructions
                        st.write("**Usage Instructions:**")
                        st.info("""
                        These class weights can be used in machine learning models to handle class imbalance.
                        Copy these weights to use in your model training code.
                        """)

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")
