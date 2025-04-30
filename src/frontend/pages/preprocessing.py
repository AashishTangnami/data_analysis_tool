import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from src.shared.logging_config import get_context_logger
from src.frontend.context import FrontendContext

# Get logger for this module
logger = get_context_logger(__name__)

# Removed commented-out code for better maintainability
def render_preprocessing_interface():
    """Render the preprocessing interface with step-by-step operations."""
    st.subheader("Preprocessing Operations")

    # Get the frontend context from session state
    if "frontend_context" not in st.session_state:
        st.session_state.frontend_context = FrontendContext()
    frontend_context = st.session_state.frontend_context

    try:
        # Use the frontend context to get available operations
        # We don't use the operations directly, but this validates the API connection
        frontend_context.get_available_preprocessing_operations()

        # Create tabs for different operation categories
        preprocessing_tabs = st.tabs([
            "Missing Values",
            "Drop Columns",
            "Encoding",
            "Scaling",
            "Feature Engineering"
        ])

        # Tab 1: Missing Values
        with preprocessing_tabs[0]:
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
                    options=["mean", "median", "mode", "constant"]
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

                        # Show preview
                        st.write("**Preview of effect:**")
                        st.dataframe(pd.DataFrame(preview["preview"]))

                        # Show impact metrics
                        st.write("**Impact:**")
                        display_impact_metrics(preview.get('impact', {}))

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

                if st.button("Preview Drop Effect"):
                    operation = {
                        "type": "drop_missing",
                        "params": {
                            "how": how
                        }
                    }

                    try:
                        # Get preview of operation effect
                        preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                        # Show preview
                        st.write("**Preview of effect:**")
                        st.dataframe(pd.DataFrame(preview["preview"]))

                        # Show impact metrics
                        st.write("**Impact:**")
                        display_impact_metrics(preview.get('impact', {}))

                        # Add button to apply
                        if st.button("Apply Drop Operation"):
                            # Apply the operation and ignore the result since we'll rerun the page
                            frontend_context.apply_single_operation(st.session_state.file_id, operation)
                            st.success("Drop operation applied successfully!")
                            st.rerun()

                    except Exception as e:
                        st.error(f"Preview failed: {str(e)}")

        # Tab 2: Feature Selection
        with preprocessing_tabs[1]:
            st.write("**Feature Selection**")

            # Drop columns
            columns = st.multiselect(
                "Select columns to drop",
                options=st.session_state.data_summary.get("columns", [])
            )

            if st.button("Preview Drop Columns Effect"):
                operation = {
                    "type": "drop_columns",
                    "params": {
                        "columns": columns
                    }
                }

                try:
                    # Get preview of operation effect
                    preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                    # Show preview
                    st.write("**Preview of effect:**")
                    st.dataframe(pd.DataFrame(preview["preview"]))

                    # Show impact metrics
                    st.write("**Impact:**")
                    display_impact_metrics(preview.get('impact', {}))

                    # Add button to apply
                    if st.button("Apply Drop Columns Operation"):
                        # Apply the operation and ignore the result since we'll rerun the page
                        frontend_context.apply_single_operation(st.session_state.file_id, operation)
                        st.success("Columns dropped successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")

        # Tab 3: Encoding
        with preprocessing_tabs[2]:
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

                        # Show preview
                        st.write("**Preview of effect:**")
                        st.dataframe(pd.DataFrame(preview["preview"]))

                        # Show impact metrics
                        st.write("**Impact:**")
                        display_impact_metrics(preview.get('impact', {}))

                        # Add button to apply
                        if st.button("Apply Encoding Operation"):
                            # Apply the operation and ignore the result since we'll rerun the page
                            frontend_context.apply_single_operation(st.session_state.file_id, operation)
                            st.success("Encoding applied successfully!")
                            st.rerun()

                    except Exception as e:
                        st.error(f"Preview failed: {str(e)}")

        # Tab 4: Scaling
        with preprocessing_tabs[3]:
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
                    options=["standard", "minmax", "robust"],
                    help="""
                    standard: Standardize to mean=0, std=1
                    minmax: Scale to range [0, 1]
                    robust: Scale using median and quartiles
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

                        # Show preview
                        st.write("**Preview of effect:**")
                        st.dataframe(pd.DataFrame(preview["preview"]))

                        # Show impact metrics
                        st.write("**Impact:**")

                        # First show general impact metrics
                        display_impact_metrics(preview.get('impact', {}))

                        # Then show column-specific scaling impact
                        st.write("**Column-specific Scaling Impact:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Before Scaling:**")
                            for col in columns:
                                if col != "all":
                                    stats = preview["original_summary"].get("column_stats", {}).get(col, {})
                                    st.write(f"- {col}: mean={stats.get('mean', 'N/A')}, std={stats.get('std', 'N/A')}")

                        with col2:
                            st.write(f"**After Scaling:**")
                            for col in columns:
                                if col != "all":
                                    stats = preview["processed_summary"].get("column_stats", {}).get(col, {})
                                    st.write(f"- {col}: mean={stats.get('mean', 'N/A')}, std={stats.get('std', 'N/A')}")

                        # Add button to apply
                        if st.button("Apply Scaling Operation"):
                            # Apply the operation and ignore the result since we'll rerun the page
                            frontend_context.apply_single_operation(st.session_state.file_id, operation)
                            st.success("Scaling applied successfully!")
                            st.rerun()

                    except Exception as e:
                        st.error(f"Preview failed: {str(e)}")

        # Tab 5: Feature Engineering
        with preprocessing_tabs[4]:
            st.write("**Feature Engineering**")
            st.warning("This section is limited to basic function applications for safety reasons.")

            function = st.selectbox(
                "Function to apply",
                options=["log", "sqrt", "square", "absolute"],
                help="""
                log: Natural logarithm (log(x+1) for negative values)
                sqrt: Square root (of absolute value for negative inputs)
                square: Square (x^2)
                absolute: Absolute value (|x|)
                """
            )

            # Get numeric columns
            col_dtypes = st.session_state.data_summary.get("dtypes", {})
            numeric_cols = [
                col for col, dtype in col_dtypes.items()
                if "int" in dtype or "float" in dtype
            ]

            if not numeric_cols:
                st.info("No numeric columns detected for feature engineering.")
            else:
                columns = st.multiselect(
                    "Select columns to transform",
                    options=numeric_cols
                )

                if st.button("Preview Transformation Effect"):
                    operation = {
                        "type": "apply_function",
                        "params": {
                            "columns": columns,
                            "function": function
                        }
                    }

                    try:
                        # Get preview of operation effect
                        preview = frontend_context.preview_operation(st.session_state.file_id, operation)

                        # Show preview
                        st.write("**Preview of effect:**")
                        st.dataframe(pd.DataFrame(preview["preview"]))

                        # Show general impact metrics
                        st.write("**General Impact:**")
                        display_impact_metrics(preview.get('impact', {}))

                        # Show impact metrics for each column
                        st.write("**Impact on specific columns:**")
                        for col in columns:
                            before_stats = preview["original_summary"].get("column_stats", {}).get(col, {})
                            after_stats = preview["processed_summary"].get("column_stats", {}).get(col, {})

                            st.write(f"**{col}:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Before:**")
                                st.write(f"- Min: {before_stats.get('min', 'N/A')}")
                                st.write(f"- Max: {before_stats.get('max', 'N/A')}")
                                st.write(f"- Mean: {before_stats.get('mean', 'N/A')}")
                            with col2:
                                st.write("**After:**")
                                st.write(f"- Min: {after_stats.get('min', 'N/A')}")
                                st.write(f"- Max: {after_stats.get('max', 'N/A')}")
                                st.write(f"- Mean: {after_stats.get('mean', 'N/A')}")

                        # Add button to apply
                        if st.button("Apply Transformation"):
                            # Apply the operation and ignore the result since we'll rerun the page
                            frontend_context.apply_single_operation(st.session_state.file_id, operation)
                            st.success("Transformation applied successfully!")
                            st.rerun()

                    except Exception as e:
                        st.error(f"Preview failed: {str(e)}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def add_undo_redo_functionality():
    # Store operation history in session state
    if "operation_history" not in st.session_state:
        st.session_state.operation_history = []
        st.session_state.history_position = -1

    col1, col2 = st.columns(2)

    with col1:
        if st.button("↩️ Undo") and st.session_state.history_position > 0:
            # Move back in history
            st.session_state.history_position -= 1
            # Apply operations up to this point
            operations = st.session_state.operation_history[:st.session_state.history_position]
            # Get the frontend context from session state
            if "frontend_context" not in st.session_state:
                st.session_state.frontend_context = FrontendContext()
            frontend_context = st.session_state.frontend_context
            frontend_context.preprocess_data(operations)
            st.rerun()

    with col2:
        if st.button("↪️ Redo") and st.session_state.history_position < len(st.session_state.operation_history) - 1:
            # Move forward in history
            st.session_state.history_position += 1
            # Apply operations up to this point
            operations = st.session_state.operation_history[:st.session_state.history_position + 1]
            # Get the frontend context from session state
            if "frontend_context" not in st.session_state:
                st.session_state.frontend_context = FrontendContext()
            frontend_context = st.session_state.frontend_context
            frontend_context.preprocess_data(operations)
            st.rerun()

def render_operations_summary():
    """Render a summary of the operations to be applied."""

    def render_operations_order_control():
        st.subheader("Operation Order")
        st.write("Drag operations to reorder them:")

        if "preprocessing_operations" not in st.session_state:
            return

        operations = st.session_state.preprocessing_operations

        # Create sortable list
        for i, op in enumerate(operations):
            col1, col2, col3 = st.columns([0.1, 0.8, 0.1])

            with col1:
                # Up arrow
                if st.button("⬆️", key=f"up_{i}") and i > 0:
                    operations[i], operations[i-1] = operations[i-1], operations[i]
                    st.rerun()

            with col2:
                st.write(f"{i+1}. **{op['type'].replace('_', ' ').title()}**")

            with col3:
                # Down arrow
                if st.button("⬇️", key=f"down_{i}") and i < len(operations) - 1:
                    operations[i], operations[i+1] = operations[i+1], operations[i]
                    st.rerun()
    render_operations_order_control()
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
            # Get the frontend context from session state
            if "frontend_context" not in st.session_state:
                st.session_state.frontend_context = FrontendContext()
            frontend_context = st.session_state.frontend_context

            # Apply preprocessing operations using the frontend context
            try:
                result = frontend_context.preprocess_data(st.session_state.preprocessing_operations)

                # Save results to session state
                st.session_state.preprocessed_data_preview = result["preview"]
                st.session_state.original_summary = result["original_summary"]
                st.session_state.processed_summary = result["processed_summary"]
                st.session_state.preprocessing_applied = True

                # Update the columns list after preprocessing
                new_columns = result.get("processed_summary", {}).get("columns", [])
                if new_columns:
                    st.session_state.data_summary["columns"] = new_columns

                st.success("Preprocessing operations applied successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

        # Add clear all button
        if st.button("Clear All Operations", key="clear_all"):
            st.session_state.preprocessing_operations = []
            st.rerun()

    else:
        st.info("No preprocessing operations selected yet. Add operations from the list above.")

def display_impact_metrics(impact: Dict[str, Any]):
    """
    Display impact metrics in a consistent format.

    Args:
        impact: Dictionary containing impact metrics
    """
    if not impact:
        st.info("No impact metrics available.")
        return

    # Create columns for before/after comparison
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Before:**")
        st.write(f"- Rows: {impact.get('rows_before', 'N/A')}")
        st.write(f"- Columns: {impact.get('columns_before', 'N/A')}")
        st.write(f"- Missing values: {impact.get('missing_values_before', 'N/A')}")

    with col2:
        st.write("**After:**")
        st.write(f"- Rows: {impact.get('rows_after', 'N/A')}")
        st.write(f"- Columns: {impact.get('columns_after', 'N/A')}")
        st.write(f"- Missing values: {impact.get('missing_values_after', 'N/A')}")

    # Calculate and display changes
    row_change = impact.get('rows_after', 0) - impact.get('rows_before', 0)
    col_change = impact.get('columns_after', 0) - impact.get('columns_before', 0)
    missing_change = impact.get('missing_values_after', 0) - impact.get('missing_values_before', 0)

    st.write("**Changes:**")
    st.write(f"- Rows: {row_change:+d}")
    st.write(f"- Columns: {col_change:+d}")
    st.write(f"- Missing values: {missing_change:+d}")

def render_preprocessing_results():
    """Render before and after preprocessing results."""
    if "preprocessing_applied" in st.session_state and st.session_state.preprocessing_applied:
        st.subheader("Preprocessing Results")

        # Create tabs for before/after
        tab1, tab2, tab3 = st.tabs(["Before Preprocessing", "After Preprocessing", "Impact"])

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

        with tab3:
            # Display impact metrics
            st.write("**Impact of Preprocessing**")
            if "impact" in st.session_state:
                display_impact_metrics(st.session_state.impact)
            else:
                # Calculate impact from summaries
                impact = {
                    "rows_before": st.session_state.original_summary.get("row_count", 0),
                    "rows_after": st.session_state.processed_summary.get("row_count", 0),
                    "columns_before": st.session_state.original_summary.get("column_count", 0),
                    "columns_after": st.session_state.processed_summary.get("column_count", 0),
                    "missing_values_before": st.session_state.original_summary.get("missing_count", 0),
                    "missing_values_after": st.session_state.processed_summary.get("missing_count", 0)
                }
                display_impact_metrics(impact)

        # Add button to use preprocessed data for analysis
        if st.button("Proceed to Analysis with Preprocessed Data"):
            st.session_state.use_preprocessed = True
            st.session_state.page = "Analysis"
            st.rerun()

# Removed commented-out code for better maintainability


def render_preprocessing_page():
    """Render the complete preprocessing page with step-by-step operations."""
    st.title("Data Preprocessing")

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

    # Display current data info
    st.write(f"**Current Engine:** {engine_type}")
    st.write(f"**File:** {file_name}")

    # Add toggle for showing original vs. preprocessed data
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Data View:**")
    with col2:
        use_preprocessed = st.toggle("Show Preprocessed Data", value=False)
        if use_preprocessed and "preprocessed_data_preview" in st.session_state:
            # Update the data preview to show preprocessed data
            st.session_state.current_view_data = st.session_state.preprocessed_data_preview
        else:
            # Show original data
            st.session_state.current_view_data = st.session_state.data_preview

    # Create tabs for data preview and operations
    main_tabs = st.tabs(["Data Preview", "Preprocessing Operations", "Operation History"])

    with main_tabs[0]:
        # Display current data preview
        st.subheader("Current Data Preview")

        # Use the current view data (original or preprocessed)
        if "current_view_data" in st.session_state:
            st.dataframe(pd.DataFrame(st.session_state.current_view_data))
        else:
            st.dataframe(pd.DataFrame(st.session_state.data_preview))

        # Display data summary
        with st.expander("Data Summary"):
            if use_preprocessed and "processed_summary" in st.session_state:
                st.json(st.session_state.processed_summary)
            else:
                st.json(st.session_state.data_summary)

        # If we have both original and preprocessed data, show impact
        if use_preprocessed and "preprocessed_data_preview" in st.session_state and "impact" in st.session_state:
            with st.expander("Impact of Preprocessing"):
                display_impact_metrics(st.session_state.impact)

    with main_tabs[1]:
        # Render preprocessing interface with operations
        render_preprocessing_interface()

    with main_tabs[2]:
        # Render operation history with undo/redo functionality
        st.subheader("Operation History")

        if "preprocessing_operations" in st.session_state and st.session_state.preprocessing_operations:
            operations = st.session_state.preprocessing_operations

            # Display operations in a table
            for i, op in enumerate(operations):
                op_type = op["type"].replace("_", " ").title()
                params = op.get("params", {})

                # Create an expander for each operation
                with st.expander(f"{i+1}. {op_type}", expanded=i==0):
                    # Display parameters
                    st.write("**Parameters:**")
                    for param_name, param_value in params.items():
                        if isinstance(param_value, list):
                            st.write(f"- **{param_name.replace('_', ' ').title()}**: {', '.join(str(x) for x in param_value)}")
                        else:
                            st.write(f"- **{param_name.replace('_', ' ').title()}**: {param_value}")

                    # Add a button to preview this operation again
                    if st.button(f"Preview Impact", key=f"preview_impact_{i}"):
                        # Get the frontend context
                        frontend_context = st.session_state.frontend_context

                        # Preview the operation
                        preview = frontend_context.preview_operation(st.session_state.file_id, op)

                        # Show impact metrics
                        st.write("**Impact of this operation:**")
                        display_impact_metrics(preview.get('impact', {}))

            # Add undo button
            if st.button("Undo Last Operation"):
                if operations:
                    operations.pop()
                    st.success("Last operation removed")
                    st.rerun()
        else:
            st.info("No preprocessing operations have been applied yet.")

    # Add button to proceed to analysis
    if st.button("Proceed to Analysis"):
        st.session_state.page = "Analysis"
        st.rerun()