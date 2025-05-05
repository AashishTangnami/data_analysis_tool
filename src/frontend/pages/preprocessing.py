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

                    # Show side-by-side preview
                    st.write("**Preview of effect:**")
                    display_side_by_side_preview(preview)

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

                        # Show side-by-side preview
                        st.write("**Preview of effect:**")
                        display_side_by_side_preview(preview)

                        # Show column-specific scaling impact
                        st.write("**Column-specific Scaling Impact:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Before Scaling:**")
                            for col in columns:
                                if col != "all":
                                    stats = preview["original_summary"].get("numeric_summary", {}).get(col, {})
                                    st.write(f"- {col}: mean={stats.get('mean', 'N/A')}, std={stats.get('std', 'N/A')}")

                        with col2:
                            st.write(f"**After Scaling:**")
                            for col in columns:
                                if col != "all":
                                    stats = preview["processed_summary"].get("numeric_summary", {}).get(col, {})
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

                        # Show side-by-side preview
                        st.write("**Preview of effect:**")
                        display_side_by_side_preview(preview)

                        # Show impact metrics for each column
                        st.write("**Impact on specific columns:**")
                        for col in columns:
                            before_stats = preview["original_summary"].get("numeric_summary", {}).get(col, {})
                            after_stats = preview["processed_summary"].get("numeric_summary", {}).get(col, {})

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

def display_side_by_side_preview(preview: Dict[str, Any]):
    """
    Display side-by-side comparison of original and processed data with impact metrics.

    Args:
        preview: Dictionary containing preview information
    """
    if not preview:
        st.info("No preview information available.")
        return

    # Show column changes if available - display this first for better visibility
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

    # Get operation type if available
    operation_type = None
    if "operations_applied" in preview and preview["operations_applied"]:
        if isinstance(preview["operations_applied"], list) and preview["operations_applied"]:
            operation_type = preview["operations_applied"][0].get("type", "")

    # Customize preview based on operation type
    if "original_summary" in preview and "processed_summary" in preview:
        original_cols = set(preview["original_summary"].get("columns", []))
        processed_cols = set(preview["processed_summary"].get("columns", []))
        removed_cols = original_cols - processed_cols
        added_cols = processed_cols - original_cols

        with col1:
            st.write("**Original Data**")

            # Create a styled dataframe based on operation type
            if not original_data.empty:
                # Create a copy to avoid modifying the original
                styled_df = original_data.copy()

                # Create a base styler
                styler = styled_df.style

                # Apply styling based on operation type
                if operation_type == "drop_columns" and removed_cols:
                    # Highlight columns that will be removed
                    for col in removed_cols:
                        if col in styled_df.columns:
                            styler = styler.set_properties(
                                subset=[col],
                                **{'background-color': '#FFCCCB'}  # Light red
                            )

                    # Add a note about highlighted columns
                    st.dataframe(styler, height=200)
                    st.caption("🔴 Columns highlighted in red will be removed")

                elif operation_type == "fill_missing":
                    # Try to highlight cells with missing values
                    # Apply styling to highlight missing values
                    def highlight_missing(val):
                        return 'background-color: #FFCCCB' if pd.isna(val) else ''

                    styler = styler.map(highlight_missing)

                    # Display the styled dataframe
                    st.dataframe(styler, height=200)
                    st.caption("🔴 Cells highlighted in red contain missing values that will be filled")

                elif operation_type == "encode_categorical":
                    # Try to highlight categorical columns that will be encoded
                    params = preview["operations_applied"][0].get("params", {})
                    columns_to_encode = params.get("columns", [])

                    if columns_to_encode:
                        for col in columns_to_encode:
                            if col in styled_df.columns:
                                styler = styler.set_properties(
                                    subset=[col],
                                    **{'background-color': '#E6F3FF'}  # Light blue
                                )

                        # Display the styled dataframe
                        st.dataframe(styler, height=200)
                        st.caption("🔵 Columns highlighted in blue will be encoded")
                    else:
                        st.dataframe(styled_df, height=200)

                else:
                    # Just show the original data without styling
                    st.dataframe(styled_df, height=200)
            else:
                # Just show the original data without styling
                st.dataframe(original_data, height=200)

            # Show column count
            if "original_summary" in preview:
                col_count = len(preview["original_summary"].get("columns", []))
                st.caption(f"Columns: {col_count}")

        with col2:
            st.write("**Processed Data (Preview)**")

            # For added columns, highlight them in the processed data
            if not processed_data.empty and added_cols:
                # Create a copy to avoid modifying the original
                styled_df = processed_data.copy()

                # Create a base styler
                styler = styled_df.style

                # Highlight added columns
                for col in added_cols:
                    if col in styled_df.columns:
                        styler = styler.set_properties(
                            subset=[col],
                            **{'background-color': '#CCFFCC'}  # Light green
                        )

                # Display the styled dataframe
                st.dataframe(styler, height=200)

                # Add a note about highlighted columns
                if added_cols:
                    st.caption("🟢 Columns highlighted in green were added during processing")
            else:
                # Display the processed data
                st.dataframe(processed_data, height=200)

            # Show column count
            if "processed_summary" in preview:
                col_count = len(preview["processed_summary"].get("columns", []))
                st.caption(f"Columns: {col_count}")
    else:
        # If we don't have column information, just show the data without styling
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

    # Display impact metrics
    # st.write("**Impact Metrics**")
    # display_impact_metrics(preview.get('impact', {}))

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
    """Render before and after preprocessing results side-by-side."""
    if "preprocessing_applied" in st.session_state and st.session_state.preprocessing_applied:
        st.subheader("Preprocessing Results")

        # Display impact metrics at the top
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

        # Create side-by-side columns for before/after data preview
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Original Data**")
            st.dataframe(pd.DataFrame(st.session_state.data_preview), height=300)
            with st.expander("Original Data Summary"):
                st.json(st.session_state.original_summary)

        with col2:
            st.write("**Processed Data**")
            st.dataframe(pd.DataFrame(st.session_state.preprocessed_data_preview), height=300)
            with st.expander("Processed Data Summary"):
                st.json(st.session_state.processed_summary)

        # Add a section to highlight changes
        st.write("**Column Changes**")

        # Get column lists
        original_cols = set(st.session_state.original_summary.get("columns", []))
        processed_cols = set(st.session_state.processed_summary.get("columns", []))

        # Find added and removed columns
        added_cols = processed_cols - original_cols
        removed_cols = original_cols - processed_cols

        # Display changes
        if added_cols:
            st.success(f"Added columns: {', '.join(added_cols)}")
        if removed_cols:
            st.warning(f"Removed columns: {', '.join(removed_cols)}")
        if not added_cols and not removed_cols:
            st.info("No columns were added or removed")

        # Add buttons to proceed to analysis with data choice
        st.write("**Proceed to Analysis:**")

        # Create columns for the data choice options
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Use Original Data", key="results_use_original", type="secondary"):
                st.session_state.use_preprocessed = False
                st.session_state.page = "Analysis"
                st.rerun()

        with col2:
            if st.button("Use Preprocessed Data", key="results_use_preprocessed", type="primary"):
                st.session_state.use_preprocessed = True
                st.session_state.page = "Analysis"
                st.rerun()

        # Add a note about the data choice
        st.info("💡 You can always switch between original and preprocessed data on the Analysis page.")

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

    # =====================================================================
    # DATA PREVIEW SECTION
    # =====================================================================
    st.header("Data Preview")

    # Check if preprocessed data is available
    has_preprocessed = "preprocessed_data_preview" in st.session_state

    if has_preprocessed:
        # Show side-by-side comparison
        st.write("**Side-by-Side Comparison**")

        # Create columns for original and processed data
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Original Data**")
            st.dataframe(pd.DataFrame(st.session_state.data_preview), height=300)
            with st.expander("Original Data Summary"):
                st.json(st.session_state.data_summary)

        with col2:
            st.write("**Processed Data**")
            st.dataframe(pd.DataFrame(st.session_state.preprocessed_data_preview), height=300)
            with st.expander("Processed Data Summary"):
                st.json(st.session_state.processed_summary)

        # Show impact metrics
        if "impact" in st.session_state:
            st.write("**Impact of Preprocessing**")
            display_impact_metrics(st.session_state.impact)

        # Show column changes
        st.write("**Column Changes**")

        # Get column lists
        original_cols = set(st.session_state.data_summary.get("columns", []))
        processed_cols = set(st.session_state.processed_summary.get("columns", []))

        # Find added and removed columns
        added_cols = processed_cols - original_cols
        removed_cols = original_cols - processed_cols

        # Display changes with better formatting
        if added_cols or removed_cols:
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
        else:
            st.info("No columns were added or removed")
    else:
        # Just show original data
        st.write("**Original Data**")
        st.dataframe(pd.DataFrame(st.session_state.data_preview))
        with st.expander("Data Summary"):
            st.json(st.session_state.data_summary)

        st.info("Apply preprocessing operations below to see side-by-side comparison")

    # =====================================================================
    # PREPROCESSING OPERATIONS SECTION
    # =====================================================================
    st.header("Preprocessing Operations")
    st.write("Apply preprocessing operations step-by-step:")

    # Render preprocessing interface with operations
    render_preprocessing_interface()

    # =====================================================================
    # OPERATION HISTORY SECTION
    # =====================================================================
    st.header("Operation History")

    if "preprocessing_operations" in st.session_state and st.session_state.preprocessing_operations:
        operations = st.session_state.preprocessing_operations

        # Display operations in a table
        for i, op in enumerate(operations):
            op_type = op["type"].replace("_", " ").title()
            params = op.get("params", {})

            # Create a more descriptive title for the operation
            operation_title = f"{i+1}. {op_type}"

            # Add specific details to the title based on operation type
            if op_type == "Drop Columns":
                columns = params.get("columns", [])
                if columns:
                    if len(columns) <= 3:
                        operation_title += f": {', '.join(columns)}"
                    else:
                        operation_title += f": {len(columns)} columns"
            elif op_type == "Fill Missing":
                method = params.get("method", "")
                operation_title += f" ({method})"
            elif op_type == "Encode Categorical":
                method = params.get("method", "")
                operation_title += f" ({method})"
            elif op_type == "Scale Numeric":
                method = params.get("method", "")
                operation_title += f" ({method})"
            elif op_type == "Apply Function":
                function = params.get("function", "")
                operation_title += f" ({function})"

            # Create an expander for each operation
            with st.expander(operation_title, expanded=i==0):
                # Display parameters with better formatting
                st.write("**Parameters:**")

                # Format parameters based on operation type
                for param_name, param_value in params.items():
                    if param_name == "columns":
                        if isinstance(param_value, list):
                            if param_value:
                                if len(param_value) > 10:
                                    # Show count for large lists
                                    st.write(f"- **Columns**: {len(param_value)} columns")
                                    with st.expander("Show all columns"):
                                        st.write(", ".join(str(x) for x in param_value))
                                else:
                                    # Show all for smaller lists
                                    st.write(f"- **Columns**: {', '.join(str(x) for x in param_value)}")
                            else:
                                st.write("- **Columns**: None")
                        else:
                            st.write(f"- **Columns**: {param_value}")
                    else:
                        # Format other parameters
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

                    # Show side-by-side preview
                    st.write("**Impact of this operation:**")
                    display_side_by_side_preview(preview)

                    # For drop columns operation, explicitly show dropped columns
                    if op_type == "Drop Columns":
                        columns = params.get("columns", [])
                        if columns:
                            st.warning(f"**Dropped Columns**: {', '.join(columns)}")

        # Add operation controls
        col1, col2 = st.columns(2)

        with col1:
            # Add undo button
            if st.button("Undo Last Operation"):
                if operations:
                    last_op = operations.pop()
                    op_type = last_op["type"].replace("_", " ").title()
                    st.success(f"Removed: {op_type}")
                    st.rerun()

        with col2:
            # Add clear all button
            if st.button("Clear All Operations"):
                st.session_state.preprocessing_operations = []
                st.success("All operations cleared")
                st.rerun()
    else:
        st.info("No preprocessing operations have been applied yet.")

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