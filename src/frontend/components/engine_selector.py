import streamlit as st
from src.frontend.context import FrontendContext

def render_engine_selector():
    """
    Render a component for selecting the data processing engine.
    Uses the Strategy pattern through the FrontendContext.

    Returns:
        str: Selected engine type
    """
    # Get the frontend context from session state
    frontend_context = st.session_state.frontend_context

    st.subheader("Select Data Processing Engine")

    engine_options = {
        "pandas": "Pandas - Python Data Analysis Library",
        "polars": "Polars - Fast DataFrame Library",
        # "pyspark": "PySpark - Apache Spark Python API"
    }

    # Get current engine from the context
    current_engine = frontend_context.get_current_engine()

    # Set default index based on current engine
    default_idx = 0
    if current_engine == "polars":
        default_idx = 1
    # elif current_engine == "pyspark":
    #     default_idx = 2

    engine_selection = st.radio(
        "Choose an engine:",
        list(engine_options.keys()),
        format_func=lambda x: engine_options[x],
        index=default_idx,
        horizontal=True
    )

    # Add tooltip with engine description
    if engine_selection == "pandas":
        st.info("Pandas is a powerful Python data analysis toolkit with easy-to-use data structures.")
    elif engine_selection == "polars":
        st.info("Polars is a lightning-fast DataFrame library using Apache Arrow.")
    # elif engine_selection == "pyspark":
    #     st.info("PySpark is the Python API for Apache Spark, used for big data processing.")

    # Update the engine in the context if changed
    if engine_selection != current_engine:
        frontend_context.set_engine(engine_selection)

    return engine_selection