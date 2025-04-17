import streamlit as st

def render_engine_selector():
    """
    Render a component for selecting the data processing engine.
    
    Returns:
        str: Selected engine type
    """
    st.subheader("Select Data Processing Engine")
    
    engine_options = {
        "pandas": "Pandas - Python Data Analysis Library",
        "polars": "Polars - Fast DataFrame Library",
        # "pyspark": "PySpark - Apache Spark Python API"
    }
    
    # If engine type is already in session state, use it as default
    default_idx = 0
    if "engine_type" in st.session_state:
        if st.session_state.engine_type == "polars":
            default_idx = 1
        # elif st.session_state.engine_type == "pyspark":
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
    elif engine_selection == "pyspark":
        st.info("PySpark is the Python API for Apache Spark, used for big data processing.")
    
    # Save selected engine to session state
    st.session_state.engine_type = engine_selection
    
    return engine_selection