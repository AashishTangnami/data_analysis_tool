import streamlit as st

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "data" not in st.session_state:
        st.session_state.data = None
    if "current_step" not in st.session_state:
        st.session_state.current_step = "upload"
    if "engine" not in st.session_state:
        st.session_state.engine = "pandas"

def render_sidebar():
    """Render the sidebar with configuration options."""
    # Initialize session state
    initialize_session_state()
    
    with st.sidebar:
        st.title("Navigation")
        
        # Step indicator
        steps = {
            "upload": "1. Upload Data 📤",
            "preprocess": "2. Preprocess 🔧",
            "analyze": "3. Analyze 🔍",
            "visualize": "4. Results 📊"
        }
        
        current_step = st.session_state.current_step
        
        # Display steps with current step highlighted
        for step, label in steps.items():
            if step == current_step:
                st.markdown(f"**→ {label}**")
            else:
                st.markdown(f"  {label}")
        
        st.divider()
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        step_list = list(steps.keys())
        current_idx = step_list.index(current_step)
        
        with col1:
            if current_idx > 0:
                if st.button("⬅️ Previous"):
                    st.session_state.current_step = step_list[current_idx - 1]
                    st.rerun()
        
        with col2:
            if current_idx < len(step_list) - 1:
                next_disabled = (
                    (current_step == "upload" and st.session_state.data is None) or
                    (current_step == "preprocess" and st.session_state.preprocessed_data is None) or
                    (current_step == "analyze" and st.session_state.analysis_results is None)
                )
                
                if not next_disabled and st.button("Next ➡️"):
                    st.session_state.current_step = step_list[current_idx + 1]
                    st.rerun()

        # Help section
        with st.expander("ℹ️ Help"):
            st.markdown("""
            **Quick Guide:**
            1. Upload your data file
            2. Select processing engine
            3. Configure preprocessing steps
            4. Choose analysis methods
            5. Visualize results
            """)
