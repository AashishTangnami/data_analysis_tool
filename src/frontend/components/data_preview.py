# frontend/components/data_preview.py
import streamlit as st
import pandas as pd
import json

def render_data_preview(preview_data, summary_data):
    """
    Render a preview of the data and its summary.
    
    Args:
        preview_data: A list of dictionaries containing the data preview
        summary_data: A dictionary containing the data summary
    """
    if preview_data and summary_data:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Data Preview", "Data Summary"])
        
        with tab1:
            # Convert preview data to DataFrame and display
            df_preview = pd.DataFrame(preview_data)
            st.dataframe(df_preview, use_container_width=True)
            st.caption(f"Showing first {len(preview_data)} rows of the data")
        
        with tab2:
            # Display summary information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Information")
                st.write(f"**Number of Rows:** {summary_data['shape'][0]}")
                st.write(f"**Number of Columns:** {summary_data['shape'][1]}")
                
                # Display data types
                st.subheader("Data Types")
                dtype_df = pd.DataFrame(
                    {"Data Type": summary_data["dtypes"].values()},
                    index=summary_data["dtypes"].keys()
                )
                st.dataframe(dtype_df)
            
            with col2:
                st.subheader("Missing Values")
                missing_df = pd.DataFrame(
                    {"Missing Count": summary_data["missing_values"].values()},
                    index=summary_data["missing_values"].keys()
                )
                st.dataframe(missing_df)
                
                # Display numeric summary if available
                if "numeric_summary" in summary_data and summary_data["numeric_summary"]:
                    st.subheader("Numeric Summary")
                    
                    # Format the numeric summary for display
                    numeric_summary = {}
                    for col, stats in summary_data["numeric_summary"].items():
                        for stat, value in stats.items():
                            if stat not in numeric_summary:
                                numeric_summary[stat] = {}
                            numeric_summary[stat][col] = value
                    
                    # Display each statistic
                    for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
                        if stat in numeric_summary:
                            st.write(f"**{stat}**")
                            stat_df = pd.DataFrame(
                                {"Value": numeric_summary[stat].values()},
                                index=numeric_summary[stat].keys()
                            )
                            st.dataframe(stat_df)