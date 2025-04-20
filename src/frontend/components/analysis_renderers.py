import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from components.visualization import (
    render_distribution_plot, 
    render_correlation_heatmap,
    render_scatter_plot,
    render_categorical_plot
)

def render_diagnostic_results(results=None):
    """
    Render diagnostic analysis results with visualizations.
    
    Args:
        results: Optional results dictionary. If None, uses st.session_state.analysis_results
    """
    if results is None:
        if 'analysis_results' not in st.session_state or not st.session_state.analysis_results:
            st.warning("No diagnostic analysis results available. Please run the analysis first.")
            return
        results = st.session_state.analysis_results
    
    st.header("Diagnostic Analysis Results")
    
    # Create tabs for different sections of the analysis
    tab1, tab2, tab3, tab4 = st.tabs([
        "Feature Importance", 
        "Correlation Analysis", 
        "Outlier Detection",
        "Raw Results"
    ])
    
    with tab1:
        render_feature_importance(results)
    
    with tab2:
        render_correlation_analysis(results)
    
    with tab3:
        render_outlier_detection(results)
    
    with tab4:
        st.subheader("Raw Analysis Results")
        st.json(results)

def render_feature_importance(results):
    """
    Render feature importance visualization.
    
    Args:
        results: Dictionary containing analysis results with feature_importance key
    """
    st.subheader("Feature Importance")
    
    if not results.get("feature_importance") or "error" in results.get("feature_importance", {}):
        error_msg = results.get("feature_importance", {}).get("error", "Unknown error")
        st.error(f"Failed to calculate feature importance: {error_msg}")
        return
    
    # Create DataFrame for feature importance
    feature_importance = results["feature_importance"]
    df_importance = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    })
    df_importance = df_importance.sort_values(by='Importance', ascending=False)
    
    # Create Plotly horizontal bar chart
    fig = px.bar(
        df_importance, 
        y='Feature', 
        x='Importance', 
        orientation='h',
        title='Feature Importance',
        color='Importance',
        color_continuous_scale='viridis',
        height=400
    )
    
    fig.update_layout(
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance insights
    st.subheader("Insights")
    top_features = df_importance.head(2)['Feature'].tolist()
    
    st.write(f"**Top influential features**: {', '.join(top_features)}")
    
    # Show metrics for top feature
    if len(top_features) > 0:
        top_feature = top_features[0]
        top_importance = feature_importance[top_feature]
        col1, col2 = st.columns(2)
        col1.metric("Top Feature", top_feature)
        col2.metric("Importance Score", f"{top_importance:.4f}")
        
        # Add explanation about what feature importance means
        st.info(
            "Feature importance indicates how much each feature contributes to the model's predictions. "
            "Higher values suggest greater influence on the target variable."
        )
    
    # Display data as table
    st.dataframe(
        df_importance,
        column_config={
            "Feature": st.column_config.TextColumn("Feature"),
            "Importance": st.column_config.ProgressColumn(
                "Importance Score",
                format="%.4f",
                min_value=0,
                max_value=max(feature_importance.values())
            )
        },
        hide_index=True
    )

def render_correlation_analysis(results):
    """
    Render correlation analysis visualization.
    
    Args:
        results: Dictionary containing analysis results with correlation_analysis key
    """
    st.subheader("Correlation Analysis")
    
    if not results.get("correlation_analysis"):
        st.error("No correlation analysis results available.")
        return
    
    # Create DataFrame for correlation analysis
    corr_analysis = results["correlation_analysis"]
    df_corr = pd.DataFrame({
        'Feature': list(corr_analysis.keys()),
        'Correlation': [item.get("correlation", 0) for item in corr_analysis.values()],
        'P-Value': [item.get("p_value", 1) for item in corr_analysis.values()],
        'Significance': ['Significant (p<0.05)' if item.get("p_value", 1) < 0.05 else 'Not Significant' 
                        for item in corr_analysis.values()]
    })
    
    # Sort by absolute correlation
    df_corr['Abs_Correlation'] = df_corr['Correlation'].abs()
    df_corr = df_corr.sort_values(by='Abs_Correlation', ascending=False)
    df_corr = df_corr.drop(columns=['Abs_Correlation'])
    
    # Create bar chart with colors indicating significance
    fig = px.bar(
        df_corr, 
        y='Feature', 
        x='Correlation',
        color='Significance',
        title='Feature Correlation with Target',
        color_discrete_map={
            'Significant (p<0.05)': '#1E88E5',  # Blue
            'Not Significant': '#9E9E9E'  # Gray
        },
        height=400
    )
    
    fig.update_layout(
        xaxis_title='Correlation Coefficient',
        yaxis_title='Feature',
        yaxis=dict(autorange="reversed"),
    )
    
    # Add a vertical line at x=0
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis insights
    st.subheader("Insights")
    
    # Find significant correlations
    significant_corr = df_corr[df_corr['Significance'] == 'Significant (p<0.05)']
    
    if len(significant_corr) > 0:
        st.write(f"**Statistically significant correlations found**: {len(significant_corr)} features")
        top_corr = significant_corr.iloc[0]
        st.write(f"Strongest significant correlation: **{top_corr['Feature']}** with correlation of {top_corr['Correlation']:.4f} (p-value: {top_corr['P-Value']:.4f})")
    else:
        st.write("No statistically significant correlations found at the 0.05 level.")
        # Show the strongest correlation anyway
        if not df_corr.empty:
            top_corr = df_corr.iloc[0]
            st.write(f"Strongest correlation (not significant): **{top_corr['Feature']}** with correlation of {top_corr['Correlation']:.4f} (p-value: {top_corr['P-Value']:.4f})")
    
    # Correlation interpretation guide
    with st.expander("Correlation Interpretation Guide"):
        st.write("""
        - **Correlation Coefficient**: Ranges from -1 to 1
            - 1: Perfect positive correlation
            - 0: No correlation
            - -1: Perfect negative correlation
        - **P-value**: Indicates statistical significance
            - p < 0.05: Typically considered statistically significant
            - Lower p-values suggest stronger evidence against the null hypothesis
        """)
    
    # Display data as table
    st.dataframe(
        df_corr,
        column_config={
            "Feature": st.column_config.TextColumn("Feature"),
            "Correlation": st.column_config.NumberColumn("Correlation", format="%.4f"),
            "P-Value": st.column_config.NumberColumn("P-Value", format="%.4f"),
            "Significance": st.column_config.TextColumn("Significance")
        },
        hide_index=True
    )

def render_outlier_detection(results):
    """
    Render outlier detection visualization.
    
    Args:
        results: Dictionary containing analysis results with outlier_detection key
    """
    st.subheader("Outlier Detection")
    
    if not results.get("outlier_detection"):
        st.error("No outlier detection results available.")
        return
    
    outlier_detection = results["outlier_detection"]
    
    # Create aggregated DataFrame for outlier summary
    df_outliers = pd.DataFrame({
        'Feature': list(outlier_detection.keys()),
        'Mean': [item.get("mean", 0) for item in outlier_detection.values()],
        'Std': [item.get("std", 0) for item in outlier_detection.values()],
        'Outlier Count': [item.get("outlier_count", 0) for item in outlier_detection.values()],
        'Outlier Percentage': [item.get("outlier_percentage", 0) for item in outlier_detection.values()]
    })
    
    # Only continue if there are features analyzed
    if df_outliers.empty:
        st.info("No features suitable for outlier detection.")
        return
    
    # Sort by outlier percentage
    df_outliers = df_outliers.sort_values(by='Outlier Percentage', ascending=False)
    
    # Check if there are any outliers detected
    total_outliers = df_outliers['Outlier Count'].sum()
    
    if total_outliers == 0:
        st.success("No outliers detected in the analyzed features using Z-score method (threshold: |z| > 3).")
    else:
        st.warning(f"Found {total_outliers} outliers across all features.")
    
    # Create a gauge chart for each feature showing distribution width
    st.subheader("Feature Distribution Range")
    
    # Display distribution ranges as a table with visualizations
    for idx, row in df_outliers.iterrows():
        feature = row['Feature']
        mean = row['Mean']
        std = row['Std']
        outlier_count = row['Outlier Count']
        
        # Only show detailed info for features with outliers
        if outlier_count > 0 or idx < 4:  # Show at least a few features regardless of outliers
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.metric(
                    label=feature,
                    value=f"{mean:.2f}",
                    delta=f"±{std:.2f} SD"
                )
                st.write(f"Outliers: {outlier_count}")
            
            with col2:
                # Create a simple gauge to show the mean and range
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = mean,
                    number = {'suffix': f" (±{std:.2f})"},
                    gauge = {
                        'axis': {'range': [mean - 3*std, mean + 3*std]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [mean - 3*std, mean - 2*std], 'color': "lightgray"},
                            {'range': [mean - 2*std, mean - std], 'color': "gray"},
                            {'range': [mean - std, mean + std], 'color': "lightgreen"},
                            {'range': [mean + std, mean + 2*std], 'color': "gray"},
                            {'range': [mean + 2*std, mean + 3*std], 'color': "lightgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': mean
                        }
                    }
                ))
                
                fig.update_layout(
                    height=150,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Outlier detection summary table
    st.subheader("Outlier Detection Summary")
    
    st.dataframe(
        df_outliers,
        column_config={
            "Feature": st.column_config.TextColumn("Feature"),
            "Mean": st.column_config.NumberColumn("Mean", format="%.2f"),
            "Std": st.column_config.NumberColumn("Std Dev", format="%.2f"),
            "Outlier Count": st.column_config.NumberColumn("Outliers"),
            "Outlier Percentage": st.column_config.ProgressColumn(
                "Outlier %",
                format="%.2f%%",
                min_value=0,
                max_value=max(df_outliers['Outlier Percentage']) if not df_outliers.empty and max(df_outliers['Outlier Percentage']) > 0 else 5
            )
        },
        hide_index=True
    )
    
    # Explanation of outlier detection method
    st.info("""
    **Z-score Outlier Detection Method**:
    - Data points with Z-score > 3 or < -3 are considered outliers (more than 3 standard deviations from the mean)
    - In a normal distribution, approximately 99.7% of the data falls within ±3 standard deviations of the mean
    - Outliers may indicate measurement errors, data entry issues, or genuinely unusual cases
    """)

# ========================== COMMON RENDERING UTILITIES ==========================

def create_tabs_for_results(results, tab_definitions):
    """
    Create tabs for displaying different parts of analysis results.
    
    Args:
        results: Analysis results dictionary
        tab_definitions: List of dictionaries with keys:
                         - key: Key in results dictionary
                         - title: Tab title
                         - render_func: Function to call for rendering this tab
    
    Returns:
        Tuple of (created_tabs, tab_keys) - created tabs and their corresponding keys
    """
    # Filter to only include tabs with data
    available_tabs = [
        tab for tab in tab_definitions
        if tab["key"] in results and results[tab["key"]]
    ]
    
    if not available_tabs:
        st.warning("No analysis results available to display.")
        return [], []
    
    # Create tabs
    tab_titles = [tab["title"] for tab in available_tabs]
    created_tabs = st.tabs(tab_titles)
    tab_keys = [tab["key"] for tab in available_tabs]
    
    # Render content in each tab
    for i, tab in enumerate(available_tabs):
        with created_tabs[i]:
            tab["render_func"](results[tab["key"]])
    
    return created_tabs, tab_keys