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

# Import utility functions
from src.frontend.utils.streamlit_helpers import generate_unique_key

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

    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("feature_importance"))

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
    try:
        df_corr = pd.DataFrame({
            'Feature': list(corr_analysis.keys()),
            'Correlation': [item.get("correlation", 0) for item in corr_analysis.values()],
            'P-Value': [item.get("p_value", 1) for item in corr_analysis.values()],
            'Significance': ['Significant (p<0.05)' if item.get("p_value") is not None and item.get("p_value", 1) < 0.05 else 'Not Significant'
                            for item in corr_analysis.values()]
        })
    except Exception as e:
        st.error(f"Failed to create correlation analysis DataFrame: {str(e)}")
        return

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

    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("correlation_chart"))

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

                # Use our function to generate a truly unique key
                unique_key = generate_unique_key(f"outlier_gauge_{feature}_{idx}")
                st.plotly_chart(fig, use_container_width=True, key=unique_key)

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

def render_descriptive_results(results=None):
    """
    Render descriptive analysis results.

    Args:
        results: Optional results dictionary. If None, uses st.session_state.analysis_results
    """
    if results is None:
        if 'analysis_results' not in st.session_state or not st.session_state.analysis_results:
            st.warning("No descriptive analysis results available. Please run the analysis first.")
            return
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

def render_predictive_results(results=None):
    """
    Render predictive analysis results.

    Args:
        results: Optional results dictionary. If None, uses st.session_state.analysis_results
    """
    if results is None:
        if 'analysis_results' not in st.session_state or not st.session_state.analysis_results:
            st.warning("No predictive analysis results available. Please run the analysis first.")
            return
        results = st.session_state.analysis_results

    st.info("Predictive analysis results rendering will be implemented in the next phase.")

    # Display raw results in JSON format
    with st.expander("Raw Results"):
        st.json(results)

def render_prescriptive_results(results=None):
    """
    Render prescriptive analysis results.

    Args:
        results: Optional results dictionary. If None, uses st.session_state.analysis_results
    """
    if results is None:
        if 'analysis_results' not in st.session_state or not st.session_state.analysis_results:
            st.warning("No prescriptive analysis results available. Please run the analysis first.")
            return
        results = st.session_state.analysis_results

    st.info("Prescriptive analysis results rendering will be implemented in the next phase.")

    # Display raw results in JSON format
    with st.expander("Raw Results"):
        st.json(results)

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