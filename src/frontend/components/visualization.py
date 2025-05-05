import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Union, List, Dict, Any, Optional

def render_distribution_plot(data: pd.DataFrame, column_name: str, use_plotly: bool = False) -> None:
    """
    Render a distribution plot for a numeric column.

    Args:
        data: Pandas DataFrame
        column_name: Name of the column to visualize
        use_plotly: Whether to use Plotly (interactive) or Matplotlib (static)
    """
    # Check if column exists
    if column_name not in data.columns:
        st.error(f"Column '{column_name}' not found in data")
        return

    # Check if column has numeric data
    if not pd.api.types.is_numeric_dtype(data[column_name]):
        st.warning(f"Column '{column_name}' is not numeric. Converting to numeric if possible.")
        try:
            # Try to convert to numeric
            data = data.copy()
            data[column_name] = pd.to_numeric(data[column_name], errors='coerce')
        except:
            st.error(f"Cannot convert column '{column_name}' to numeric type")
            return

    # Handle missing values
    if data[column_name].isna().any():
        st.info(f"Column '{column_name}' contains {data[column_name].isna().sum()} missing values which will be excluded")

    if use_plotly:
        # Create interactive Plotly histogram
        fig = px.histogram(
            data,
            x=column_name,
            marginal="box",  # Add a box plot at the margin
            title=f"Distribution of {column_name}",
            labels={column_name: column_name},
            opacity=0.7,
            color_discrete_sequence=['#3366CC']
        )

        # Add KDE curve
        fig.update_layout(
            xaxis_title=column_name,
            yaxis_title="Frequency",
            bargap=0.1
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        # Create static Matplotlib plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data[column_name].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {column_name}")
        ax.set_xlabel(column_name)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

def render_correlation_heatmap(data: pd.DataFrame, use_plotly: bool = False,
                          max_columns: int = 20, correlation_threshold: Optional[float] = None) -> None:
    """
    Render a correlation heatmap for numeric columns.

    Args:
        data: Pandas DataFrame
        use_plotly: Whether to use Plotly (interactive) or Matplotlib (static)
        max_columns: Maximum number of columns to include in the heatmap
        correlation_threshold: Optional threshold to filter correlations (e.g., 0.5 to show only |corr| >= 0.5)
    """
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.shape[1] < 2:
        st.warning("Not enough numeric columns for correlation analysis.")
        return

    # Limit the number of columns if there are too many
    if numeric_data.shape[1] > max_columns:
        st.warning(f"Too many numeric columns ({numeric_data.shape[1]}). Limiting to {max_columns} columns.")
        # Select columns with highest variance
        variances = numeric_data.var().sort_values(ascending=False)
        selected_columns = variances.index[:max_columns]
        numeric_data = numeric_data[selected_columns]

    # Compute correlation
    corr = numeric_data.corr()

    # Apply threshold if specified
    if correlation_threshold is not None:
        # Create a mask for correlations below threshold (absolute value)
        mask = np.abs(corr) < correlation_threshold
        # Replace correlations below threshold with NaN
        corr_filtered = corr.copy()
        corr_filtered[mask] = np.nan
        # If all correlations are below threshold, show warning
        if np.isnan(corr_filtered).all().all():
            st.warning(f"No correlations with absolute value >= {correlation_threshold} found.")
            # Fall back to original correlation matrix
            corr_filtered = corr
    else:
        corr_filtered = corr

    if use_plotly:
        # Create interactive Plotly heatmap
        fig = px.imshow(
            corr_filtered,
            color_continuous_scale="RdBu_r",
            labels=dict(color="Correlation"),
            title="Correlation Heatmap",
            text_auto=".2f",
            aspect="auto"
        )

        fig.update_layout(
            height=600,
            width=800,
            xaxis_title="",
            yaxis_title=""
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        # Create static Matplotlib plot
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_filtered, dtype=bool))  # Create mask for upper triangle
        sns.heatmap(
            corr_filtered,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax,
            mask=mask,  # Only show lower triangle
            vmin=-1,
            vmax=1
        )
        ax.set_title("Correlation Heatmap")
        plt.tight_layout()
        st.pyplot(fig)

def render_scatter_plot(data: pd.DataFrame, x_column: str, y_column: str,
                      color_column: Optional[str] = None, use_plotly: bool = False,
                      add_trendline: bool = False, sample_size: Optional[int] = None) -> None:
    """
    Render a scatter plot between two numeric columns.

    Args:
        data: Pandas DataFrame
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        color_column: Optional column name for color coding
        use_plotly: Whether to use Plotly (interactive) or Matplotlib (static)
        add_trendline: Whether to add a trendline to the plot
        sample_size: Optional sample size to limit the number of points (for large datasets)
    """
    # Check if columns exist
    for col in [x_column, y_column] + ([color_column] if color_column else []):
        if col not in data.columns:
            st.error(f"Column '{col}' not found in data")
            return

    # Check if columns have numeric data
    for col in [x_column, y_column]:
        if not pd.api.types.is_numeric_dtype(data[col]):
            st.warning(f"Column '{col}' is not numeric. Converting to numeric if possible.")
            try:
                # Try to convert to numeric
                data = data.copy()
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except:
                st.error(f"Cannot convert column '{col}' to numeric type")
                return

    # Handle missing values
    missing_x = data[x_column].isna().sum()
    missing_y = data[y_column].isna().sum()
    if missing_x > 0 or missing_y > 0:
        st.info(f"Missing values: {missing_x} in {x_column}, {missing_y} in {y_column}. These will be excluded.")

    # Create a copy of the data with only the required columns and drop missing values
    plot_data = data[[x_column, y_column] + ([color_column] if color_column else [])].dropna()

    # Sample data if needed
    if sample_size and len(plot_data) > sample_size:
        st.info(f"Sampling {sample_size} points from {len(plot_data)} total points")
        plot_data = plot_data.sample(sample_size, random_state=42)

    if use_plotly:
        # Create interactive Plotly scatter plot
        if color_column:
            fig = px.scatter(
                plot_data,
                x=x_column,
                y=y_column,
                color=color_column,
                title=f"{y_column} vs {x_column}",
                labels={x_column: x_column, y_column: y_column},
                opacity=0.7,
                trendline="ols" if add_trendline else None
            )
        else:
            fig = px.scatter(
                plot_data,
                x=x_column,
                y=y_column,
                title=f"{y_column} vs {x_column}",
                labels={x_column: x_column, y_column: y_column},
                opacity=0.7,
                trendline="ols" if add_trendline else None
            )

        fig.update_layout(
            height=500,
            width=700,
            xaxis_title=x_column,
            yaxis_title=y_column
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        # Create static Matplotlib plot
        fig, ax = plt.subplots(figsize=(10, 6))

        if color_column:
            scatter = sns.scatterplot(x=x_column, y=y_column, hue=color_column, data=plot_data, ax=ax)
            # Add legend outside the plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            scatter = sns.scatterplot(x=x_column, y=y_column, data=plot_data, ax=ax)

        # Add trendline if requested
        if add_trendline:
            sns.regplot(x=x_column, y=y_column, data=plot_data, ax=ax, scatter=False, line_kws={"color": "red"})

        ax.set_title(f"{y_column} vs {x_column}")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        plt.tight_layout()
        st.pyplot(fig)

def render_categorical_plot(data: pd.DataFrame, column_name: str,
                         use_plotly: bool = False, max_categories: int = 15,
                         horizontal: bool = False) -> None:
    """
    Render a bar plot for a categorical column.

    Args:
        data: Pandas DataFrame
        column_name: Name of the column to visualize
        use_plotly: Whether to use Plotly (interactive) or Matplotlib (static)
        max_categories: Maximum number of categories to display
        horizontal: Whether to display the bars horizontally (better for long category names)
    """
    # Check if column exists
    if column_name not in data.columns:
        st.error(f"Column '{column_name}' not found in data")
        return

    # Count values
    value_counts = data[column_name].value_counts().sort_values(ascending=False)

    # Check if there are too many categories
    if len(value_counts) > max_categories:
        st.info(f"Column '{column_name}' has {len(value_counts)} categories. Showing top {max_categories}.")
        value_counts = value_counts.head(max_categories)
        # Add an "Other" category for the rest
        other_count = data[column_name].value_counts().sort_values(ascending=False)[max_categories:].sum()
        if other_count > 0:
            value_counts["Other"] = other_count

    if use_plotly:
        # Create interactive Plotly bar chart
        if horizontal:
            fig = px.bar(
                y=value_counts.index,
                x=value_counts.values,
                title=f"Distribution of {column_name}",
                labels={"y": column_name, "x": "Count"},
                orientation='h',
                color=value_counts.values,
                color_continuous_scale="Viridis"
            )
        else:
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {column_name}",
                labels={"x": column_name, "y": "Count"},
                color=value_counts.values,
                color_continuous_scale="Viridis"
            )

        fig.update_layout(
            height=500,
            width=700,
            coloraxis_showscale=False
        )

        # Rotate x-axis labels if not horizontal
        if not horizontal:
            fig.update_layout(xaxis_tickangle=-45)

        st.plotly_chart(fig, use_container_width=True)
    else:
        # Create static Matplotlib plot
        fig, ax = plt.subplots(figsize=(12, 8))

        if horizontal:
            # Horizontal bar plot
            sns.barplot(y=value_counts.index, x=value_counts.values, ax=ax)
            ax.set_title(f"Distribution of {column_name}")
            ax.set_ylabel(column_name)
            ax.set_xlabel("Count")
        else:
            # Vertical bar plot
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
            ax.set_title(f"Distribution of {column_name}")
            ax.set_xlabel(column_name)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        st.pyplot(fig)