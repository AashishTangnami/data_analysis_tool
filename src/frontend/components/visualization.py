# frontend/components/visualization.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def render_distribution_plot(data, column_name):
    """
    Render a distribution plot for a numeric column.
    
    Args:
        data: Pandas DataFrame
        column_name: Name of the column to visualize
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[column_name], kde=True, ax=ax)
    ax.set_title(f"Distribution of {column_name}")
    ax.set_xlabel(column_name)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def render_correlation_heatmap(data):
    """
    Render a correlation heatmap for numeric columns.
    
    Args:
        data: Pandas DataFrame
    """
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.shape[1] < 2:
        st.warning("Not enough numeric columns for correlation analysis.")
        return
    
    # Compute correlation
    corr = numeric_data.corr()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

def render_scatter_plot(data, x_column, y_column, color_column=None):
    """
    Render a scatter plot between two numeric columns.
    
    Args:
        data: Pandas DataFrame
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        color_column: Optional column name for color coding
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if color_column:
        sns.scatterplot(x=x_column, y=y_column, hue=color_column, data=data, ax=ax)
    else:
        sns.scatterplot(x=x_column, y=y_column, data=data, ax=ax)
    
    ax.set_title(f"{y_column} vs {x_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    st.pyplot(fig)

def render_categorical_plot(data, column_name):
    """
    Render a bar plot for a categorical column.
    
    Args:
        data: Pandas DataFrame
        column_name: Name of the column to visualize
    """
    # Count values
    value_counts = data[column_name].value_counts().sort_values(ascending=False)
    
    # Limit to top 15 categories if there are too many
    if len(value_counts) > 15:
        value_counts = value_counts.head(15)
        
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
    ax.set_title(f"Distribution of {column_name}")
    ax.set_xlabel(column_name)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)