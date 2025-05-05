import streamlit as st
import plotly.graph_objects as go
from typing import Optional, List

def render_gauge_chart(value: float, title: str = "Gauge Chart", 
                      min_value: float = 0, max_value: float = 100, 
                      thresholds: Optional[List[float]] = None,
                      colors: Optional[List[str]] = None,
                      show_value: bool = True) -> None:
    """
    Render a gauge chart with the given value.
    
    Args:
        value: The value to display on the gauge
        title: Title of the gauge chart
        min_value: Minimum value on the gauge
        max_value: Maximum value on the gauge
        thresholds: List of threshold values for color changes (e.g., [33, 66] for a 3-color gauge)
        colors: List of colors for each threshold range (e.g., ["red", "yellow", "green"])
        show_value: Whether to show the numeric value on the gauge
    """
    # Validate inputs
    if value < min_value or value > max_value:
        st.warning(f"Value {value} is outside the range [{min_value}, {max_value}]")
        value = max(min_value, min(value, max_value))  # Clamp value to range
    
    # Set default thresholds and colors if not provided
    if thresholds is None:
        thresholds = [min_value + (max_value - min_value) / 3, min_value + 2 * (max_value - min_value) / 3]
    
    if colors is None:
        colors = ["#FF4B4B", "#FFA500", "#00B050"]  # Red, Yellow, Green
    
    # Ensure we have one more color than thresholds
    if len(colors) != len(thresholds) + 1:
        st.warning(f"Number of colors ({len(colors)}) should be one more than number of thresholds ({len(thresholds)})")
        # Adjust colors to match thresholds
        if len(colors) > len(thresholds) + 1:
            colors = colors[:len(thresholds) + 1]
        else:
            colors = colors + ["#00B050"] * (len(thresholds) + 1 - len(colors))
    
    # Create a gauge chart using Plotly
    # Calculate the position on the gauge
    position = (value - min_value) / (max_value - min_value) * 100
    
    # Create the gauge chart
    fig = go.Figure()
    
    # Add the gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number" if show_value else "gauge",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [min_value, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_value, thresholds[0]], 'color': colors[0]},
                {'range': [thresholds[0], thresholds[-1]], 'color': colors[1]},
                {'range': [thresholds[-1], max_value], 'color': colors[2]}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    # Update layout for a cleaner look
    fig.update_layout(
        height=300,
        width=500,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
