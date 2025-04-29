from src.frontend.utils.streamlit_helpers import render_data_preview as streamlit_render_data_preview

def render_data_preview(preview_data, summary_data):
    """
    Render a preview of the data and its summary.

    Args:
        preview_data: A list of dictionaries containing the data preview
        summary_data: A dictionary containing the data summary
    """
    # Use the centralized utility function
    streamlit_render_data_preview(preview_data, summary_data)