import streamlit as st
from src.frontend.pages.upload_page import upload_page
from src.frontend.pages.preprocess_page import preprocess_page
from src.frontend.pages.analysis_page import render_analysis_page
from src.frontend.pages.results_page import view_results_page
from src.frontend.components.sidebar import render_sidebar

# Set page configuration
st.set_page_config(
    page_title="Dynamic Data Analysis Platform",
    page_icon="📊",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables."""
    if "data" not in st.session_state:
        st.session_state.data = None
    if "current_step" not in st.session_state:
        st.session_state.current_step = "upload"
    if "engine" not in st.session_state:
        st.session_state.engine = "pandas"
    if "uploaded_file_info" not in st.session_state:
        st.session_state.uploaded_file_info = None
    if "preprocessed_data" not in st.session_state:
        st.session_state.preprocessed_data = None
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

def main():
    """Main Streamlit application.
  
  
        Upload Data & Ingestion
        1. Data Collection and Integration: 
            Gather data from various sources such as web scraping, APIs, or file imports, and merge/integrate data if needed.
        2. Data Cleaning: 
            Remove duplicates, correct errors (e.g., typos), and handle missing values through imputation or removal.
        3. Data Transformation: 
            Convert data types, aggregate data, encode categorical variables, scale numerical values, and create new features.
        4. Data Storage: 
            Store data in a suitable repository (relational database, NoSQL, or file system) while capturing metadata (source, schema, timestamp) and ensuring security/privacy.
        5. Data Access: 
            Provide users or systems with access to the stored data via APIs, database connections, or direct file retrieval.

        Data Cleaning, Preparation, and Validation
        1. Handling Missing Values: Detect and appropriately impute or remove missing data based on contextual requirements.
        2. Outlier Detection and Treatment: Identify and address outliers based on statistical methods or domain knowledge.
        3. Data Validation: Ensure data meets expected formats, quality standards, and integrity constraints.
        4. Feature Engineering: Generate new features from existing data to better support analysis.
        5. Data Normalization: Standardize data scales to improve consistency for analysis.
        6. Data Splitting: Partition data into training, validation, and testing sets for modeling or further analysis.
        7. Data Aggregation: Summarize data at a higher granularity when necessary.
        8. Data Quality Assessment: Continuously review and improve data quality through systematic checks.

        EDA
        1. Univariate Analysis: Examine individual variables using summary statistics, histograms, and box plots.
        2. Bivariate Analysis: Explore relationships between pairs of variables with scatter plots, correlation coefficients, or heatmaps.
        3. Multivariate Analysis: Analyze interactions among multiple variables using techniques like pair plots or principal component analysis.
        4. Data Visualization: Create diverse visualizations (e.g., bar charts, line graphs, heatmaps) to illustrate insights.
        5. Hypothesis Testing: Use statistical tests to evaluate assumptions and determine the significance of observed relationships.
        6. Data Storytelling: Develop narratives that effectively communicate insights and contextualize the findings.

        Drawing Conclusions and Making Recommendations
        1. Summarize Findings: Consolidate key insights from the analysis into a clear, comprehensive report.
        2. Make Recommendations: Propose actionable steps based on the analytical insights.
        3. Communicate Results: Present findings through compelling visualizations and narratives tailored to stakeholders.
        4. Monitor Results: Track the implementation and impact of the recommendations over time.
        5. Integration into Systems: Embed the insights and recommendations into business processes or operational systems.
        6. Data Versioning and Lineage: Maintain version control and track data provenance to support reproducibility and iterative improvements.
    """
    initialize_session_state()
    st.title("Dynamic Data Analysis Platform")
    
    # # Render sidebar
    # render_sidebar()
    
    # Main content based on current step
    if st.session_state.current_step == "upload":
        upload_page()
    elif st.session_state.current_step == "preprocess":
        preprocess_page()
    elif st.session_state.current_step == "analyze":
        render_analysis_page()
    elif st.session_state.current_step == "visualize":
        view_results_page()

if __name__ == "__main__":
    main()
