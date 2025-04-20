# Dynamic Data Analysis Platform

***This project is under heavy development and is not yet ready for use.***

## 📊 Overview
A state-of-the-art, comprehensive platform for advanced automated data analysis that supports the complete analytics lifecycle: descriptive, diagnostic, [Future : predictive, and prescriptive analytics]. This powerful tool allows users to upload data in a variety of formats and seamlessly progress through each stage of data analysis with sophisticated algorithms and visualizations.

## 🌟 Features

### Data Ingestion
- **Multiple File Format Support**: Upload and analyze data from various formats:
  - [x] ✅ Excel (xlsx, xls)
  - [x] ✅ CSV
  - [x] ✅ JSON
  - Parquet
  - Avro
  - XML
  - ORC

**Engines**
-   [x] ✅ Pandas
-   [x] ✅ Polars
-   [ ] PySpark

**Preprocessing and Analysis**
- [x] ✅ **Descriptive Analytics**  
  Gain insights into past data with detailed summaries and visualizations.
  
- [ ] 🔄 **Diagnostic Analytics**  
  Identify the root causes of trends and anomalies in your data.

- [ ] 🔄 **Predictive Analytics** *(Coming Soon)*  
  Use machine learning models to forecast future outcomes.

- [ ] 🔄 **Prescriptive Analytics** *(Coming Soon)*  
  Get recommendations for decision-making based on predictive analytics.

- **Data Validation**: Automatic validation and error detection in uploaded data
- **Integration**: Connect directly with databases and APIs

- **Mutiple Engine Support**: Pandas, Polars, PySpark

### Data Processing
- **Automated Data Cleaning**: Smart detection and handling of missing values, outliers, and duplicates
- **Data Transformation**: Feature engineering, normalization, and encoding
- **Data Enrichment**: Merge external data sources for enhanced analysis

### Advanced Analytics Suite
- **Descriptive Analytics**: Understand what happened
  - Statistical summaries
  - Distribution analysis
  - Correlation matrices
  - Pattern identification
  
- **Diagnostic Analytics**: Understand why it happened
  - Root cause analysis
  - Anomaly detection
  - Variance analysis
  - Segmentation
  
- **Predictive Analytics**: Understand what might happen
  - Time series forecasting
  - Regression analysis
  - Classification models
  - Clustering
  
- **Prescriptive Analytics**: Understand what actions to take
  - Optimization algorithms
  - Simulation modeling
  - Decision trees
  - Recommendation engines

### Visualization & Reporting
- **Interactive Dashboards**: Drag-and-drop interface to create custom dashboards
- **Advanced Visualizations**: From simple charts to complex network graphs
- **Report Generation**: Export findings as PDF, PowerPoint, or interactive HTML [Advance Feature]

### Multiple Interfaces
- **Web UI**: User-friendly interface for non-technical users, [POC : Streamlit , Advance Feature : Next.js] 
- **REST API**: Programmatic access for integration with other systems
- **CLI**: Command-line tools for automation and scripting


## Core Objective of the Project.
## 🔍 Step-by-Step Analysis Capabilities

The platform guides users through a structured data analysis workflow:

### 1. Descriptive Analysis
- Data profiling and summary statistics [x]
- Correlation analysis [x] and relationship mapping
- Trend identification and pattern recognition
- Distribution analysis and visualization [x]
- Time-series decomposition (trend, seasonality, residual) [Advance Feature]
- Geospatial analysis for location-based insights [Advance Feature]

### 2. Diagnostic Analysis
- Drill-down capabilities for deeper investigation
- Anomaly detection and outlier analysis
- Comparative analysis across dimensions
- Factor analysis to determine influence weights
- Causal inference techniques (e.g., Granger causality, A/B testing) [Advance Feature]
- Hypothesis testing for statistical validation [Advance Feature]

### 3. Predictive Analysis [Advance Feature]
- Automated model selection and hyperparameter tuning
- Feature importance ranking
- Cross-validation and model evaluation
- Confidence intervals and prediction probabilities
- Ensemble methods (e.g., Random Forest, Gradient Boosting) 
- Deep learning models for unstructured data 
- NLP techniques for text-based predictions

### 4. Prescriptive Analysis [Advance Feature]
- "What-if" scenario modeling
- Optimization for business objectives
- Action recommendation prioritization
- Risk assessment and mitigation strategies  
- Real-time optimization for dynamic decision-making  
- Game theory models for competitive strategy analysis  
- Cost-benefit analysis for prioritizing recommendations 

## 🚀 Installation and Setup

### Prerequisites
- Python 3.13 
- Git

### Using pip with uv (recommended)
```bash
# Install uv if not already installed
pip install uv

# Clone the repository
git clone https://github.com/AashishTangnami/data_analysis_tool.git
cd data_analysis_tool

# Install the package
uv pip install -e .

# For development
uv pip install -e ".[dev]"
```

### Environment Variables
Create a `.env` file in the root directory based on the provided `.env.example`:
```
API_KEY=your_api_key
DATABASE_URL=your_database_url
```

## 📊 Usage Examples

### Command Line Interface (CLI)
```bash
-------
```

### Web Interface
1. Start the web server:
   ```bash
   ------
   ```
2. Open your browser at http://localhost:8000
3. Upload your data file
4. Navigate through the analysis tabs

### API Usage
```python
import requests

# Upload file
files = {'file': open('data.csv', 'rb')}
response = requests.post('http://localhost:8000/api/upload', files=files)
file_id = response.json()['file_id']

# Get descriptive analysis
analysis = requests.get(f'http://localhost:8000/api/analysis/{file_id}/descriptive')
print(analysis.json())
```

## 👨‍💻 Development

### Project Structure
```
data_analysis_tool/
├── src/                 # Source code
│   ├── api/             # FastAPI implementation
│   ├── cli/             # CLI implementation
│   ├── core/            # Core analysis algorithms
│   ├── frontend/        # Streamlit frontend
│   └── utils/           # Utility functions
├── tests/               # Test suite
├── data/                # Sample datasets
├── docs/                # Documentation
└── pyproject.toml       # Project configuration
```

### Running Tests
```bash
pytest
```

### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support
For support and questions, please open an issue on the GitHub repository.
