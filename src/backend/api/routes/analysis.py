from typing import Dict, Any, List
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from core.context import EngineContext
from api.models.responses import AnalysisResponse
from src.shared.constants import EngineType, AnalysisType
from src.shared.session import SessionManager

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Get the session manager instance from ingestion
from api.routes.ingestion import session_manager

class AnalysisRequest(BaseModel):
    """Request model for analysis operations"""
    file_id: str = Field(..., description="ID of the file to analyze")
    analysis_type: str = Field(..., description="Type of analysis to perform (descriptive, diagnostic, predictive, prescriptive)")
    params: Dict[str, Any] = Field({}, description="Parameters for the analysis")
    use_preprocessed: bool = Field(False, description="Whether to use preprocessed data if available")

    @field_validator('analysis_type')
    def validate_analysis_type(cls, v):
        """Validate that the analysis type is supported."""
        try:
            return AnalysisType(v).value
        except ValueError:
            supported_types = [t.value for t in AnalysisType]
            raise ValueError(f"Unsupported analysis type: {v}. Supported types: {', '.join(supported_types)}")

    @field_validator('file_id')
    def validate_file_id(cls, v):
        """Validate that the file ID has a valid engine type."""
        try:
            parts = v.split('_', 1)
            if len(parts) < 2:
                raise ValueError("File ID must be in format 'engine_filename'")

            engine_type = parts[0]
            EngineType(engine_type)  # Will raise ValueError if invalid
            return v
        except ValueError as e:
            supported_engines = [t.value for t in EngineType]
            raise ValueError(f"Invalid file ID format or unsupported engine type. Supported engines: {', '.join(supported_engines)}")

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest):
    """
    Analyze data according to specified analysis type.

    Args:
        request: AnalysisRequest with file_id, analysis_type, and params

    Returns:
        AnalysisResponse with analysis results
    """
    # Check if data exists in session
    data = None

    # Use preprocessed data if requested and available
    if request.use_preprocessed:
        data = await session_manager.get_preprocessed_data(request.file_id)
        if data is not None:
            logger.info(f"Using preprocessed data for file_id: {request.file_id}")

    # If no preprocessed data or not requested, use original data
    if data is None:
        data = await session_manager.get_data(request.file_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"File not found with ID: {request.file_id}")

    try:
        logger.info(f"Analyzing data for file_id: {request.file_id}, analysis type: {request.analysis_type}")

        # Get engine type from file_id
        engine_type = request.file_id.split("_")[0]

        # Initialize engine context
        engine_context = EngineContext(engine_type)

        # Perform analysis
        analysis_results = engine_context.analyze_data(
            data,
            request.analysis_type,
            request.params
        )

        # Generate visualization data based on analysis type and results
        visualizations = generate_visualizations(request.analysis_type, analysis_results)

        logger.info(f"Analysis completed successfully for file_id: {request.file_id}")

        return AnalysisResponse(
            file_id=request.file_id,
            analysis_type=request.analysis_type,
            results=analysis_results,
            visualizations=visualizations,
            message=f"{request.analysis_type.capitalize()} analysis completed successfully"
        )

    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def generate_visualizations(analysis_type: str, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate visualization configurations based on analysis type and results.

    Args:
        analysis_type: Type of analysis performed
        analysis_results: Results from the analysis

    Returns:
        List of visualization configurations
    """
    visualizations = []

    if analysis_type == "descriptive":
        # Add visualizations for descriptive analysis
        if "numeric_analysis" in analysis_results and "statistics" in analysis_results["numeric_analysis"]:
            stats = analysis_results["numeric_analysis"]["statistics"]
            # Add histogram for each numeric column
            for col in stats:
                visualizations.append({
                    "type": "histogram",
                    "title": f"Distribution of {col}",
                    "column": col
                })

        # Add correlation heatmap if correlations exist
        if "correlations" in analysis_results and analysis_results["correlations"]:
            visualizations.append({
                "type": "correlation_heatmap",
                "title": "Correlation Heatmap",
                "data": analysis_results["correlations"]
            })

    elif analysis_type == "diagnostic":
        # Add visualizations for diagnostic analysis
        if "feature_importance" in analysis_results:
            visualizations.append({
                "type": "bar_chart",
                "title": "Feature Importance",
                "data": analysis_results["feature_importance"]
            })

        if "outlier_detection" in analysis_results:
            visualizations.append({
                "type": "scatter_plot",
                "title": "Outlier Detection",
                "data": analysis_results["outlier_detection"]
            })

        if "correlation_analysis" in analysis_results:
            visualizations.append({
                "type": "bar_chart",
                "title": "Correlation with Target",
                "data": analysis_results["correlation_analysis"]
            })

    elif analysis_type == "predictive":
        # Add visualizations for predictive analysis
        if "model_performance" in analysis_results:
            visualizations.append({
                "type": "metrics_table",
                "title": "Model Performance Metrics",
                "data": analysis_results["model_performance"]
            })

        if "feature_importance" in analysis_results:
            visualizations.append({
                "type": "bar_chart",
                "title": "Feature Importance",
                "data": analysis_results["feature_importance"]
            })

        if "predictions" in analysis_results and analysis_results["predictions"]:
            actual_values = [pred["actual"] for pred in analysis_results["predictions"]]
            predicted_values = [pred["predicted"] for pred in analysis_results["predictions"]]

            visualizations.append({
                "type": "scatter_plot",
                "title": "Predictions vs Actual Values",
                "data": {
                    "actual": actual_values,
                    "predicted": predicted_values
                }
            })

    elif analysis_type == "prescriptive":
        # Add visualizations for prescriptive analysis
        if "optimization_results" in analysis_results:
            visualizations.append({
                "type": "bar_chart",
                "title": "Optimal Variable Values",
                "data": analysis_results["optimization_results"].get("optimal_values", {})
            })

        if "scenario_comparison" in analysis_results:
            visualizations.append({
                "type": "radar_chart",
                "title": "Scenario Comparison",
                "data": analysis_results["scenario_comparison"]
            })

        if "sensitivity_analysis" in analysis_results:
            visualizations.append({
                "type": "heat_map",
                "title": "Sensitivity Analysis",
                "data": analysis_results["sensitivity_analysis"]
            })

    return visualizations

