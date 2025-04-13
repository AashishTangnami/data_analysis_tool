"""
CLI commands for the Dynamic Data Analysis Platform.
"""
import typer
from pathlib import Path
from typing import Optional

app = typer.Typer(help="Dynamic Data Analysis Platform CLI")

@app.command()
def upload(
    file_path: Path = typer.Argument(..., help="Path to the data file to upload")
):
    """
    Upload a data file for analysis.
    """
    if not file_path.exists():
        typer.echo(f"Error: File {file_path} does not exist.")
        raise typer.Exit(code=1)
    
    from ..core.data_ingestion import validate_file_format, save_uploaded_file
    
    if not validate_file_format(str(file_path)):
        typer.echo(f"Error: Unsupported file format. Supported formats: CSV, JSON, Excel.")
        raise typer.Exit(code=1)
    
    # Read the file content
    with open(file_path, "rb") as f:
        file_content = f.read()
    
    # Save to the raw data directory
    saved_path = save_uploaded_file(file_content, file_path.name)
    
    typer.echo(f"File uploaded successfully: {saved_path}")
    return saved_path

@app.command()
def analyze(
    file_path: Path = typer.Argument(..., help="Path to the data file to analyze"),
    descriptive: bool = typer.Option(True, help="Run descriptive analysis"),
    diagnostic: bool = typer.Option(False, help="Run diagnostic analysis"),
    predictive: bool = typer.Option(False, help="Run predictive analysis"),
    prescriptive: bool = typer.Option(False, help="Run prescriptive analysis"),
    output: Optional[Path] = typer.Option(None, help="Output directory for results")
):
    """
    Analyze a data file using the specified analysis types.
    """
    if not file_path.exists():
        typer.echo(f"Error: File {file_path} does not exist.")
        raise typer.Exit(code=1)
    
    typer.echo(f"Analyzing file: {file_path}")
    
    analysis_types = []
    if descriptive:
        analysis_types.append("Descriptive")
    if diagnostic:
        analysis_types.append("Diagnostic")
    if predictive:
        analysis_types.append("Predictive")
    if prescriptive:
        analysis_types.append("Prescriptive")
    
    typer.echo(f"Selected analysis types: {', '.join(analysis_types)}")
    
    # Placeholder for actual analysis logic
    typer.echo("Analysis processing would happen here.")
    typer.echo("Analysis completed successfully.")

if __name__ == "__main__":
    app()
