"""
Main entry point for the Dynamic Data Analysis Platform.
"""
import typer
import sys
import multiprocessing
from typing import Optional
import subprocess
import os
import time
import requests
from requests.exceptions import ConnectionError

app = typer.Typer(help="Dynamic Data Analysis Platform")

def wait_for_api(url: str, timeout: int = 30, interval: float = 0.5):
    """Wait for API to become available."""
    start_time = time.time()
    while True:
        try:
            requests.get(url)
            return True
        except ConnectionError:
            if time.time() - start_time > timeout:
                raise TimeoutError("API server failed to start")
            time.sleep(interval)

def run_api():
    """Start the FastAPI backend server."""
    from src.api.main import start_api
    start_api()

def run_frontend():
    """Start the Streamlit frontend server."""
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", "src/frontend/app.py"]
    stcli.main()

def run_cli():
    """Start the CLI interface."""
    from src.cli.commands import app as cli_app
    cli_app()

@app.command()
def start(
    component: str = typer.Argument(
        "all",
        help="Component to run: 'api', 'frontend', 'cli', or 'all'"
    ),
):
    """
    Start the Dynamic Data Analysis Platform components.
    """
    if component == "api":
        typer.echo("Starting API server...")
        run_api()
    elif component == "frontend":
        typer.echo("Starting frontend server...")
        run_frontend()
    elif component == "cli":
        typer.echo("Starting CLI interface...")
        run_cli()
    elif component == "all":
        typer.echo("Starting both API and frontend servers...")
        # Start API in a separate process using multiprocessing
        api_process = multiprocessing.Process(target=run_api)
        api_process.start()
        
        try:
            typer.echo("Waiting for API to start...")
            wait_for_api("http://localhost:8000")
            typer.echo("API is ready!")
            
            # Start frontend in the main process
            run_frontend()
        finally:
            # Ensure API process is terminated when the frontend exits
            api_process.terminate()
            api_process.join()
    else:
        typer.echo(f"Invalid component: {component}")
        raise typer.Exit(1)

@app.command()
def version():
    """Display the current version of the platform."""
    typer.echo("Dynamic Data Analysis Platform v0.1.0")

if __name__ == "__main__":
    app()
