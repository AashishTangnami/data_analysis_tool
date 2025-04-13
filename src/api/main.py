"""
FastAPI main application module.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from typing import List

from ..config import API_HOST, API_PORT, API_WORKERS
from ..core.data_ingestion import validate_file_format, save_uploaded_file, load_data

app = FastAPI(
    title="Dynamic Data Analysis Platform API",
    description="API for uploading and analyzing data with descriptive, diagnostic, predictive, and prescriptive analytics.",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def start_api():
    """Start the FastAPI application with uvicorn."""
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        workers=1  # Changed from API_WORKERS to 1
    )

if __name__ == "__main__":
    start_api()
