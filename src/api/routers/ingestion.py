import uuid
from typing import Optional, Dict
from pathlib import Path
from src.api.dependencies import get_engine
from src.core.engine_context import EngineContext
from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Depends
from pydantic import BaseModel

router = APIRouter(prefix="/ingestion", tags=["ingestion"])

# Define constants for file paths
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    status: str
    rows: int
    columns: Optional[Dict]  = None

@router.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    engine: EngineContext = Depends(get_engine)
):
    """
    Upload and ingest a data file.
    The engine type can be specified using the X-Engine-Type header.
    """
    try:
        # Generate a unique file ID
        file_id = str(uuid.uuid4())
        
        # Create a secure filename with file_id prefix
        secure_filename = f"{file_id}_{file.filename}"
        file_path = RAW_DATA_DIR / secure_filename  # Save to RAW_DATA_DIR instead
        
        # Read and write file safely
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        try:
            # Load data using the selected engine
            df = engine.load_data(str(file_path))
            
            return UploadResponse(
                file_id=file_id,
                filename=secure_filename,
                status="success",
                rows=len(df) if hasattr(df, '__len__') else df.height,
                columns=list(df.columns) if hasattr(df, 'columns') else []
            )
            
        except Exception as e:
            # If there's an error loading the data, clean up the file and raise
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"Error loading file: {str(e)}"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
