from fastapi import APIRouter, UploadFile, File, Depends
import uuid
import os
from src.core.engine_context import EngineContext
from src.api.dependencies import get_engine
router = APIRouter()



# Determine the base directory (parent of 'src')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the path to the 'data/uploaded_files' directory
UPLOAD_DIR = os.path.join(BASE_DIR, 'data', 'uploaded_files')

# Create the directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    engine: EngineContext = Depends(get_engine),
    handle_missing: bool = True,
    remove_duplicates: bool = True
):
    try:
        # Read file contents
        file_contents = await file.read()
        
        # Generate a unique file ID and save the file
        file_id = str(uuid.uuid4())
        save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        
        with open(save_path, "wb") as f:
            f.write(file_contents)

        # Load and process the data using the provided engine
        df = engine.load_data(save_path)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "message": "File processed successfully"
        }
    except Exception as e:
        return {"error": str(e)}
