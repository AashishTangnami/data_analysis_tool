from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/transformation", tags=["transformation"])



@router.post("/{filename}")
async def transform_data(filename: str):
    """Simple transformation endpoint"""
    return {"status": "ok", "filename": filename}
