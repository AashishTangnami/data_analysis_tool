
from fastapi import APIRouter, Depends, Query
from src.api.dependencies import get_engine
from src.core.engine_context import EngineContext

router = APIRouter()

@router.get("/analyze")
def analyze_file(
    filename: str = Query(...),
    engine: EngineContext = Depends(get_engine)
):
    file_path = f"data/raw/{filename}"
    df = engine.load_data(file_path)
    result = engine.describe_data()
    return {"status": "success", "analysis": result}
