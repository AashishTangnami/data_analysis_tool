from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routers import upload, ingestion, preprocessing, transformation
from src.api.middleware import TimingMiddleware

app = FastAPI(title="Data Analysis Tool API", 
              version="1.0", 
              description="API for processing and analyzing data files using different engine strategies.")

# Adding middleware for request timing
app.add_middleware(TimingMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingestion.router, prefix="/api/v1/ingestion", tags=["ingestion"])
app.include_router(preprocessing.router, prefix="/api/v1/preprocessing", tags=["preprocessing"])
app.include_router(transformation.router, prefix="/api/v1/transformation", tags=["transformation"])

def start_api():
    """Start the FastAPI server"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start_api()

