from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import ingestion, preprocessing, transformation, analysis  

app = FastAPI(
    title="Dynamic Data Analysis Tool API",
    description="API for dynamic data analysis with multiple engines",
    version="1.0.0"
)

# Add CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(ingestion.router, prefix="/api/ingestion", tags=["ingestion"])
app.include_router(preprocessing.router, prefix="/api/preprocessing", tags=["preprocessing"])
app.include_router(transformation.router, prefix="/api/transformation", tags=["transformation"])  # Added transformation router
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])

@app.get("/")
async def root():
    return {"message": "Welcome to Dynamic Data Analysis Tool API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)