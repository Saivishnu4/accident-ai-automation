"""
FastAPI application for AI-Driven Accident Reporting & FIR Automation
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
import uvicorn
from typing import List, Optional
import logging
from datetime import datetime

from ..models.yolo_detector import YOLODetector
from ..models.ocr_engine import OCREngine
from ..models.damage_classifier import DamageClassifier
from ..utils.config import settings
from ..utils.logger import setup_logger
from .schemas.requests import AnalyzeRequest, FIRGenerateRequest
from .schemas.responses import AnalyzeResponse, FIRResponse
from .routes import analysis, fir, health

# Setup logging
logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Accident FIR Automation API",
    description="AI-powered accident detection and FIR automation system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
app.include_router(fir.router, prefix="/api/v1", tags=["fir"])


@app.on_event("startup")
async def startup_event():
    """Initialize models and resources on startup"""
    logger.info("Starting Accident FIR Automation API...")
    
    # Initialize ML models
    try:
        app.state.yolo_detector = YOLODetector(
            model_path=settings.YOLO_MODEL_PATH,
            confidence_threshold=settings.YOLO_CONFIDENCE
        )
        app.state.ocr_engine = OCREngine(
            use_gpu=settings.USE_GPU
        )
        app.state.damage_classifier = DamageClassifier(
            model_path=settings.DAMAGE_MODEL_PATH
        )
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise
    
    # Initialize database connection
    # app.state.db = await init_db()
    
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("Shutting down Accident FIR Automation API...")
    # Cleanup code here
    logger.info("Shutdown complete")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Accident FIR Automation API",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )
