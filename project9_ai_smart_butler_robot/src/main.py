"""
AI Smart Butler Robot - Main Entry Point
"""
import uvicorn
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging

from src.config import settings
from src.database import Base, engine

load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI智能管家机器人",
    description="极轻量级大模型本地部署的可移动AI家庭机器人",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers (to be implemented)
# from src.api import router
# app.include_router(router)


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "AI智能管家机器人 API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "local_llm": settings.USE_LOCAL_LLM,
        "personality": settings.DEFAULT_PERSONALITY
    }


if __name__ == "__main__":
    logger.info("Starting AI Smart Butler Robot...")
    logger.info(f"API URL: {settings.API_URL}")
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )

