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

from src.api import router
from src.models import Base, engine

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="多模态RAG智能文档中心",
    description="支持图文混合的企业级智能文档系统",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "多模态RAG智能文档中心 API",
        "version": "1.0.0",
        "docs": "/docs",
        "gradiourl": "http://localhost:7860"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)