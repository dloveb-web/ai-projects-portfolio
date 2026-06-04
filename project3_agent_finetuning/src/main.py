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
from src.database import Base, engine

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="智能体与微调一体化平台",
    description="微调 + Agent 一体化开发平台",
    version="1.0.0"
)

# CORS配置 - 从环境变量读取允许的来源
# 注意：allow_credentials=True 时不能使用通配符 "*"
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:7860,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Agent Finetune Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "gradiourl": "http://localhost:7860"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

