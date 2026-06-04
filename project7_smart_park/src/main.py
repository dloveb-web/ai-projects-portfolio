#!/usr/bin/env python3
"""
智慧园区系统主入口
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.api.routes import auth, park, device, access, energy, security


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    yield
    # 关闭时执行


app = FastAPI(
    title="Smart Park API",
    description="智慧园区综合运管平台API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth.router, prefix="/api")
app.include_router(park.router, prefix="/api")
app.include_router(device.router, prefix="/api")
app.include_router(access.router, prefix="/api")
app.include_router(energy.router, prefix="/api")
app.include_router(security.router, prefix="/api")


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "Smart Park API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
