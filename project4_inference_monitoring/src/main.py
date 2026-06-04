import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn

from src.api.routes import router, set_engine
from src.core.inference import InferenceEngine
from src.core.monitor import get_prometheus_metrics
from src.core.tracing import init_tracing
from src.models.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting inference monitoring platform...")

    init_tracing(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host
    )

    engine = InferenceEngine(
        model_path=settings.model_path,
        gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
        max_model_len=settings.vllm_max_model_len
    )
    set_engine(engine)

    logger.info("Inference engine initialized successfully")

    yield

    logger.info("Shutting down inference monitoring platform...")


app = FastAPI(
    title="Enterprise AI Inference & Monitoring Platform",
    description="Production-grade LLM inference service with comprehensive monitoring",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "Enterprise AI Inference & Monitoring Platform",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    prometheus = get_prometheus_metrics()
    metrics_text = prometheus.format_prometheus()
    return Response(content=metrics_text, media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        reload=False
    )
