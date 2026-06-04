from fastapi import APIRouter, HTTPException, Request, Header, Depends
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
import time
import uuid
import subprocess
import json

from src.models.schemas import (
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest,
    ChatCompletionResponse, CompletionResponse, EmbeddingResponse,
    ModelList, ModelInfo, HealthStatus
)
from src.core.inference import InferenceEngine
from src.core.monitor import get_metrics_collector
from src.core.version_manager import get_version_manager
from src.core.tracing import get_tracing_service

router = APIRouter()

_engine: Optional[InferenceEngine] = None

API_KEYS = {"test-api-key": "test-user"}


def set_engine(engine: InferenceEngine):
    global _engine
    _engine = engine


def _check_api_key(authorization: Optional[str] = Header(None)) -> str:
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            api_key = parts[1]
            if api_key in API_KEYS:
                return API_KEYS[api_key]
    raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API key")


async def get_current_user(user_id: str = Depends(_check_api_key)) -> str:
    return user_id


def _detect_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, _: str = Depends(get_current_user)):
    if not _engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    tracing = get_tracing_service()
    trace_id = None

    if tracing:
        trace_id = tracing.start_trace(
            name="chat_completion",
            user_id=request.user,
            metadata={"model": request.model}
        )

    try:
        start_time = time.time()
        response = await _engine.chat_completion(request, trace_id)
        latency = time.time() - start_time

        metrics = get_metrics_collector()
        metrics.record_endpoint_request("/v1/chat/completions", latency, success=True)

        if trace_id:
            tracing.record_generation(
                trace_id,
                model=request.model,
                prompt=str(request.messages),
                completion=str(response.choices[0].message.content),
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                latency=latency
            )
            tracing.end_trace(trace_id, success=True)

        return response

    except Exception as e:
        latency = time.time() - start_time
        metrics = get_metrics_collector()
        metrics.record_endpoint_request("/v1/chat/completions", latency, success=False)

        if trace_id:
            tracing.end_trace(trace_id, success=False, error=str(e))

        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest, _: str = Depends(get_current_user)):
    if not _engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    tracing = get_tracing_service()
    trace_id = tracing.start_trace(name="completion") if tracing else None

    try:
        start_time = time.time()
        response = await _engine.completion(request, trace_id)
        latency = time.time() - start_time

        metrics = get_metrics_collector()
        metrics.record_endpoint_request("/v1/completions", latency, success=True)

        if trace_id:
            tracing.end_trace(trace_id, success=True)

        return response

    except Exception as e:
        if trace_id:
            tracing.end_trace(trace_id, success=False, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(request: EmbeddingRequest, _: str = Depends(get_current_user)):
    if not _engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    try:
        response = await _engine.embedding(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/chat/completions/stream")
async def chat_completions_stream(request: ChatCompletionRequest, _: str = Depends(get_current_user)):
    if not _engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    async def stream_generator():
        try:
            async for chunk in _engine.chat_completion_stream(request):
                yield chunk
        except Exception as e:
            error_chunk = json.dumps({"error": {"message": str(e), "type": "streaming_error"}})
            yield f"data: {error_chunk}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )


@router.get("/v1/models", response_model=ModelList)
async def list_models(_: str = Depends(get_current_user)):
    if not _engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    models = _engine.get_available_models()
    model_list = [
        ModelInfo(
            id=model,
            object="model",
            created=int(time.time()),
            owned_by="enterprise-ai"
        )
        for model in models
    ]

    return ModelList(data=model_list)


@router.get("/health", response_model=HealthStatus)
async def health_check():
    version_manager = get_version_manager()
    active_version = version_manager.get_active_version()

    return HealthStatus(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
        version="1.0.0",
        models=[active_version["model_name"] if active_version else "unknown"],
        gpu_available=_detect_gpu()
    )


@router.get("/metrics")
async def get_metrics(_: str = Depends(get_current_user)):
    metrics = get_metrics_collector()
    return metrics.get_metrics()


@router.get("/api/versions")
async def get_versions(_: str = Depends(get_current_user)):
    version_manager = get_version_manager()
    return {
        "versions": version_manager.get_all_versions(),
        "active_version": version_manager.get_active_version(),
        "traffic_distribution": version_manager.get_traffic_distribution()
    }


@router.post("/api/versions/grayscale/enable")
async def enable_grayscale(percentage: int = 10, _: str = Depends(get_current_user)):
    version_manager = get_version_manager()
    config = version_manager.enable_grayscale(percentage)
    return {"status": "success", "config": config}


@router.post("/api/versions/grayscale/disable")
async def disable_grayscale(_: str = Depends(get_current_user)):
    version_manager = get_version_manager()
    version_manager.disable_grayscale()
    return {"status": "success"}


@router.post("/api/versions/{version}/activate")
async def activate_version(version: str, _: str = Depends(get_current_user)):
    version_manager = get_version_manager()
    version_manager.activate_version(version)
    return {"status": "success", "version": version}


@router.post("/api/versions/rollback")
async def rollback_version(_: str = Depends(get_current_user)):
    version_manager = get_version_manager()
    success = version_manager.rollback_version()
    return {"status": "success" if success else "failed"}
