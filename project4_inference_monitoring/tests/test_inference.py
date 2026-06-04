import pytest
import asyncio
import sys
from unittest.mock import Mock, patch
from src.models.schemas import ChatCompletionRequest, Message, MessageRole


@pytest.fixture
def mock_vllm():
    """Mock vLLM to avoid actual model loading"""
    with patch('src.core.inference.VLLM_AVAILABLE', False):
        yield


@pytest.fixture
def engine(mock_vllm):
    from src.core.inference import InferenceEngine
    return InferenceEngine(
        model_path="/models/test-model",
        gpu_memory_utilization=0.5,
        max_model_len=2048
    )


@pytest.mark.asyncio
async def test_chat_completion(engine):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            Message(role=MessageRole.USER, content="Hello, how are you?")
        ],
        temperature=0.7,
        max_tokens=100
    )

    response = await engine.chat_completion(request)

    assert response.id is not None
    assert response.object == "chat.completion"
    assert len(response.choices) > 0
    assert response.choices[0].message.role == "assistant"
    assert response.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_chat_completion_with_system_message(engine):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            Message(role=MessageRole.SYSTEM, content="You are a pirate."),
            Message(role=MessageRole.USER, content="Hello!")
        ],
        max_tokens=50
    )

    response = await engine.chat_completion(request)

    assert response.choices[0].message.content is not None


@pytest.mark.asyncio
async def test_metrics_collection(engine):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            Message(role=MessageRole.USER, content="Test")
        ]
    )

    await engine.chat_completion(request)

    metrics = engine.get_metrics()

    assert metrics["total_requests"] >= 1
    assert "p50_latency" in metrics
    assert "p95_latency" in metrics
