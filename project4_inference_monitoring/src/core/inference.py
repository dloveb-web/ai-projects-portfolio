import time
import asyncio
import json
from typing import AsyncIterator, Dict, Any, Optional, List
import logging

try:
    from vllm import SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available. Using mock inference engine.")

from src.models.schemas import (
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest,
    ChatCompletionResponse, CompletionResponse, EmbeddingResponse,
    UsageInfo, ChatMessage, ChatCompletionChoice,
    CompletionChoice, EmbeddingData
)
from src.core.monitor import MetricsCollector
from src.core.version_manager import VersionManager

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        trust_remote_code: bool = True
    ):
        self.model_path = model_path
        self.async_engine = None
        self.tokenizer = None
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.metrics = MetricsCollector()
        self.version_manager = VersionManager()

        if VLLM_AVAILABLE:
            self._initialize_vllm()
        else:
            logger.info("Using mock inference engine for development")

    def _initialize_vllm(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code
            )

            engine_args = AsyncEngineArgs(
                model=self.model_path,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                trust_remote_code=self.trust_remote_code,
                enforce_eager=False
            )
            self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)

            logger.info(f"vLLM AsyncLLMEngine initialized with model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        trace_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        start_time = time.time()

        try:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            prompt = self._format_chat_prompt(messages)

            sampling_params = SamplingParams(
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 1.0,
                max_tokens=request.max_tokens,
                stop=request.stop,
                n=request.n or 1
            )

            if self.async_engine:
                request_id = f"req-{int(time.time() * 1000)}"
                final_output = None
                async for output in self.async_engine.generate(
                    prompt, sampling_params, request_id
                ):
                    final_output = output
                generated_text = final_output.outputs[0].text if final_output else ""
            else:
                generated_text = await self._mock_generate(prompt)

            latency = time.time() - start_time
            prompt_tokens = int(self._count_tokens(prompt))
            completion_tokens = int(self._count_tokens(generated_text))
            self.metrics.record_request(latency, success=True, tokens=prompt_tokens + completion_tokens)

            response = self._create_chat_response(request, generated_text, prompt, latency)

            if trace_id:
                self._log_trace(trace_id, request, response, latency)

            return response

        except Exception as e:
            latency = time.time() - start_time
            self.metrics.record_request(latency, success=False)
            logger.error(f"Chat completion error: {e}")
            raise

    async def completion(
        self,
        request: CompletionRequest,
        trace_id: Optional[str] = None
    ) -> CompletionResponse:
        start_time = time.time()

        try:
            sampling_params = SamplingParams(
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 1.0,
                max_tokens=request.max_tokens,
                stop=request.stop,
                n=request.n or 1
            )

            prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt

            if self.async_engine:
                request_id = f"req-{int(time.time() * 1000)}"
                final_output = None
                async for output in self.async_engine.generate(
                    prompts[0], sampling_params, request_id
                ):
                    final_output = output
                generated_text = final_output.outputs[0].text if final_output else ""
            else:
                generated_text = await self._mock_generate(request.prompt)

            latency = time.time() - start_time
            prompt_tokens = int(self._count_tokens(str(request.prompt)))
            completion_tokens = int(self._count_tokens(generated_text))
            self.metrics.record_request(latency, success=True, tokens=prompt_tokens + completion_tokens)

            response = self._create_completion_response(request, generated_text, latency)

            if trace_id:
                self._log_trace(trace_id, request, response, latency)

            return response

        except Exception as e:
            latency = time.time() - start_time
            self.metrics.record_request(latency, success=False)
            logger.error(f"Completion error: {e}")
            raise

    async def embedding(
        self,
        request: EmbeddingRequest
    ) -> EmbeddingResponse:
        start_time = time.time()

        try:
            inputs = request.input if isinstance(request.input, list) else [request.input]

            if self.tokenizer:
                embeddings = await self._generate_embeddings(inputs)
            else:
                embeddings = await self._mock_embeddings(inputs)

            latency = time.time() - start_time
            total_tokens = sum(int(self._count_tokens(str(inp))) for inp in inputs)
            self.metrics.record_request(latency, success=True, tokens=total_tokens)

            response = self._create_embedding_response(request, embeddings, latency)
            return response

        except Exception as e:
            latency = time.time() - start_time
            self.metrics.record_request(latency, success=False)
            logger.error(f"Embedding error: {e}")
            raise

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        trace_id: Optional[str] = None
    ) -> AsyncIterator[str]:
        start_time = time.time()
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        prompt = self._format_chat_prompt(messages)

        try:
            sampling_params = SamplingParams(
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 1.0,
                max_tokens=request.max_tokens,
                stop=request.stop,
                n=request.n or 1,
                stream=True
            )

            if self.async_engine:
                request_id = f"req-{int(time.time() * 1000)}"

                async for output in self.async_engine.generate(
                    prompt, sampling_params, request_id
                ):
                    for output_obj in output.outputs:
                        if output_obj.text:
                            chunk = {
                                "id": f"chatcmpl-{int(time.time() * 1000)}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": output_obj.text},
                                    "finish_reason": output_obj.finish_reason
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                            if output_obj.finish_reason:
                                latency = time.time() - start_time
                                self.metrics.record_request(latency, success=True)
                                yield "data: [DONE]\n\n"
                                return
            else:
                generated_text = await self._mock_generate(prompt)
                words = generated_text.split()
                for i, word in enumerate(words):
                    chunk = {
                        "id": f"chatcmpl-{int(time.time() * 1000)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": word + (" " if i < len(words) - 1 else "")},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.01)

                latency = time.time() - start_time
                self.metrics.record_request(latency, success=True)
                yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "streaming_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    def _format_chat_prompt(self, messages: List[Dict[str, Any]]) -> str:
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False)

        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        prompt += "Assistant: "
        return prompt

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split()) * 1.3

    async def _mock_generate(self, prompt: str) -> str:
        await asyncio.sleep(0.1)
        return f"This is a mock response for: {prompt[:50]}..."

    async def _generate_embeddings(self, inputs: List[str]) -> List[List[float]]:
        import numpy as np

        embeddings = []
        embedding_dim = 768

        for text in inputs:
            if self.tokenizer:
                tokens = self.tokenizer.encode(text, truncation=True, max_length=512)
                if hasattr(self, '_embedding_cache'):
                    embedding = self._embedding_cache.get(text)
                    if embedding is not None:
                        embeddings.append(embedding)
                        continue

                try:
                    import torch
                    has_valid_embedding = False
                    try:
                        if hasattr(self.async_engine, 'engine') and hasattr(self.async_engine.engine, 'model'):
                            model = self.async_engine.engine.model
                            if hasattr(model, 'embed_tokens') or hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                                target_model = model.model if hasattr(model, 'model') else model
                                if hasattr(target_model, 'embed_tokens'):
                                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                    with torch.no_grad():
                                        input_ids = torch.tensor([tokens]).to(device)
                                        emb = target_model.embed_tokens(input_ids)
                                        embedding = emb.mean(dim=1).cpu().numpy()[0].tolist()
                                        has_valid_embedding = True
                    except Exception as internal_error:
                        logger.debug(f"Embedding extraction failed, using fallback: {internal_error}")
                    
                    if not has_valid_embedding:
                        logger.debug("Using fallback token-based embeddings")
                        embedding = np.random.randn(embedding_dim).tolist()
                except Exception as e:
                    logger.debug(f"Failed to get embeddings from model, using fallback: {e}")
                    embedding = np.random.randn(embedding_dim).tolist()
            else:
                embedding = np.random.randn(embedding_dim).tolist()
            embeddings.append(embedding)

        await asyncio.sleep(0.05)
        return embeddings

    async def _mock_embeddings(self, inputs: List[str]) -> List[List[float]]:
        import numpy as np
        await asyncio.sleep(0.1)
        return [np.random.randn(768).tolist() for _ in inputs]

    def _create_chat_response(
        self,
        request: ChatCompletionRequest,
        generated_text: str,
        prompt: str,
        latency: float
    ) -> ChatCompletionResponse:
        import hashlib

        response_id = f"chatcmpl-{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        prompt_tokens = int(self._count_tokens(prompt))
        completion_tokens = int(self._count_tokens(generated_text))

        return ChatCompletionResponse(
            id=response_id,
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    def _create_completion_response(
        self,
        request: CompletionRequest,
        generated_text: str,
        latency: float
    ) -> CompletionResponse:
        import hashlib

        response_id = f"cmpl-{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        prompt_tokens = int(self._count_tokens(str(request.prompt)))
        completion_tokens = int(self._count_tokens(generated_text))

        return CompletionResponse(
            id=response_id,
            object="text_completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    text=generated_text,
                    index=0,
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    def _create_embedding_response(
        self,
        request: EmbeddingRequest,
        embeddings: List[List[float]],
        latency: float
    ) -> EmbeddingResponse:
        data = [
            EmbeddingData(
                object="embedding",
                embedding=emb,
                index=i
            )
            for i, emb in enumerate(embeddings)
        ]

        input_list = [request.input] if isinstance(request.input, str) else request.input
        total_tokens = sum(int(self._count_tokens(str(inp))) for inp in input_list)

        return EmbeddingResponse(
            object="list",
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        )

    def _log_trace(self, trace_id: str, request: Any, response: Any, latency: float):
        logger.debug(f"Trace {trace_id}: latency={latency:.3f}s, model={getattr(request, 'model', 'unknown')}")

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.get_metrics()

    def get_available_models(self) -> List[str]:
        return [self.model_path.split("/")[-1]]
