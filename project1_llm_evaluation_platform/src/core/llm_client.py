import os
import time
import json
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    import dashscope
    from dashscope import Generation
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if dashscope_api_key:
        dashscope.api_key = dashscope_api_key
except ImportError:
    dashscope = None
    logger.warning("dashscope not installed, Qwen models will not be available")

try:
    import qianfan
    qianfan_access_key = os.getenv("QIANFAN_ACCESS_KEY")
    qianfan_secret_key = os.getenv("QIANFAN_SECRET_KEY")
    if qianfan_access_key and qianfan_secret_key:
        qianfan.AK(qianfan_access_key)
        qianfan.SK(qianfan_secret_key)
except ImportError:
    qianfan = None
    logger.warning("qianfan not installed, ERNIE models will not be available")

try:
    import zhipuai
    zhipuai_api_key = os.getenv("ZHIPUAI_API_KEY")
    if zhipuai_api_key:
        zhipuai.api_key = zhipuai_api_key
except ImportError:
    zhipuai = None
    logger.warning("zhipuai not installed, GLM models will not be available")


SUPPORTED_MODELS = {
    "qwen-turbo": {"provider": "qwen", "name": "通义千问Turbo", "cost_per_1k": 0.002},
    "qwen-plus": {"provider": "qwen", "name": "通义千问Plus", "cost_per_1k": 0.02},
    "qwen-max": {"provider": "qwen", "name": "通义千问Max", "cost_per_1k": 0.2},
    "ernie-bot": {"provider": "ernie", "name": "文心一言", "cost_per_1k": 0.012},
    "ernie-bot-turbo": {"provider": "ernie", "name": "文心一言Turbo", "cost_per_1k": 0.008},
    "glm-4": {"provider": "glm", "name": "智谱GLM-4", "cost_per_1k": 0.1},
    "glm-4-flash": {"provider": "glm", "name": "智谱GLM-4-Flash", "cost_per_1k": 0.001},
}


class LLMClient:
    def __init__(self, model: str = "qwen-turbo"):
        self.model = model
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}")
        self.model_info = SUPPORTED_MODELS[model]
        self.total_cost = 0.0
        self.total_tokens = 0
        self.call_count = 0

    def chat(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> Dict[str, Any]:
        start_time = time.time()
        try:
            response = self._call_model(prompt, system_prompt, temperature)
            latency_ms = (time.time() - start_time) * 1000

            tokens = len(prompt.split()) + len(response.get("content", "").split())
            cost = (tokens / 1000) * self.model_info["cost_per_1k"]

            self.total_cost += cost
            self.total_tokens += tokens
            self.call_count += 1

            return {
                "success": True,
                "content": response.get("content", ""),
                "model": self.model,
                "latency_ms": latency_ms,
                "tokens": tokens,
                "cost": cost,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error calling {self.model}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model,
                "latency_ms": (time.time() - start_time) * 1000,
                "timestamp": datetime.now().isoformat(),
            }

    def _call_model(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> Dict:
        provider = self.model_info["provider"]

        if provider == "qwen" and dashscope:
            return self._call_qwen(prompt, system_prompt, temperature)
        elif provider == "ernie" and qianfan:
            return self._call_ernie(prompt, system_prompt, temperature)
        elif provider == "glm" and zhipuai:
            return self._call_glm(prompt, system_prompt, temperature)
        else:
            raise RuntimeError(f"Provider {provider} not available or not configured")

    def _call_qwen(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> Dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = Generation.call(
            model=self.model,
            messages=messages,
            temperature=temperature,
            result_format="message",
        )

        if response.status_code == 200:
            return {"content": response.output.choices[0].message.content}
        else:
            raise RuntimeError(f"Qwen API error: {response.message}")

    def _call_ernie(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> Dict:
        chat_comp = qianfan.ChatCompletion()

        messages = []
        if system_prompt:
            messages.append({"role": "user", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = chat_comp.do(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )

        return {"content": response.body["result"]}

    def _call_glm(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> Dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = zhipuai.model_api.invoke(
            model=self.model,
            prompt=messages,
            temperature=temperature,
        )

        if response.get("code") == 200:
            return {"content": response["data"]["choices"]["text"][0]["content"]}
        else:
            raise RuntimeError(f"GLM API error: {response.get('msg')}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "total_cost": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "avg_cost_per_call": round(self.total_cost / self.call_count, 6) if self.call_count > 0 else 0,
        }

    def reset_stats(self):
        self.total_cost = 0.0
        self.total_tokens = 0
        self.call_count = 0


_llm_client_instance = None


def get_llm_client(model: str = "qwen-turbo") -> LLMClient:
    global _llm_client_instance
    if _llm_client_instance is None or _llm_client_instance.model != model:
        _llm_client_instance = LLMClient(model)
    return _llm_client_instance


def list_available_models() -> List[Dict[str, str]]:
    available = []
    for model_id, info in SUPPORTED_MODELS.items():
        if info["provider"] == "qwen" and dashscope and dashscope_api_key:
            available.append({"id": model_id, "name": info["name"], "provider": "通义千问"})
        elif info["provider"] == "ernie" and qianfan and qianfan_access_key:
            available.append({"id": model_id, "name": info["name"], "provider": "文心一言"})
        elif info["provider"] == "glm" and zhipuai and zhipuai_api_key:
            available.append({"id": model_id, "name": info["name"], "provider": "智谱GLM"})
    return available


def list_all_models() -> List[Dict[str, str]]:
    return [{"id": model_id, "name": info["name"], "provider": info["provider"]}
            for model_id, info in SUPPORTED_MODELS.items()]
