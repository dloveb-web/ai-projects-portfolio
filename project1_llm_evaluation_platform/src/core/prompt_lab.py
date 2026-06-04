import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from sqlalchemy.orm import Session

from .llm_client import LLMClient
from ..database.models import Prompt, ABTest

logger = logging.getLogger(__name__)


class PromptLab:
    def __init__(self, db: Session):
        self.db = db

    def save_prompt(self, name: str, content: str, category: str = "general") -> Prompt:
        prompt = Prompt(name=name, content=content, category=category)
        self.db.add(prompt)
        self.db.commit()
        self.db.refresh(prompt)
        return prompt

    def get_prompts(self, category: Optional[str] = None) -> List[Prompt]:
        query = self.db.query(Prompt)
        if category:
            query = query.filter(Prompt.category == category)
        return query.order_by(Prompt.created_at.desc()).all()

    def update_prompt(self, prompt_id: int, content: str) -> Optional[Prompt]:
        prompt = self.db.query(Prompt).filter(Prompt.id == prompt_id).first()
        if prompt:
            prompt.content = content
            prompt.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(prompt)
        return prompt

    def delete_prompt(self, prompt_id: int) -> bool:
        prompt = self.db.query(Prompt).filter(Prompt.id == prompt_id).first()
        if prompt:
            self.db.delete(prompt)
            self.db.commit()
            return True
        return False

    def ab_test(
        self,
        prompt_a: str,
        prompt_b: str,
        test_prompt: str,
        model: str = "qwen-turbo",
        system_prompt: Optional[str] = None,
        blind_mode: bool = False
    ) -> Dict[str, Any]:
        logger.info("Starting A/B test")

        client_a = LLMClient(model)
        result_a = client_a.chat(test_prompt, system_prompt)
        result_a["prompt"] = prompt_a

        client_b = LLMClient(model)
        result_b = client_b.chat(test_prompt, system_prompt)
        result_b["prompt"] = prompt_b

        test_record = ABTest(
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            test_prompt=test_prompt,
            result_a={"response": result_a.get("content"), "latency": result_a.get("latency_ms")},
            result_b={"response": result_b.get("content"), "latency": result_b.get("latency_ms")},
            winner=None,
        )
        self.db.add(test_record)
        self.db.commit()
        self.db.refresh(test_record)

        response = {
            "test_id": test_record.id,
            "test_prompt": test_prompt,
            "model": model,
            "timestamp": datetime.now().isoformat(),
        }

        if blind_mode:
            response["results"] = {
                "version_a": {"response": result_a.get("content"), "latency_ms": result_a.get("latency_ms")},
                "version_b": {"response": result_b.get("content"), "latency_ms": result_b.get("latency_ms")},
            }
        else:
            response["results"] = {
                "prompt_a": {
                    "prompt": prompt_a,
                    "response": result_a.get("content"),
                    "latency_ms": result_a.get("latency_ms"),
                },
                "prompt_b": {
                    "prompt": prompt_b,
                    "response": result_b.get("content"),
                    "latency_ms": result_b.get("latency_ms"),
                },
            }

        return response

    def batch_ab_test(
        self,
        prompt_a: str,
        prompt_b: str,
        test_prompts: List[str],
        model: str = "qwen-turbo",
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info(f"Starting batch A/B test with {len(test_prompts)} prompts")

        results = []
        for i, test_prompt in enumerate(test_prompts):
            logger.info(f"Testing prompt {i+1}/{len(test_prompts)}")
            result = self.ab_test(prompt_a, prompt_b, test_prompt, model, system_prompt)
            results.append(result)

        total_a_latency = sum(r["results"]["prompt_a"]["latency_ms"] for r in results)
        total_b_latency = sum(r["results"]["prompt_b"]["latency_ms"] for r in results)

        summary = {
            "total_tests": len(test_prompts),
            "model": model,
            "avg_latency_a": round(total_a_latency / len(test_prompts), 2),
            "avg_latency_b": round(total_b_latency / len(test_prompts), 2),
            "faster_version": "A" if total_a_latency < total_b_latency else "B",
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }

        return summary

    def evaluate_prompt(
        self,
        prompt: str,
        model: str,
        evaluation_questions: List[str],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info(f"Evaluating prompt with {len(evaluation_questions)} questions")

        client = LLMClient(model)
        results = []

        for question in evaluation_questions:
            full_prompt = f"{prompt}\n\n问题: {question}"
            response = client.chat(full_prompt, system_prompt)
            results.append({
                "question": question,
                "response": response.get("content", ""),
                "success": response.get("success", False),
                "latency_ms": response.get("latency_ms", 0),
            })

        success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)

        return {
            "prompt": prompt,
            "model": model,
            "total_questions": len(evaluation_questions),
            "success_rate": round(success_rate, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }

    def get_ab_test_history(self, limit: int = 10) -> List[ABTest]:
        return self.db.query(ABTest).order_by(ABTest.created_at.desc()).limit(limit).all()
