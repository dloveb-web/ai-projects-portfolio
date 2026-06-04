import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    method: str = "qlora"
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    output_dir: str = "./data/finetuned_models"
    learning_rate: float = 2e-4
    batch_size: int = 4
    epochs: int = 3
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 512


class FinetuneManager:
    def __init__(self):
        self.tasks: Dict[int, Dict[str, Any]] = {}
        self.output_base_path = Path("./data/finetuned_models")
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        
        # 检查是否在模拟模式
        self.mock_mode = os.getenv("MOCK_MODE", "true").lower() == "true"
        if self.mock_mode:
            logger.warning(
                "⚠️  微调模块运行在 MOCK 模式！"
                "所有训练过程都是模拟的，不会真正调用 LLaMA-Factory 或其他训练框架。"
                "如需启用真实训练，请设置环境变量 MOCK_MODE=false 并配置相关API。"
            )

    def create_task(self, task_id: int, model_name: str, method: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建微调任务"""
        task = {
            "id": task_id,
            "model_name": model_name,
            "method": method,
            "status": "pending",
            "config": config,
            "metrics": {},
            "progress": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_mock": self.mock_mode,
        }
        self.tasks[task_id] = task
        logger.info(f"Created finetune task {task_id} for model {model_name} (mock={self.mock_mode})")
        return task

    def start_task(self, task_id: int) -> Dict[str, Any]:
        """启动微调任务"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        task["status"] = "running"
        task["started_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Starting finetune task {task_id} (mock={self.mock_mode})")

        try:
            if self.mock_mode:
                result = self._simulate_finetuning(task)
            else:
                # TODO: 调用真实的LLaMA-Factory或PEFT训练代码
                logger.error("真实训练模式尚未实现，使用模拟模式")
                result = self._simulate_finetuning(task)
            
            task["status"] = "completed"
            task["completed_at"] = datetime.now(timezone.utc).isoformat()
            task["metrics"] = result["metrics"]
            task["model_path"] = result["model_path"]
        except Exception as e:
            task["status"] = "failed"
            task["error"] = str(e)
            logger.error(f"Task {task_id} failed: {e}")

        return task

    def _simulate_finetuning(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """模拟微调过程（实际项目中这里会调用真实的训练代码）"""
        import time
        import random

        config = task.get("config", {})
        method = task["method"]

        for i in range(1, 11):
            task["progress"] = i * 10
            time.sleep(0.5)
            logger.info(f"Task {task['id']} progress: {i * 10}%")

        model_name = f"{task['model_name'].replace('/', '_')}_{method}_{task['id']}"
        model_path = str(self.output_base_path / model_name)

        metrics = {
            "final_loss": random.uniform(0.1, 0.5),
            "train_loss": random.uniform(0.15, 0.6),
            "eval_loss": random.uniform(0.2, 0.7),
            "perplexity": random.uniform(1.5, 3.0),
            "method": method,
            "duration_seconds": 5,
        }

        Path(model_path).mkdir(parents=True, exist_ok=True)

        return {
            "model_path": model_path,
            "metrics": metrics
        }

    def get_task_status(self, task_id: int) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        return self.tasks.get(task_id)

    def list_tasks(self) -> List[Dict[str, Any]]:
        """列出所有任务"""
        return list(self.tasks.values())

    def get_available_methods(self) -> List[str]:
        """获取可用的微调方法"""
        return ["lora", "qlora", "adapter", "dpo"]

    def compare_methods(self, model_name: str, test_data: List[Dict]) -> Dict[str, Any]:
        """对比不同PEFT方法"""
        results = {}
        methods = self.get_available_methods()

        for method in methods:
            results[method] = {
                "perplexity": 2.0 + hash(method) % 10 * 0.1,
                "memory_usage_mb": 4000 + hash(method) % 2000,
                "train_time_min": 30 + hash(method) % 30,
                "score": 85 + hash(method) % 10,
                "is_mock": self.mock_mode,
            }
        
        if self.mock_mode:
            logger.warning("⚠️ 方法对比结果为模拟数据，非真实实验结果")
        
        return results


finetune_manager = FinetuneManager()

