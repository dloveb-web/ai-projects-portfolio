import logging
import random
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self):
        self.test_cases = self._load_test_cases()
        
        # 检查是否在模拟模式
        self.mock_mode = os.getenv("MOCK_MODE", "true").lower() == "true"
        if self.mock_mode:
            logger.warning(
                "⚠️  评估模块运行在 MOCK 模式！"
                "所有评估指标都是随机生成的，非真实模型评估结果。"
                "如需启用真实评估，请设置环境变量 MOCK_MODE=false 并实现真实的评估逻辑。"
            )

    def _load_test_cases(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载测试用例"""
        return {
            "general": [
                {"question": "什么是合同？", "type": "knowledge"},
                {"question": "1+1等于几？", "type": "reasoning"},
            ],
            "domain_law": [
                {"question": "合同违约怎么办？", "type": "legal"},
                {"question": "劳动法规定工作时间是多少？", "type": "legal"},
            ],
            "agent": [
                {"question": "帮我看看这个合同有没有问题", "type": "tool_use"},
            ]
        }

    def evaluate_model(self, model_id: Optional[int], task_type: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """评估模型"""
        if task_type not in self.test_cases:
            task_type = "general"

        test_cases = self.test_cases[task_type]

        metrics = {
            "accuracy": random.uniform(0.75, 0.95),
            "precision": random.uniform(0.7, 0.9),
            "recall": random.uniform(0.72, 0.92),
            "f1_score": random.uniform(0.71, 0.91),
            "avg_response_time_ms": random.uniform(200, 800),
            "total_tests": len(test_cases),
            "passed_tests": int(len(test_cases) * random.uniform(0.8, 1.0)),
        }

        result = {
            "model_id": model_id,
            "task_type": task_type,
            "metrics": metrics,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "summary": f"模型在{task_type}任务上的综合得分为{metrics['accuracy']:.2%} (MOCK数据)",
            "is_mock": self.mock_mode,
        }

        logger.info(f"Evaluated model {model_id} on {task_type}: {metrics} (mock={self.mock_mode})")
        return result

    def compare_models(self, model_ids: List[int], task_type: str) -> Dict[str, Any]:
        """对比多个模型"""
        results = {}
        for model_id in model_ids:
            results[model_id] = self.evaluate_model(model_id, task_type)

        comparison = {
            "task_type": task_type,
            "results": results,
            "best_model": max(results.items(), key=lambda x: x[1]["metrics"]["accuracy"])[0],
            "compared_at": datetime.now(timezone.utc).isoformat(),
            "is_mock": self.mock_mode,
        }

        return comparison

    def evaluate_agent(self, session_id: int, test_cases: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估Agent"""
        test_cases = test_cases or self.test_cases["agent"]

        metrics = {
            "tool_call_accuracy": random.uniform(0.8, 0.95),
            "task_completion_rate": random.uniform(0.85, 0.98),
            "average_steps": random.randint(2, 5),
            "user_satisfaction_score": random.uniform(4.0, 4.8),
            "is_mock": self.mock_mode,
        }

        return {
            "session_id": session_id,
            "metrics": metrics,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "is_mock": self.mock_mode,
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成评估报告"""
        mock_warning = "⚠️ 警告：当前为模拟数据" if results.get("is_mock") else ""
        report = f"""# 模型评估报告

## 评估时间
{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
{mock_warning}

## 总体指标
"""
        if "metrics" in results:
            m = results["metrics"]
            report += f"""
- 准确率: {m.get('accuracy', 0):.2%}
- 精确率: {m.get('precision', 0):.2%}
- 召回率: {m.get('recall', 0):.2%}
- F1分数: {m.get('f1_score', 0):.2%}
- 平均响应时间: {m.get('avg_response_time_ms', 0):.0f}ms
"""

        if "summary" in results:
            report += f"\n## 总结\n{results['summary']}\n"

        return report


evaluator = Evaluator()

