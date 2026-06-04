import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import logging
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..database.models import EvaluationResult
from .llm_client import list_all_models

logger = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, db: Session):
        self.db = db

    def generate_markdown_report(
        self,
        model: str,
        benchmark_type: str,
        results: Dict[str, Any]
    ) -> str:
        report = f"""# 大模型评测报告

## 基本信息

- **模型名称**: {model}
- **评测类型**: {benchmark_type}
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **评测框架**: 企业AI全栈实战项目集 - 项目1

---

## 评测摘要

### 整体表现

| 指标 | 数值 |
|------|------|
| 总题数 | {results.get('total_questions', 'N/A')} |
| 成功数 | {results.get('success_count', 'N/A')} |
| 成功率 | {results.get('success_rate', 'N/A')}% |
| 平均延迟 | {results.get('avg_latency_ms', 'N/A')}ms |
| 总成本 | ¥{results.get('total_cost', 'N/A')} |
| 平均得分 | {results.get('avg_score', 'N/A')}分 |

---

## 分类评分

"""
        category_scores = results.get("category_scores", {})
        if category_scores:
            report += "| 类别 | 得分 |\n|------|------|\n"
            for category, score in category_scores.items():
                report += f"| {category} | {score}分 |\n"
        report += "\n---\n\n"

        report += self._generate_detailed_results(results.get("results", []))
        report += self._generate_recommendations(results)

        return report

    def _generate_detailed_results(self, results: List[Dict]) -> str:
        if not results:
            return "### 详细结果\n\n暂无详细结果数据。\n\n"

        report = "## 详细评测结果\n\n"
        for i, result in enumerate(results, 1):
            report += f"### {i}. 问题 #{result.get('question_id', i)}\n\n"
            report += f"**问题**: {result.get('question', 'N/A')}\n\n"
            report += f"**类别**: {result.get('category', 'N/A')}\n\n"
            report += f"**得分**: {result.get('score', 0)}分\n\n"
            report += f"**延迟**: {result.get('latency_ms', 0):.2f}ms\n\n"

            if result.get("response"):
                report += f"**回答**: \n```\n{result['response']}\n```\n\n"
            elif result.get("error"):
                report += f"**错误**: {result.get('error')}\n\n"

            report += "---\n\n"

        return report

    def _generate_recommendations(self, results: Dict) -> str:
        report = "## 优化建议\n\n"

        avg_score = results.get("avg_score", 0)
        avg_latency = results.get("avg_latency_ms", 0)

        recommendations = []

        if avg_score < 60:
            recommendations.append("1. **模型性能偏低**，建议更换为更强大的模型版本（如 qwen-plus 或 qwen-max）")
        elif avg_score < 80:
            recommendations.append("1. **模型性能一般**，可以通过优化提示词进一步提升效果")

        if avg_latency > 3000:
            recommendations.append("2. **响应延迟较高**，可以考虑使用响应速度更快的模型（如 qwen-turbo）")

        category_scores = results.get("category_scores", {})
        weak_categories = [cat for cat, score in category_scores.items() if score < 70]
        if weak_categories:
            recommendations.append(f"3. **薄弱领域**: {', '.join(weak_categories)}，建议针对这些领域进行提示词优化或选择更专业的模型")

        if not recommendations:
            recommendations.append("1. 模型表现优秀，继续保持当前配置")

        report += "\n".join(recommendations)
        report += "\n\n---\n\n"

        report += f"## 附录\n\n"
        report += f"- 报告生成时间: {datetime.now().isoformat()}\n"
        report += f"- 评测平台版本: 1.0.0\n"

        return report

    def generate_comparison_report(
        self,
        models: List[str],
        comparisons: List[Dict[str, Any]]
    ) -> str:
        report = f"""# 大模型对比评测报告

## 基本信息

- **对比模型**: {', '.join(models)}
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **评测框架**: 企业AI全栈实战项目集 - 项目1

---

## 模型排名

| 排名 | 模型 | 平均得分 | 平均延迟 | 总成本 |
|------|------|----------|----------|--------|
"""

        scored_models = [(c["model"], c["avg_score"], c["avg_latency_ms"], c["total_cost"]) for c in comparisons]
        scored_models.sort(key=lambda x: x[1], reverse=True)

        for rank, (model, score, latency, cost) in enumerate(scored_models, 1):
            report += f"| {rank} | {model} | {score}分 | {latency}ms | ¥{cost:.6f} |\n"

        winner = scored_models[0][0] if scored_models else None
        report += f"""

## 最佳模型

🏆 **{winner}** 在本次评测中表现最佳！

---

## 各模型详细分析

"""

        for comparison in comparisons:
            rank = scored_models.index((comparison["model"], comparison["avg_score"],
                                       comparison["avg_latency_ms"], comparison["total_cost"])) + 1
            report += f"### {rank}. {comparison['model']}\n\n"
            report += f"- **成功率**: {comparison['success_rate']}%\n"
            report += f"- **平均得分**: {comparison['avg_score']}分\n"
            report += f"- **平均延迟**: {comparison['avg_latency_ms']}ms\n"
            report += f"- **总成本**: ¥{comparison['total_cost']:.6f}\n\n"

            category_scores = comparison.get("category_scores", {})
            if category_scores:
                report += "**分类得分**:\n"
                for cat, score in category_scores.items():
                    report += f"- {cat}: {score}分\n"
            report += "\n---\n\n"

        report += self._generate_comparison_recommendations(scored_models)

        return report

    def _generate_comparison_recommendations(self, scored_models: List) -> str:
        report = "## 选型建议\n\n"

        if len(scored_models) >= 2:
            best_model = scored_models[0][0]
            best_score = scored_models[0][1]
            second_model = scored_models[1][0]
            second_score = scored_models[1][1]

            if best_score - second_score > 20:
                report += f"### 推荐模型: {best_model}\n\n"
                report += f"{best_model} 在性能上显著优于其他模型（得分差距: {best_score - second_score:.1f}分），建议作为首选。\n\n"
            else:
                report += f"### 推荐模型: {best_model}\n\n"
                report += f"综合考虑性能和成本，{best_model} 是最佳选择。\n\n"

            report += "### 使用场景建议\n\n"
            for model, score, latency, cost in scored_models:
                if latency < 2000:
                    report += f"- **{model}**: 适合实时对话场景（延迟低）\n"
                elif score > 80:
                    report += f"- **{model}**: 适合高质量输出场景（性能优）\n"
                else:
                    report += f"- **{model}**: 适合成本敏感场景（成本低）\n"

        report += "\n---\n\n"
        report += f"## 附录\n\n"
        report += f"- 报告生成时间: {datetime.now().isoformat()}\n"
        report += f"- 评测平台版本: 1.0.0\n"

        return report

    def save_report(self, report_content: str, filename: Optional[str] = None) -> str:
        if not filename:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        reports_dir = Path(__file__).parent.parent.parent / "data" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        filepath = reports_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Report saved to: {filepath}")
        return str(filepath)

    def get_historical_results(self, model: Optional[str] = None, limit: int = 50) -> List[Dict]:
        query = self.db.query(EvaluationResult)
        if model:
            query = query.filter(EvaluationResult.model_name == model)
        results = query.order_by(desc(EvaluationResult.created_at)).limit(limit).all()

        return [{
            "id": r.id,
            "model": r.model_name,
            "prompt": r.prompt[:100] + "..." if len(r.prompt) > 100 else r.prompt,
            "benchmark_type": r.benchmark_type,
            "score": r.metrics.get("score") if r.metrics else None,
            "latency_ms": r.latency_ms,
            "cost": r.cost,
            "created_at": r.created_at.isoformat(),
        } for r in results]
