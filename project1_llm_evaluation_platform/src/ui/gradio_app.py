import gradio as gr
import requests
import json
from typing import Optional
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def get_api_url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def get_models():
    try:
        response = requests.get(get_api_url("/api/v1/models"), timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return [m["id"] for m in models] if models else ["qwen-turbo"]
        return ["qwen-turbo"]
    except Exception as e:
        return ["qwen-turbo"]


def chat_with_model(model: str, prompt: str, temperature: float = 0.7):
    try:
        response = requests.post(
            get_api_url("/api/v1/chat"),
            json={"prompt": prompt, "model": model, "temperature": temperature},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return f"**{model}**:\n\n{data.get('content', '')}\n\n---\n⏱ {data.get('latency_ms', 0):.2f}ms | 💰 ¥{data.get('cost', 0):.6f}"
            else:
                return f"❌ 错误: {data.get('error', 'Unknown error')}"
        return f"❌ API错误: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到API服务器，请确保后端服务已启动"
    except Exception as e:
        return f"❌ 请求失败: {str(e)}"


def multi_model_chat(models: list, prompt: str, temperature: float = 0.7):
    if not models:
        return "请至少选择一个模型"

    results = []
    for model in models:
        result = chat_with_model(model, prompt, temperature)
        results.append(result)

    return "\n\n".join(results)


def security_check(text: str):
    try:
        response = requests.post(
            get_api_url("/api/v1/security/check"),
            json={"text": text},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("is_safe"):
                return f"✅ 安全\n\n威胁等级: {data.get('threat_level', 'none')}\n检测时间: {data.get('checked_at', '')}"
            else:
                threats = data.get("threats", [])
                threat_info = "\n".join([f"- **{t.get('type', 'unknown')}**: {t.get('description', '')}" for t in threats])
                return f"⚠️ 不安全\n\n威胁等级: {data.get('threat_level', 'unknown')}\n\n检测到的威胁:\n{threat_info}\n\n检测时间: {data.get('checked_at', '')}"
        return f"❌ API错误: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到API服务器"
    except Exception as e:
        return f"❌ 请求失败: {str(e)}"


def run_evaluation(model: str, benchmark_type: str, temperature: float = 0.7):
    try:
        response = requests.post(
            get_api_url("/api/v1/evaluate"),
            json={"model": model, "benchmark_type": benchmark_type, "temperature": temperature},
            timeout=300  # 增加超时时间到5分钟
        )
        if response.status_code == 200:
            data = response.json()
            result_text = f"""## 评测结果

**模型**: {data.get('model')}
**评测类型**: {data.get('benchmark_type')}
**总题数**: {data.get('total_questions')}
**成功数**: {data.get('success_count')}
**成功率**: {data.get('success_rate')}%
**平均延迟**: {data.get('avg_latency_ms')}ms
**总成本**: ¥{data.get('total_cost')}
**平均得分**: {data.get('avg_score')}分

### 分类评分
"""
            category_scores = data.get('category_scores', {})
            for category, score in category_scores.items():
                result_text += f"- **{category}**: {score}分\n"

            return result_text
        return f"❌ API错误: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到API服务器"
    except Exception as e:
        return f"❌ 请求失败: {str(e)}"


def compare_models(models: list, benchmark_type: str, temperature: float = 0.7):
    if not models or len(models) < 2:
        return "请至少选择两个模型进行对比"

    try:
        response = requests.post(
            get_api_url("/api/v1/compare"),
            json={"models": models, "benchmark_type": benchmark_type, "temperature": temperature},
            timeout=600  # 多模型对比增加到10分钟
        )
        if response.status_code == 200:
            data = response.json()
            result_text = f"""## 模型对比评测结果

**评测类型**: {data.get('benchmark_type')}
**最佳模型**: 🏆 {data.get('winner')}

### 各模型表现
"""
            comparisons = data.get("comparisons", [])
            for comp in comparisons:
                result_text += f"""\n#### {comp.get('model')}
- 成功率: {comp.get('success_rate')}%
- 平均得分: {comp.get('avg_score')}分
- 平均延迟: {comp.get('avg_latency_ms')}ms
- 总成本: ¥{comp.get('total_cost')}
"""
                category_scores = comp.get("category_scores", {})
                if category_scores:
                    result_text += "**分类得分**:\n"
                    for cat, score in category_scores.items():
                        result_text += f"- {cat}: {score}分\n"

            return result_text
        return f"❌ API错误: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到API服务器"
    except Exception as e:
        return f"❌ 请求失败: {str(e)}"


def ab_test(prompt_a: str, prompt_b: str, test_prompt: str, model: str, blind_mode: bool = False):
    if not prompt_a or not prompt_b or not test_prompt:
        return "请填写所有必填字段"

    try:
        response = requests.post(
            get_api_url("/api/v1/prompts/compare"),
            json={
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "test_prompt": test_prompt,
                "model": model,
                "blind_mode": blind_mode
            },
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            result_text = f"""## A/B测试结果

**测试问题**: {data.get('test_prompt')}
**模型**: {data.get('model')}

"""
            results = data.get("results", {})
            if blind_mode:
                result_text += "### 盲测结果\n\n"
                result_text += f"**版本A响应**:\n{results.get('version_a', {}).get('response', 'N/A')}\n\n"
                result_text += f"**版本B响应**:\n{results.get('version_b', {}).get('response', 'N/A')}\n\n"
            else:
                result_text += "### 完整对比\n\n"
                result_text += f"**提示词A**:\n{prompt_a}\n\n**响应A**:\n{results.get('prompt_a', {}).get('response', 'N/A')}\n\n---\n\n"
                result_text += f"**提示词B**:\n{prompt_b}\n\n**响应B**:\n{results.get('prompt_b', {}).get('response', 'N/A')}\n\n"

            return result_text
        return f"❌ API错误: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到API服务器"
    except Exception as e:
        return f"❌ 请求失败: {str(e)}"


def create_gradio_app():
    with gr.Blocks(title="大模型评测平台", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🤖 大模型评测平台")
        gr.Markdown("支持通义千问、文心一言、智谱GLM三大国产模型的对比评测")

        with gr.Tabs():
            with gr.TabItem("💬 聊天对比"):
                with gr.Row():
                    with gr.Column():
                        models_multi = gr.Dropdown(
                            choices=get_models(),
                            value="qwen-turbo",
                            label="选择模型",
                            multiselect=True
                        )
                        prompt_input = gr.Textbox(
                            label="输入问题",
                            placeholder="请输入您的问题...",
                            lines=4
                        )
                        temperature_slider = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                        chat_btn = gr.Button("发送", variant="primary")

                    with gr.Column():
                        chat_output = gr.Markdown(label="模型响应")

                chat_btn.click(
                    fn=multi_model_chat,
                    inputs=[models_multi, prompt_input, temperature_slider],
                    outputs=chat_output
                )

            with gr.TabItem("📊 评测中心"):
                with gr.Row():
                    with gr.Column():
                        eval_model = gr.Dropdown(
                            choices=get_models(),
                            value="qwen-turbo",
                            label="选择模型"
                        )
                        benchmark_type = gr.Dropdown(
                            choices=["general", "domain", "security"],
                            value="general",
                            label="评测维度",
                            info="general: 通用能力, domain: 领域专业性, security: 安全测试"
                        )
                        eval_temp = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                        run_eval_btn = gr.Button("开始评测", variant="primary")

                    with gr.Column():
                        eval_output = gr.Markdown(label="评测结果")

                gr.Markdown("### 模型对比")
                with gr.Row():
                    with gr.Column():
                        compare_models_list = gr.Dropdown(
                            choices=get_models(),
                            value=["qwen-turbo", "glm-4-flash"],
                            label="选择对比模型（至少2个）",
                            multiselect=True
                        )
                        compare_btn = gr.Button("对比评测", variant="secondary")
                    with gr.Column():
                        compare_output = gr.Markdown(label="对比结果")

                run_eval_btn.click(
                    fn=run_evaluation,
                    inputs=[eval_model, benchmark_type, eval_temp],
                    outputs=eval_output
                )

                compare_btn.click(
                    fn=compare_models,
                    inputs=[compare_models_list, benchmark_type, eval_temp],
                    outputs=compare_output
                )

            with gr.TabItem("🧪 提示词实验室"):
                with gr.Row():
                    with gr.Column():
                        prompt_a = gr.Textbox(
                            label="提示词 A",
                            placeholder="输入第一个提示词...",
                            lines=3
                        )
                        prompt_b = gr.Textbox(
                            label="提示词 B",
                            placeholder="输入第二个提示词...",
                            lines=3
                        )
                        test_prompt = gr.Textbox(
                            label="测试问题",
                            placeholder="输入用于测试的问题...",
                            lines=2
                        )
                        ab_model = gr.Dropdown(
                            choices=get_models(),
                            value="qwen-turbo",
                            label="测试模型"
                        )
                        blind_mode = gr.Checkbox(label="盲测模式（不显示提示词）", value=False)
                        ab_test_btn = gr.Button("运行A/B测试", variant="primary")

                    with gr.Column():
                        ab_output = gr.Markdown(label="测试结果")

                ab_test_btn.click(
                    fn=ab_test,
                    inputs=[prompt_a, prompt_b, test_prompt, ab_model, blind_mode],
                    outputs=ab_output
                )

            with gr.TabItem("🔒 安全测试"):
                gr.Markdown("### 内容安全检测")
                gr.Markdown("检测提示注入、敏感内容和RAG攻击")

                with gr.Row():
                    with gr.Column():
                        security_input = gr.Textbox(
                            label="待检测文本",
                            placeholder="输入要检测的文本...",
                            lines=4
                        )
                        security_btn = gr.Button("检测", variant="primary")
                    with gr.Column():
                        security_output = gr.Textbox(label="检测结果", lines=8)

                security_btn.click(
                    fn=security_check,
                    inputs=security_input,
                    outputs=security_output
                )

        gr.Markdown("""
        ---
        ### 使用说明

        1. **聊天对比**: 选择多个模型，输入问题，查看各模型的响应
        2. **评测中心**: 对单个模型或多个模型进行自动化评测
        3. **提示词实验室**: 对比不同提示词的效果
        4. **安全测试**: 检测输入内容的安全性

        ⚠️ 请确保后端服务已启动: `uvicorn src.main:app --reload`
        """)

    return app


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
