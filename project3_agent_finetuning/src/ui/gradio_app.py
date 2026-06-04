import gradio as gr
import requests
import json
from typing import Optional
import os
import time

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def get_api_url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def get_finetune_methods():
    try:
        response = requests.get(get_api_url("/api/v1/finetune/methods"), timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("methods", ["lora", "qlora", "adapter", "dpo"])
        return ["lora", "qlora", "adapter", "dpo"]
    except Exception as e:
        return ["lora", "qlora", "adapter", "dpo"]


def create_finetune_task(model_name: str, method: str, learning_rate: float, batch_size: int, epochs: int):
    try:
        response = requests.post(
            get_api_url("/api/v1/finetune/create"),
            json={
                "model_name": model_name,
                "method": method,
                "config": {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": epochs
                }
            },
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return f"✅ 任务创建成功!\n任务ID: {data['id']}\n模型: {data['model_name']}\n方法: {data['method']}\n\n点击'开始任务'启动训练"
        return f"❌ API错误: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到API服务器"
    except Exception as e:
        return f"❌ 请求失败: {str(e)}"


def start_finetune_task(task_id: int):
    try:
        if not task_id:
            return "请先创建任务"
        response = requests.post(
            get_api_url(f"/api/v1/finetune/tasks/{task_id}/start"),
            timeout=10
        )
        if response.status_code == 200:
            return f"✅ 任务已启动! 请在任务列表中查看进度"
        return f"❌ API错误: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到API服务器"
    except Exception as e:
        return f"❌ 请求失败: {str(e)}"


def list_finetune_tasks():
    try:
        response = requests.get(get_api_url("/api/v1/finetune/tasks"), timeout=5)
        if response.status_code == 200:
            tasks = response.json()
            if not tasks:
                return "暂无任务"
            result = "## 微调任务列表\n\n"
            for task in tasks[:10]:
                status_icon = "⏳" if task["status"] == "pending" else "🔄" if task["status"] == "running" else "✅" if task["status"] == "completed" else "❌"
                result += f"### {status_icon} 任务 {task['id']}\n- 模型: {task['model_name']}\n- 方法: {task['method']}\n- 状态: {task['status']}\n"
                if task.get('metrics'):
                    result += f"- 指标: {json.dumps(task['metrics'], indent=2, ensure_ascii=False)}\n"
                result += "\n"
            return result
        return f"❌ API错误: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到API服务器"
    except Exception as e:
        return f"❌ 请求失败: {str(e)}"


def compare_methods(model_name: str):
    try:
        response = requests.post(
            get_api_url("/api/v1/finetune/compare?model_name=" + model_name),
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            result = "## 方法对比结果\n\n"
            for method, metrics in data.items():
                result += f"### {method}\n- Perplexity: {metrics.get('perplexity', 'N/A')}\n- 内存: {metrics.get('memory_usage_mb', 'N/A')}MB\n- 训练时间: {metrics.get('train_time_min', 'N/A')}min\n- 评分: {metrics.get('score', 'N/A')}\n\n"
            return result
        return f"❌ API错误: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到API服务器"
    except Exception as e:
        return f"❌ 请求失败: {str(e)}"


def get_agent_types():
    try:
        response = requests.get(get_api_url("/api/v1/agent/types"), timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("types", ["react", "multiagent", "visual"])
        return ["react", "multiagent", "visual"]
    except Exception as e:
        return ["react", "multiagent", "visual"]


def create_agent_session(agent_type: str):
    try:
        response = requests.post(
            get_api_url("/api/v1/agent/create"),
            json={"agent_type": agent_type, "config": {}},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return (data["id"], f"✅ Session创建成功! ID: {data['id']}")
        return (None, f"❌ API错误: {response.status_code}")
    except requests.exceptions.ConnectionError:
        return (None, "❌ 无法连接到API服务器")
    except Exception as e:
        return (None, f"❌ 请求失败: {str(e)}")


def chat_with_agent(session_id: int, user_input: str, image: Optional = None):
    if not session_id:
        return "请先创建Session"
    try:
        payload = {"session_id": session_id, "user_input": user_input}
        if image:
            payload["image"] = image
        response = requests.post(
            get_api_url("/api/v1/agent/chat"),
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                tools_used = data.get("tools_used", [])
                tool_info = ""
                if tools_used:
                    tool_info = f"\n\n🛠 工具使用:\n" + "\n".join([f"- {json.dumps(t, ensure_ascii=False)}" for t in tools_used])
                return data["output"] + tool_info
            else:
                return f"❌ 错误: {data.get('error')}"
        return f"❌ API错误: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到API服务器"
    except Exception as e:
        return f"❌ 请求失败: {str(e)}"


def list_agent_sessions():
    try:
        response = requests.get(get_api_url("/api/v1/agent/sessions"), timeout=5)
        if response.status_code == 200:
            sessions = response.json()
            if not sessions:
                return "暂无Session"
            result = "## Agent Session列表\n\n"
            for session in sessions:
                result += f"- ID: {session['id']}, 类型: {session['agent_type']}, 创建: {session['created_at']}\n"
            return result
        return f"❌ API错误: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到API服务器"
    except Exception as e:
        return f"❌ 请求失败: {str(e)}"


def evaluate_model(model_id: Optional[int], task_type: str):
    try:
        response = requests.post(
            get_api_url("/api/v1/evaluate"),
            json={"model_id": model_id, "task_type": task_type, "config": {}},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            metrics = data["metrics"]
            result = f"## 评估结果\n\n"
            result += f"**任务类型**: {data['task_type']}\n\n"
            for key, value in metrics.items():
                result += f"- {key}: {value}\n"
            return result
        return f"❌ API错误: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到API服务器"
    except Exception as e:
        return f"❌ 请求失败: {str(e)}"


def get_evaluation_reports(model_id: Optional[int] = None):
    try:
        url = get_api_url("/api/v1/evaluate/reports")
        if model_id:
            url += f"?model_id={model_id}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            reports = data.get("reports", [])
            if not reports:
                return "暂无报告"
            result = "## 评估报告列表\n\n"
            for report in reports:
                result += f"### 报告 {report['id']}\n- 模型: {report['model_id']}\n- 类型: {report['task_type']}\n- 指标: {json.dumps(report['metrics'], indent=2, ensure_ascii=False)}\n- 时间: {report['created_at']}\n\n"
            return result
        return f"❌ API错误: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到API服务器"
    except Exception as e:
        return f"❌ 请求失败: {str(e)}"


def create_gradio_app():
    with gr.Blocks(title="智能体与微调一体化平台", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🚀 智能体与微调一体化平台")
        gr.Markdown("微调 + Agent 一体化开发平台")

        with gr.Tabs():
            with gr.TabItem("🔧 微调工作台"):
                gr.Markdown("### 创建微调任务")
                with gr.Row():
                    with gr.Column():
                        ft_model_name = gr.Textbox(
                            label="基座模型",
                            value="Qwen/Qwen2-0.5B-Instruct",
                            placeholder="输入模型名称..."
                        )
                        ft_method = gr.Dropdown(
                            choices=get_finetune_methods(),
                            value="qlora",
                            label="微调方法"
                        )
                        ft_lr = gr.Number(value=2e-4, label="学习率")
                        ft_batch = gr.Number(value=4, label="Batch Size")
                        ft_epochs = gr.Number(value=3, label="Epochs")
                        create_ft_btn = gr.Button("创建任务", variant="primary")

                    with gr.Column():
                        ft_task_id = gr.Number(label="任务ID (用于启动任务)")
                        start_ft_btn = gr.Button("开始任务", variant="secondary")
                        ft_output = gr.Markdown(label="输出")

                gr.Markdown("---")
                gr.Markdown("### 任务列表")
                refresh_ft_btn = gr.Button("刷新列表")
                ft_list_output = gr.Markdown(label="任务列表")

                gr.Markdown("---")
                gr.Markdown("### 方法对比")
                with gr.Row():
                    compare_model = gr.Textbox(label="模型名称", value="Qwen/Qwen2-0.5B-Instruct")
                    compare_btn = gr.Button("对比方法")
                compare_output = gr.Markdown(label="对比结果")

                create_ft_btn.click(
                    fn=create_finetune_task,
                    inputs=[ft_model_name, ft_method, ft_lr, ft_batch, ft_epochs],
                    outputs=ft_output
                )

                start_ft_btn.click(
                    fn=start_finetune_task,
                    inputs=[ft_task_id],
                    outputs=ft_output
                )

                refresh_ft_btn.click(
                    fn=list_finetune_tasks,
                    outputs=ft_list_output
                )

                compare_btn.click(
                    fn=compare_methods,
                    inputs=[compare_model],
                    outputs=compare_output
                )

            with gr.TabItem("🤖 ReAct Agent"):
                gr.Markdown("### 法律助手Agent")
                with gr.Row():
                    with gr.Column():
                        react_agent_type = gr.Dropdown(
                            choices=get_agent_types(),
                            value="react",
                            label="Agent类型",
                            interactive=False
                        )
                        create_react_btn = gr.Button("创建Session", variant="primary")
                        react_session_id = gr.Number(label="Session ID")
                        react_user_input = gr.Textbox(
                            label="输入问题",
                            placeholder="请输入法律相关问题...",
                            lines=3
                        )
                        react_chat_btn = gr.Button("发送", variant="primary")

                    with gr.Column():
                        react_output = gr.Markdown(label="Agent响应")

                gr.Markdown("---")
                refresh_react_sessions_btn = gr.Button("刷新Session列表")
                react_sessions_output = gr.Markdown(label="Session列表")

                create_react_btn.click(
                    fn=create_agent_session,
                    inputs=[react_agent_type],
                    outputs=[react_session_id, react_output]
                )

                react_chat_btn.click(
                    fn=chat_with_agent,
                    inputs=[react_session_id, react_user_input],
                    outputs=react_output
                )

                refresh_react_sessions_btn.click(
                    fn=list_agent_sessions,
                    outputs=react_sessions_output
                )

            with gr.TabItem("👥 多Agent协作"):
                gr.Markdown("### 任务分解与执行")
                with gr.Row():
                    with gr.Column():
                        multi_agent_type = gr.Dropdown(
                            choices=get_agent_types(),
                            value="multiagent",
                            label="Agent类型",
                            interactive=False
                        )
                        create_multi_btn = gr.Button("创建Session", variant="primary")
                        multi_session_id = gr.Number(label="Session ID")
                        multi_user_input = gr.Textbox(
                            label="输入任务",
                            placeholder="请描述需要完成的任务...",
                            lines=3
                        )
                        multi_chat_btn = gr.Button("执行", variant="primary")

                    with gr.Column():
                        multi_output = gr.Markdown(label="执行结果")

                create_multi_btn.click(
                    fn=create_agent_session,
                    inputs=[multi_agent_type],
                    outputs=[multi_session_id, multi_output]
                )

                multi_chat_btn.click(
                    fn=chat_with_agent,
                    inputs=[multi_session_id, multi_user_input],
                    outputs=multi_output
                )

            with gr.TabItem("👁️ 视觉Agent"):
                gr.Markdown("### 合同风险点分析")
                with gr.Row():
                    with gr.Column():
                        visual_agent_type = gr.Dropdown(
                            choices=get_agent_types(),
                            value="visual",
                            label="Agent类型",
                            interactive=False
                        )
                        create_visual_btn = gr.Button("创建Session", variant="primary")
                        visual_session_id = gr.Number(label="Session ID")
                        visual_image = gr.Image(label="上传合同图片", type="filepath")
                        visual_user_input = gr.Textbox(
                            label="问题 (可选)",
                            placeholder="分析这张合同图片有什么风险？",
                            lines=2
                        )
                        visual_analyze_btn = gr.Button("分析", variant="primary")

                    with gr.Column():
                        visual_output = gr.Markdown(label="分析结果")

                create_visual_btn.click(
                    fn=create_agent_session,
                    inputs=[visual_agent_type],
                    outputs=[visual_session_id, visual_output]
                )

                visual_analyze_btn.click(
                    fn=chat_with_agent,
                    inputs=[visual_session_id, visual_user_input, visual_image],
                    outputs=visual_output
                )

            with gr.TabItem("📊 模型评估"):
                gr.Markdown("### 评估微调模型")
                with gr.Row():
                    with gr.Column():
                        eval_model_id = gr.Number(label="模型ID (可选)")
                        eval_task_type = gr.Dropdown(
                            choices=["general", "domain_law", "agent"],
                            value="general",
                            label="任务类型"
                        )
                        eval_btn = gr.Button("开始评估", variant="primary")

                    with gr.Column():
                        eval_output = gr.Markdown(label="评估结果")

                gr.Markdown("---")
                gr.Markdown("### 评估报告")
                refresh_eval_btn = gr.Button("刷新报告")
                eval_reports_output = gr.Markdown(label="报告列表")

                eval_btn.click(
                    fn=evaluate_model,
                    inputs=[eval_model_id, eval_task_type],
                    outputs=eval_output
                )

                refresh_eval_btn.click(
                    fn=get_evaluation_reports,
                    inputs=[eval_model_id],
                    outputs=eval_reports_output
                )

        gr.Markdown("""
        ---
        ### 使用说明

        1. **微调工作台**: 创建和管理微调任务，对比不同方法
        2. **ReAct Agent**: 法律助手，支持工具调用
        3. **多Agent协作**: 任务分解与执行
        4. **视觉Agent**: 合同图片风险点分析
        5. **模型评估**: 评估微调后的模型效果

        ⚠️ 请确保后端服务已启动: `python src/main.py`
        """)

    return app


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860)

