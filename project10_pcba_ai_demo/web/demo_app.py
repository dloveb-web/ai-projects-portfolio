"""PCBA 工业质检 AI 演示系统 — Streamlit Web 界面"""

import io
import json
import time
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ═══ 页面配置 ═══
st.set_page_config(
    page_title="PCBA 缺陷检测 Demo",
    page_icon="🔬",
    layout="wide",
)

# ═══ 常量 ═══
API_URL = "http://localhost:8000"

CLASS_NAMES = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper",
}

CLASS_NAMES_ZH = {
    0: "漏孔",
    1: "鼠咬",
    2: "断路",
    3: "短路",
    4: "毛刺",
    5: "残铜",
}

CLASS_COLORS = {
    0: "#FF6B6B", 1: "#4ECDC4", 2: "#45B7D1",
    3: "#96CEB4", 4: "#FFEAA7", 5: "#DDA0DD",
}

MODEL_OPTIONS = ["YOLOv8n", "YOLOv11n", "YOLOv8s"]

# ═══ 会话状态 ═══
if "history" not in st.session_state:
    st.session_state.history = []


# ═══ 辅助函数 ═══
def draw_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    """在原图上绘制检测框"""
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = det.get("color", "#FF0000")
        name = CLASS_NAMES_ZH.get(det["class_id"], det["class_name"])
        label = f"{name} {det['confidence']:.2f}"

        # 画框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # 画标签背景
        draw.rectangle([x1, y1 - 22, x1 + len(label) * 8, y1], fill=color)
        draw.text((x1 + 2, y1 - 20), label, fill="#FFFFFF")

    return image


def check_api_health() -> bool:
    """检查 API 是否可用"""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def load_default_model():
    """加载默认模型"""
    try:
        r = requests.post(
            f"{API_URL}/model/load",
            params={"version": "v1.0", "model_name": "yolov8n"},
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False


# ═══ 侧边栏 ═══
with st.sidebar:
    st.header("⚙️ 模型设置")

    model_choice = st.selectbox(
        "选择模型",
        MODEL_OPTIONS,
        index=0,
        help="切换不同的 YOLO 模型进行推理",
    )

    conf_threshold = st.slider(
        "置信度阈值",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="低于此阈值的检测结果将被过滤",
    )

    st.divider()

    st.header("📊 模型性能 (训练指标)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("mAP@0.5", "--", help="训练完成后填入")
    with col2:
        st.metric("Recall", "--")

    col3, col4 = st.columns(2)
    with col3:
        st.metric("F1-Score", "--")
    with col4:
        st.metric("推理延迟", "-- ms")

    st.divider()

    # API 状态
    api_ok = check_api_health()
    if api_ok:
        st.success("🟢 API 服务在线")
    else:
        st.error("🔴 API 服务离线")
        st.info("请先启动 API 服务:\n```bash\npython api/app.py\n```")
        if st.button("🔄 重试连接"):
            st.rerun()

    st.divider()
    st.caption(f"API 地址: {API_URL}")

# ═══ 主区域 ═══
st.title("🔬 PCBA 工业质检 AI 演示系统")
st.caption("基于 YOLO 的 PCB 缺陷自动检测 | Mac M5 · MPS | 产线部署 Jetson/TensorRT")

# ─── 上传区域 ───
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📷 原始图片")
    uploaded_file = st.file_uploader(
        "拖拽或点击上传 PCB 图片",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption=f"原始图片 ({image.size[0]}×{image.size[1]})")

with col_right:
    st.subheader("🔍 AI 检测结果")

    if uploaded_file:
        if not api_ok:
            st.error("API 服务未连接，请先启动服务")
        else:
            result_placeholder = st.empty()

            with st.spinner("正在进行 AI 推理..."):
                try:
                    # 调用 API
                    files = {"file": uploaded_file.getvalue()}
                    params = {
                        "conf": conf_threshold,
                        "model_name": model_choice.lower(),
                    }

                    t0 = time.time()
                    response = requests.post(
                        f"{API_URL}/predict",
                        files=files,
                        params=params,
                        timeout=30,
                    )
                    total_time = round((time.time() - t0) * 1000, 1)

                    if response.status_code == 200:
                        result = response.json()

                        # 绘制带标注框的结果
                        result_image = image.copy()
                        result_image = draw_boxes(result_image, result["detections"])

                        result_placeholder.image(
                            result_image,
                            use_container_width=True,
                            caption=f"检测结果 ({result['count']} 个缺陷)",
                        )

                        # 结果摘要
                        st.write(f"**检出缺陷**: {result['count']} 个 | "
                                 f"**推理耗时**: {result['inference_time_ms']}ms | "
                                 f"**总响应**: {total_time}ms")

                        # 缺陷列表
                        if result["detections"]:
                            for det in result["detections"]:
                                name = CLASS_NAMES_ZH.get(det["class_id"], det["class_name"])
                                color = det.get("color", "#FFF")
                                st.markdown(
                                    f"· <span style='color:{color}'>●</span> "
                                    f"**{name}** (`{det['class_name']}`) — "
                                    f"置信度: {det['confidence']:.1%} | "
                                    f"位置: [{det['bbox'][0]}, {det['bbox'][1]}, "
                                    f"{det['bbox'][2]}, {det['bbox'][3]}]",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.success("✅ 未检出缺陷")

                        # 加入历史
                        st.session_state.history.append({
                            "time": time.strftime("%H:%M:%S"),
                            "file": uploaded_file.name,
                            "count": result["count"],
                            "latency": result["inference_time_ms"],
                        })

                    elif response.status_code == 400:
                        st.warning(f"请先加载模型: {response.json().get('error')}")
                    else:
                        st.error(f"推理失败: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("无法连接到 API 服务，请确认服务已启动")
                except Exception as e:
                    st.error(f"发生错误: {e}")

    else:
        st.info("👈 请先上传一张 PCB 图片开始检测")
        st.markdown("""
        ### 支持的缺陷类型
        | 缺陷 | 说明 |
        |------|------|
        | 🔴 漏孔 | PCB 上缺少应有的孔位 |
        | 🟢 鼠咬 | 铜箔边缘不规则缺口 |
        | 🔵 断路 | 线路意外断开 |
        | 🟣 短路 | 两条线路意外连接 |
        | 🟡 毛刺 | 铜箔边缘突出细丝 |
        | 🟠 残铜 | 不该有铜的区域残留铜箔 |
        """)

# ═══ 底部：检测历史 ═══
if st.session_state.history:
    st.divider()
    st.subheader("📋 检测历史")
    history_df = [
        {**h, "count_str": f"{h['count']}个缺陷", "latency_str": f"{h['latency']}ms"}
        for h in st.session_state.history[-10:]
    ]
    st.dataframe(
        history_df,
        column_config={
            "time": "时间",
            "file": "文件",
            "count_str": "检出",
            "latency_str": "延迟",
        },
        use_container_width=True,
        hide_index=True,
    )

# ═══ 页脚 ═══
st.divider()
st.caption(
    "PCBA 工业质检 AI Demo · 开发环境 Mac M5 (MPS) · "
    "模型格式 ONNX · 产线部署 Jetson/TensorRT · "
    "PKU PCB 缺陷数据集 (6类/10,668张)"
)
