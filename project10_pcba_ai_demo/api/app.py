"""PCBA 缺陷检测 — FastAPI 推理服务"""

import io
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ═══ 配置 ═══
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CURRENT = MODELS_DIR / "current"

CLASS_NAMES = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper",
}

CLASS_COLORS = {
    0: "#FF6B6B", 1: "#4ECDC4", 2: "#45B7D1",
    3: "#96CEB4", 4: "#FFEAA7", 5: "#DDA0DD",
}

app = FastAPI(
    title="PCBA Defect Detector API",
    version="1.0",
    description="基于 YOLO 的 PCB 缺陷自动检测推理服务",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══ 模型管理器 ═══
class ModelManager:
    def __init__(self):
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.active_version: str | None = None
        self.active_name: str | None = None
        self.input_name: str = "images"
        self.input_shape: tuple = (1, 3, 640, 640)

    def load(self, version: str = "v1.0", model_name: str = "yolov8n") -> None:
        """加载 ONNX 模型"""
        onnx_path = MODELS_DIR / version / model_name / "best.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"模型未找到: {onnx_path}")

        session = ort.InferenceSession(str(onnx_path))
        key = f"{version}/{model_name}"
        self.sessions[key] = session
        self.active_version = version
        self.active_name = model_name

        # 获取输入信息
        self.input_name = session.get_inputs()[0].name
        self.input_shape = session.get_inputs()[0].shape

    @property
    def session(self) -> ort.InferenceSession:
        if not self.active_version or not self.active_name:
            raise RuntimeError("未加载模型，请先调用 /model/load")
        key = f"{self.active_version}/{self.active_name}"
        return self.sessions[key]

    @property
    def info(self) -> dict:
        return {
            "active_version": self.active_version,
            "active_model": self.active_name,
            "models_loaded": list(self.sessions.keys()),
        }


model_mgr = ModelManager()


# ═══ 预处理 ═══
def preprocess(
    image: Image.Image, target_size: tuple[int, int] = (640, 640)
) -> tuple[np.ndarray, int, int]:
    """图像预处理: 缩放 + 归一化"""
    orig_w, orig_h = image.size
    image = image.convert("RGB").resize(target_size)
    img_array = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return img_array[np.newaxis, ...], orig_w, orig_h


# ═══ 后处理 ═══
def postprocess(
    outputs: list[np.ndarray],
    orig_w: int,
    orig_h: int,
    conf_threshold: float = 0.5,
) -> list[dict]:
    """后处理: ONNX 原始输出 → NMS + bbox 缩放 + 格式化

    YOLO ONNX 输出 shape: (1, 4 + nc, 8400)
    - 前4列: cx, cy, w, h (归一化到 [0,1])
    - 后nc列: 各类别置信度
    """
    preds = outputs[0][0]  # shape: (4 + nc, 8400)
    num_classes = len(CLASS_NAMES)  # 6

    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0

    # 转置为 (8400, 4 + nc)
    preds = preds.T

    # 提取 bbox 中心点坐标 和 类别置信度
    cx = preds[:, 0]
    cy = preds[:, 1]
    w = preds[:, 2]
    h = preds[:, 3]
    class_scores = preds[:, 4:4 + num_classes]  # (8400, nc)

    # 找出每行的最大置信度和对应类别
    max_scores = class_scores.max(axis=1)  # (8400,)
    class_ids = class_scores.argmax(axis=1)  # (8400,)

    # 转换为 x1, y1, x2, y2 (归一化)
    x1 = (cx - w / 2.0)
    y1 = (cy - h / 2.0)
    x2 = (cx + w / 2.0)
    y2 = (cy + h / 2.0)

    # 尺度还原
    x1 = x1 * scale_x * 640.0
    y1 = y1 * scale_y * 640.0
    x2 = x2 * scale_x * 640.0
    y2 = y2 * scale_y * 640.0

    # 过滤低置信度
    mask = max_scores >= conf_threshold
    x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
    max_scores, class_ids = max_scores[mask], class_ids[mask]

    # 简单 NMS（按类别去重，IoU > 0.5 的去重）
    keep_indices = _nms(x1, y1, x2, y2, max_scores, class_ids, iou_threshold=0.5)

    detections = []
    for idx in keep_indices:
        cls_id = int(class_ids[idx])
        detections.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES[cls_id],
            "confidence": round(float(max_scores[idx]), 4),
            "bbox": [
                round(float(x1[idx]), 1),
                round(float(y1[idx]), 1),
                round(float(x2[idx]), 1),
                round(float(y2[idx]), 1),
            ],
            "color": CLASS_COLORS.get(cls_id, "#FFFFFF"),
        })

    return detections


def _nms(
    x1: np.ndarray, y1: np.ndarray,
    x2: np.ndarray, y2: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.5,
) -> list[int]:
    """简单类内 NMS"""
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # 按置信度降序

    keep = []
    while len(order) > 0:
        idx = order[0]
        keep.append(idx)
        if len(order) == 1:
            break

        # 计算当前框与剩余框的 IoU
        xx1 = np.maximum(x1[idx], x1[order[1:]])
        yy1 = np.maximum(y1[idx], y1[order[1:]])
        xx2 = np.minimum(x2[idx], x2[order[1:]])
        yy2 = np.minimum(y2[idx], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[idx] + areas[order[1:]] - inter)

        # 只保留同类中 IoU 低的框（不同类别不抑制）
        same_class = class_ids[order[1:]] == class_ids[idx]
        inds = np.where(~same_class | (ovr <= iou_threshold))[0]
        order = order[inds + 1]

    return keep


# ═══ API 端点 ═══
@app.get("/health")
def health():
    return {
        "status": "ok",
        **model_mgr.info,
    }


@app.get("/models")
def list_models():
    """列出所有可用模型"""
    available = []
    for model_dir in sorted(MODELS_DIR.glob("v*/*/")):
        onnx_file = model_dir / "best.onnx"
        pt_file = model_dir / "best.pt"
        if onnx_file.exists() or pt_file.exists():
            meta = {}
            meta_file = model_dir / "metadata.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
            available.append({
                "version": model_dir.parent.name,
                "model": model_dir.name,
                "has_onnx": onnx_file.exists(),
                "has_pt": pt_file.exists(),
                "metadata": meta,
            })
    return {"models": available}


@app.post("/model/load")
def load_model(
    version: str = Query("v1.0", description="模型版本"),
    model_name: str = Query("yolov8n", description="模型名称"),
):
    """加载/切换模型"""
    try:
        model_mgr.load(version, model_name)
        return {
            "status": "switched",
            "version": version,
            "model": model_name,
        }
    except FileNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content={"error": str(e)},
        )


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="PCB 图片"),
    conf: float = Query(0.5, ge=0.1, le=0.9, description="置信度阈值"),
):
    """PCB 缺陷检测推理"""
    if model_mgr.active_version is None:
        return JSONResponse(
            status_code=400,
            content={"error": "模型未加载，请先调用 POST /model/load"},
        )

    # 读取图像
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"无法读取图片: {e}"},
        )

    # 推理
    t0 = time.time()
    input_tensor, orig_w, orig_h = preprocess(image)

    try:
        outputs = model_mgr.session.run(
            None, {model_mgr.input_name: input_tensor}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"推理失败: {e}"},
        )

    detections = postprocess(outputs, orig_w, orig_h, conf)
    latency = round((time.time() - t0) * 1000, 2)

    return {
        "model_version": model_mgr.active_version,
        "model_name": model_mgr.active_name,
        "inference_time_ms": latency,
        "image_size": {"width": orig_w, "height": orig_h},
        "detections": detections,
        "count": len(detections),
    }


# ═══ 启动 ═══
if __name__ == "__main__":
    import uvicorn

    # 启动时自动加载默认模型
    default_model = MODELS_DIR / "v1.0" / "yolov8n" / "best.onnx"
    if default_model.exists():
        model_mgr.load("v1.0", "yolov8n")
        print(f"✅ 默认模型加载: v1.0/yolov8n")
    else:
        print("⚠ 未找到默认模型，请训练后导出 ONNX")

    uvicorn.run(app, host="0.0.0.0", port=8000)
