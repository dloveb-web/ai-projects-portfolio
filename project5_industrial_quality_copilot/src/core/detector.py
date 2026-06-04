"""
Defect detection module — YOLO-based object detection + VLM-based visual analysis.

Provides:
- DefectDetector: primary YOLO-based detector (numpy / file / base64 / bytes)
- VLMDetector: vision-language-model detector via Qwen-VL API
- create_detector(): factory function
"""

from __future__ import annotations

import base64
import json
import logging
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..database.settings import settings

# Lazy import — cv2 is a heavy dependency not needed for all code paths
_cv2 = None


def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as _cv2_mod
        _cv2 = _cv2_mod
    return _cv2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class name mappings
# ---------------------------------------------------------------------------

CLASS_NAMES: Dict[int, str] = {
    0: "crazing",
    1: "inclusion",
    2: "pitted_surface",
    3: "scratches",
    4: "patches",
    5: "rolled-in_scale",
}

CLASS_NAMES_CN: Dict[int, str] = {
    0: "龟裂",
    1: "夹杂",
    2: "点蚀",
    3: "划痕",
    4: "斑块",
    5: "氧化铁皮压入",
}

CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (255, 100, 100),
    1: (100, 255, 100),
    2: (100, 100, 255),
    3: (255, 255, 100),
    4: (255, 100, 255),
    5: (100, 255, 255),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectionBox:
    """Single detection bounding-box."""
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": [self.x1, self.y1, self.x2, self.y2],
        }


@dataclass
class DetectionResult:
    """Full detection result for one image."""
    model_type: str
    image_path: str
    boxes: List[DetectionBox] = field(default_factory=list)
    vlm_text: str = ""
    vlm_defect_types: List[str] = field(default_factory=list)
    inference_time: float = 0.0
    annotated_image: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "image_path": self.image_path,
            "boxes": [b.to_dict() for b in self.boxes],
            "vlm_text": self.vlm_text,
            "vlm_defect_types": self.vlm_defect_types,
            "inference_time": self.inference_time,
        }


# ---------------------------------------------------------------------------
# DefectDetector — primary YOLO-based detector
# ---------------------------------------------------------------------------

class DefectDetector:
    """YOLO-based surface defect detector.

    Supports input as file path, numpy array, base64 string, or raw bytes.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        self.model_path = model_path or settings.yolo_model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self._load_model()

    # -- model loading -------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                logger.info("Loaded YOLO model from %s", self.model_path)
            else:
                self.model = YOLO("yolov8n.pt")
                logger.info("Loaded default YOLO model (yolov8n)")
        except Exception as exc:
            logger.warning("Failed to load YOLO model: %s", exc)
            self.model = None

    # -- unified detect (handles str path OR numpy array) --------------------

    def detect(
        self,
        image: Any,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        augment: bool = False,
        conf_threshold: Optional[float] = None,  # alias for test compatibility
    ) -> Dict[str, Any]:
        """Detect defects from a file path (str) or numpy image array.

        Returns a dict with keys:
            has_defect, defects, annotated_image, processing_time
        """
        # Support both 'conf' and 'conf_threshold' kwarg names
        conf = conf if conf is not None else conf_threshold

        if isinstance(image, (str, Path)):
            return self.detect_from_path(str(image), conf=conf, iou=iou, augment=augment)

        # numpy array path
        if isinstance(image, np.ndarray):
            return self._detect_from_array(image, conf=conf, iou=iou, augment=augment)

        raise TypeError(f"Unsupported image type: {type(image)}")

    # -- file-path-based detection -------------------------------------------

    def detect_from_path(
        self,
        image_path: str,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        augment: bool = False,
    ) -> Dict[str, Any]:
        """Detect defects from a file path. Returns simplified dict."""
        result = self._detect_yolo(image_path, conf=conf, iou=iou, augment=augment)

        defects = [
            {
                "class_id": b.class_id,
                "class_name": b.class_name,
                "confidence": b.confidence,
                "bbox": [b.x1, b.y1, b.x2, b.y2],
            }
            for b in result.boxes
        ]

        return {
            "has_defect": len(defects) > 0,
            "defects": defects,
            "annotated_image": result.annotated_image,
            "processing_time": result.inference_time,
        }

    # -- base64 detection ----------------------------------------------------

    def detect_from_base64(self, image_base64: str) -> Dict[str, Any]:
        """Detect defects from a base64-encoded image string."""
        image_bytes = base64.b64decode(image_base64)
        return self.detect_from_bytes(image_bytes)

    def detect_from_bytes(self, content: bytes) -> Dict[str, Any]:
        """Detect defects from raw image bytes."""
        nparr = np.frombuffer(content, np.uint8)
        cv2 = _get_cv2()
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return {
                "has_defect": False,
                "defects": [],
                "annotated_image": None,
                "processing_time": 0.0,
                "error": "Failed to decode image bytes",
            }
        return self._detect_from_array(image)

    # -- batch detection -----------------------------------------------------

    def detect_batch(
        self,
        image_paths: List[str],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        augment: bool = False,
    ) -> List[Dict[str, Any]]:
        return [self.detect_from_path(p, conf=conf, iou=iou, augment=augment) for p in image_paths]

    # -- drawing & region helpers (public) -----------------------------------

    def draw_defect_boxes(
        self,
        image: np.ndarray,
        defects: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Draw bounding boxes and labels on an image (numpy array)."""
        pil_img = Image.fromarray(image.astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

        for d in defects:
            bbox = d.get("bbox", [0, 0, 0, 0])
            class_name = d.get("class_name", "unknown")
            confidence = d.get("confidence", 0.0)
            # match color by name
            color = (255, 255, 255)
            for cid, cname in CLASS_NAMES.items():
                if cname.lower() == class_name.lower():
                    color = CLASS_COLORS.get(cid, (255, 255, 255))
                    cn_name = CLASS_NAMES_CN.get(cid, class_name)
                    label = f"{cn_name} ({confidence:.2f})"
                    break
            else:
                label = f"{class_name} ({confidence:.2f})"

            draw.rectangle(bbox, outline=color, width=2)
            draw.text((bbox[0], max(0, bbox[1] - 16)), label, fill=color, font=font)

        return np.array(pil_img)

    def get_defect_regions(
        self,
        image: np.ndarray,
        defects: List[Dict[str, Any]],
    ) -> List[np.ndarray]:
        """Crop and return each defect region from the image."""
        regions = []
        for d in defects:
            bbox = d.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            region = image[y1:y2, x1:x2].copy()
            regions.append(region)
        return regions

    def image_to_base64(self, image: np.ndarray, fmt: str = ".jpg") -> str:
        """Encode a numpy image array to a base64 string."""
        cv2 = _get_cv2()
        _, buffer = cv2.imencode(fmt, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode("utf-8")

    # -- internal helpers ----------------------------------------------------

    def _detect_from_array(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        augment: bool = False,
    ) -> Dict[str, Any]:
        """Detect defects from a numpy array by saving to temp file first."""
        # Ensure RGB
        cv2 = _get_cv2()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Save to temp file for YOLO
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        try:
            result = self._detect_yolo(tmp_path, conf=conf, iou=iou, augment=augment)

            defects = [
                {
                    "class_id": b.class_id,
                    "class_name": b.class_name,
                    "confidence": b.confidence,
                    "bbox": [b.x1, b.y1, b.x2, b.y2],
                }
                for b in result.boxes
            ]

            return {
                "has_defect": len(defects) > 0,
                "defects": defects,
                "annotated_image": result.annotated_image,
                "processing_time": result.inference_time,
            }
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _detect_yolo(
        self,
        image_path: str,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        augment: bool = False,
    ) -> DetectionResult:
        """Core YOLO inference returning a DetectionResult."""
        if self.model is None:
            return DetectionResult(
                model_type="yolo",
                image_path=image_path,
                vlm_text="[错误] YOLO模型未加载",
            )

        conf = conf if conf is not None else self.confidence_threshold
        iou = iou if iou is not None else self.iou_threshold

        start_time = time.time()

        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            augment=augment,
            verbose=False,
        )

        elapsed = time.time() - start_time

        boxes: List[DetectionBox] = []
        if len(results) > 0 and results[0].boxes is not None:
            result_boxes = results[0].boxes
            xyxy = result_boxes.xyxy.cpu().numpy()
            confs = result_boxes.conf.cpu().numpy()
            classes = result_boxes.cls.cpu().numpy()

            for box, conf_val, cls_id in zip(xyxy, confs, classes):
                cls_id_int = int(cls_id)
                boxes.append(DetectionBox(
                    class_id=cls_id_int,
                    class_name=CLASS_NAMES.get(cls_id_int, f"class_{cls_id_int}"),
                    confidence=float(conf_val),
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                ))

        result = DetectionResult(
            model_type="yolo",
            image_path=image_path,
            boxes=boxes,
            vlm_text=f"YOLO检测到 {len(boxes)} 个缺陷区域",
            inference_time=elapsed,
        )

        result.annotated_image = self._draw_boxes_pil(image_path, boxes)
        return result

    def _draw_boxes_pil(self, image_path: str, boxes: List[DetectionBox]) -> np.ndarray:
        """Draw bounding boxes using PIL (used internally by YOLO path)."""
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

        for box in boxes:
            color = CLASS_COLORS.get(box.class_id, (255, 255, 255))
            draw.rectangle(
                [box.x1, box.y1, box.x2, box.y2], outline=color, width=2
            )
            cn_name = CLASS_NAMES_CN.get(box.class_id, box.class_name)
            label = f"{cn_name} ({box.confidence:.2f})"
            draw.text(
                (box.x1, max(0, box.y1 - 16)), label, fill=color, font=font
            )

        return np.array(img)


# ---------------------------------------------------------------------------
# VLMDetector
# ---------------------------------------------------------------------------

class VLMDetector:
    """Vision-Language-Model-based defect detector (Qwen-VL via DashScope)."""

    DEFAULT_PROMPT = """你是一个钢铁表面缺陷检测专家。请分析这张图片，检测是否存在以下缺陷类型：

缺陷类型：
- crazing (龟裂): 细小的网状裂纹
- inclusion (夹杂): 材料中的异物
- pitted_surface (点蚀): 小坑状腐蚀
- scratches (划痕): 表面线性划伤
- patches (斑块): 表面色斑或污渍
- rolled-in_scale (氧化铁皮压入): 氧化皮压入表面

请以JSON格式返回检测结果，bbox_2d使用0-1000的归一化坐标：

```json
[
  {
    "label": "缺陷类型英文名",
    "bbox_2d": [x1, y1, x2, y2],
    "description": "缺陷的详细描述"
  }
]
```

如果未检测到缺陷，请返回空数组 []。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or settings.qwen_api_key
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = model_name or "qwen-vl-plus"
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        if not self.api_key:
            logger.warning("未配置API Key，VLM检测不可用")
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info("VLM客户端初始化成功: %s", self.model_name)
        except ImportError:
            logger.warning("未安装openai库，VLM检测不可用")
        except Exception as exc:
            logger.warning("VLM客户端初始化失败: %s", exc)

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _extract_json(self, text: str) -> str:
        text = re.sub(r"```json\s*", "```", text, flags=re.DOTALL)
        text = re.sub(r"```\s*", "", text, flags=re.DOTALL).strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts[1::2]:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                return cleaned
        return text.strip()

    def _convert_bbox(
        self, bbox_2d: List[float], img_width: int, img_height: int
    ) -> Tuple[float, float, float, float]:
        if "qwen3" in self.model_name.lower():
            x1 = bbox_2d[0] / 1000 * img_width
            y1 = bbox_2d[1] / 1000 * img_height
            x2 = bbox_2d[2] / 1000 * img_width
            y2 = bbox_2d[3] / 1000 * img_height
        else:
            x1, y1, x2, y2 = bbox_2d
        return x1, y1, x2, y2

    def _parse_bbox_response(
        self, reply: str, img_width: int, img_height: int
    ) -> Tuple[List[DetectionBox], List[str], str]:
        json_str = self._extract_json(reply)
        parsed = json.loads(json_str)

        if isinstance(parsed, dict):
            parsed = [parsed]

        boxes: List[DetectionBox] = []
        defect_types: List[str] = []
        desc_lines: List[str] = []

        for item in parsed:
            bbox_2d = item.get("bbox_2d")
            label = item.get("label", "unknown")
            description = item.get("description", "")

            if not bbox_2d or len(bbox_2d) != 4:
                continue

            x1, y1, x2, y2 = self._convert_bbox(bbox_2d, img_width, img_height)
            class_id = next(
                (k for k, v in CLASS_NAMES.items() if v.lower() == label.lower()), -1
            )

            boxes.append(DetectionBox(
                class_id=class_id,
                class_name=label,
                confidence=1.0,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))

            if label not in defect_types:
                defect_types.append(label)

            cn_name = CLASS_NAMES_CN.get(class_id, label)
            desc_lines.append(f"- {cn_name} ({label}): {description}")

        has_defect = len(boxes) > 0
        summary = f"缺陷检测: {'有缺陷' if has_defect else '无缺陷'}\n"
        summary += f"检测到 {len(boxes)} 个缺陷区域\n"
        if desc_lines:
            summary += "\n".join(desc_lines)

        return boxes, defect_types, summary

    def detect(
        self, image_path: str, prompt: Optional[str] = None
    ) -> DetectionResult:
        if self.client is None:
            return DetectionResult(
                model_type="vlm",
                image_path=image_path,
                vlm_text="[错误] VLM客户端未初始化，请检查API配置",
            )

        prompt = prompt or self.DEFAULT_PROMPT
        b64_image = self._encode_image(image_path)

        img = Image.open(image_path)
        img_width, img_height = img.size

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            elapsed = time.time() - start_time
            reply = response.choices[0].message.content.strip()
        except Exception as exc:
            elapsed = time.time() - start_time
            return DetectionResult(
                model_type="vlm",
                image_path=image_path,
                vlm_text=f"[API调用失败] {exc}",
                inference_time=elapsed,
            )

        boxes: List[DetectionBox] = []
        defect_types: List[str] = []
        description = reply
        try:
            boxes, defect_types, description = self._parse_bbox_response(
                reply, img_width, img_height
            )
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            description = f"VLM原始回复 (JSON解析失败):\n{reply}"

        result = DetectionResult(
            model_type="vlm",
            image_path=image_path,
            boxes=boxes,
            vlm_text=description,
            vlm_defect_types=defect_types,
            inference_time=elapsed,
        )

        result.annotated_image = self._draw_boxes(image_path, boxes)
        return result

    def _draw_boxes(
        self, image_path: str, boxes: List[DetectionBox]
    ) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

        for box in boxes:
            color = CLASS_COLORS.get(box.class_id, (255, 255, 255))
            draw.rectangle(
                [box.x1, box.y1, box.x2, box.y2], outline=color, width=2
            )
            cn_name = CLASS_NAMES_CN.get(box.class_id, box.class_name)
            label = f"{cn_name} ({box.class_name})"
            draw.text(
                (box.x1, max(0, box.y1 - 16)), label, fill=color, font=font
            )

        return np.array(img)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_detector(model_type: str, **kwargs: Any) -> Any:
    """Create a detector instance by type name."""
    if model_type == "yolo":
        return DefectDetector(
            model_path=kwargs.get("model_path"),
            confidence_threshold=kwargs.get("confidence_threshold", 0.5),
        )
    if model_type == "vlm":
        return VLMDetector(
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
            model_name=kwargs.get("model_name"),
        )
    raise ValueError(f"不支持的模型类型: {model_type}")


# ---------------------------------------------------------------------------
# Module-level singletons (lazy-friendly)
# ---------------------------------------------------------------------------

detector: Optional[DefectDetector] = None
vlm_detector: Optional[VLMDetector] = None


def _get_detector() -> DefectDetector:
    global detector
    if detector is None:
        detector = DefectDetector()
    return detector


def _get_vlm_detector() -> VLMDetector:
    global vlm_detector
    if vlm_detector is None:
        vlm_detector = VLMDetector()
    return vlm_detector
