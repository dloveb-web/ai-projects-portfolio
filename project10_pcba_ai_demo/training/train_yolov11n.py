#!/usr/bin/env python3
"""训练 YOLOv11 Nano — 最新架构对比"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

DATA_YAML = PROJECT_ROOT / "data" / "processed" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "models" / "v1.0" / "yolov11n"

def main():
    print("=" * 60)
    print("YOLOv11n 训练 (最新架构)")
    print("=" * 60)

    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        device="mps",
        project=str(OUTPUT_DIR.parent.parent),
        name="v1.0/yolov11n",
        exist_ok=True,
        patience=10,
        save=True,
        save_period=10,
        val=True,
        plots=True,
    )

    metrics = model.val()
    print(f"\n📊 YOLOv11n 最终指标:")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")

    print(f"\n✅ 模型保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
