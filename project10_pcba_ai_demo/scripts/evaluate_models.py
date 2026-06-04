#!/usr/bin/env python3
"""多模型批量评估对比脚本"""

import json
import time
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_YAML = PROJECT_ROOT / "data" / "processed" / "data.yaml"


def evaluate_model(model_path: Path, device: str = "mps") -> dict:
    """评估单个模型并返回指标"""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"评估: {model_path.parent.name}")
    print(f"{'='*60}")

    model = YOLO(str(model_path))
    t0 = time.time()

    results = model.val(data=str(DATA_YAML), device=device)

    eval_time = time.time() - t0

    metrics = {
        "model": model_path.parent.name,
        "pt_path": str(model_path),
        "mAP_50": round(float(results.box.map50), 4),
        "mAP_50_95": round(float(results.box.map), 4),
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
        "f1_score": 0.0,
        "eval_time_sec": round(eval_time, 1),
    }

    # 计算 F1
    p = metrics["precision"]
    r = metrics["recall"]
    if p + r > 0:
        metrics["f1_score"] = round(2 * p * r / (p + r), 4)

    return metrics


def infer_speed(model_path: Path, device: str = "mps", n_runs: int = 100) -> float:
    """测量推理延迟 (ms/frame)"""
    from ultralytics import YOLO
    import numpy as np

    model = YOLO(str(model_path))
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        model(dummy, device=device)

    # Benchmark
    t0 = time.time()
    for _ in range(n_runs):
        model(dummy, device=device, verbose=False)
    elapsed = time.time() - t0

    return round(elapsed / n_runs * 1000, 2)


def main():
    print("=" * 60)
    print("PCBA 缺陷检测 — 多模型评估对比")
    print("=" * 60)

    all_metrics = []

    # 查找所有 best.pt
    for model_dir in sorted(MODELS_DIR.glob("v*/*/")):
        pt_file = model_dir / "best.pt"
        if not pt_file.exists():
            print(f"  ⏭ 跳过 {model_dir}: 未找到 best.pt")
            continue

        metrics = evaluate_model(pt_file)
        metrics["inference_ms"] = infer_speed(pt_file)
        all_metrics.append(metrics)

        print(f"\n  📊 {metrics['model']}:")
        print(f"    mAP@0.5:      {metrics['mAP_50']}")
        print(f"    mAP@0.5:0.95: {metrics['mAP_50_95']}")
        print(f"    Precision:    {metrics['precision']}")
        print(f"    Recall:       {metrics['recall']}")
        print(f"    F1-Score:     {metrics['f1_score']}")
        print(f"    推理延迟:      {metrics['inference_ms']}ms")

    # 保存对比结果
    output_path = PROJECT_ROOT / "docs" / "model_comparison.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 评估结果已保存到 {output_path}")

    # 打印对比表
    if len(all_metrics) > 1:
        print("\n" + "=" * 80)
        print("模型对比总表")
        print("=" * 80)
        header = f"{'模型':<12} {'mAP@0.5':<10} {'mAP@.5:.95':<12} {'Recall':<10} {'F1':<10} {'延迟(ms)':<10} {'参数量':<10}"
        print(header)
        print("-" * 80)
        for m in sorted(all_metrics, key=lambda x: x["mAP_50"], reverse=True):
            params = {"yolov8n": "3.2M", "yolov11n": "2.6M", "yolov8s": "11.2M", "yolov8m": "25.9M"}
            pcount = params.get(m["model"], "?M")
            print(f"{m['model']:<12} {m['mAP_50']:<10} {m['mAP_50_95']:<12} {m['recall']:<10} {m['f1_score']:<10} {m['inference_ms']:<10} {pcount:<10}")


if __name__ == "__main__":
    main()
