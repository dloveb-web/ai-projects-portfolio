#!/usr/bin/env python3
"""模型导出: PyTorch (.pt) → ONNX + CoreML"""
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

MODELS_DIR = PROJECT_ROOT / "models" / "v1.0"


def export_model(pt_path: Path):
    """导出单个模型到 ONNX 和 CoreML 格式"""
    model_name = pt_path.parent.name
    print(f"\n{'='*60}")
    print(f"导出: {model_name}")
    print(f"{'='*60}")

    model = YOLO(str(pt_path))

    results = {}

    # 1. ONNX 导出 (产线通用格式)
    print("\n[1/2] 导出 ONNX ...")
    onnx_path = pt_path.parent / "best.onnx"
    t0 = time.time()
    model.export(format="onnx", simplify=True, opset=12)
    results["onnx_export_time"] = round(time.time() - t0, 1)
    results["onnx_size_mb"] = round(onnx_path.stat().st_size / 1024 / 1024, 2)
    print(f"  ✅ ONNX: {onnx_path} ({results['onnx_size_mb']}MB)")

    # 2. CoreML 导出 (M5 专属，ANE 加速验证)
    print("[2/2] 导出 CoreML ...")
    try:
        t0 = time.time()
        model.export(format="coreml", nms=True)
        results["coreml_export_time"] = round(time.time() - t0, 1)

        mlpackage = pt_path.parent / "best.mlpackage"
        if mlpackage.exists():
            results["coreml_size_mb"] = round(
                sum(f.stat().st_size for f in mlpackage.rglob("*") if f.is_file())
                / 1024 / 1024, 2
            )
            print(f"  ✅ CoreML: {mlpackage} ({results['coreml_size_mb']}MB)")
    except Exception as e:
        print(f"  ⚠ CoreML 导出失败: {e}")
        results["coreml_export_time"] = None

    # 3. 保存 metadata
    metadata = {
        "model": model_name,
        "pt_path": str(pt_path),
        "export_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        **results,
    }
    metadata_path = pt_path.parent / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  📄 metadata: {metadata_path}")
    return results


def main():
    models = sorted(MODELS_DIR.glob("*/best.pt"))
    if not models:
        print("❌ 未找到任何 best.pt 文件")
        print("请先运行训练脚本: python training/train_yolov8n.py")
        return

    print(f"找到 {len(models)} 个模型待导出")

    all_results = {}
    for pt_path in models:
        all_results[pt_path.parent.name] = export_model(pt_path)

    print(f"\n{'='*60}")
    print("导出汇总")
    print(f"{'='*60}")
    for name, r in all_results.items():
        print(f"  {name}: ONNX={r.get('onnx_size_mb', '?')}MB | CoreML={r.get('coreml_size_mb', '?')}MB")


if __name__ == "__main__":
    main()
