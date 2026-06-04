#!/usr/bin/env python3
"""下载 PKU PCB 缺陷数据集

数据来源: 北大 Open Lab on Human-Robot Interaction
规模: 1,386 原始 + 增强 → 10,668 张, 6 类缺陷
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def download_from_kaggle():
    """通过 Kaggle API 下载"""
    import subprocess

    os.makedirs(DATA_RAW, exist_ok=True)

    print("[1/2] 下载 PKU PCB 数据集...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "norbertelter/pcb-defect-dataset",
            "-p", str(DATA_RAW),
        ],
        check=True,
    )

    zip_path = DATA_RAW / "pcb-defect-dataset.zip"
    if zip_path.exists():
        print(f"[2/2] 解压到 {DATA_RAW / 'PCB_DATASET'} ...")
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(DATA_RAW / "PCB_DATASET")
        os.remove(zip_path)
        print("✅ 下载完成")
        return True
    else:
        print("❌ 压缩包未找到，请检查 Kaggle API 配置")
        return False


def download_from_modelscope():
    """通过 ModelScope 下载 (备选)"""
    import subprocess

    os.makedirs(DATA_RAW, exist_ok=True)

    print("[1/2] 通过 ModelScope 下载...")
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        snapshot_download(
            "ModelBulider/PCB_DATASET_JSON",
            cache_dir=str(DATA_RAW),
        )
        print("✅ 下载完成 (ModelScope)")
        return True
    except ImportError:
        print("modelscope 未安装，尝试 pip install modelscope")
        print("或使用 Kaggle 方式: --source kaggle")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 PKU PCB 缺陷数据集")
    parser.add_argument(
        "--source",
        choices=["kaggle", "modelscope"],
        default="kaggle",
        help="下载源 (默认: kaggle)",
    )
    args = parser.parse_args()

    if args.source == "kaggle":
        success = download_from_kaggle()
    else:
        success = download_from_modelscope()

    if success:
        # 统计数据集
        dataset_dir = DATA_RAW / "PCB_DATASET"
        if dataset_dir.exists():
            print("\n📊 数据集概览:")
            for root, dirs, files in os.walk(dataset_dir):
                depth = root[len(str(dataset_dir)):].count(os.sep)
                if depth <= 2:
                    print(f"  {'  ' * depth}{os.path.basename(root)}/ ({len(dirs) + len(files)} 项)")


if __name__ == "__main__":
    main()
