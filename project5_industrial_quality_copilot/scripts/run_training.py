#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型训练脚本

用法:
    python scripts/run_training.py --epochs 100 --batch 16
    python scripts/run_training.py --epochs 150 --batch 16 --enhanced
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.core.training import trainer


def main():
    parser = argparse.ArgumentParser(description="YOLO模型训练")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--device", type=str, default="0", help="设备")
    parser.add_argument("--enhanced", action="store_true", help="启用增强模式")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    args = parser.parse_args()

    print("=" * 60)
    print("工业AI质检 - 模型训练")
    print("=" * 60)
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch}")
    print(f"设备: {args.device}")
    print(f"增强模式: {args.enhanced}")
    print("=" * 60)

    result = trainer.train(
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        enhanced=args.enhanced,
        imgsz=args.imgsz,
    )

    print("=" * 60)
    print("训练完成!")
    print(f"最佳模型: {result.get('best_model')}")
    print(f"本地备份: {result.get('local_copy')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
