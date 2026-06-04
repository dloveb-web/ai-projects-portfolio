#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量预测脚本

用法:
    python scripts/batch_predict.py
    python scripts/batch_predict.py --model ./models/best.pt --conf 0.2 --tta
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.core.training import trainer


def main():
    parser = argparse.ArgumentParser(description="批量预测")
    parser.add_argument("--model", type=str, default=None, help="模型路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU阈值")
    parser.add_argument("--tta", action="store_true", help="启用TTA")
    args = parser.parse_args()

    print("=" * 60)
    print("工业AI质检 - 批量预测")
    print("=" * 60)

    result = trainer.predict(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        tta=args.tta,
    )

    print("=" * 60)
    print("预测完成!")
    print(f"输出文件: {result.get('output_path')}")
    print(f"检测框总数: {result.get('total_boxes')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
