#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EDA分析脚本

用法:
    python scripts/run_eda.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.eda import eda_analyzer


def main():
    print("=" * 60)
    print("工业AI质检 - 数据集EDA分析")
    print("=" * 60)

    summary = eda_analyzer.run_full_analysis()

    print("=" * 60)
    print("分析完成!")
    print(f"可视化图表: data/eda_output/eda_summary.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
