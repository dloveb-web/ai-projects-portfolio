#!/usr/bin/env python3
"""数据集划分: 训练/验证/测试 = 7:2:1，按类别分层抽样"""

import os
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ALL_IMAGES = DATA_PROCESSED / "images" / "all"
ALL_LABELS = DATA_PROCESSED / "labels" / "all"


def get_dominant_class(label_path: Path) -> int:
    """获取标注文件的优势类别 (用于分层抽样)"""
    class_counts = defaultdict(int)
    with open(label_path) as f:
        for line in f:
            cls_id = int(line.strip().split()[0])
            class_counts[cls_id] += 1
    if not class_counts:
        return -1
    return max(class_counts, key=class_counts.get)


def split_dataset(seed: int = 42):
    """执行分层抽样划分"""
    if not ALL_LABELS.exists():
        print("❌ 请先运行 voc_to_yolo.py 转换数据")
        return

    label_files = sorted(ALL_LABELS.glob("*.txt"))
    if not label_files:
        print("❌ 未找到标注文件")
        return

    print(f"📁 总样本数: {len(label_files)}")

    # 获取每个样本的优势类别 (分层标签)
    stratify_labels = []
    for lf in label_files:
        cls_id = get_dominant_class(lf)
        stratify_labels.append(cls_id)

    # 统计类别分布
    class_dist = defaultdict(int)
    for sl in stratify_labels:
        class_dist[sl] += 1
    print(f"  类别分布: {dict(class_dist)}")

    # 第一步: 7:3 → train : temp
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, temp_idx = next(sss1.split(label_files, stratify_labels))

    # 第二步: 2:1 → val : test (从 temp 中分)
    temp_files = [label_files[i] for i in temp_idx]
    temp_labels = [stratify_labels[i] for i in temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1/3, random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_files, temp_labels))

    train_files = [label_files[i] for i in train_idx]
    val_files = [temp_files[i] for i in val_idx]
    test_files = [temp_files[i] for i in test_idx]

    print(f"  划分结果: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # 复制文件到对应目录
    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        img_dir = DATA_PROCESSED / "images" / split_name
        lbl_dir = DATA_PROCESSED / "labels" / split_name
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for label_file in files:
            stem = label_file.stem
            # 复制标注
            shutil.copy2(label_file, lbl_dir / f"{stem}.txt")
            # 复制图像 (尝试多种扩展名)
            img_copied = False
            for ext in [".jpg", ".png", ".bmp", ".JPG", ".PNG", ".jpeg"]:
                img_src = ALL_IMAGES / f"{stem}{ext}"
                if img_src.exists():
                    shutil.copy2(img_src, img_dir / f"{stem}{ext}")
                    img_copied = True
                    break
            if not img_copied:
                print(f"  ⚠ 未找到图像: {stem}")

        print(f"  ✅ {split_name}: {len(files)} samples")

    print("\n✅ 数据集划分完成")
    print(f"\n📊 输出目录:")
    print(f"  {DATA_PROCESSED / 'images' / 'train'}/")
    print(f"  {DATA_PROCESSED / 'images' / 'val'}/")
    print(f"  {DATA_PROCESSED / 'images' / 'test'}/")


def main():
    parser = argparse.ArgumentParser(description="数据集划分 (7:2:1 分层抽样)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    args = parser.parse_args()
    split_dataset(seed=args.seed)


if __name__ == "__main__":
    main()
