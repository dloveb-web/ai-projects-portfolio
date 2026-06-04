#!/usr/bin/env python3
"""VOC XML 格式 → YOLO txt 格式转换

核心: bbox 从 (x1, y1, x2, y2) 转为 (cx, cy, w, h) 并归一化到 [0, 1]
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "PCB_DATASET"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# PKU 数据集缺陷类别映射
CLASS_MAP = {
    "missing_hole": 0,
    "mouse_bite": 1,
    "open_circuit": 2,
    "short": 3,
    "spur": 4,
    "spurious_copper": 5,
}


def voc_to_yolo_boxes(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """将 VOC XML 标注转为 YOLO 格式行列表"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in CLASS_MAP:
            print(f"  ⚠ 未知类别: {cls_name} in {xml_path.name}")
            continue

        cls_id = CLASS_MAP[cls_name]
        bbox = obj.find("bndbox")
        x1 = float(bbox.find("xmin").text)
        y1 = float(bbox.find("ymin").text)
        x2 = float(bbox.find("xmax").text)
        y2 = float(bbox.find("ymax").text)

        # 转为归一化中心点坐标
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        # 裁剪到 [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return labels


def find_image_size(xml_path: Path) -> tuple[int, int] | None:
    """从 XML 中读取图像尺寸"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is not None:
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        return w, h
    return None


def convert_dataset():
    """转换全部数据集"""
    # 查找所有 XML 标注文件
    annotations_dir = None
    images_dir = None

    # 尝试常见的目录结构
    for candidate in DATA_RAW.rglob("Annotations"):
        if candidate.is_dir():
            annotations_dir = candidate
            break

    if annotations_dir is None:
        # 尝试直接查找 XML 文件
        xml_files = list(DATA_RAW.rglob("*.xml"))
        if xml_files:
            annotations_dir = xml_files[0].parent

    if annotations_dir is None:
        print("❌ 未找到标注文件 (XML)")
        print("请确认数据集已解压到 data/raw/PCB_DATASET/")
        sys.exit(1)

    xml_files = sorted(annotations_dir.glob("*.xml"))
    print(f"📁 找到 {len(xml_files)} 个标注文件")

    # 创建输出目录
    os.makedirs(DATA_PROCESSED / "images" / "all", exist_ok=True)
    os.makedirs(DATA_PROCESSED / "labels" / "all", exist_ok=True)

    stats = {"total_boxes": 0, "per_class": {name: 0 for name in CLASS_MAP}}
    skipped = 0

    for xml_path in tqdm(xml_files, desc="转换 VOC → YOLO"):
        img_size = find_image_size(xml_path)
        if img_size is None:
            skipped += 1
            continue

        img_w, img_h = img_size
        labels = voc_to_yolo_boxes(xml_path, img_w, img_h)

        if not labels:
            skipped += 1
            continue

        # 写入 YOLO 格式标注
        label_file = DATA_PROCESSED / "labels" / "all" / (xml_path.stem + ".txt")
        with open(label_file, "w") as f:
            f.write("\n".join(labels))

        # 复制/软链图像 (如果存在)
        # 尝试找到同名图像
        for ext in [".jpg", ".png", ".bmp", ".JPG", ".PNG"]:
            img_path = None
            for candidate in DATA_RAW.rglob(xml_path.stem + ext):
                img_path = candidate
                break
            if img_path:
                import shutil
                dest = DATA_PROCESSED / "images" / "all" / (xml_path.stem + ext)
                if not dest.exists():
                    shutil.copy2(img_path, dest)
                break

        stats["total_boxes"] += len(labels)
        for i, name in CLASS_MAP.items():
            stats["per_class"][name] += sum(
                1 for l in labels if l.startswith(str(i))
            )

    print(f"\n📊 转换统计:")
    print(f"  有效文件: {len(xml_files) - skipped}")
    print(f"  跳过(无尺寸/无标签): {skipped}")
    print(f"  总标注框: {stats['total_boxes']}")
    print(f"  各类别分布:")
    for name, count in stats["per_class"].items():
        print(f"    {name}: {count}")

    print(f"\n✅ 转换完成 → {DATA_PROCESSED}")
    return stats


if __name__ == "__main__":
    convert_dataset()
