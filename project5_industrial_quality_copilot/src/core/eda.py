import os
import glob
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "steel_data"

CLASS_NAMES = {
    0: "crazing (龟裂)",
    1: "inclusion (夹杂)",
    2: "pitted_surface (点蚀)",
    3: "scratches (划痕)",
    4: "patches (斑块)",
    5: "rolled-in_scale (氧化铁皮压入)",
}

CLASS_NAMES_SHORT = {
    0: "crazing",
    1: "inclusion",
    2: "pitted_surface",
    3: "scratches",
    4: "patches",
    5: "rolled-in_scale",
}


class EDAAnalyzer:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.labels_dir = self.data_dir / "train" / "labels"
        self.images_dir = self.data_dir / "train" / "images"
        self.output_dir = BASE_DIR / "data" / "eda_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_all_labels(self) -> Dict[str, List[Tuple]]:
        annotations = {}
        for label_file in glob.glob(str(self.labels_dir / "*.txt")):
            basename = Path(label_file).stem
            boxes = []
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        boxes.append((cls_id, x, y, w, h))
            annotations[basename] = boxes
        return annotations

    def analyze_class_distribution(self, annotations: Dict) -> Tuple[Dict, List]:
        class_counts = Counter()
        boxes_per_image = []

        for basename, boxes in annotations.items():
            boxes_per_image.append(len(boxes))
            for cls_id, *_ in boxes:
                class_counts[cls_id] += 1

        return class_counts, boxes_per_image

    def analyze_box_sizes(self, annotations: Dict) -> Tuple[List, List, List, List, Dict]:
        widths = []
        heights = []
        areas = []
        aspect_ratios = []
        class_areas = defaultdict(list)

        for basename, boxes in annotations.items():
            for cls_id, x, y, w, h in boxes:
                widths.append(w)
                heights.append(h)
                area = w * h
                areas.append(area)
                aspect_ratios.append(w / max(h, 1e-6))
                class_areas[cls_id].append(area)

        return widths, heights, areas, aspect_ratios, class_areas

    def check_class_imbalance(self, class_counts: Dict) -> Dict[str, Any]:
        counts = list(class_counts.values())
        max_count = max(counts) if counts else 0
        min_count = min(counts) if counts else 0
        ratio = max_count / max(min_count, 1)

        imbalance_level = "严重" if ratio > 5 else ("中等" if ratio > 2 else "轻微")
        suggestions = []

        if ratio > 2:
            suggestions.extend([
                "对少数类别进行过采样或数据增强",
                "使用类别权重平衡损失函数",
                "考虑使用Focal Loss",
            ])

        return {
            "max_count": max_count,
            "min_count": min_count,
            "imbalance_ratio": round(ratio, 2),
            "imbalance_level": imbalance_level,
            "suggestions": suggestions,
        }

    def generate_summary(self, annotations: Dict) -> Dict[str, Any]:
        class_counts, boxes_per_image = self.analyze_class_distribution(annotations)
        widths, heights, areas, aspect_ratios, class_areas = self.analyze_box_sizes(annotations)
        imbalance_info = self.check_class_imbalance(class_counts)

        summary = {
            "dataset_info": {
                "total_images": len(annotations),
                "total_boxes": sum(len(boxes) for boxes in annotations.values()),
                "avg_boxes_per_image": round(np.mean(boxes_per_image), 2),
                "min_boxes_per_image": min(boxes_per_image) if boxes_per_image else 0,
                "max_boxes_per_image": max(boxes_per_image) if boxes_per_image else 0,
            },
            "class_distribution": {
                CLASS_NAMES_SHORT[cls_id]: {
                    "count": count,
                    "percentage": round(count / sum(class_counts.values()) * 100, 2)
                }
                for cls_id, count in sorted(class_counts.items())
            },
            "box_size_stats": {
                "width": {
                    "min": round(min(widths), 4) if widths else 0,
                    "max": round(max(widths), 4) if widths else 0,
                    "mean": round(np.mean(widths), 4) if widths else 0,
                    "median": round(np.median(widths), 4) if widths else 0,
                },
                "height": {
                    "min": round(min(heights), 4) if heights else 0,
                    "max": round(max(heights), 4) if heights else 0,
                    "mean": round(np.mean(heights), 4) if heights else 0,
                    "median": round(np.median(heights), 4) if heights else 0,
                },
                "area": {
                    "min": round(min(areas), 6) if areas else 0,
                    "max": round(max(areas), 6) if areas else 0,
                    "mean": round(np.mean(areas), 6) if areas else 0,
                    "median": round(np.median(areas), 6) if areas else 0,
                },
                "aspect_ratio": {
                    "min": round(min(aspect_ratios), 2) if aspect_ratios else 0,
                    "max": round(max(aspect_ratios), 2) if aspect_ratios else 0,
                    "mean": round(np.mean(aspect_ratios), 2) if aspect_ratios else 0,
                },
            },
            "class_areas": {
                CLASS_NAMES_SHORT[cls_id]: {
                    "mean": round(np.mean(areas), 6) if areas else 0,
                    "median": round(np.median(areas), 6) if areas else 0,
                }
                for cls_id, areas in class_areas.items()
            },
            "imbalance": imbalance_info,
        }

        return summary

    def print_summary(self, summary: Dict[str, Any]):
        print("=" * 60)
        print("钢铁表面缺陷数据集 EDA 分析报告")
        print("=" * 60)

        di = summary["dataset_info"]
        print(f"\n【数据集概况】")
        print(f"  总图片数: {di['total_images']}")
        print(f"  总标注框: {di['total_boxes']}")
        print(f"  每图框数: 平均{di['avg_boxes_per_image']}, "
              f"最少{di['min_boxes_per_image']}, 最多{di['max_boxes_per_image']}")

        print(f"\n【类别分布】")
        for cls_name, info in summary["class_distribution"].items():
            cn_name = CLASS_NAMES.get(
                [k for k, v in CLASS_NAMES_SHORT.items() if v == cls_name][0], ""
            ).split("(")[0]
            print(f"  {cls_name:20s} ({cn_name:4s}) | {info['count']:4d} ({info['percentage']:5.1f}%)")

        print(f"\n【标注框尺寸】")
        bs = summary["box_size_stats"]
        print(f"  宽度: {bs['width']['min']:.3f} ~ {bs['width']['max']:.3f} (平均{bs['width']['mean']:.3f})")
        print(f"  高度: {bs['height']['min']:.3f} ~ {bs['height']['max']:.3f} (平均{bs['height']['mean']:.3f})")
        print(f"  宽高比: {bs['aspect_ratio']['min']:.2f} ~ {bs['aspect_ratio']['max']:.2f} (平均{bs['aspect_ratio']['mean']:.2f})")

        imb = summary["imbalance"]
        print(f"\n【类别不均衡】")
        print(f"  不均衡比例: {imb['imbalance_ratio']}:1 ({imb['imbalance_level']}不均衡)")
        if imb['suggestions']:
            print("  建议:")
            for s in imb['suggestions']:
                print(f"    - {s}")

        print()

    def plot_visualizations(self, annotations: Dict):
        try:
            import matplotlib.pyplot as plt
            matplotlib_available = True
        except ImportError:
            logger.warning("matplotlib未安装，跳过可视化")
            matplotlib_available = False
            return

        class_counts, boxes_per_image = self.analyze_class_distribution(annotations)
        widths, heights, areas, aspect_ratios, class_areas = self.analyze_box_sizes(annotations)

        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        classes = sorted(class_counts.keys())
        names = [CLASS_NAMES_SHORT[c] for c in classes]
        counts = [class_counts[c] for c in classes]

        colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
        bars = axes[0, 0].bar(names, counts, color=colors, edgecolor='black', linewidth=0.5)
        axes[0, 0].set_title("Class Distribution", fontsize=12)
        axes[0, 0].set_ylabel("Box Count")
        axes[0, 0].tick_params(axis='x', rotation=30)
        for bar, count in zip(bars, counts):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                           str(count), ha='center', va='bottom', fontsize=9)

        axes[0, 1].pie(counts, labels=names, colors=colors, autopct='%1.1f%%',
                       startangle=90, textprops={'fontsize': 8})
        axes[0, 1].set_title("Class Percentage", fontsize=12)

        axes[1, 0].hist(areas, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title("Box Area Distribution", fontsize=12)
        axes[1, 0].set_xlabel("Normalized Area")

        axes[1, 1].scatter(widths, heights, alpha=0.3, s=10, c='purple')
        axes[1, 1].set_title("Width vs Height", fontsize=12)
        axes[1, 1].set_xlabel("Width")
        axes[1, 1].set_ylabel("Height")

        plt.tight_layout()
        output_path = self.output_dir / "eda_summary.png"
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"EDA可视化已保存: {output_path}")

    def run_full_analysis(self) -> Dict[str, Any]:
        logger.info("开始EDA分析...")
        annotations = self.load_all_labels()
        summary = self.generate_summary(annotations)
        self.print_summary(summary)
        self.plot_visualizations(annotations)
        return summary


eda_analyzer = EDAAnalyzer()
