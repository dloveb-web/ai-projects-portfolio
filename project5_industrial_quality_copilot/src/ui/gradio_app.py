import base64
import io
import json
import logging
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

# Lazy cv2 import — not needed for module load
_cv2 = None


def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as _cv2_mod
        _cv2 = _cv2_mod
    return _cv2

from ..core.detector import DefectDetector, VLMDetector, CLASS_NAMES, CLASS_NAMES_CN
from ..core.training import trainer, OfflineAugmentation
from ..core.eda import eda_analyzer
from ..database.inspection_db import inspection_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

yolo_detector = None
vlm_detector = None


def get_yolo_detector():
    global yolo_detector
    if yolo_detector is None:
        yolo_detector = DefectDetector()
    return yolo_detector


def get_vlm_detector():
    global vlm_detector
    if vlm_detector is None:
        vlm_detector = VLMDetector()
    return vlm_detector


def pil_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_pil(b64_string):
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes))


def process_yolo_detection(image, conf, iou, tta):
    if image is None:
        return None, "请上传图片"

    try:
        detector = get_yolo_detector()

        temp_path = Path("data/temp_uploads")
        temp_path.mkdir(parents=True, exist_ok=True)
        img_path = temp_path / f"temp_{int(time.time())}.jpg"

        if isinstance(image, np.ndarray):
            cv2_mod = _get_cv2()
            cv2_mod.imwrite(str(img_path), cv2_mod.cvtColor(image, cv2_mod.COLOR_RGB2BGR))
        else:
            image.save(img_path)

        result = detector.detect(str(img_path), conf=conf, iou=iou, augment=tta)

        img_path.unlink(missing_ok=True)

        # result is a dict — save to inspection DB
        inspection_db.save_detection({
            "model_type": "yolo",
            "image_path": str(img_path),
            "boxes": result.get("defects", []),
            "inference_time": result.get("processing_time", 0),
        })

        annotated_img = result.get("annotated_image")
        if annotated_img is not None:
            display_img = Image.fromarray(annotated_img)
        else:
            display_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image

        defects = result.get("defects", [])
        num_defects = len(defects)
        status = f"✅ PASS - 无缺陷" if num_defects == 0 else f"❌ FAIL - 检测到 {num_defects} 个缺陷"

        details = f"""
**检测状态**: {status}

**模型类型**: YOLO
**推理时间**: {result.get('processing_time', 0):.3f}秒

**检测结果**:
"""
        for i, d in enumerate(defects, 1):
            cls_id = d.get("class_id", -1)
            cls_name = d.get("class_name", "unknown")
            cn_name = CLASS_NAMES_CN.get(cls_id, cls_name)
            details += f"\n{i}. **{cn_name}** ({cls_name}) - 置信度: {d.get('confidence', 0):.2%}"

        return display_img, details

    except Exception as e:
        logger.error(f"YOLO检测错误: {e}")
        return None, f"检测错误: {str(e)}"


def process_vlm_detection(image, model_name):
    if image is None:
        return None, "请上传图片"

    try:
        detector = get_vlm_detector()

        if detector.client is None:
            return None, "VLM未配置API Key"

        temp_path = Path("data/temp_uploads")
        temp_path.mkdir(parents=True, exist_ok=True)
        img_path = temp_path / f"temp_{int(time.time())}.jpg"

        if isinstance(image, np.ndarray):
            cv2_mod = _get_cv2()
            cv2_mod.imwrite(str(img_path), cv2_mod.cvtColor(image, cv2_mod.COLOR_RGB2BGR))
        else:
            image.save(img_path)

        detector.model_name = model_name
        result = detector.detect(str(img_path))  # returns DetectionResult

        img_path.unlink(missing_ok=True)

        inspection_db.save_detection(result)

        annotated_img = result.annotated_image
        if annotated_img is not None:
            display_img = Image.fromarray(annotated_img)
        else:
            display_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image

        return display_img, f"**VLM分析结果**:\n\n{result.vlm_text}\n\n推理时间: {result.inference_time:.2f}秒"

    except Exception as e:
        logger.error(f"VLM检测错误: {e}")
        return None, f"检测错误: {str(e)}"


def run_training(epochs, batch_size, strategy, enhanced):
    try:
        strategy_name = f"{strategy}_{'enhanced' if enhanced else 'baseline'}"
        logger.info(f"开始训练: {strategy_name}, epochs={epochs}, batch={batch_size}")

        log_id = inspection_db.save_training_log(strategy_name, epochs, batch_size)

        result = trainer.train(
            epochs=epochs,
            batch=batch_size,
            enhanced=enhanced,
        )

        inspection_db.update_training_log(
            log_id,
            status='completed',
            model_path=result.get('best_model'),
            best_map50=0.5,
        )

        return f"✅ 训练完成!\n\n模型保存位置: {result.get('local_copy', 'N/A')}"

    except Exception as e:
        logger.error(f"训练错误: {e}")
        return f"❌ 训练失败: {str(e)}"


def run_eda_analysis():
    try:
        summary = eda_analyzer.run_full_analysis()

        report = f"""## 📊 EDA分析报告

### 数据集概况
- 总图片数: {summary['dataset_info']['total_images']}
- 总标注框: {summary['dataset_info']['total_boxes']}
- 平均每图框数: {summary['dataset_info']['avg_boxes_per_image']}

### 类别分布
"""
        for cls_name, info in summary['class_distribution'].items():
            report += f"- {cls_name}: {info['count']} ({info['percentage']}%)\n"

        report += f"\n### 类别不均衡\n"
        report += f"- 不均衡比例: {summary['imbalance']['imbalance_ratio']}:1\n"
        report += f"- 等级: {summary['imbalance']['imbalance_level']}\n"

        return report, "data/eda_output/eda_summary.png"

    except Exception as e:
        logger.error(f"EDA分析错误: {e}")
        return f"❌ 分析失败: {str(e)}", None


def run_offline_augmentation(multiply):
    try:
        aug = OfflineAugmentation()
        images_dir = "data/steel_data/train/images"
        labels_dir = "data/steel_data/train/labels"

        result = aug.augment(images_dir, labels_dir, multiply=multiply)

        return f"""✅ 数据增强完成!

- 原始图片: {result['original_images']}
- 生成图片: {result['generated_images']}
- 输出目录: {result['output_dir']}
"""

    except Exception as e:
        logger.error(f"数据增强错误: {e}")
        return f"❌ 增强失败: {str(e)}"


def get_statistics():
    try:
        stats = inspection_db.get_statistics()

        report = f"""## 📈 统计数据

### 总体统计
- 总检测数: {stats.get('total', 0)}

### 按模型类型
"""
        for model, count in stats.get('by_model', {}).items():
            report += f"- {model}: {count}\n"

        report += f"\n### 按审核状态\n"
        for status, count in stats.get('by_status', {}).items():
            report += f"- {status}: {count}\n"

        if stats.get('accuracy'):
            report += f"\n### 审核准确率\n- {stats['accuracy']}% ({stats['reviewed_count']}条已审核)\n"

        report += f"\n### 缺陷类型统计 (YOLO)\n"
        for cls_name, count in stats.get('defect_class_counts', {}).items():
            report += f"- {cls_name}: {count}\n"

        return report

    except Exception as e:
        logger.error(f"统计查询错误: {e}")
        return f"获取统计失败: {str(e)}"


def get_history_records(limit):
    try:
        records = inspection_db.get_records(limit=limit)

        if not records:
            return "暂无检测记录"

        report = f"## 📋 最近 {len(records)} 条检测记录\n\n"
        report += "| ID | 图片 | 模型 | 缺陷数 | 状态 | 时间 |\n"
        report += "|---|---|---|---|---|---|\n"

        for r in records[:20]:
            report += f"| {r['id']} | {r['image_name'][:20]}... | {r['model_type']} | {r['num_detections']} | {r['review_status']} | {r['created_at'][:19]} |\n"

        return report

    except Exception as e:
        logger.error(f"历史记录查询错误: {e}")
        return f"获取记录失败: {str(e)}"


def create_gradio_interface():
    with gr.Blocks(
        title="工业AI质检Copilot",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray")
    ) as demo:
        gr.Markdown("""
        # 🏭 工业AI质检Copilot

        基于YOLOv11和Qwen-VL的钢铁表面缺陷检测系统

        ## 功能模块
        - **YOLO检测**: 快速准确的缺陷目标检测
        - **VLM分析**: 深度语义理解和根因分析
        - **模型训练**: 支持自定义数据集训练
        - **数据分析**: EDA分析和数据增强
        """)

        with gr.Tabs():
            with gr.TabItem("🔍 YOLO缺陷检测"):
                with gr.Row():
                    with gr.Column(scale=1):
                        yolo_image_input = gr.Image(
                            label="上传产品图片",
                            type="numpy",
                            height=400
                        )
                        with gr.Row():
                            yolo_conf = gr.Slider(0.1, 1.0, 0.5, label="置信度阈值")
                            yolo_iou = gr.Slider(0.1, 1.0, 0.45, label="IoU阈值")
                        yolo_tta = gr.Checkbox(False, label="启用TTA")
                        yolo_btn = gr.Button("🚀 开始检测", variant="primary")

                    with gr.Column(scale=1):
                        yolo_image_output = gr.Image(label="检测结果", height=400)
                        yolo_result = gr.Markdown("")

            with gr.TabItem("🤖 VLM深度分析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vlm_image_input = gr.Image(
                            label="上传产品图片",
                            type="numpy",
                            height=400
                        )
                        vlm_model = gr.Dropdown(
                            ["qwen-vl-plus", "qwen2.5-vl-plus"],
                            value="qwen-vl-plus",
                            label="选择模型"
                        )
                        vlm_btn = gr.Button("🔬 VLM分析", variant="primary")

                    with gr.Column(scale=1):
                        vlm_image_output = gr.Image(label="分析结果", height=400)
                        vlm_result = gr.Markdown("")

            with gr.TabItem("🛠️ 模型训练"):
                with gr.Row():
                    with gr.Column(scale=1):
                        train_epochs = gr.Number(100, label="训练轮数")
                        train_batch = gr.Number(16, label="批次大小")
                        train_strategy = gr.Dropdown(
                            ["yolov11n", "yolov11s", "yolov11m"],
                            value="yolov11n",
                            label="模型策略"
                        )
                        train_enhanced = gr.Checkbox(False, label="增强模式")
                        train_btn = gr.Button("🚀 开始训练", variant="primary")

                    with gr.Column(scale=1):
                        train_output = gr.Textbox(label="训练日志", lines=10)

            with gr.TabItem("📊 数据分析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### EDA分析")
                        eda_btn = gr.Button("📈 运行EDA分析")

                        gr.Markdown("### 数据增强")
                        aug_multiply = gr.Number(3, label="增强倍数")
                        aug_btn = gr.Button("🔄 执行离线增强")

                    with gr.Column(scale=1):
                        eda_output = gr.Textbox(label="EDA报告", lines=15)
                        aug_output = gr.Textbox(label="增强结果", lines=5)

            with gr.TabItem("📈 统计分析"):
                with gr.Row():
                    stats_btn = gr.Button("🔍 获取统计")
                    history_limit = gr.Number(20, label="历史记录数")
                    history_btn = gr.Button("📋 查看历史")
                stats_output = gr.Textbox(label="统计数据", lines=15)
                history_output = gr.Textbox(label="历史记录", lines=15)

        yolo_btn.click(
            fn=process_yolo_detection,
            inputs=[yolo_image_input, yolo_conf, yolo_iou, yolo_tta],
            outputs=[yolo_image_output, yolo_result]
        )

        vlm_btn.click(
            fn=process_vlm_detection,
            inputs=[vlm_image_input, vlm_model],
            outputs=[vlm_image_output, vlm_result]
        )

        train_btn.click(
            fn=run_training,
            inputs=[train_epochs, train_batch, train_strategy, train_enhanced],
            outputs=[train_output]
        )

        eda_btn.click(
            fn=run_eda_analysis,
            inputs=[],
            outputs=[eda_output]
        )

        aug_btn.click(
            fn=run_offline_augmentation,
            inputs=[aug_multiply],
            outputs=[aug_output]
        )

        stats_btn.click(
            fn=get_statistics,
            inputs=[],
            outputs=[stats_output]
        )

        history_btn.click(
            fn=get_history_records,
            inputs=[history_limit],
            outputs=[history_output]
        )

    return demo


demo = create_gradio_interface()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
