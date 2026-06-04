import os
import csv
import time
import yaml
import shutil
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from ..database.settings import settings

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
STEEL_DATA_DIR = BASE_DIR / "data" / "steel_data"
MODELS_DIR = BASE_DIR / "models"

CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "pitted_surface",
    3: "scratches",
    4: "patches",
    5: "rolled-in_scale",
}

CLASS_NAMES_CN = {
    0: "龟裂",
    1: "夹杂",
    2: "点蚀",
    3: "划痕",
    4: "斑块",
    5: "氧化铁皮压入",
}


class YOLOTrainer:
    def __init__(
        self,
        data_dir: Optional[str] = None,
        model_dir: Optional[str] = None
    ):
        self.data_dir = Path(data_dir) if data_dir else STEEL_DATA_DIR
        self.model_dir = Path(model_dir) if model_dir else MODELS_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = BASE_DIR / "runs"
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["train/images", "train/labels", "test/images"]:
            (self.data_dir / subdir).mkdir(parents=True, exist_ok=True)

    def detect_device(self, device_arg: Optional[str] = None) -> str:
        import torch
        if device_arg is not None:
            return str(device_arg)
        elif torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
            logger.info(f"检测到GPU: {gpu_name} ({gpu_mem:.1f}GB)")
            return "0"
        else:
            logger.warning("未检测到GPU，将使用CPU训练")
            return "cpu"

    def create_dataset_yaml(self) -> str:
        yaml_content = {
            "path": str(self.data_dir),
            "train": "train/train.txt",
            "val": "train/val.txt",
            "test": "train/test.txt",
            "names": CLASS_NAMES,
        }
        yaml_path = BASE_DIR / "dataset_local.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"数据集配置已生成: {yaml_path}")
        return str(yaml_path)

    def train(
        self,
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 640,
        device: str = "0",
        enhanced: bool = False,
        patience: int = 50,
        pretrained: bool = False
    ) -> Dict[str, Any]:
        from ultralytics import YOLO

        is_cpu = (device == "cpu")
        if is_cpu:
            epochs = min(epochs, 10)
            batch = min(batch, 4)

        dataset_yaml = self.create_dataset_yaml()
        yolo_yaml = BASE_DIR / "yolov11.yaml"

        logger.info("=" * 60)
        logger.info("YOLOv11 钢铁缺陷检测 - 开始训练")
        logger.info("=" * 60)
        logger.info(f"  设备: {device}")
        logger.info(f"  轮数: {epochs}")
        logger.info(f"  批次: {batch}")
        logger.info(f"  输入尺寸: {imgsz}")
        logger.info(f"  增强模式: {'增强版' if enhanced else '基线版'}")

        if yolo_yaml.exists():
            model = YOLO(str(yolo_yaml))
        else:
            model = YOLO("yolov8n.pt")

        train_kwargs = dict(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            patience=patience,
            device=device,
            workers=0 if is_cpu else 8,
            pretrained=pretrained,
            save=True,
            verbose=True,
            plots=True,
            project=str(self.results_dir),
            name="steel_train",
            exist_ok=True,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            scale=0.5,
            fliplr=0.5,
        )

        if enhanced:
            train_kwargs.update(
                flipud=0.5,
                degrees=15.0,
                translate=0.15,
                hsv_h=0.02,
                hsv_s=0.7,
                hsv_v=0.5,
                perspective=0.001,
                shear=5.0,
                erasing=0.3,
                cos_lr=True,
            )

        results = model.train(**train_kwargs)

        best_pt = self.results_dir / "steel_train" / "weights" / "best.pt"
        if best_pt.exists():
            shutil.copy(best_pt, self.model_dir / "best.pt")

        return {
            "status": "completed",
            "best_model": str(best_pt),
            "local_copy": str(self.model_dir / "best.pt"),
            "results": results,
        }

    def predict(
        self,
        model_path: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        tta: bool = False,
        save_results: bool = True
    ) -> Dict[str, Any]:
        from ultralytics import YOLO

        if model_path is None:
            model_path = self.model_dir / "best.pt"
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            return {"status": "error", "message": f"模型文件不存在: {model_path}"}

        model = YOLO(str(model_path))
        test_dir = self.data_dir / "test" / "images"

        if not test_dir.exists():
            return {"status": "error", "message": f"测试集目录不存在: {test_dir}"}

        test_files = sorted([
            f for f in os.listdir(test_dir)
            if f.endswith(('.jpg', '.png'))
        ])

        logger.info("=" * 60)
        logger.info("批量预测测试集")
        logger.info("=" * 60)
        logger.info(f"  模型: {model_path}")
        logger.info(f"  测试图片: {len(test_files)} 张")
        logger.info(f"  conf={conf}, iou={iou}, TTA={tta}")

        all_rows = []
        start = time.time()

        for idx, img_file in enumerate(test_files):
            img_path = test_dir / img_file
            image_id = int(Path(img_file).stem)

            results = model.predict(
                source=str(img_path),
                conf=conf,
                iou=iou,
                augment=tta,
                verbose=False,
            )

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                for box, conf_val, cls_id in zip(xyxy, confs, classes):
                    bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                    all_rows.append({
                        "image_id": image_id,
                        "bbox": str(bbox),
                        "category_id": int(cls_id),
                        "confidence": round(float(conf_val), 6),
                    })

            if (idx + 1) % 50 == 0 or idx == len(test_files) - 1:
                elapsed = time.time() - start
                logger.info(f"  进度: {idx + 1}/{len(test_files)} 检测框: {len(all_rows)} 耗时: {elapsed:.1f}s")

        output_path = BASE_DIR / "data" / "submission.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "bbox", "category_id", "confidence"])
            writer.writeheader()
            writer.writerows(all_rows)

        elapsed = time.time() - start

        class_counter = {}
        for row in all_rows:
            cid = row["category_id"]
            cname = CLASS_NAMES.get(cid, f"class_{cid}")
            class_counter[cname] = class_counter.get(cname, 0) + 1

        return {
            "status": "completed",
            "total_boxes": len(all_rows),
            "output_path": str(output_path),
            "elapsed_time": elapsed,
            "avg_boxes_per_image": len(all_rows) / max(len(test_files), 1),
            "class_distribution": class_counter,
        }

    def optimize_confidence(
        self,
        model_path: Optional[str] = None,
        thresholds: List[float] = None
    ) -> Dict[str, Any]:
        from ultralytics import YOLO

        if model_path is None:
            model_path = self.model_dir / "best.pt"

        if thresholds is None:
            thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

        model = YOLO(str(model_path))
        dataset_yaml = self.create_dataset_yaml()

        results = []
        best_map50 = 0
        best_conf = 0.25

        logger.info("置信度阈值调优")
        for conf in thresholds:
            val_results = model.val(data=dataset_yaml, conf=conf, verbose=False)
            map50 = val_results.box.map50
            map5095 = val_results.box.map
            precision = val_results.box.mp
            recall = val_results.box.mr

            results.append({
                "threshold": conf,
                "map50": map50,
                "map5095": map5095,
                "precision": precision,
                "recall": recall,
            })

            if map50 > best_map50:
                best_map50 = map50
                best_conf = conf

        return {
            "results": results,
            "best_threshold": best_conf,
            "best_map50": best_map50,
        }


class OfflineAugmentation:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else (BASE_DIR / "data" / "augmented")

    def augment(
        self,
        images_dir: str,
        labels_dir: str,
        multiply: int = 3
    ) -> Dict[str, Any]:
        try:
            import cv2
            import albumentations as A
        except ImportError:
            return {"status": "error", "message": "需要安装: pip install albumentations opencv-python"}

        out_images = self.output_dir / "images"
        out_labels = self.output_dir / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30,
                              border_mode=cv2.BORDER_REFLECT, p=0.5),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3,
        ))

        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        total_generated = 0

        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, Path(img_file).stem + ".txt")

            if not os.path.exists(label_path):
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bboxes = []
            class_labels = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_labels.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:5]])

            basename = Path(img_file).stem
            shutil.copy2(img_path, out_images / img_file)
            shutil.copy2(label_path, out_labels / f"{basename}.txt")

            for i in range(multiply):
                try:
                    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']

                    if len(aug_bboxes) == 0:
                        continue

                    aug_name = f"{basename}_aug{i}"
                    aug_img = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(out_images / f"{aug_name}.jpg"), aug_img)

                    with open(out_labels / f"{aug_name}.txt", "w") as f:
                        for cls_id, bbox in zip(aug_labels, aug_bboxes):
                            f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                    total_generated += 1
                except Exception:
                    continue

        return {
            "status": "completed",
            "original_images": len(image_files),
            "generated_images": total_generated,
            "output_dir": str(self.output_dir),
        }


trainer = YOLOTrainer()
offline_augmentation = OfflineAugmentation()
