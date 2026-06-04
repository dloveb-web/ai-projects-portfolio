# 🔬 PCBA 工业质检 AI Demo

基于 YOLO 的 PCB 缺陷自动检测系统，支持 6 类常见 PCB 缺陷的自动识别与定位。

> **开发环境**: MacBook Air M5 / 32GB / MPS
> **产线部署**: Jetson Orin NX / TensorRT

## 快速开始

```bash
# 1. 安装依赖
make install

# 2. 下载数据集 (需要 Kaggle API)
venv/bin/python scripts/download_dataset.py

# 3. 数据预处理
venv/bin/python scripts/voc_to_yolo.py
venv/bin/python scripts/split_dataset.py

# 4. 训练模型 (选一个)
make train-v8n      # YOLOv8n (30~60min)
make train-v11n     # YOLOv11n (30~60min)
make train-v8s      # YOLOv8s (1~2h)

# 5. 导出 ONNX
make export

# 6. 启动推理服务
make serve          # API: http://localhost:8000

# 7. 打开 Web 界面
make web            # UI: http://localhost:8501
```

## 项目结构

```
PCBA_AI_Demo/
├── README.md
├── requirements.txt          # Python 依赖
├── Makefile                  # 快捷命令
├── Dockerfile                # API 服务镜像
├── docker-compose.yml        # Docker 编排
│
├── data/
│   ├── raw/                  # 原始数据集 (PKU PCB)
│   ├── processed/            # YOLO 格式数据
│   │   ├── images/{train,val,test}/
│   │   ├── labels/{train,val,test}/
│   │   └── data.yaml
│   └── test_samples/         # 推理测试图
│
├── notebooks/
│   └── 01_eda.ipynb          # EDA 分析
│
├── scripts/
│   ├── download_dataset.py   # 数据集下载
│   ├── voc_to_yolo.py        # VOC → YOLO 转换
│   ├── split_dataset.py      # 7:2:1 划分
│   └── evaluate_models.py    # 多模型评估
│
├── training/
│   ├── train_yolov8n.py      # YOLOv8n 训练
│   ├── train_yolov11n.py     # YOLOv11n 训练
│   ├── train_yolov8s.py      # YOLOv8s 训练
│   └── export_models.py      # ONNX/CoreML 导出
│
├── models/
│   └── v1.0/{yolov8n,yolov11n,yolov8s}/
│       ├── best.pt
│       ├── best.onnx
│       ├── best.mlpackage/   # CoreML (可选)
│       └── metadata.json
│
├── api/
│   └── app.py                # FastAPI 推理服务
│
├── web/
│   └── demo_app.py           # Streamlit Web 界面
│
└── docs/                     # 文档
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 + 模型状态 |
| `GET` | `/models` | 列出所有可用模型 |
| `POST` | `/model/load` | 加载/切换模型 |
| `POST` | `/predict` | PCB 缺陷检测推理 |

### 示例

```bash
# 健康检查
curl http://localhost:8000/health

# 推理
curl -X POST http://localhost:8000/predict \
  -F "file=@data/test_samples/sample_01.jpg" \
  -F "conf=0.5"

# 响应
{
  "model_version": "v1.0",
  "model_name": "yolov8n",
  "inference_time_ms": 28.5,
  "detections": [
    {
      "class_id": 3,
      "class_name": "short",
      "confidence": 0.92,
      "bbox": [120, 340, 200, 400]
    }
  ],
  "count": 1
}
```

## 缺陷类型

| 类别 | 英文 | 说明 |
|------|------|------|
| 漏孔 | missing_hole | PCB 上缺少应有的孔位 |
| 鼠咬 | mouse_bite | 铜箔边缘不规则缺口 |
| 断路 | open_circuit | 线路意外断开 |
| 短路 | short | 两条线路意外连接 |
| 毛刺 | spur | 铜箔边缘突出细丝 |
| 残铜 | spurious_copper | 不该有铜的区域残留铜箔 |

## 从 Demo 到产线

```
M5 Demo                        产线部署                    周期
───────                        ────────                   ────
数据适配     ────────────→     真实产线数据                 1~2周
模型迁移     ────────────→     ONNX→TensorRT+INT8量化      1~2周
服务升级     ────────────→     Triton+DeepStream           1周
系统集成     ────────────→     MES/QMS/EAP 全对接          1~2周
运维保障     ────────────→     7×24 运行+监控              持续
```

详见 `docs/` 目录下的部署文档。

## License

MIT
