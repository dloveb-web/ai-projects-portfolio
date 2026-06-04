# Industrial AI Quality Copilot

End-to-end industrial product surface defect detection Copilot system, implementing "defect detection → defect analysis → report generation → knowledge base update" full-process automation.

## Core Features

1. **YOLOv11 Defect Detection** - Accuracy ≥95%, video stream support
2. **Model Quantization Acceleration** - INT8 quantization + TensorRT acceleration
3. **Qwen-VL Defect Analysis** - Description + severity + cause
4. **Multi-modal RAG Knowledge Base** - Reuses Project 2 RAG architecture
5. **Three-Agent Collaboration** - Detection + Analysis + Report, reuses Project 3 Agent architecture
6. **Structured Quality Reports** - With images and data
7. **Web Management System** - User/task/history/statistics
8. **Production Deployment** - Docker+Nginx+Prometheus
9. **Edge Device Deployment** - New feature
10. **ROI Analysis Report** - Labor cost savings estimation

## Tech Stack

- **Backend**: FastAPI
- **Frontend**: React / Gradio
- **Object Detection**: YOLOv11
- **Multi-modal**: Qwen-VL
- **RAG**: Reuses Project 2
- **Agent**: Reuses Project 3
- **Database**: SQLite + ChromaDB
- **Deployment**: Docker, Nginx

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models (optional)
# YOLO: Place model in models/yolov11/best.pt
# Qwen-VL: Auto-downloaded on first use

# 4. Start backend
python src/main.py

# 5. Start Gradio interface (new terminal)
python src/ui/gradio_app.py
```

## Project Structure

```
project5_industrial_quality_copilot/
├── src/
│   ├── api/              # FastAPI routes & models
│   ├── core/
│   │   ├── detector.py   # YOLOv11 detection
│   │   ├── analyzer.py   # Qwen-VL analysis
│   │   ├── rag_engine.py # RAG engine
│   │   ├── agents.py     # Three-agent collaboration
│   │   └── report_generator.py
│   ├── database/         # Database models
│   └── ui/              # Gradio interface
├── models/              # AI models
├── data/                # Datasets & storage
├── docker/              # Docker config
└── README.md
```

## License

MIT
