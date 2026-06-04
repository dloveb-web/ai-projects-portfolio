.PHONY: help install train evaluate export serve web docker-build docker-up clean

help: ## 显示帮助信息
	@echo "PCBA AI Demo — 可用命令:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## 安装 Python 依赖
	venv/bin/pip install -r requirements.txt

train: ## 训练所有模型
	venv/bin/python training/train_yolov8n.py
	venv/bin/python training/train_yolov11n.py
	venv/bin/python training/train_yolov8s.py

train-v8n: ## 仅训练 YOLOv8n
	venv/bin/python training/train_yolov8n.py

train-v11n: ## 仅训练 YOLOv11n
	venv/bin/python training/train_yolov11n.py

train-v8s: ## 仅训练 YOLOv8s
	venv/bin/python training/train_yolov8s.py

evaluate: ## 评估所有模型
	venv/bin/python scripts/evaluate_models.py

export: ## 导出 ONNX/CoreML
	venv/bin/python training/export_models.py

serve: ## 启动 FastAPI 推理服务
	venv/bin/python api/app.py

web: ## 启动 Streamlit Web 界面
	venv/bin/streamlit run web/demo_app.py

docker-build: ## 构建 Docker 镜像
	docker compose build

docker-up: ## 启动 Docker 服务
	docker compose up -d

docker-down: ## 停止 Docker 服务
	docker compose down

clean: ## 清理临时文件
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
