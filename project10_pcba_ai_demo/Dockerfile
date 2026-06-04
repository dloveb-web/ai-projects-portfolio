# M5 是 ARM64，用 arm64 原生镜像性能最好
FROM python:3.11-slim-bookworm

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖 (M5 ARM64: onnxruntime 而非 onnxruntime-gpu)
RUN pip install --no-cache-dir \
    fastapi==0.115.* \
    uvicorn[standard]==0.34.* \
    onnxruntime==1.19.* \
    pillow==11.* \
    numpy==2.2.*

# 复制代码和模型
COPY api/ /app/api/
COPY models/ /app/models/

WORKDIR /app/api
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
