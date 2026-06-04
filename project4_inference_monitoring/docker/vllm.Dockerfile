FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV VLLM_VERSION=0.2.0

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

RUN pip3 install \
    torch==2.1.0 \
    torchvision==0.16.0 \
    xformers==0.0.23 \
    vllm==${VLLM_VERSION} \
    accelerate==0.24.0 \
    transformers==4.35.0

RUN mkdir -p /models

WORKDIR /app

COPY src /app/src

ENV PYTHONPATH=/app
ENV VLLM_HOST=0.0.0.0
ENV VLLM_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
