# =============================================================================
# Industrial AI Quality Copilot — Docker image
# =============================================================================
# Build:
#   docker build -t quality-copilot:latest -f docker/Dockerfile .
#
# Run:
#   docker run -p 8000:8000 -v ./data:/app/data -v ./models:/app/models quality-copilot:latest
# =============================================================================

FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="Industrial AI Quality Copilot"
LABEL org.opencontainers.image.description="End-to-end industrial surface defect detection system"
LABEL org.opencontainers.image.version="1.0.0"

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
WORKDIR /app

# Install Python dependencies (cache-friendly layer ordering)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY pyproject.toml README.md ./

# Create directories for runtime data
RUN mkdir -p /app/data/uploads /app/data/reports /app/data/chroma_db /app/models

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
