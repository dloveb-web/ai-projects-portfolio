# 项目4: 企业级推理与监控平台

生产级大模型推理服务平台，解决AI应用落地的"最后一公里"问题。

## 核心功能

1. **vLLM推理框架** - 主推理框架，SGLang作为可选补充
2. **OpenAI兼容API** - 100%兼容标准接口
3. **模型版本管理** - 新增功能：灰度发布
4. **Locust压测** - 高并发测试 + 性能报告
5. **LangFuse追踪** - 全链路请求追踪
6. **Prometheus+Grafana** - 实时监控面板
7. **RAGAS+Garak** - 自动化评测 + 安全扫描
8. **Docker一键部署** - Docker Compose

## 技术栈

- **推理引擎**: vLLM (主) / SGLang (补充)
- **后端**: FastAPI
- **监控**: Prometheus, Grafana
- **追踪**: LangFuse
- **部署**: Docker, Docker Compose

## 环境要求

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU (推荐A100/H100)
- CUDA 12.1+

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，配置必要的参数
```

### 3. 启动服务

```bash
# 使用Docker Compose一键启动
docker-compose -f docker/docker-compose.yml up -d

# 或本地开发模式
python src/main.py
```

### 4. 访问服务

- API服务: http://localhost:8000
- API文档: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin123)

## 目录结构

```
project4_inference_monitoring/
├── src/
│   ├── api/
│   │   └── routes.py          # OpenAI兼容API路由
│   ├── core/
│   │   ├── inference.py       # 推理服务核心
│   │   ├── monitor.py         # Prometheus监控指标
│   │   ├── tracing.py         # LangFuse全链路追踪
│   │   └── version_manager.py # 模型版本管理
│   ├── models/
│   │   ├── schemas.py         # Pydantic数据模型
│   │   └── config.py           # 配置管理
│   └── main.py                 # FastAPI应用入口
├── docker/
│   ├── docker-compose.yml      # 容器编排
│   ├── vllm.Dockerfile        # vLLM镜像
│   └── api.Dockerfile         # API服务镜像
├── prometheus/
│   └── prometheus.yml         # Prometheus配置
├── grafana/
│   ├── datasources/           # 数据源配置
│   └── dashboards/           # 监控面板
├── tests/
│   ├── test_inference.py      # 推理服务测试
│   ├── test_monitor.py        # 监控模块测试
│   └── load_test.py          # Locust压测脚本
├── scripts/
│   ├── benchmark.py           # 性能基准测试
│   ├── evaluate.py            # RAGAS评测
│   └── security_scan.py      # 安全扫描
├── .env.example              # 环境变量示例
├── requirements.txt         # Python依赖
└── README.md
```

## API使用

### Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-api-key" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

### Completions

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-api-key" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": "Once upon a time",
    "max_tokens": 100
  }'
```

### Embeddings

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-api-key" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "input": "Hello, world!"
  }'
```

## 模型版本管理

### 启用灰度发布

```bash
curl -X POST "http://localhost:8000/api/versions/grayscale/enable?percentage=10" \
  -H "Authorization: Bearer test-api-key"
```

### 激活新版本

```bash
curl -X POST "http://localhost:8000/api/versions/v2.0/activate" \
  -H "Authorization: Bearer test-api-key"
```

### 回滚版本

```bash
curl -X POST "http://localhost:8000/api/versions/rollback" \
  -H "Authorization: Bearer test-api-key"
```

## 性能测试

### Locust压测

```bash
locust -f tests/load_test.py --host=http://localhost:8000

# Headless模式
locust -f tests/load_test.py --host=http://localhost:8000 \
  --users 100 --spawn-rate 10 --run-time 60s --headless
```

### 基准测试

```bash
python scripts/benchmark.py
```

## 自动化评测

### RAGAS评测

```bash
python scripts/evaluate.py
```

### 安全扫描

```bash
python scripts/security_scan.py
```

## 监控面板

Grafana仪表盘包含以下监控指标：

- **请求QPS**: 实时请求速率
- **响应延迟**: P50/P95/P99延迟分布
- **成功率**: 请求成功/失败率
- **Token吞吐**: 每秒处理的Token数量
- **在途请求**: 当前处理中的请求数

## 环境变量

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| MODEL_PATH | 模型路径 | /models/Qwen2.5-7B-Instruct |
| VLLM_GPU_MEMORY_UTILIZATION | GPU内存利用率 | 0.9 |
| LANGFUSE_PUBLIC_KEY | LangFuse公钥 | - |
| LANGFUSE_SECRET_KEY | LangFuse密钥 | - |
| API_PORT | API服务端口 | 8000 |
| PROMETHEUS_PORT | Prometheus端口 | 9090 |
| GRAFANA_PORT | Grafana端口 | 3000 |

## 开发指南

### 本地开发

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行开发服务器
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 运行测试
pytest tests/ -v

# 运行压测
locust -f tests/load_test.py
```

### Docker部署

```bash
# 构建镜像
docker build -f docker/api.Dockerfile -t inference-api:latest .

# 运行容器
docker run -d -p 8000:8000 --gpus all inference-api:latest
```

## 生产部署

建议使用Docker Compose进行生产部署：

1. 配置环境变量
2. 挂载模型存储卷
3. 配置GPU资源
4. 设置监控告警

## 许可证

MIT License
