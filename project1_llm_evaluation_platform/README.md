# 项目1: 大模型基础与评测平台

一站式大模型对比与选型平台，解决企业"选模型难、调prompt难"的痛点。

## 核心功能

1. **多模型统一调用接口** - 支持通义千问、文心一言、智谱GLM三大国产模型
2. **提示词实验室** - 多版本并行编辑 + 盲测A/B测试
3. **三维度自动化评测系统** - 通用/领域/安全
4. **推理性能评测模块** - 吞吐量、延迟、成本对比
5. **可视化对比面板** - 准确率/延迟/成本
6. **大模型选型报告** - 可直接用于企业决策
7. **安全防护** - 提示注入防御 + 敏感内容过滤 + RAG防护

## 技术栈

- **后端**: FastAPI
- **前端**: Gradio
- **数据库**: SQLite
- **支持模型**:
  - 通义千问（阿里云）- qwen-turbo, qwen-plus, qwen-max
  - 文心一言（百度）- ernie-bot, ernie-bot-turbo
  - 智谱GLM（智谱AI）- glm-4, glm-4-flash

## 快速开始

### 1. 安装依赖

```bash
cd project1_llm_evaluation_platform
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件填入API密钥
```

### 3. 启动服务

**方式一：分别启动API和Gradio界面**

```bash
# 终端1: 启动FastAPI后端
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 终端2: 启动Gradio界面
python src/ui/gradio_app.py
```

**方式二：直接启动Gradio（包含API）**

```bash
python src/ui/gradio_app.py
```

### 4. 访问服务

- **Gradio界面**: http://localhost:7860
- **API文档**: http://localhost:8000/docs
- **API根路径**: http://localhost:8000

## 项目结构

```
project1_llm_evaluation_platform/
├── src/
│   ├── api/              # API路由和模型定义
│   │   ├── __init__.py
│   │   ├── routes.py     # FastAPI路由
│   │   └── models.py     # Pydantic模型
│   ├── core/             # 核心业务逻辑
│   │   ├── __init__.py
│   │   ├── llm_client.py      # 多模型客户端
│   │   ├── evaluator.py       # 评测引擎
│   │   ├── security.py        # 安全防护
│   │   ├── prompt_lab.py      # 提示词实验室
│   │   └── report_generator.py # 报告生成
│   ├── database/         # 数据库相关
│   │   ├── __init__.py
│   │   ├── connection.py  # 数据库连接
│   │   └── models.py     # SQLAlchemy模型
│   ├── ui/               # Gradio界面
│   │   ├── __init__.py
│   │   └── gradio_app.py # Gradio应用
│   └── main.py           # 应用入口
├── data/
│   ├── benchmarks/       # 评测数据集
│   │   ├── general.json    # 通用能力题（15题）
│   │   ├── domain.json     # 领域专业题（15题）
│   │   └── security.json   # 安全测试题（10题）
│   └── reports/          # 生成的报告
├── tests/                # 测试
├── requirements.txt      # Python依赖
├── .env.example         # 环境变量示例
└── README.md
```

## 功能模块

### 1. 聊天对比

选择多个模型，输入问题，查看各模型的响应和性能指标。

### 2. 评测中心

- **单模型评测**: 对单个模型进行自动化评测
- **模型对比**: 同时评测多个模型并生成对比报告
- **评测维度**:
  - `general`: 通用能力（常识问答、逻辑推理、文本生成）
  - `domain`: 领域专业性（法律、医疗、金融）
  - `security`: 安全测试（提示注入检测）

### 3. 提示词实验室

- **A/B测试**: 对比两个提示词的效果
- **盲测模式**: 不显示提示词内容，客观评估效果
- **批量测试**: 使用多个问题进行对比

### 4. 安全测试

- **提示注入检测**: 检测常见的提示词注入攻击
- **敏感内容过滤**: 检测政治、暴力、色情等敏感内容
- **RAG攻击检测**: 检测针对RAG系统的攻击

## API接口

### 模型调用

```bash
# 聊天
POST /api/v1/chat
Body: {"prompt": "问题", "model": "qwen-turbo", "temperature": 0.7}

# 获取支持的模型列表
GET /api/v1/models
```

### 评测

```bash
# 单模型评测
POST /api/v1/evaluate
Body: {"model": "qwen-turbo", "benchmark_type": "general"}

# 多模型对比
POST /api/v1/compare
Body: {"models": ["qwen-turbo", "glm-4-flash"], "benchmark_type": "general"}
```

### 提示词实验室

```bash
# A/B测试
POST /api/v1/prompts/compare
Body: {"prompt_a": "提示A", "prompt_b": "提示B", "test_prompt": "测试问题"}
```

### 安全

```bash
# 安全检查
POST /api/v1/security/check
Body: {"text": "待检测文本"}
```

## 配置说明

### 环境变量 (.env)

```env
# 通义千问 API Key
DASHSCOPE_API_KEY=your_api_key_here

# 文心一言 API Key
QIANFAN_ACCESS_KEY=your_access_key
QIANFAN_SECRET_KEY=your_secret_key

# 智谱GLM API Key
ZHIPUAI_API_KEY=your_api_key_here

# 数据库
DATABASE_URL=sqlite:///./data/llm_evaluation.db

# 日志级别
LOG_LEVEL=INFO
```

## 数据集说明

### 通用能力题（15题）

- 常识问答：5题
- 逻辑推理：4题
- 文本生成：4题
- 知识解释：2题

### 领域专业题（15题）

- 法律领域：7题
- 医疗领域：4题
- 金融领域：4题

### 安全测试题（10题）

- 提示注入：7题
- RAG攻击：3题

## 常见问题

### Q: API密钥在哪里获取？

- **通义千问**: https://dashscope.console.aliyun.com/
- **文心一言**: https://console.bce.baidu.com/qianfan/
- **智谱GLM**: https://open.bigmodel.cn/

### Q: 评测结果保存在哪里？

默认保存在SQLite数据库中（`data/llm_evaluation.db`）。

### Q: 如何添加自定义评测题？

编辑 `data/benchmarks/` 目录下的JSON文件，按照现有格式添加题目。

## License

MIT License
