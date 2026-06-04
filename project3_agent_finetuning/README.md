# 项目3: 智能体与微调一体化平台

"微调 + Agent"一体化开发平台，先解决通用模型的垂直领域适配问题，再构建多智能体协作系统。

## ⚠️ 重要说明

**当前版本为 MVP/教学原型**，所有核心功能（微调、评估、LLM调用）默认运行在 **MOCK 模式**：
- 所有训练过程都是模拟的，不会真正调用 LLaMA-Factory 或其他训练框架
- 所有评估指标都是随机生成的，非真实模型评估结果
- 所有LLM调用在没有dashscope API时使用预设回复

**如需启用真实功能**，请设置环境变量：
```bash
export MOCK_MODE=false
export DASHSCOPE_API_KEY=your_api_key_here
```

## 核心功能

1. **QLoRA一键微调** - 微调任务管理（模拟模式）
2. **PEFT方法对比** - LoRA vs QLoRA vs Adapter 对比分析
3. **DPO偏好对齐** - 偏好对齐任务管理
4. **微调后自动评估** - 模型评估系统
5. **ReAct单Agent** - 法律助手+检索工具（支持工具调用）
6. **双Agent协作** - 任务分解 + 执行
7. **视觉Agent** - 合同图片风险点分析

## 技术栈

- **后端**: FastAPI
- **前端**: Gradio
- **微调框架**: PEFT（当前为模拟实现）
- **Agent框架**: 自定义实现（基于LangChain设计理念）
- **数据库**: SQLite（通过SQLAlchemy）
- **LLM**: DashScope（可选，当前为模拟模式）

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 复制环境变量配置
cp .env.example .env
# 编辑 .env 文件，填入您的 API Key

# 4. 启动服务（后端API）
python src/main.py

# 5. 启动Gradio界面（新终端）
python src/ui/gradio_app.py
```

## 目录结构

```
project3_agent_finetuning/
├── src/
│   ├── api/                  # API路由和Pydantic模型
│   │   ├── routes.py         # API路由定义
│   │   └── models.py         # Pydantic请求/响应模型
│   ├── core/                 # 核心业务逻辑
│   │   ├── finetune.py       # 微调任务管理（模拟实现）
│   │   ├── agent.py          # Agent框架（ReAct/多Agent/视觉）
│   │   ├── evaluate.py       # 评估模块（模拟实现）
│   │   └── config.py         # 配置管理
│   ├── database/             # 数据库层
│   │   ├── connection.py     # 数据库连接和Session管理
│   │   └── models.py         # SQLAlchemy ORM模型
│   ├── ui/                   # Gradio界面
│   │   └── gradio_app.py     # Gradio应用
│   └── main.py               # FastAPI应用入口
├── tests/                    # 测试文件（pytest格式）
│   ├── test_finetune.py
│   └── test_agent.py
├── data/                     # 数据存储目录
│   └── finetuned_models/     # 微调后模型存储
├── .env.example              # 环境变量示例
├── requirements.txt          # Python依赖
└── README.md
```

## API接口

访问 `http://localhost:8000/docs` 查看完整的API文档。

主要接口：

### 微调管理
- `POST /api/v1/finetune/create` - 创建微调任务
- `POST /api/v1/finetune/tasks/{id}/start` - 启动微调任务
- `GET /api/v1/finetune/tasks` - 获取任务列表
- `GET /api/v1/finetune/methods` - 获取可用微调方法

### Agent管理
- `POST /api/v1/agent/create` - 创建Agent会话
- `POST /api/v1/agent/chat` - 与Agent对话
- `GET /api/v1/agent/types` - 获取可用Agent类型

### 评估
- `POST /api/v1/evaluate` - 评估模型
- `GET /api/v1/evaluate/reports` - 获取评估报告

## 环境变量配置

在 `.env` 文件中配置：

```env
# API配置
DASHSCOPE_API_KEY=your_api_key_here

# 数据库
DATABASE_URL=sqlite:///./data/agent_finetuning.db

# 模拟模式（true=使用模拟数据，false=使用真实服务）
MOCK_MODE=true

# 模型配置
DEFAULT_MODEL=Qwen/Qwen2-0.5B-Instruct
FINETUNED_MODEL_PATH=./data/finetuned_models

# CORS配置（允许的前端域名，逗号分隔）
CORS_ORIGINS=http://localhost:7860,http://localhost:3000
```

## 测试

```bash
# 安装pytest
pip install pytest

# 运行测试
python -m pytest tests/ -v
```

## 后续开发方向

1. **真实微调集成**：对接 LLaMA-Factory 或 PEFT 真实训练代码
2. **真实LLM调用**：集成 DashScope 或其他 LLM API
3. **真实评估框架**：集成 EleutherAI lm-evaluation-harness
4. **Agent工具扩展**：添加更多工具（搜索、数据库查询等）
5. **部署优化**：Docker容器化、Kubernetes部署

## 许可证

MIT License
