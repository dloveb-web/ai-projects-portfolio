# 项目2: 多模态RAG智能文档中心

支持图文混合的企业级智能文档系统，完美适配产品说明书、技术手册、财报等复杂文档。

## 核心功能

1. **多格式文档解析** - 自动提取文本/图片/表格/流程图
2. **LayoutLM文档元素检测** - PDF解析时自动检测文档布局元素（标题、段落、表格等）
3. **语义分块** - 使用RecursiveCharacterTextSplitter进行智能文本分块
4. **混合检索 + HyDE优化** - BM25 + 向量检索，支持元数据过滤
5. **图文混合问答** - 支持"看图提问""查表提问"
6. **引用溯源** - 自动标注来源
7. **文档版本管理** - 版本对比与回滚
8. **文档选择过滤** - 可选择特定文档进行问答

## 技术栈

- **后端**: FastAPI
- **前端**: Gradio
- **向量数据库**: FAISS
- **文档解析**: LayoutLMv3, PyMuPDF, python-docx
- **OCR**: Tesseract
- **Embedding**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: 通义千问 (Qwen)

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 DASHSCOPE_API_KEY

# 4. 启动服务
python src/main.py
```

## 目录结构

```
project2_multimodal_rag/
├── src/
│   ├── api/              # API路由
│   ├── core/
│   │   ├── parser.py     # 文档解析 (PDF/DOCX/图片 + LayoutLM)
│   │   ├── chunker.py    # 语义分块
│   │   ├── retriever.py  # 混合检索引擎 (BM25 + FAISS)
│   │   ├── rag_chain.py  # RAG链
│   │   └── version.py    # 版本管理
│   ├── models/           # 数据模型
│   ├── ui/               # Gradio界面
│   └── config.py         # 配置管理
├── data/                 # 文档存储
├── .env.example          # 环境变量示例
└── README.md
```

## 配置说明

主要配置项（通过 `.env` 文件或环境变量设置）：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `API_HOST` | API服务地址 | `0.0.0.0` |
| `API_PORT` | API服务端口 | `8000` |
| `INIT_LAYOUTLM` | 是否启用LayoutLM | `true` |
| `USE_VECTOR_SEARCH` | 是否启用向量检索 | `true` |
| `CHUNK_SIZE` | 文本分块大小 | `500` |
| `TOP_K_RESULTS` | 检索返回结果数 | `5` |
| `DASHSCOPE_API_KEY` | 通义千问API密钥 | - |

## API端点

- `POST /documents/upload` - 上传文档
- `GET /documents` - 获取文档列表
- `DELETE /documents/{id}` - 删除文档
- `POST /query` - 智能问答
- `POST /query-with-filter` - 带过滤条件的问答
- `GET /documents/{id}/versions` - 获取文档版本
