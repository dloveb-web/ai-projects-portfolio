# LCM MES系统 - 液晶显示模组制造执行系统

基于AI升级的制造执行系统，支持PCS级追溯、智能质量检测、预测性维护等功能。

## 技术栈

- 后端: FastAPI + SQLAlchemy + SQLite/PostgreSQL
- 前端: React + TypeScript + Ant Design
- 实时通信: WebSocket + MQTT
- AI能力: 智能质量检测、预测性维护、智能排程

## 项目结构

```
project6_lcm_mes/
├── src/
│   ├── api/              # API路由
│   ├── core/             # 核心配置和安全
│   ├── services/         # 业务逻辑层
│   ├── database/         # 数据库模型
│   ├── websocket/        # WebSocket管理
│   └── main.py           # 入口文件
├── frontend/             # React前端
├── tests/                # 测试
└── requirements.txt      # Python依赖
```

## 快速开始

```bash
cd project6_lcm_mes
cp .env.example .env
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/init_db.py
python -m src.main
```

## 访问地址

- API: http://localhost:8000
- Swagger文档: http://localhost:8000/docs

## 责任部门

| 部门 | 核心职责 |
|------|----------|
| 工程部 | 全局管理、设备管理、系统配置 |
| 生产部 | 在线操作、上料、检测、维修 |
| PMC | 生产计划、物料状态、排程 |
| 品质部 | 来料检验、过程质量监控 |
| 货仓 | 来料录入、出货管理、库存 |

## 产品编码体系

| 阶段 | 编码位数 | 说明 |
|------|----------|------|
| 基板 | 8位 | 切割后基板序号 |
| 半成品 | 31位 | 8位基板码 + 玻璃盖板编码 |
| 成品 | 71位 | 31位 + 背光码 + 工厂序号 |
