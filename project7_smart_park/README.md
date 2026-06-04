# Smart Park - 智慧园区

基于某大型智慧园区综合运管平台设计理念的现代智慧园区系统。

## 技术栈

- 后端: FastAPI + SQLAlchemy + PostgreSQL
- 前端: React + TypeScript + Ant Design
- 实时通信: WebSocket + MQTT
- 缓存: Redis
- 物联网: LoRa / Modbus

## 项目架构

### 架构分层
```
┌─────────────────────────────────────────────────────────────────┐
│                     运营管理平台层                              │
│  领导驾驶舱 | 园区综合监测 | 设备集中监控 | 对客服务平台        │
├─────────────────────────────────────────────────────────────────┤
│                         应用层                                │
│  智慧安防 | 智慧消防 | 智慧能源 | 智慧通行 | 设备管理          │
├─────────────────────────────────────────────────────────────────┤
│                        数字平台层                              │
│  数据集成平台 | 大数据支撑 | 物联网支撑 | 视频支撑              │
├─────────────────────────────────────────────────────────────────┤
│                         传输层                                │
│  有线网络 | WiFi | 蓝牙/LoRa | NB-IoT/4G/5G                  │
├─────────────────────────────────────────────────────────────────┤
│                      园区基础设施                              │
│  视频监控 | 防盗报警 | 电子巡更 | 公共广播 | 楼宇自控          │
└─────────────────────────────────────────────────────────────────┘
```

## 项目结构

```
project7_smart_park/
├── src/
│   ├── api/                    # API层
│   │   ├── routes/             # 路由模块
│   │   │   ├── auth.py         # 认证路由
│   │   │   ├── park.py         # 园区管理路由
│   │   │   ├── device.py       # 设备管理路由
│   │   │   ├── access.py       # 通行管理路由
│   │   │   ├── energy.py       # 能源管理路由
│   │   │   └── security.py     # 安防监控路由
│   │   ├── models.py           # Pydantic数据模型
│   │   └── dependencies.py     # 依赖注入
│   ├── core/                   # 核心模块
│   │   ├── config.py           # 配置管理
│   │   └── security.py         # 安全工具
│   ├── database/               # 数据库层
│   │   ├── models/             # 数据模型
│   │   │   ├── user.py         # 用户模型
│   │   │   ├── park.py         # 园区模型
│   │   │   ├── device.py       # 设备模型
│   │   │   ├── access.py       # 通行模型
│   │   │   ├── energy.py       # 能源模型
│   │   │   └── security.py     # 安防模型
│   │   ├── base.py             # ORM基类
│   │   └── session.py          # 会话管理
│   ├── services/               # 业务逻辑层
│   ├── websocket/              # WebSocket管理
│   ├── tasks/                  # 后台任务
│   └── main.py                 # 应用入口
├── tests/                      # 测试
├── docker/                     # Docker配置
├── requirements.txt            # Python依赖
├── .env.example                # 环境变量示例
├── alembic.ini                 # 数据库迁移配置
└── README.md                   # 项目说明
```

## 功能模块

| 模块 | 功能说明 |
|------|----------|
| **园区管理** | 园区、楼栋、房间信息管理 |
| **设备管理** | 设备注册、状态监控、告警管理 |
| **智慧通行** | 人员管理、访客管理、通行记录 |
| **能源管控** | 能耗计量、统计分析、节能管理 |
| **安防监控** | 摄像头管理、告警处理、巡更管理 |

## API接口

| 模块 | 路径 | 说明 |
|------|------|------|
| 认证 | `/api/auth/*` | 登录、注册 |
| 园区 | `/api/park/*` | 园区、楼栋、房间管理 |
| 设备 | `/api/device/*` | 设备类型、设备管理 |
| 通行 | `/api/access/*` | 人员、访客管理 |
| 能源 | `/api/energy/*` | 计量表、能耗记录 |
| 安防 | `/api/security/*` | 摄像头、告警管理 |

## 快速开始

### 后端启动

```bash
cd project7_smart_park
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 配置数据库连接
python -m src.main
```

### 启动数据库服务

```bash
cd docker
docker-compose up -d
```

### API文档

访问 http://localhost:8000/docs 查看Swagger文档

## 数据库表结构

```
用户表(users)          园区表(parks)        楼栋表(buildings)
├─ id                  ├─ id                ├─ id
├─ username            ├─ name              ├─ park_id
├─ email               ├─ code              ├─ name
├─ hashed_password     ├─ address           ├─ code
├─ full_name           ├─ area              └─ ...
├─ is_active           └─ ...
└─ ...

房间表(rooms)          设备表(devices)      人员表(persons)
├─ id                  ├─ id                ├─ id
├─ building_id         ├─ device_type_id    ├─ name
├─ name                ├─ room_id           ├─ id_card
├─ code                ├─ name              └─ ...
└─ ...                 └─ ...

访客表(visitors)       能耗表(energy_consumptions)   安防告警表(security_alerts)
├─ id                  ├─ id                        ├─ id
├─ name                ├─ meter_id                  ├─ camera_id
├─ company             ├─ value                     ├─ alert_type
└─ ...                 └─ ...                       └─ ...
```
