# LCM MES 前端

基于 React + TypeScript + Vite + Ant Design 的 LCM MES 前端项目

## 技术栈

- React 18
- TypeScript
- Vite 5
- Ant Design 5
- React Router 6
- React Query
- ECharts
- Axios

## 功能模块

### 1. 用户登录
- 支持多角色登录
- Token 认证

### 2. 工程部（admin）
- 全局监控仪表盘
- 设备管理
- 设备维护记录

### 3. 生产部（operator）
- 工单看板
- 工单操作（开始、报工、完成）
- 产品追溯
- 不良品维修

### 4. PMC（manager）
- 生产计划看板
- 订单管理
- 物料管理
- 库存预警

### 5. 品质部（inspector）
- 质量监控中心
- SPC控制图
- 检测记录
- 不良品管理

### 6. 货仓部（warehouse）
- 库存看板
- Lot入库
- 装箱管理
- 出货管理

## 开发

### 安装依赖

```bash
npm install
```

### 启动开发服务器

```bash
npm run dev
```

前端会运行在 http://localhost:3000

### 构建

```bash
npm run build
```

## 接口说明

后端API运行在 http://localhost:8000，前端通过代理访问

主要API：
- `/api/v1/auth/*` - 认证相关
- `/api/v1/production/*` - 生产相关
- `/api/v1/equipment/*` - 设备相关
- `/api/v1/quality/*` - 质量相关
- `/api/v1/material/*` - 物料相关
- `/api/v1/warehouse/*` - 货仓相关
- `/api/v1/reports/*` - 报表相关

## 测试账号

| 角色 | 用户名 | 密码 | 部门 |
|------|--------|------|------|
| 工程部 | admin | admin123 | 工程部 |
| PMC | manager | manager123 | PMC |
| 生产部 | operator | operator123 | 生产部 |
| 品质部 | inspector | inspector123 | 品质部 |
| 货仓部 | warehouse | warehouse123 | 货仓部 |
