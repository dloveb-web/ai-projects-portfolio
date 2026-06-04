# PCBA 缺陷检测 — 边缘部署决策手册 v1.0

> 基于 M5 Mac Demo 验证 | 目标部署: Jetson Orin NX / TensorRT

---

## 0. 训练与验证环境

| 项目 | 配置 |
|------|------|
| 训练平台 | MacBook Air M5 / 32GB / MPS 后端 |
| 导出格式 | ONNX (通用) + CoreML (M5 验证用) |
| 性能基准 | M5 ONNX CPU 延迟 ~30ms (仅作格式验证，不代表产线性能) |
| 产线部署格式 | ONNX → TensorRT Engine (Jetson) 或 ONNX → MindSpore (华为) |

---

## 1. 模型选择

| 属性 | 值 |
|------|-----|
| 推荐模型 | YOLOv8n (轻量高速) / YOLOv8s (精度优先) |
| 输入尺寸 | 640×640 |
| 参数量 | 3.2M (nano) / 11.2M (small) |
| ONNX 文件大小 | ~6MB (nano) / ~24MB (small) |
| 格式 | ONNX opset 12, simplified |

---

## 2. 硬件选型建议

| 平台 | 推理引擎 | INT8 吞吐 | 单路 25fps 路数 | 预估售价 | 适用场景 |
|------|---------|----------|----------------|---------|---------|
| Jetson Orin Nano 8GB | TensorRT | ~200 fps | 2~4 路 | ¥3-4K | 单工位 AOI 复判 |
| Jetson Orin NX 16GB | TensorRT | ~500 fps | 8~16 路 | ¥5-7K | 多工位边缘推理 |
| T4 服务器 | TensorRT | ~2000 fps | 40+ 路 | ¥20-30K | 车间级集中推理 |
| 华为 Atlas 200 | MindSpore Lite | ~150 fps | 2~3 路 | ¥3-5K | 华为生态 |

> ⚠️ M5 本地 ONNX CPU 推理 ≠ 产线 TensorRT 性能！
> TensorRT INT8 量化后通常比 ONNX FP32 快 3~5 倍

---

## 3. 兼容性矩阵

```
训练 (M5 MPS)          导出            产线部署
─────────────────      ────          ──────────────
PyTorch .pt      →     ONNX    →    Jetson: TensorRT Engine
                       ↑            GPU Server: TensorRT Engine
                       │            Huawei: MindSpore Lite
PyTorch .pt      →     CoreML  →    Apple 生态专有 (产线不用)
```

---

## 4. ONNX → TensorRT 转换流程 (在 Jetson 上执行)

```bash
# 1. 将 M5 导出的 ONNX 传输到 Jetson
scp best.onnx user@jetson-ip:/home/user/models/

# 2. FP16 (精度损失极小，速度快 1.5~2x) ⭐ 推荐
trtexec --onnx=best.onnx --saveEngine=best_fp16.engine \
  --workspace=1024 --fp16

# 3. INT8 (速度最快，需校准数据)
trtexec --onnx=best.onnx --saveEngine=best_int8.engine \
  --workspace=1024 --int8 --calib=calibration.bin

# 4. 性能验证
trtexec --loadEngine=best_fp16.engine --batch=1 --iterations=1000
```

| 精度模式 | 速度 | 精度损失 | 推荐 |
|---------|------|---------|------|
| FP32 | 1x | 无 | 基线 |
| FP16 | 1.5~2x | < 0.1% | ⭐ 生产首选 |
| INT8 | 2~4x | 0.3~1.0% | 多路并发 |

---

## 5. 扩展性设计

- 单 Jetson Orin NX 16GB 可支撑 8~16 路 25fps 推理 (YOLOv8n INT8)
- 模型热更新: API 切换 TensorRT Engine 路径，新旧引擎共存
- 边缘节点管理: MQTT 统一模型分发 + 配置同步
- 断网续传: 本地 SQLite 缓存 + 恢复自动补传

---

## 6. 从 M5 Demo 到产线部署的 Gap 清单

| Gap | 现状 (M5) | 产线要求 | 补缺方式 |
|-----|----------|---------|---------|
| 推理引擎 | ONNX CPU/MPS | TensorRT INT8 | Jetson 上做 PTQ 量化 |
| 视频输入 | 单图片上传 | RTSP 多路流 | DeepStream pipeline |
| 模型服务 | FastAPI 单进程 | Triton + 动态 batch | 升级推理服务器 |
| 监控告警 | 无 | Prometheus + Grafana | 产线部署时加入 |
| 可靠性 | 无要求 | ≥99% 可用率 | 主备热切换 + 健康检查 |
| 模型更新 | 手动文件替换 | CI/CD 自动推送 | Jenkins + 模型注册中心 |
| 安全 | 无认证 | RBAC + 审计日志 | 集成企业统一认证 |
| 系统集成 | 独立运行 | MES/QMS 对接 | REST API 适配层 |

---

## 7. 单工位硬件部署清单

```
□ Jetson Orin NX 16GB ×1
□ 工业相机 (200万像素 POE) ×1-2
□ POE 交换机 ×1
□ 声光报警器 ×1 (可选)
□ 边缘终端机柜 ×1
□ 网线/电源线
```
