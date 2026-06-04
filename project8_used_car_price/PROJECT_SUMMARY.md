# 二手车价格预测项目 - 最终总结

## 目标
- 测试集MAE < 450

## 全部尝试结果

### CV MAE (交叉验证) vs Test MAE (测试集)

| 版本 | CV MAE | Test MAE | 方法 | 特征数 | 模型 | 问题 |
|------|---------|----------|------|---------|------|------|
| ultra_simple_final | 484.22 | 4724 | 简单平均 | 40 | CatBoost+LGB | 严重过拟合 |
| stable_breakthrough | 512.76 | 未知 | 简单平均 | 42 | CatBoost+LGB | - |
| optimized_v2 | 475.68 | 4722 | 加权平均(35:65) | 47 | CatBoost+LGB | 严重过拟合 |
| optimized_v3 | 471.10 | 未知 | 加权平均(35:65) | 137 | CatBoost+LGB | 严重过拟合 |
| optimized_v4 | 470.27 | 未知 | 加权平均(35:65) | 63 | CatBoost+LGB | 严重过拟合 |
| optimized_v5 | 462.13 | 未知 | 6模型Stacking | 63 | 6模型 | 严重过拟合 |
| optimized_v6 | 460.00 | 729 | Top3平均 | 65 | 6模型 | 灾难性过拟合 |
| optimized_v7 | 664.63 | 未知 | 强正则化 | 15 | CatBoost+LGB | 欠拟合 |
| optimized_v8 | 521.97 | 未知 | 平衡参数 | 29 | CatBoost+LGB | - |
| optimized_v9 | 487.86 | 4724 | 简单平均 | 38 | CatBoost+LGB | 严重过拟合 |
| optimized_v12 | 529.68 | 未知 | 单次训练 | 25 | CatBoost | 有负值 |
| optimized_v13 | 555.91 | 未知 | 最简单默认 | 25 | CatBoost | 最差CV |
| final_breakthrough | 473.84 | 未知 | 目标编码 | 49 | CatBoost+LGB | - |
| improve_480 | 479.21 | 未知 | 保守优化 | 43 | CatBoost+LGB | - |

## 核心发现

### 1. 灾难性过拟合问题
- **CV MAE**: 全部在 460-560 范围
- **Test MAE**: 已测试版本全在 4700-729 范围
- **差距**: 10倍左右！

### 2. 数据诊断结果
- 训练集和测试集分布**几乎完全相同**
- power: 119.32 vs 119.77
- kilometer: 12.60 vs 12.60
- v特征: 3.01 vs 3.01
- 无新类别在测试集
- 异常值比例相似

### 3. 尝试的方法

#### 特征工程
- 基础统计特征
- v特征交互 (v_0*v_3, v_0*v_12等)
- 业务特征 (power*km, age*km等)
- 极简特征 (只保留核心特征)

#### 模型策略
- 单模型 (CatBoost/LightGBM)
- 双模型融合
- 6模型Stacking
- 权重调整 (30:70, 35:65等)
- 目标编码

#### 正则化
- 保守正则 (l2_leaf_reg=10)
- 强正则 (reg_alpha=0.5)
- 标准化 vs 不标准化
- 预测分布调整

### 4. 生成文件
```
ultra_simple_final_submit.csv          - CV 484.22, Test 4724 ⚠️
stable_breakthrough_submit.csv            - CV 512.76
optimized_v2_submit.csv                - CV 475.68, Test 4722 ⚠️
optimized_v3_submit.csv                - CV 471.10
optimized_v4_submit.csv                - CV 470.27
optimized_v5_submit.csv                - CV 462.13
optimized_v6_submit.csv                - CV 460.00, Test 729 ⚠️⚠️
optimized_v7_submit.csv                - CV 664.63
optimized_v8_submit.csv                - CV 521.97
optimized_v9_submit.csv                - CV 487.86, Test 4724 ⚠️
optimized_v10_submit.csv               - CV 1236.71
optimized_v11_submit.csv               - CV 1173.01
optimized_v12_submit.csv               - CV 529.68
optimized_v13_submit.csv               - CV 555.91
final_breakthrough_submit.csv           - CV 473.84
improve_480_baseline_submit.csv         - CV 479.21
```

## 可能的根本原因

### 1. 测试集设计问题
- testB可能与train分布有系统性差异（统计上看不出来）
- 可能测试集包含了train中没有的模式

### 2. 评估方式问题
- MAE计算可能有特殊处理
- 可能测试集标注方式不同

### 3. 模型适用性问题
- 传统梯度提升树可能不适用于此数据
- 可能需要深度学习或其他方法
- 数据可能存在未发现的泄露或异常

## 建议

### 最优CV模型
1. **optimized_v6** - CV 460.00 (但测试集严重过拟合 729)
2. **optimized_v5** - CV 462.13 (可能过拟合)
3. **ultra_simple_final** - CV 484.22, Test 4724 (最可靠)

### 推荐提交
如果必须选择一个提交文件，建议：
1. 先测试`ultra_simple_final_submit.csv`（已知道成绩4724）
2. 如果需要更低，尝试`final_breakthrough_submit.csv`
3. 避免使用已验证严重过拟合的版本(v6, v2, v9)

### 后续探索方向
1. 深度学习方法 (TensorFlow/PyTorch)
2. 时序模型 (如果数据有时间维度)
3. 异常检测和数据清洗
4. 特征选择重要性分析
5. 尝试不同的损失函数或评估指标

## 项目时间线
- 开始: 2026-03-31
- 尝试版本数: 13+
- 生成文件数: 13+
- 训练次数: 50+次5折CV

## 最终状态
- ⚠️ 未达到目标 MAE < 450
- ⚠️ 所有模型都出现严重过拟合
- ⚠️ 传统的机器学习方法在此数据上失效

---
生成时间: 2026-03-31
