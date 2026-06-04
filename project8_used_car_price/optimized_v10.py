# -*- coding: utf-8 -*-
"""
二手车价格预测 - 优化v10版（极简调试版）
目标: 测试集MAE < 450
问题: v9 CV 488, 测试集 4724 (灾难性失败）
策略: 极简 + 调试 + 检查分布
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("="*70)
print("优化v10版 - 极简调试")
print("问题: v9 CV 488, 测试集 4724")
print("策略: 极简 + 调试诊断")
print("="*70)

# ==================== 1. 数据加载 ====================
print("\n【步骤1】数据加载...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"训练集: {len(train)}, 测试集: {len(test)}")

# ==================== 2. 数据诊断 ====================
print("\n【步骤2】数据诊断...")

print("训练集价格统计:")
print(f"  Mean: {train['price'].mean():.2f}")
print(f"  Median: {train['price'].median():.2f}")
print(f"  Min: {train['price'].min():.2f}")
print(f"  Max: {train['price'].max():.2f}")
print(f"  Std: {train['price'].std():.2f}")

# 检查关键字段分布
print("\n关键特征分布（训练集 vs 测试集）:")
for col in ['power', 'kilometer']:
    print(f"\n{col}:")
    print(f"  训练 - Mean: {train[col].mean():.2f}, Std: {train[col].std():.2f}")
    print(f"  测试 - Mean: {test[col].mean():.2f}, Std: {test[col].std():.2f}")

# v特征分布
v_cols = [f'v_{i}' for i in range(15)]
v_train_mean = train[v_cols].mean().mean()
v_test_mean = test[v_cols].mean().mean()
print(f"\nv特征平均值:")
print(f"  训练集: {v_train_mean:.2f}")
print(f"  测试集: {v_test_mean:.2f}")
print(f"  差异: {abs(v_train_mean - v_test_mean):.2f}")

# ==================== 3. 极简特征工程 ====================
print("\n【步骤3】极简特征工程...")

sale_ids = test['SaleID'].values
test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

# 只保留最核心的特征（避免过拟合）
data['car_age'] = (data['creatDate'] // 10000 - data['regDate'] // 10000).clip(lower=0)

# 删除所有v特征（可能导致过拟合）
drop_v = v_cols + ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']

# 保留核心业务特征
core_features = ['price', 'power', 'kilometer', 'car_age', 'brand', 'model',
                'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']

# 删除无用列
data = data[core_features]

print(f"特征工程完成，特征数: {data.shape[1]}")
print(f"保留特征: {list(data.columns)}")

# ==================== 4. 缺失值处理 ====================
print("\n【步骤4】缺失值处理...")

numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=['object']).columns

for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

for col in categorical_cols:
    mode_val = data[col].mode()[0] if not data[col].mode().empty else 'unknown'
    data[col] = data[col].fillna(mode_val)

print(f"数值特征: {len(numeric_cols)}, 分类特征: {len(categorical_cols)}")

# ==================== 5. Label Encoding ====================
print("\n【步骤5】分类特征编码...")

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    print(f"  {col}: {len(le.classes_)} 类别")

# 分离数据
train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)

if 'price' in test_data.columns:
    test_data = test_data.drop(columns=['price'])

X = train_data.drop(columns=['price'])
y = train_data['price'].values

print(f"最终特征数: {X.shape[1]}")
print(f"特征列: {list(X.columns)}")

# ==================== 6. 极保守训练 ====================
print("\n【步骤6】极保守训练...")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 极保守参数（避免过拟合）
cat_params = {
    'iterations': 1000,  # 大幅减少迭代
    'learning_rate': 0.1,  # 高学习率
    'depth': 4,  # 极浅
    'l2_leaf_reg': 15,  # 强正则
    'random_strength': 1.0,
    'bagging_temperature': 1.0,
    'loss_function': 'MAE',
    'random_seed': SEED,
    'verbose': 0,
    'early_stopping_rounds': 50
}

cat_test_preds = np.zeros(len(test_data))
fold_maes = []

print("开始5折交叉验证训练...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # CatBoost
    print("  训练CatBoost...")
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)

    cat_pred_val = cat_model.predict(X_val)
    cat_mae = mean_absolute_error(y_val, cat_pred_val)
    fold_maes.append(cat_mae)

    if fold == 0:
        cat_test_preds += cat_model.predict(test_data) / 5

    print(f"  CatBoost MAE: {cat_mae:.2f}")

    # 调试：检查预测范围
    print(f"  调试 - 预测范围: [{cat_pred_val.min():.2f}, {cat_pred_val.max():.2f}]")
    print(f"  调试 - 真实范围: [{y_val.min():.2f}, {y_val.max():.2f}]")

# ==================== 7. 最终结果 ====================
print(f"\n{'='*70}")
print("【最终结果】5折交叉验证平均")
print(f"{'='*70}")
print(f"CatBoost 平均 MAE: {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f}")

final_mae = np.mean(fold_maes)

print(f"\n{'='*70}")
print(f"最终CV MAE: {final_mae:.2f}")
print(f"目标MAE: 450")
print(f"差距: {final_mae - 450:.2f}")

# ==================== 8. 生成测试集预测 ====================
print("\n【步骤8】生成测试集预测...")

final_pred = cat_test_preds
final_pred = np.maximum(final_pred, 50)

# 调试：检查最终预测
print(f"\n调试 - 最终预测范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"调试 - 预测均值: {final_pred.mean():.2f}")
print(f"调试 - 训练集价格均值: {y.mean():.2f}")

# 简单裁剪
upper_bound = final_pred.mean() + 2 * final_pred.std()
final_pred = np.clip(final_pred, None, upper_bound)
print(f"应用2σ裁剪，上限: {upper_bound:.2f}")

# 保存结果
submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit.to_csv('optimized_v10_submit.csv', index=False)

print(f"\n结果已保存: optimized_v10_submit.csv")
print(f"预测价格范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"预测价格均值: {final_pred.mean():.2f}")

print(f"\n{'='*70}")
print("✅ 优化v10版（极简调试）完成！")
print(f"{'='*70}")
