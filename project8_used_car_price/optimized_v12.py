# -*- coding: utf-8 -*-
"""
二手车价格预测 - 优化v12版（完全重置）
目标: 测试集MAE < 450
问题: 全部模型失败 (CV 480, Test 4700+)
策略: 完全不同方法 - 简单特征 + 纯CatBoost
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("="*70)
print("优化v12版 - 完全重置")
print("策略: 最简单方法 - 保留所有原始特征")
print("="*70)

# ==================== 1. 数据加载 ====================
print("\n【步骤1】数据加载...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"训练集: {len(train)}, 测试集: {len(test)}")

# ==================== 2. 极简处理 ====================
print("\n【步骤2】极简处理...")

sale_ids = test['SaleID'].values

# 只做最基础的处理
train['car_age'] = (train['creatDate'] // 10000 - train['regDate'] // 10000).clip(lower=0)
test['car_age'] = (test['creatDate'] // 10000 - test['regDate'] // 10000).clip(lower=0)

# 处理notRepairedDamage
train['notRepairedDamage'] = train['notRepairedDamage'].replace('-', '0')
test['notRepairedDamage'] = test['notRepairedDamage'].replace('-', '0')

# 选择特征（保留所有原始特征）
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']

# 明确列出特征
feature_columns = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'power', 'kilometer',
                  'notRepairedDamage', 'regionCode',
                  'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7',
                  'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14', 'car_age']

# 提取目标
y_train = train['price']

print(f"特征数: {len(feature_columns)}")
print(f"特征列: {feature_columns}")

# 从数据中提取特征
X_train = train[feature_columns].copy()
X_test = test[feature_columns].copy()

# 标记分类特征
cat_features = ['notRepairedDamage']
print(f"分类特征: {cat_features}")

# ==================== 3. 单次训练 ====================
print("\n【步骤3】单次训练（不做CV）...")

# 划分训练集和验证集
X_tr, X_val, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)

print(f"训练集: {len(X_tr)}, 验证集: {len(X_val)}")

# 最简单的CatBoost参数
cat_params = {
    'iterations': 2000,
    'learning_rate': 0.05,
    'depth': 7,
    'l2_leaf_reg': 5,
    'random_strength': 0.8,
    'bagging_temperature': 0.8,
    'loss_function': 'MAE',
    'random_seed': SEED,
    'verbose': 100
}

print("\n训练模型...")
cat_model = CatBoostRegressor(**cat_params)
cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_va), verbose=50)

# 验证
val_pred = cat_model.predict(X_val)
val_mae = mean_absolute_error(y_va, val_pred)

print(f"\n验证集 MAE: {val_mae:.2f}")
print(f"验证集预测均值: {val_pred.mean():.2f}")
print(f"验证集真实均值: {y_va.mean():.2f}")
print(f"比例: {val_pred.mean()/y_va.mean():.2f}")

# ==================== 4. 预测测试集 ====================
print("\n【步骤4】预测测试集...")

# 只使用特征列（不包含price）
feature_cols = [col for col in X_train.columns if col != 'price']

print(f"特征数: {len(feature_cols)}")
print(f"特征列: {feature_cols}")

test_pred = cat_model.predict(X_test[feature_cols])

print(f"测试集预测统计:")
print(f"  均值: {test_pred.mean():.2f}")
print(f"  中位数: {np.median(test_pred):.2f}")
print(f"  最小值: {test_pred.min():.2f}")
print(f"  最大值: {test_pred.max():.2f}")

# 简单后处理 - 确保正数
test_pred = np.maximum(test_pred, 10)

# ==================== 5. 保存结果 ====================
print("\n【步骤5】保存结果...")

submit = pd.DataFrame({'SaleID': sale_ids, 'price': test_pred})
submit.to_csv('optimized_v12_submit.csv', index=False)

print(f"\n结果已保存: optimized_v12_submit.csv")
print(f"文件大小: {len(submit)}")

print(f"\n{'='*70}")
print("✅ 优化v12版（完全重置）完成！")
print(f"{'='*70}")
