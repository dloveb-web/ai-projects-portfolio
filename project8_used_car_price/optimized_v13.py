# -*- coding: utf-8 -*-
"""
二手车价格预测 - 优化v13版（最终极简）
目标: 测试集MAE < 450
问题: 全部模型CV ~480, Test ~4700
策略: 最原始方法 - 只用基本特征 + CatBoost默认参数
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("="*70)
print("优化v13版 - 最终极简")
print("策略: 最原始方法 - 默认参数")
print("="*70)

# ==================== 1. 数据加载 ====================
print("\n【步骤1】数据加载...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

sale_ids = test['SaleID'].values
test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

print(f"训练集: {len(train)}, 测试集: {len(test)}")

# ==================== 2. 极简处理 ====================
print("\n【步骤2】极简处理...")

# 时间特征
data['car_age'] = (data['creatDate'] // 10000 - data['regDate'] // 10000).clip(lower=0)

# 处理notRepairedDamage
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0')

# 选择最基础的特征
feature_columns = ['price', 'power', 'kilometer', 'car_age', 'brand', 'model',
                  'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage',
                  'regionCode']

# 添加所有v特征
for i in range(15):
    feature_columns.append(f'v_{i}')

data = data[feature_columns]

print(f"特征数: {data.shape[1]}")

# ==================== 3. 缺失值处理 ====================
print("\n【步骤3】缺失值处理...")

numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=['object']).columns

for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

for col in categorical_cols:
    mode_val = data[col].mode()[0] if not data[col].mode().empty else 'unknown'
    data[col] = data[col].fillna(mode_val)

print(f"数值特征: {len(numeric_cols)}, 分类特征: {len(categorical_cols)}")

# ==================== 4. 分离数据 ====================
print("\n【步骤4】分离数据...")

train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)

X_train = train_data.drop(columns=['price'])
y_train = train_data['price'].values
X_test = test_data.drop(columns=['price'])

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# ==================== 5. 极简训练 ====================
print("\n【步骤5】极简训练（默认参数）...")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 使用默认参数（最简单）
cat_params = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'depth': 6,
    'random_seed': SEED,
    'verbose': 100
}

fold_maes = []
test_preds = np.zeros(len(X_test))

print("开始5折交叉验证训练...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\nFold {fold+1}/5")

    X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_va = y_train[train_idx], y_train[val_idx]

    # CatBoost
    print("  训练CatBoost...")
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=50)

    pred = cat_model.predict(X_va)
    mae = mean_absolute_error(y_va, pred)
    fold_maes.append(mae)

    print(f"  CatBoost MAE: {mae:.2f}")

    if fold == 0:
        test_preds += cat_model.predict(X_test) / 5

# ==================== 6. 最终结果 ====================
print(f"\n{'='*70}")
print("【最终结果】5折交叉验证平均")
print(f"{'='*70}")
print(f"CatBoost 平均 MAE: {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f}")

final_mae = np.mean(fold_maes)

print(f"\n{'='*70}")
print(f"最终CV MAE: {final_mae:.2f}")
print(f"目标MAE: 450")
print(f"差距: {final_mae - 450:.2f}")

# ==================== 7. 生成测试集预测 ====================
print("\n【步骤6】生成测试集预测...")

final_pred = test_preds

# 确保正数
final_pred = np.maximum(final_pred, 10)

print(f"\n预测统计:")
print(f"  均值: {final_pred.mean():.2f}")
print(f"  中位数: {np.median(final_pred):.2f}")
print(f"  最小值: {final_pred.min():.2f}")
print(f"  最大值: {final_pred.max():.2f}")

# 保存结果
submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit.to_csv('optimized_v13_submit.csv', index=False)

print(f"\n结果已保存: optimized_v13_submit.csv")
print(f"文件大小: {len(submit)}")

print(f"\n{'='*70}")
print("✅ 优化v13版（最终极简）完成！")
print(f"{'='*70}")
