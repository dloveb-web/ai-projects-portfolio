# -*- coding: utf-8 -*-
"""
二手车价格预测 - 中位数填充异常值 + 简化优化版
目标：MAE <= 450
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("中位数填充异常值 + 简化训练版")
print("="*60)

# ==================== 数据加载与处理 ====================
print("\n加载数据...")
train = pd.read_csv('train_median_fill.csv')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

# 立即处理 notRepairedDamage
for df in [train, test]:
    if 'notRepairedDamage' in df.columns and df['notRepairedDamage'].dtype == object:
        df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
        df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce').fillna(0)

print(f"训练集: {train.shape}, 测试集: {test.shape}")

# 基础特征工程（简化版）
print("\n特征工程...")
for df in [train, test]:
    # 日期特征
    df['reg_year'] = df['regDate'] // 10000
    df['creat_year'] = df['creatDate'] // 10000
    df['car_age'] = (df['creat_year'] - df['reg_year']).clip(lower=0)

    # v特征统计
    v_cols = [f'v_{i}' for i in range(15)]
    available_v_cols = [col for col in v_cols if col in df.columns]
    if len(available_v_cols) >= 3:
        df['v_mean'] = df[available_v_cols].mean(axis=1)
        df['v_0_v_3'] = df['v_0'] * df['v_3']
        df['v_0_v_12'] = df['v_0'] * df['v_12']

    # 组合特征
    df['power_km'] = df['power'] * df['kilometer']
    df['log_power'] = np.log1p(df['power'])
    df['log_km'] = np.log1p(df['kilometer'])

print(f"✓ 特征工程完成，特征数: {train.shape[1]-1}")

# 准备数据
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
X = train.drop(columns=[c for c in drop_cols if c in train.columns])
y = train['price']
test_clean = test.drop(columns=[c for c in drop_cols if c in test.columns])

# 只选择数值列（排除price）
numeric_cols = [col for col in X.columns if col != 'price' and pd.api.types.is_numeric_dtype(X[col])]
print(f"数值特征数: {len(numeric_cols)}")

X = X[numeric_cols]
test_clean = test_clean[[col for col in numeric_cols if col in test_clean.columns]]

# 填充缺失值
for col in numeric_cols:
    median_val = X[col].median()
    X[col].fillna(median_val, inplace=True)
    if col in test_clean.columns:
        test_clean[col].fillna(median_val, inplace=True)

# CatBoost 参数
cat_params = {
    'iterations': 2500,
    'learning_rate': 0.04,
    'depth': 7,
    'l2_leaf_reg': 8,
    'loss_function': 'MAE',
    'random_seed': 42,
    'verbose': 100,
    'early_stopping_rounds': 100
}

print(f"\n{'='*60}")
print("开始训练 (CatBoost)")
print(f"{'='*60}")

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cat_maes = []
test_preds = np.zeros(len(test_clean))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 训练
    model = CatBoostRegressor(**cat_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)

    # 预测
    val_pred = model.predict(X_val)
    test_preds += model.predict(test_clean) / 5

    mae = mean_absolute_error(y_val, val_pred)
    cat_maes.append(mae)
    print(f"  MAE: {mae:.2f}")

# 最终结果
print(f"\n{'='*60}")
print("最终结果")
print(f"{'='*60}")
print(f"CatBoost 平均 MAE: {np.mean(cat_maes):.2f} ± {np.std(cat_maes):.2f}")

final_pred = np.maximum(test_preds, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': test['SaleID'].values,
    'price': final_pred
})
output_file = 'median_fill_simple_submit.csv'
submit.to_csv(output_file, index=False)
print(f"\n✓ 结果已保存到: {output_file}")

if np.mean(cat_maes) <= 450:
    print(f"\n🎉 成功！MAE: {np.mean(cat_maes):.2f} ≤ 450")
else:
    print(f"\n📈 距离目标: {np.mean(cat_maes) - 450:.2f}")
