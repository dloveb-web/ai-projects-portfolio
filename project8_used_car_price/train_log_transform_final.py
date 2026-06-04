# -*- coding: utf-8 -*-
"""二手车价格预测 - 对数变换优化版 (基于原始数据)
目标：MAE <= 450
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("对数变换优化版 - 基于原始数据")
print("="*60)

# 数据加载
print("\n加载数据...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"训练集: {train.shape}")
print(f"测试集: {test.shape}")

# 简化处理 notRepairedDamage
for df in [train, test]:
    if 'notRepairedDamage' in df.columns and df['notRepairedDamage'].dtype == object:
        df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
        df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce').fillna(0)

# 对数变换
print("\n对数变换...")
train['log_price'] = np.log1p(train['price'])

# 特征工程
print("\n特征工程...")
for df in [train]:
    df['reg_year'] = df['regDate'] // 10000
    df['creat_year'] = df['creatDate'] // 10000
    df['car_age'] = (df['creat_year'] - df['reg_year']).clip(lower=0)
    df['car_age_squared'] = df['car_age'] ** 2

    v_cols = [f'v_{i}' for i in range(15)]
    available_v_cols = [col for col in v_cols if col in df.columns]
    if len(available_v_cols) >= 3:
        df['v_mean'] = df[available_v_cols].mean(axis=1)
        df['v_0_v_3'] = df['v_0'] * df['v_3']
    df['power_km'] = df['power'] * df['kilometer']
    df['log_power'] = np.log1p(df['power'])

print(f"特征数: {train.shape[1]-1}")

# 特征工程应用到测试集
print("\n对测试集应用特征工程...")
for df in [test]:
    df['reg_year'] = df['regDate'] // 10000
    df['creat_year'] = df['creatDate'] // 10000
    df['car_age'] = (df['creat_year'] - df['reg_year']).clip(lower=0)
    df['car_age_squared'] = df['car_age'] ** 2

    v_cols = [f'v_{i}' for i in range(15)]
    available_v_cols = [col for col in v_cols if col in df.columns]
    if len(available_v_cols) >= 3:
        df['v_mean'] = df[available_v_cols].mean(axis=1)
        df['v_0_v_3'] = df['v_0'] * df['v_3']
    df['power_km'] = df['power'] * df['kilometer']
    df['log_power'] = np.log1p(df['power'])

# 数据准备
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType', 'notRepairedDamage', 'price', 'log_price']
X = train.drop(columns=[c for c in drop_cols if c in train.columns])
X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])
y = train['log_price']

numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
X = X[numeric_cols]
X_test = X_test[[col for col in numeric_cols if col in X_test.columns]]

# 填充缺失值
for col in X.columns:
    median_val = X[col].median()
    X[col].fillna(median_val, inplace=True)
    if col in X_test.columns:
        X_test[col].fillna(median_val, inplace=True)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print(f"准备完成，特征数: {X_scaled.shape[1]}")

# 模型训练
print("\n开始训练 (5折交叉验证)...")
print("="*60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

all_maes = {'cat': [], 'lgb': [], 'ensemble': [],
             'cat_orig': [], 'lgb_orig': [], 'ensemble_orig': []}
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    y_val_orig = train['price'].iloc[val_idx]  # 真实价格

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=10,
        loss_function='MAE',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    cat_pred = cat_model.predict(X_val)
    cat_mae = mean_absolute_error(y_val, cat_pred)
    all_maes['cat'].append(cat_mae)

    # 真实价格空间MAE
    cat_pred_orig = np.exp(cat_pred)
    cat_mae_orig = mean_absolute_error(y_val_orig, cat_pred_orig)
    all_maes['cat_orig'].append(cat_mae_orig)

    test_preds += cat_model.predict(X_test_scaled) / 5
    print(f"CatBoost MAE (log空间): {cat_mae:.2f}")
    print(f"CatBoost MAE (真实价格): {cat_mae_orig:.2f}")

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        num_leaves=63,
        max_depth=6,
        learning_rate=0.03,
        n_estimators=2500,
        min_data_in_leaf=30,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    lgb_pred = lgb_model.predict(X_val)
    lgb_mae = mean_absolute_error(y_val, lgb_pred)
    all_maes['lgb'].append(lgb_mae)

    # 真实价格空间MAE
    lgb_pred_orig = np.exp(lgb_pred)
    lgb_mae_orig = mean_absolute_error(y_val_orig, lgb_pred_orig)
    all_maes['lgb_orig'].append(lgb_mae_orig)

    test_preds += lgb_model.predict(X_test_scaled) / 5
    print(f"LightGBM MAE (log空间): {lgb_mae:.2f}")
    print(f"LightGBM MAE (真实价格): {lgb_mae_orig:.2f}")

    # 融合
    ensemble_pred = 0.5 * cat_pred + 0.5 * lgb_pred
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    all_maes['ensemble'].append(ensemble_mae)

    # 真实价格空间MAE
    ensemble_pred_orig = np.exp(ensemble_pred)
    ensemble_mae_orig = mean_absolute_error(y_val_orig, ensemble_pred_orig)
    all_maes['ensemble_orig'].append(ensemble_mae_orig)

    print(f"Ensemble MAE (log空间): {ensemble_mae:.2f}")
    print(f"Ensemble MAE (真实价格): {ensemble_mae_orig:.2f}")

# 最终结果
print("\n" + "="*60)
print("最终结果 (5折平均)")
print("="*60)
print(f"CatBoost MAE (log空间): {np.mean(all_maes['cat']):.4f}")
print(f"LightGBM MAE (log空间): {np.mean(all_maes['lgb']):.4f}")
print(f"Ensemble MAE (log空间): {np.mean(all_maes['ensemble']):.4f}")
print("")
print(f"CatBoost MAE (真实价格): {np.mean(all_maes['cat_orig']):.2f}")
print(f"LightGBM MAE (真实价格): {np.mean(all_maes['lgb_orig']):.2f}")
print(f"Ensemble MAE (真实价格): {np.mean(all_maes['ensemble_orig']):.2f}")

# 反变换回原始空间
# test_preds 已经是所有5折模型的平均预测值（在对数空间）
final_pred = np.exp(test_preds)
final_pred = np.maximum(final_pred, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': test['SaleID'].values,
    'price': final_pred
})
submit.to_csv('log_transform_final_submit.csv', index=False)
print(f"\n结果已保存到: log_transform_final_submit.csv")

final_mae_log = np.mean(all_maes['ensemble'])
final_mae_orig = np.mean(all_maes['ensemble_orig'])
print(f"\n" + "="*60)
if final_mae_orig <= 450:
    print(f"成功！真实价格MAE: {final_mae_orig:.2f} <= 450")
    print(f"Log空间MAE: {final_mae_log:.4f}")
else:
    print(f"距离目标: {final_mae_orig - 450:.2f}")
    print(f"当前真实价格MAE: {final_mae_orig:.2f}")
print("="*60)
