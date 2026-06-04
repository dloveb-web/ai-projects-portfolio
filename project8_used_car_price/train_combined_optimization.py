# -*- coding: utf-8 -*-
"""二手车价格预测 - 组合优化版 (超参数调优 + 增强特征工程 + XGBoost集成)
目标：MAE <= 450
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("组合优化版 - 超参数调优 + 增强特征工程 + XGBoost集成")
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

# 增强特征工程
print("\n增强特征工程...")
for df in [train, test]:
    # 基础时间特征
    df['reg_year'] = df['regDate'] // 10000
    df['creat_year'] = df['creatDate'] // 10000
    df['car_age'] = (df['creat_year'] - df['reg_year']).clip(lower=0)
    df['car_age_squared'] = df['car_age'] ** 2
    df['car_age_cubed'] = df['car_age'] ** 3

    # v特征交互
    v_cols = [f'v_{i}' for i in range(15)]
    available_v_cols = [col for col in v_cols if col in df.columns]
    if len(available_v_cols) >= 3:
        df['v_mean'] = df[available_v_cols].mean(axis=1)
        df['v_std'] = df[available_v_cols].std(axis=1)
        df['v_0_v_3'] = df['v_0'] * df['v_3']
        df['v_1_v_2'] = df['v_1'] * df['v_2']
        df['v_4_v_5'] = df['v_4'] * df['v_5']

    # 功率特征
    df['log_power'] = np.log1p(df['power'])
    df['power_squared'] = df['power'] ** 2
    df['power_cubed'] = df['power'] ** 3
    df['power_km'] = df['power'] * df['kilometer']

    # 里程特征
    df['log_kilometer'] = np.log1p(df['kilometer'])
    df['kilometer_squared'] = df['kilometer'] ** 2

    # 分箱特征
    df['power_bin'] = pd.cut(df['power'], bins=[0, 100, 200, 300, 500, 1000], labels=False)
    df['car_age_bin'] = pd.cut(df['car_age'], bins=[0, 3, 6, 10, 20], labels=False)

    # 组合特征
    df['power_per_year'] = df['power'] / (df['car_age'] + 1)
    df['km_per_year'] = df['kilometer'] / (df['car_age'] + 1)
    df['power_per_km'] = df['power'] / (df['kilometer'] + 1)

    # 交互特征
    df['power_v0'] = df['power'] * df['v_0']
    df['carage_vmean'] = df['car_age'] * df['v_mean']

print(f"训练集特征数: {train.shape[1]-1}")
print(f"测试集特征数: {test.shape[1]}")

# 数据准备
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType', 'notRepairedDamage', 'price', 'log_price']
X = train.drop(columns=[c for c in drop_cols if c in train.columns])
X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])
y = train['log_price']
y_orig = train['price']

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

all_maes = {
    'cat': [], 'lgb': [], 'xgb': [], 'ensemble': [],
    'cat_orig': [], 'lgb_orig': [], 'xgb_orig': [], 'ensemble_orig': []
}
test_preds_cat = np.zeros(len(test))
test_preds_lgb = np.zeros(len(test))
test_preds_xgb = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    y_val_orig = y_orig.iloc[val_idx]

    # CatBoost - 优化超参数
    cat_model = CatBoostRegressor(
        iterations=4000,
        learning_rate=0.01,
        depth=8,
        l2_leaf_reg=5,
        loss_function='MAE',
        random_seed=42,
        verbose=200,
        early_stopping_rounds=150
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    cat_pred = cat_model.predict(X_val)
    cat_mae = mean_absolute_error(y_val, cat_pred)
    all_maes['cat'].append(cat_mae)

    cat_pred_orig = np.exp(cat_pred)
    cat_mae_orig = mean_absolute_error(y_val_orig, cat_pred_orig)
    all_maes['cat_orig'].append(cat_mae_orig)

    test_preds_cat += cat_model.predict(X_test_scaled) / 5
    print(f"CatBoost MAE (log空间): {cat_mae:.4f}, 真实价格: {cat_mae_orig:.2f}")

    # LightGBM - 优化超参数
    lgb_model = lgb.LGBMRegressor(
        num_leaves=127,
        max_depth=8,
        learning_rate=0.01,
        n_estimators=4000,
        min_data_in_leaf=20,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    lgb_pred = lgb_model.predict(X_val)
    lgb_mae = mean_absolute_error(y_val, lgb_pred)
    all_maes['lgb'].append(lgb_mae)

    lgb_pred_orig = np.exp(lgb_pred)
    lgb_mae_orig = mean_absolute_error(y_val_orig, lgb_pred_orig)
    all_maes['lgb_orig'].append(lgb_mae_orig)

    test_preds_lgb += lgb_model.predict(X_test_scaled) / 5
    print(f"LightGBM MAE (log空间): {lgb_mae:.4f}, 真实价格: {lgb_mae_orig:.2f}")

    # XGBoost - 优化超参数
    xgb_model = XGBRegressor(
        n_estimators=4000,
        learning_rate=0.01,
        max_depth=8,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1,
        objective='reg:absoluteerror',
        random_state=42,
        n_jobs=-1,
        eval_metric='mae',
        early_stopping_rounds=150,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_pred = xgb_model.predict(X_val)
    xgb_mae = mean_absolute_error(y_val, xgb_pred)
    all_maes['xgb'].append(xgb_mae)

    xgb_pred_orig = np.exp(xgb_pred)
    xgb_mae_orig = mean_absolute_error(y_val_orig, xgb_pred_orig)
    all_maes['xgb_orig'].append(xgb_mae_orig)

    test_preds_xgb += xgb_model.predict(X_test_scaled) / 5
    print(f"XGBoost MAE (log空间): {xgb_mae:.4f}, 真实价格: {xgb_mae_orig:.2f}")

    # 融合预测
    ensemble_pred = 0.4 * cat_pred + 0.35 * lgb_pred + 0.25 * xgb_pred
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    all_maes['ensemble'].append(ensemble_mae)

    ensemble_pred_orig = np.exp(ensemble_pred)
    ensemble_mae_orig = mean_absolute_error(y_val_orig, ensemble_pred_orig)
    all_maes['ensemble_orig'].append(ensemble_mae_orig)

    print(f"Ensemble MAE (log空间): {ensemble_mae:.4f}, 真实价格: {ensemble_mae_orig:.2f}")

# 最终结果
print("\n" + "="*60)
print("最终结果 (5折平均)")
print("="*60)
print(f"CatBoost MAE (log空间): {np.mean(all_maes['cat']):.4f}")
print(f"LightGBM MAE (log空间): {np.mean(all_maes['lgb']):.4f}")
print(f"XGBoost MAE (log空间): {np.mean(all_maes['xgb']):.4f}")
print(f"Ensemble MAE (log空间): {np.mean(all_maes['ensemble']):.4f}")
print("")
print(f"CatBoost MAE (真实价格): {np.mean(all_maes['cat_orig']):.2f}")
print(f"LightGBM MAE (真实价格): {np.mean(all_maes['lgb_orig']):.2f}")
print(f"XGBoost MAE (真实价格): {np.mean(all_maes['xgb_orig']):.2f}")
print(f"Ensemble MAE (真实价格): {np.mean(all_maes['ensemble_orig']):.2f}")

# 最终预测
print(f"\n融合权重: CatBoost=0.40, LightGBM=0.35, XGBoost=0.25")
test_preds_final = 0.4 * test_preds_cat + 0.35 * test_preds_lgb + 0.25 * test_preds_xgb
final_pred = np.exp(test_preds_final)
final_pred = np.maximum(final_pred, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': test['SaleID'].values,
    'price': final_pred
})
submit.to_csv('ensemble_optimized_submit.csv', index=False)
print(f"\n结果已保存到: ensemble_optimized_submit.csv")

final_mae_log = np.mean(all_maes['ensemble'])
final_mae_orig = np.mean(all_maes['ensemble_orig'])
print(f"\n" + "="*60)
if final_mae_orig <= 450:
    print(f"🎉 成功！真实价格MAE: {final_mae_orig:.2f} <= 450")
    print(f"Log空间MAE: {final_mae_log:.4f}")
else:
    print(f"⚠️ 距离目标: {final_mae_orig - 450:.2f}")
    print(f"当前真实价格MAE: {final_mae_orig:.2f}")
print("="*60)
