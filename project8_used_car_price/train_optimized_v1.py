# -*- coding: utf-8 -*-
"""
方案1: 特征筛选 + 权重优化
目标: MAE < 460
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("方案1: 特征筛选 + 权重优化")
print("=" * 60)

# ==================== 加载数据 ====================
print("\n加载预处理数据...")
X_train = joblib.load('processed_data/X_train.joblib')
X_val = joblib.load('processed_data/X_val.joblib')
y_train = joblib.load('processed_data/y_train.joblib')
y_val = joblib.load('processed_data/y_val.joblib')
test_data = joblib.load('processed_data/test_data.joblib')
sale_ids = joblib.load('processed_data/sale_ids.joblib')

print(f"原始特征数: {X_train.shape[1]}")

# 转为DataFrame便于处理
if not isinstance(X_train, pd.DataFrame):
    train_cols = joblib.load('processed_data/train_columns.joblib')
    X_train = pd.DataFrame(X_train, columns=train_cols)
    X_val = pd.DataFrame(X_val, columns=train_cols)
    test_data = pd.DataFrame(test_data, columns=train_cols)

# ==================== 特征筛选 ====================
print("\n特征筛选...")

# 基于特征重要性，删除低重要性特征
low_importance_features = [
    'regionCode', 'creat_date_diff', 'brand', 'bodyType', 
    'model', 'fuelType', 'seller', 'offerType'
]

# 只删除存在的特征
cols_to_drop = [col for col in low_importance_features if col in X_train.columns]
print(f"删除低重要性特征: {cols_to_drop}")

X_train_sel = X_train.drop(columns=cols_to_drop, errors='ignore')
X_val_sel = X_val.drop(columns=cols_to_drop, errors='ignore')
test_data_sel = test_data.drop(columns=cols_to_drop, errors='ignore')

# 删除高相关特征
numeric_cols = X_train_sel.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    corr_matrix = X_train_sel[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
    if high_corr:
        print(f"删除高相关特征: {high_corr}")
        X_train_sel = X_train_sel.drop(columns=high_corr)
        X_val_sel = X_val_sel.drop(columns=high_corr)
        test_data_sel = test_data_sel.drop(columns=high_corr)

print(f"筛选后特征数: {X_train_sel.shape[1]}")

# ==================== K折训练 ====================
print("\n" + "=" * 60)
print("K折集成训练 (5折)")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储预测
cat_oof = np.zeros(len(X_train_sel))
lgb_oof = np.zeros(len(X_train_sel))
xgb_oof = np.zeros(len(X_train_sel))
cat_test = np.zeros(len(test_data_sel))
lgb_test = np.zeros(len(test_data_sel))
xgb_test = np.zeros(len(test_data_sel))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_sel)):
    print(f"\n--- Fold {fold+1}/5 ---")
    
    X_tr, X_va = X_train_sel.iloc[tr_idx], X_train_sel.iloc[val_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[val_idx]
    
    # CatBoost
    cat_model = cb.CatBoostRegressor(
        iterations=3000, learning_rate=0.03, depth=6,
        l2_leaf_reg=10, min_data_in_leaf=30, random_seed=42,
        verbose=0, loss_function='MAE', early_stopping_rounds=100
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    cat_oof[val_idx] = cat_model.predict(X_va)
    cat_test += cat_model.predict(test_data_sel) / 5
    
    # LightGBM
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_model = lgb.train(
        {'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1,
         'learning_rate': 0.03, 'num_leaves': 63, 'feature_fraction': 0.8,
         'bagging_fraction': 0.8, 'bagging_freq': 5, 'seed': 42},
        lgb_train, num_boost_round=3000, 
        valid_sets=[lgb.Dataset(X_va, label=y_va)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    lgb_oof[val_idx] = lgb_model.predict(X_va)
    lgb_test += lgb_model.predict(test_data_sel) / 5
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.03, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, objective='reg:absoluteerror',
        early_stopping_rounds=100, verbosity=0
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_oof[val_idx] = xgb_model.predict(X_va)
    xgb_test += xgb_model.predict(test_data_sel) / 5
    
    fold_mae = mean_absolute_error(y_va, cat_oof[val_idx])
    print(f"Fold {fold+1} CatBoost MAE: {fold_mae:.2f}")

# 单模型MAE
cat_mae = mean_absolute_error(y_train, cat_oof)
lgb_mae = mean_absolute_error(y_train, lgb_oof)
xgb_mae = mean_absolute_error(y_train, xgb_oof)
print(f"\n单模型MAE:")
print(f"  CatBoost: {cat_mae:.2f}")
print(f"  LightGBM: {lgb_mae:.2f}")
print(f"  XGBoost: {xgb_mae:.2f}")

# ==================== 权重优化 ====================
print("\n" + "=" * 60)
print("权重优化")
print("=" * 60)

# 方法1: 基于MAE反比
inv_maes = [1/cat_mae, 1/lgb_mae, 1/xgb_mae]
weights_inverse = [w/sum(inv_maes) for w in inv_maes]
print(f"反比权重: CatBoost={weights_inverse[0]:.3f}, LightGBM={weights_inverse[1]:.3f}, XGBoost={weights_inverse[2]:.3f}")

# 方法2: 网格搜索最优权重
print("\n网格搜索最优权重...")
best_mae = float('inf')
best_weights = None

for w1 in np.arange(0.3, 0.7, 0.05):
    for w2 in np.arange(0.2, 0.5, 0.05):
        w3 = 1 - w1 - w2
        if w3 > 0 and w3 < 0.4:
            ensemble = w1 * cat_oof + w2 * lgb_oof + w3 * xgb_oof
            mae = mean_absolute_error(y_train, ensemble)
            if mae < best_mae:
                best_mae = mae
                best_weights = (w1, w2, w3)

print(f"最优权重: CatBoost={best_weights[0]:.3f}, LightGBM={best_weights[1]:.3f}, XGBoost={best_weights[2]:.3f}")

# 使用最优权重
ensemble_oof = best_weights[0] * cat_oof + best_weights[1] * lgb_oof + best_weights[2] * xgb_oof
ensemble_test = best_weights[0] * cat_test + best_weights[1] * lgb_test + best_weights[2] * xgb_test

final_mae = mean_absolute_error(y_train, ensemble_oof)
final_rmse = np.sqrt(mean_squared_error(y_train, ensemble_oof))
final_r2 = r2_score(y_train, ensemble_oof)

print("\n" + "=" * 60)
print("最终结果")
print("=" * 60)
print(f"MAE: {final_mae:.2f}")
print(f"RMSE: {final_rmse:.2f}")
print(f"R²: {final_r2:.4f}")

if final_mae < 460:
    print("🎉 目标达成！MAE < 460")
else:
    print(f"距离目标: {final_mae - 460:.2f}")

# ==================== 保存结果 ====================
# 裁剪预测值
ensemble_test = np.clip(ensemble_test, y_train.min() * 0.9, y_train.max() * 1.1)

submission = pd.DataFrame({
    'SaleID': sale_ids,
    'price': ensemble_test
})
submission.to_csv('optimized_submit.csv', index=False)
print(f"\n预测结果已保存: optimized_submit.csv")

# 保存模型权重
joblib.dump({
    'catboost_weight': best_weights[0],
    'lightgbm_weight': best_weights[1],
    'xgboost_weight': best_weights[2],
    'mae': final_mae,
    'features_dropped': cols_to_drop
}, 'optimized_weights.joblib')
print(f"权重已保存: optimized_weights.joblib")
