# -*- coding: utf-8 -*-
"""
方案1: 特征筛选 + 权重优化 (快速版)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("方案1: 特征筛选 + 权重优化 (快速版)")
print("=" * 50)

# 加载数据
print("\n加载数据...")
X_train = joblib.load('processed_data/X_train.joblib')
X_val = joblib.load('processed_data/X_val.joblib')
y_train = joblib.load('processed_data/y_train.joblib')
y_val = joblib.load('processed_data/y_val.joblib')
test_data = joblib.load('processed_data/test_data.joblib')
sale_ids = joblib.load('processed_data/sale_ids.joblib')

if not isinstance(X_train, pd.DataFrame):
    train_cols = joblib.load('processed_data/train_columns.joblib')
    X_train = pd.DataFrame(X_train, columns=train_cols)
    X_val = pd.DataFrame(X_val, columns=train_cols)
    test_data = pd.DataFrame(test_data, columns=train_cols)

print(f"原始特征: {X_train.shape[1]}")

# 特征筛选 - 删除低重要性特征
drop_cols = ['regionCode', 'brand', 'bodyType', 'model', 'fuelType', 'seller', 'offerType']
drop_cols = [c for c in drop_cols if c in X_train.columns]
print(f"删除特征: {drop_cols}")

X_train = X_train.drop(columns=drop_cols, errors='ignore')
X_val = X_val.drop(columns=drop_cols, errors='ignore')
test_data = test_data.drop(columns=drop_cols, errors='ignore')

print(f"筛选后特征: {X_train.shape[1]}")

# 3折快速训练
print("\n3折训练...")
kf = KFold(n_splits=3, shuffle=True, random_state=42)

cat_oof, lgb_oof, xgb_oof = np.zeros(len(X_train)), np.zeros(len(X_train)), np.zeros(len(X_train))
cat_test, lgb_test, xgb_test = np.zeros(len(test_data)), np.zeros(len(test_data)), np.zeros(len(test_data))

for fold, (ti, vi) in enumerate(kf.split(X_train)):
    print(f"Fold {fold+1}/3")
    X_tr, X_va = X_train.iloc[ti], X_train.iloc[vi]
    y_tr, y_va = y_train.iloc[ti], y_train.iloc[vi]
    
    # CatBoost
    cat = cb.CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=5, 
                                random_seed=42, verbose=0, loss_function='MAE', early_stopping_rounds=50)
    cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    cat_oof[vi] = cat.predict(X_va)
    cat_test += cat.predict(test_data) / 3
    
    # LightGBM
    lgb_m = lgb.train({'objective': 'regression_l1', 'verbosity': -1, 'learning_rate': 0.05, 'num_leaves': 63},
                       lgb.Dataset(X_tr, label=y_tr), num_boost_round=2000,
                       valid_sets=[lgb.Dataset(X_va, label=y_va)],
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    lgb_oof[vi] = lgb_m.predict(X_va)
    lgb_test += lgb_m.predict(test_data) / 3
    
    # XGBoost
    xgb_m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8,
                              colsample_bytree=0.8, random_state=42, objective='reg:absoluteerror',
                              early_stopping_rounds=50, verbosity=0)
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_oof[vi] = xgb_m.predict(X_va)
    xgb_test += xgb_m.predict(test_data) / 3
    
    print(f"  CatBoost MAE: {mean_absolute_error(y_va, cat_oof[vi]):.2f}")

# 单模型MAE
print(f"\nCatBoost MAE: {mean_absolute_error(y_train, cat_oof):.2f}")
print(f"LightGBM MAE: {mean_absolute_error(y_train, lgb_oof):.2f}")
print(f"XGBoost MAE: {mean_absolute_error(y_train, xgb_oof):.2f}")

# 权重搜索
print("\n搜索最优权重...")
best_mae, best_w = float('inf'), None
for w1 in np.arange(0.35, 0.6, 0.05):
    for w2 in np.arange(0.25, 0.45, 0.05):
        w3 = 1 - w1 - w2
        if 0.1 < w3 < 0.35:
            mae = mean_absolute_error(y_train, w1*cat_oof + w2*lgb_oof + w3*xgb_oof)
            if mae < best_mae:
                best_mae, best_w = mae, (w1, w2, w3)

print(f"最优权重: Cat={best_w[0]:.2f}, LGB={best_w[1]:.2f}, XGB={best_w[2]:.2f}")

# 最终预测
pred = best_w[0]*cat_test + best_w[1]*lgb_test + best_w[2]*xgb_test
print(f"\n最终MAE: {best_mae:.2f}")

if best_mae < 460:
    print("🎉 目标达成！")
else:
    print(f"距离目标: {best_mae - 460:.2f}")

# 保存
pred = np.clip(pred, y_train.min()*0.9, y_train.max()*1.1)
pd.DataFrame({'SaleID': sale_ids, 'price': pred}).to_csv('opt_v1.csv', index=False)
print("保存: opt_v1.csv")
