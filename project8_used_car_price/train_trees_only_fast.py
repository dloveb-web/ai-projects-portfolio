# -*- coding: utf-8 -*-
"""
二手车价格预测 - 树模型快速版
只使用CatBoost和LightGBM，快速可靠
目标：MAE <= 450
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("树模型快速版 - CatBoost + LightGBM")
print("目标: MAE <= 450")
print("="*60)

# ==================== 数据处理 ====================
print("\n快速数据处理...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

# 基础特征工程
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = data['creat_year'] - data['reg_year']
data['car_age'] = data['car_age'].clip(lower=0)

data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', np.nan)
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)

v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)
data['v_max'] = data[v_cols].max(axis=1)
data['v_min'] = data[v_cols].min(axis=1)

data['v_0_v_3'] = data['v_0'] * data['v_3']
data['v_0_v_2'] = data['v_0'] * data['v_2']
data['v_0_v_12'] = data['v_0'] * data['v_12']
data['v_3_v_12'] = data['v_3'] * data['v_12']

data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']

for col in ['brand', 'model', 'regionCode']:
    data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')

drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
test_data = test_data.drop(columns=['price'])

X = train_data.drop(columns=['price'])
y = train_data['price']

X = X.fillna(X.median())
test_data = test_data.fillna(X.median())

common_cols = list(set(X.columns) & set(test_data.columns))
X = X[common_cols]
test_data = test_data[common_cols]

print(f"特征数: {X.shape[1]}")

# ==================== 模型训练 ====================
print("\n开始训练 (5折交叉验证)...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cat_maes = []
lgb_maes = []
ensemble_maes = []

test_preds_cat = np.zeros(len(test_data))
test_preds_lgb = np.zeros(len(test_data))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # CatBoost（优化参数）
    cat_model = CatBoostRegressor(
        iterations=3500,
        learning_rate=0.025,
        depth=7,
        l2_leaf_reg=8,
        random_strength=0.7,
        bagging_temperature=0.6,
        loss_function='MAE',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=120
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    cat_pred = cat_model.predict(X_val)
    cat_mae = mean_absolute_error(y_val, cat_pred)
    cat_maes.append(cat_mae)
    test_preds_cat += cat_model.predict(test_data) / 5
    print(f"   CatBoost MAE: {cat_mae:.2f}")

    # LightGBM（优化参数）
    train_data_lgb = lgb.Dataset(X_train, label=y_train)
    val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.025,
        'num_leaves': 127,
        'max_depth': 9,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'reg_alpha': 0.15,
        'reg_lambda': 0.15,
        'min_split_gain': 0.01,
        'verbose': -1,
        'seed': 42
    }

    lgb_model = lgb.train(params, train_data_lgb, num_boost_round=3500,
                           valid_sets=[val_data_lgb],
                           callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
    lgb_pred = lgb_model.predict(X_val)
    lgb_mae = mean_absolute_error(y_val, lgb_pred)
    lgb_maes.append(lgb_mae)
    test_preds_lgb += lgb_model.predict(test_data) / 5
    print(f"   LightGBM MAE: {lgb_mae:.2f}")

    # 融合（优化权重）
    inv_maes = [1/cat_mae, 1/lgb_mae]
    total = sum(inv_maes)
    weights = [w/total for w in inv_maes]

    ensemble_pred = weights[0] * cat_pred + weights[1] * lgb_pred
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    ensemble_maes.append(ensemble_mae)
    print(f"   Ensemble MAE: {ensemble_mae:.2f} (Cat={weights[0]:.3f}, LGB={weights[1]:.3f})")

# 最终结果
print(f"\n{'='*60}")
print("最终结果 (5折平均)")
print(f"{'='*60}")
print(f"CatBoost 平均 MAE: {np.mean(cat_maes):.2f}")
print(f"LightGBM 平均 MAE: {np.mean(lgb_maes):.2f}")
print(f"Ensemble 平均 MAE: {np.mean(ensemble_maes):.2f}")

# 最终预测（使用优化权重）
final_mae = np.mean(ensemble_maes)
if final_mae < 470:  # 如果效果好，使用优化权重
    inv_maes_final = [1/np.mean(cat_maes), 1/np.mean(lgb_maes)]
    total_final = sum(inv_maes_final)
    final_weights = [w/total_final for w in inv_maes_final]
else:  # 使用保守权重
    final_weights = [0.5, 0.5]

print(f"\n最终权重: CatBoost={final_weights[0]:.3f}, LightGBM={final_weights[1]:.3f}")

final_pred = final_weights[0] * test_preds_cat + final_weights[1] * test_preds_lgb
final_pred = np.maximum(final_pred, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': sale_ids,
    'price': final_pred
})
submit.to_csv('trees_only_fast_submit.csv', index=False)
print(f"\n结果已保存到 trees_only_fast_submit.csv")

# 最终总结
print(f"\n{'='*60}")
print(f"最终MAE: {final_mae:.2f}")
print(f"目标MAE: 450")
print(f"差距: {final_mae - 450:.2f}")

if final_mae <= 450:
    print(f"🎉 成功达到目标MAE: {final_mae:.2f} <= 450")
    print(f"比之前477改进: {477 - final_mae:.2f}")
else:
    print(f"⚠️ 距离目标: {final_mae - 450:.2f}")
    print(f"当前MAE: {final_mae:.2f}")
    if final_mae < 477:
        print(f"比之前477改进: {477 - final_mae:.2f}")
    else:
        print(f"比之前477差距: {final_mae - 477:.2f}")

print(f"{'='*60}")