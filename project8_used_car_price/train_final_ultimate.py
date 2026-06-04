# -*- coding: utf-8 -*-
"""
二手车价格预测 - 最终完成版
基于480.64成功策略
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
print("最终完成版 - 基于480.64成功策略")
print("目标：MAE <= 450")
print("="*60)

# 数据处理
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

# 特征工程
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)

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
data['v_1_v_5'] = data['v_1'] * data['v_5']
data['v_2_v_4'] = data['v_2'] * data['v_4']

data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']
data['power_age'] = data['power'] * data['car_age']

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

# 模型训练 (5折CV)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cat_maes = []
lgb_maes = []
ensemble_maes = []

test_preds_cat = np.zeros(len(test_data))
test_preds_lgb = np.zeros(len(test_data))

# 480.64成功参数
catboost_params = {
    'iterations': 4000,
    'learning_rate': 0.022,
    'depth': 7,
    'l2_leaf_reg': 7,
    'loss_function': 'MAE',
    'random_seed': 42,
    'verbose': 0
}

lightgbm_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.022,
    'num_leaves': 115,
    'max_depth': 8,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'reg_alpha': 0.12,
    'reg_lambda': 0.12,
    'n_estimators': 4000,
    'verbose': -1,
    'seed': 42
}

print("开始训练...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}/5")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # CatBoost
    cat_model = CatBoostRegressor(**catboost_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    cat_pred = cat_model.predict(X_val)
    cat_mae = mean_absolute_error(y_val, cat_pred)
    cat_maes.append(cat_mae)
    test_preds_cat += cat_model.predict(test_data) / 5
    print(f"  CatBoost MAE: {cat_mae:.2f}")

    # LightGBM
    train_data_lgb = lgb.Dataset(X_train, label=y_train)
    val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
    lgb_model = lgb.train(**lightgbm_params, train_data_lgb, num_boost_round=4000,
                       valid_sets=[val_data_lgb], callbacks=[lgb.early_stopping(130), lgb.log_evaluation(0)])
    lgb_pred = lgb_model.predict(X_val)
    lgb_mae = mean_absolute_error(y_val, lgb_pred)
    lgb_maes.append(lgb_mae)
    test_preds_lgb += lgb_model.predict(test_data) / 5
    print(f"  LightGBM MAE: {lgb_mae:.2f}")

    # 集成
    ensemble_pred = (cat_pred + lgb_pred) / 2
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    ensemble_maes.append(ensemble_mae)
    print(f"  Ensemble MAE: {ensemble_mae:.2f}")

# 最终结果
print("\n最终结果 (5折平均)")
print(f"CatBoost 平均 MAE: {np.mean(cat_maes):.2f}")
print(f"LightGBM 平均 MAE: {np.mean(lgb_maes):.2f}")
print(f"Ensemble 平均 MAE: {np.mean(ensemble_maes):.2f}")

# 最终预测
final_pred = (test_preds_cat + test_preds_lgb) / 2
final_pred = np.maximum(final_pred, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': sale_ids,
    'price': final_pred
})
submit.to_csv('final_ultimate_submit.csv', index=False)
print(f"\n结果已保存到 final_ultimate_submit.csv")

# 最终总结
print(f"\n{'='*60}")
print(f"最终MAE: {np.mean(ensemble_maes):.2f}")
print(f"目标MAE: 450")
print(f"差距: {np.mean(ensemble_maes) - 450:.2f}")

if np.mean(ensemble_maes) <= 450:
    print(f"\n🎉 成功达到目标MAE: {np.mean(ensemble_maes):.2f} <= 450")
else:
    print(f"\n⚠️ 距离目标: {np.mean(ensemble_maes) - 450:.2f}")
    print(f"\n当前MAE: {np.mean(ensemble_maes):.2f}")