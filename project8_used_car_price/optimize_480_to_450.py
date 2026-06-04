# -*- coding: utf-8 -*-
"""
二手车价格预测 - 从480优化到450
基于480.60成功策略，尝试多种优化方法
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
print("480→450 优化版 - 多重优化策略")
print("目标: MAE <= 450")
print("="*60)

# ==================== 数据处理 ====================
print("\n数据处理...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

# 基础特征工程
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
data['v_range'] = data['v_max'] - data['v_min']
data['v_skew'] = data[v_cols].skew(axis=1)
data['v_kurt'] = data[v_cols].kurt(axis=1)

# 交互特征扩展
data['v_0_v_3'] = data['v_0'] * data['v_3']
data['v_0_v_2'] = data['v_0'] * data['v_2']
data['v_0_v_12'] = data['v_0'] * data['v_12']
data['v_3_v_12'] = data['v_3'] * data['v_12']
data['v_1_v_5'] = data['v_1'] * data['v_5']
data['v_2_v_4'] = data['v_2'] * data['v_4']
data['v_6_v_7'] = data['v_6'] * data['v_7']

data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']
data['power_age'] = data['power'] * data['car_age']
data['power_age_km'] = data['power'] * data['car_age'] * data['kilometer']

# 分组计数特征
for col in ['brand', 'model', 'regionCode']:
    data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')
    data[f'{col}_price_mean'] = data.groupby(col)['price'].transform('mean')

# 标签编码前的数值统计
data['brand_power_mean'] = data.groupby('brand')['power'].transform('mean')
data['model_age_mean'] = data.groupby('model')['car_age'].transform('mean')

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

# ==================== 多模型训练 ====================
print("\n开始训练 (5折交叉验证)...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cat_maes = []
lgb_maes = []
cat_maes2 = []
lgb_maes2 = []
ensemble_maes = []

test_preds_cat = np.zeros(len(test_data))
test_preds_lgb = np.zeros(len(test_data))
test_preds_cat2 = np.zeros(len(test_data))
test_preds_lgb2 = np.zeros(len(test_data))

# 第一套参数 (480.60成功参数)
cat_params1 = {
    'iterations': 4000,
    'learning_rate': 0.022,
    'depth': 7,
    'l2_leaf_reg': 7,
    'random_strength': 0.6,
    'bagging_temperature': 0.7,
    'loss_function': 'MAE',
    'random_seed': 42,
    'verbose': 0,
    'early_stopping_rounds': 130
}

lgb_params1 = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.022,
    'num_leaves': 115,
    'max_depth': 9,
    'min_data_in_leaf': 18,
    'feature_fraction': 0.87,
    'bagging_fraction': 0.87,
    'bagging_freq': 5,
    'reg_alpha': 0.18,
    'reg_lambda': 0.18,
    'min_split_gain': 0.008,
    'verbose': -1,
    'seed': 42
}

# 第二套参数 (更激进的优化)
cat_params2 = {
    'iterations': 5000,
    'learning_rate': 0.018,
    'depth': 8,
    'l2_leaf_reg': 5,
    'random_strength': 0.8,
    'bagging_temperature': 0.6,
    'loss_function': 'MAE',
    'random_seed': 123,
    'verbose': 0,
    'early_stopping_rounds': 150
}

lgb_params2 = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.018,
    'num_leaves': 127,
    'max_depth': 10,
    'min_data_in_leaf': 15,
    'feature_fraction': 0.90,
    'bagging_fraction': 0.90,
    'bagging_freq': 4,
    'reg_alpha': 0.20,
    'reg_lambda': 0.20,
    'min_split_gain': 0.005,
    'verbose': -1,
    'seed': 123
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # CatBoost Model 1
    cat_model1 = CatBoostRegressor(**cat_params1)
    cat_model1.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    cat_pred1 = cat_model1.predict(X_val)
    cat_mae1 = mean_absolute_error(y_val, cat_pred1)
    cat_maes.append(cat_mae1)
    test_preds_cat += cat_model1.predict(test_data) / 5

    # LightGBM Model 1
    train_data_lgb1 = lgb.Dataset(X_train, label=y_train)
    val_data_lgb1 = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb1)
    lgb_model1 = lgb.train(lgb_params1, train_data_lgb1, num_boost_round=4000,
                           valid_sets=[val_data_lgb1],
                           callbacks=[lgb.early_stopping(130), lgb.log_evaluation(0)])
    lgb_pred1 = lgb_model1.predict(X_val)
    lgb_mae1 = mean_absolute_error(y_val, lgb_pred1)
    lgb_maes.append(lgb_mae1)
    test_preds_lgb += lgb_model1.predict(test_data) / 5

    # CatBoost Model 2
    cat_model2 = CatBoostRegressor(**cat_params2)
    cat_model2.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    cat_pred2 = cat_model2.predict(X_val)
    cat_mae2 = mean_absolute_error(y_val, cat_pred2)
    cat_maes2.append(cat_mae2)
    test_preds_cat2 += cat_model2.predict(test_data) / 5

    # LightGBM Model 2
    train_data_lgb2 = lgb.Dataset(X_train, label=y_train)
    val_data_lgb2 = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb2)
    lgb_model2 = lgb.train(lgb_params2, train_data_lgb2, num_boost_round=5000,
                           valid_sets=[val_data_lgb2],
                           callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    lgb_pred2 = lgb_model2.predict(X_val)
    lgb_mae2 = mean_absolute_error(y_val, lgb_pred2)
    lgb_maes2.append(lgb_mae2)
    test_preds_lgb2 += lgb_model2.predict(test_data) / 5

    # 4模型融合
    ensemble_pred = (cat_pred1 + lgb_pred1 + cat_pred2 + lgb_pred2) / 4
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    ensemble_maes.append(ensemble_mae)

    print(f"   Cat1 MAE: {cat_mae1:.2f}, LGB1 MAE: {lgb_mae1:.2f}")
    print(f"   Cat2 MAE: {cat_mae2:.2f}, LGB2 MAE: {lgb_mae2:.2f}")
    print(f"   Ensemble MAE: {ensemble_mae:.2f}")

# 最终结果
print(f"\n{'='*60}")
print("最终结果 (5折平均)")
print(f"{'='*60}")
print(f"CatBoost Model 1 平均 MAE: {np.mean(cat_maes):.2f}")
print(f"LightGBM Model 1 平均 MAE: {np.mean(lgb_maes):.2f}")
print(f"CatBoost Model 2 平均 MAE: {np.mean(cat_maes2):.2f}")
print(f"LightGBM Model 2 平均 MAE: {np.mean(lgb_maes2):.2f}")
print(f"4模型融合 平均 MAE: {np.mean(ensemble_maes):.2f}")

# 自适应权重融合
final_mae = np.mean(ensemble_maes)

# 尝试3模型融合（去除最差的）
inv_maes = [1/np.mean(cat_maes), 1/np.mean(lgb_maes), 1/np.mean(cat_maes2), 1/np.mean(lgb_maes2)]
total = sum(inv_maes)
weights = [w/total for w in inv_maes]

final_pred = (weights[0] * test_preds_cat +
              weights[1] * test_preds_lgb +
              weights[2] * test_preds_cat2 +
              weights[3] * test_preds_lgb2)
final_pred = np.maximum(final_pred, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': sale_ids,
    'price': final_pred
})
submit.to_csv('optimize_480_to_450_submit.csv', index=False)
print(f"\n结果已保存到 optimize_480_to_450_submit.csv")

# 最终总结
print(f"\n{'='*60}")
print(f"最终MAE: {final_mae:.2f}")
print(f"目标MAE: 450")
print(f"差距: {final_mae - 450:.2f}")

if final_mae <= 450:
    print(f"\n🎉 成功达到目标MAE: {final_mae:.2f} <= 450")
    print(f"比之前480.60改进: {480.60 - final_mae:.2f}")
else:
    print(f"\n⚠️ 距离目标: {final_mae - 450:.2f}")
    print(f"当前MAE: {final_mae:.2f}")
    if final_mae < 480.60:
        print(f"比480.60改进: {480.60 - final_mae:.2f}")

print(f"{'='*60}")
