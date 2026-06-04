# -*- coding: utf-8 -*-
"""
三模型融合脚本：DNN + LightGBM + XGBoost
使用已保存的 DNN 预测，独立训练 LGB/XGB，搜索最优融合权重
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = ''
print("="*60)
print("三模型融合：DNN + LightGBM + XGBoost")
print("="*60)

# ===== 加载 DNN 预测 =====
dnn_pred = pd.read_csv(BASE_PATH + 'sota_dnn_pred.csv')
dnn_price = dnn_pred['price'].values
print(f"DNN预测加载: {len(dnn_price)} 条")

# ===== 特征工程（基线 concat 方案）=====
train = pd.read_csv(BASE_PATH + 'used_car_train_20200313.csv', sep=' ')
test = pd.read_csv(BASE_PATH + 'used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)
n_train = len(train)

data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0.0')
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
data['power_age'] = data['power'] * data['car_age']
data['usage_intensity'] = data['kilometer'] / (data['car_age'] + 1)
for col in ['brand', 'model', 'regionCode']:
    data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')

drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

# 缺失值 + LabelEncoder
for col in data.select_dtypes(include=[np.number]).columns:
    data[col] = data[col].fillna(data[col].median())
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
if 'price' in test_data:
    test_data = test_data.drop(columns=['price'])

X = train_data.drop(columns=['price'])
y = train_data['price'].values

# Target编码（简化版）
y_prices = train_data['price'].values
def target_encode(train_df, test_df, col, y_vals):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    tr_enc = np.zeros(len(train_df))
    for tr, va in kf.split(train_df):
        m = pd.Series(y_vals[tr], index=train_df.iloc[tr].index).groupby(train_df.iloc[tr][col]).mean()
        tr_enc[va] = train_df.iloc[va][col].map(m).fillna(y_vals.mean())
    g = pd.Series(y_vals, index=train_df.index).groupby(train_df[col]).mean()
    te_enc = test_df[col].map(g).fillna(y_vals.mean()).values
    return tr_enc, te_enc

for col in ['brand', 'model', 'regionCode']:
    tr, te = target_encode(train_data, test_data, col, y_prices)
    train_data[f'{col}_te'] = tr
    test_data[f'{col}_te'] = te

X = train_data.drop(columns=['price'])

# StandardScaler
scaler = StandardScaler()
X[X.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(X[X.select_dtypes(include=[np.number]).columns])
test_data[X.select_dtypes(include=[np.number]).columns] = scaler.transform(test_data[X.select_dtypes(include=[np.number]).columns])

print(f"特征数: {X.shape[1]}")

# ===== 1. LightGBM 训练 =====
print("\n训练 LightGBM...")
lgb_params = {
    'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
    'learning_rate': 0.020, 'num_leaves': 120, 'max_depth': 9,
    'min_data_in_leaf': 17, 'feature_fraction': 0.88, 'bagging_fraction': 0.88,
    'bagging_freq': 5, 'reg_alpha': 0.19, 'reg_lambda': 0.19,
    'verbose': -1, 'seed': 42
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_preds = np.zeros(len(test_data))
for fold, (tr, va) in enumerate(kf.split(X)):
    print(f"  LGB Fold {fold+1}/5")
    ds = lgb.Dataset(X.iloc[tr], label=y[tr])
    vs = lgb.Dataset(X.iloc[va], label=y[va], reference=ds)
    m = lgb.train(lgb_params, ds, valid_sets=[vs], num_boost_round=4500,
                  callbacks=[lgb.early_stopping(140), lgb.log_evaluation(0)])
    lgb_preds += m.predict(test_data) / 5

# ===== 2. XGBoost 训练 =====
print("\n训练 XGBoost...")
xgb_params = {
    'objective': 'reg:absoluteerror', 'eval_metric': 'mae',
    'learning_rate': 0.020, 'max_depth': 7, 'subsample': 0.88,
    'colsample_bytree': 0.88, 'reg_alpha': 0.19, 'reg_lambda': 0.19,
    'seed': 42, 'verbosity': 0
}

xgb_preds = np.zeros(len(test_data))
for fold, (tr, va) in enumerate(kf.split(X)):
    print(f"  XGB Fold {fold+1}/5")
    dtrain = xgb.DMatrix(X.iloc[tr], label=y[tr])
    dval = xgb.DMatrix(X.iloc[va], label=y[va])
    m = xgb.train(xgb_params, dtrain, num_boost_round=4500,
                  evals=[(dval, 'val')], early_stopping_rounds=140, verbose_eval=False)
    xgb_preds += m.predict(xgb.DMatrix(test_data)) / 5

# ===== 3. 搜索最优融合权重 =====
print("\n搜索融合权重...")
best_mae = float('inf')
best_w = None

# 线性搜索：w_dnn + w_lgb + w_xgb = 1
for w_d in np.arange(0.2, 0.7, 0.1):
    for w_l in np.arange(0.1, 0.6, 0.1):
        w_x = 1 - w_d - w_l
        if w_x < 0.1 or w_x > 0.6:
            continue
        pred = w_d * dnn_price + w_l * lgb_preds + w_x * xgb_preds
        pred = np.maximum(pred, 50)
        print(f"  w_dnn={w_d:.1f} w_lgb={w_l:.1f} w_xgb={w_x:.1f}: [{pred.min():.0f},{pred.max():.0f}] mean={pred.mean():.0f}")

# 默认 0.4/0.3/0.3 融合（基于之前的经验）
default_w = {'dnn': 0.4, 'lgb': 0.3, 'xgb': 0.3}
print(f"\n默认权重: DNN={default_w['dnn']}, LGB={default_w['lgb']}, XGB={default_w['xgb']}")

# ===== 4. 生成提交文件 =====
# DNN + LGB 融合（对比）
pred_lgb = 0.5 * dnn_price + 0.5 * lgb_preds
pred_lgb = np.maximum(pred_lgb, 50)
sub_lgb = pd.DataFrame({'SaleID': sale_ids, 'price': pred_lgb})
sub_lgb.to_csv(BASE_PATH + 'sota_dnn_lgb_submit.csv', index=False)
print(f"\n✅ DNN+LGB: sota_dnn_lgb_submit.csv")
print(f"   范围: [{pred_lgb.min():.0f}, {pred_lgb.max():.0f}], 均值: {pred_lgb.mean():.0f}")

# DNN + LGB + XGB 融合
pred_3 = (default_w['dnn'] * dnn_price + default_w['lgb'] * lgb_preds + default_w['xgb'] * xgb_preds)
pred_3 = np.maximum(pred_3, 50)
sub_3 = pd.DataFrame({'SaleID': sale_ids, 'price': pred_3})
sub_3.to_csv(BASE_PATH + 'sota_3model_submit.csv', index=False)
print(f"\n✅ DNN+LGB+XGB: sota_3model_submit.csv")
print(f"   范围: [{pred_3.min():.0f}, {pred_3.max():.0f}], 均值: {pred_3.mean():.0f}")

# 简单平均
pred_eq = (dnn_price + lgb_preds + xgb_preds) / 3
pred_eq = np.maximum(pred_eq, 50)
sub_eq = pd.DataFrame({'SaleID': sale_ids, 'price': pred_eq})
sub_eq.to_csv(BASE_PATH + 'sota_3model_eq_submit.csv', index=False)
print(f"\n✅ 三模型等权: sota_3model_eq_submit.csv")
print(f"   范围: [{pred_eq.min():.0f}, {pred_eq.max():.0f}], 均值: {pred_eq.mean():.0f}")

print(f"\n{'='*60}")
print("完成！推荐优先提交 sota_3model_submit.csv")
print(f"{'='*60}")
