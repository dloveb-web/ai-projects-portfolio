# -*- coding: utf-8 -*-
"""
LightGBM 融合脚本 - 与已保存的 DNN 预测融合
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = ''
print("="*60)
print("LightGBM 融合 - 与 DNN 预测融合")
print("="*60)

# 加载 DNN 预测
dnn_pred = pd.read_csv(BASE_PATH + 'sota_dnn_pred.csv')
dnn_price = dnn_pred['price'].values
print(f"DNN预测加载完成: {len(dnn_price)} 条")

# 训练 LightGBM（使用基线 concat 特征方案）
train = pd.read_csv(BASE_PATH + 'used_car_train_20200313.csv', sep=' ')
test = pd.read_csv(BASE_PATH + 'used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)
n_train = len(train)

# 基线特征工程
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

# 分离
train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
if 'price' in test_data:
    test_data = test_data.drop(columns=['price'])

X = train_data.drop(columns=['price'])
y = train_data['price'].values

# Target编码
y_prices = train_data['price'].values

def target_encode(train_df, test_df, col, y_vals, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
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

print(f"LightGBM 特征数: {X.shape[1]}")

# LightGBM 训练
lgb_params = {
    'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
    'learning_rate': 0.020, 'num_leaves': 120, 'max_depth': 9,
    'min_data_in_leaf': 17, 'feature_fraction': 0.88, 'bagging_fraction': 0.88,
    'bagging_freq': 5, 'reg_alpha': 0.19, 'reg_lambda': 0.19,
    'verbose': -1, 'seed': 42
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_test_preds = np.zeros(len(test_data))

for fold, (tr, va) in enumerate(kf.split(X)):
    print(f"  Fold {fold+1}/5")
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y[tr], y[va]
    ds = lgb.Dataset(X_tr, label=y_tr)
    vs = lgb.Dataset(X_va, label=y_va, reference=ds)
    m = lgb.train(lgb_params, ds, valid_sets=[vs], num_boost_round=4500,
                  callbacks=[lgb.early_stopping(140), lgb.log_evaluation(0)])
    lgb_test_preds += m.predict(test_data) / 5

# 融合（找最优权重）
for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = w * dnn_price + (1-w) * lgb_test_preds
    pred = np.maximum(pred, 50)
    print(f"  权重 DNN={w:.1f}: 范围=[{pred.min():.0f},{pred.max():.0f}], 均值={pred.mean():.0f}")

# 默认 0.5 融合
final_w = 0.5
final_pred = final_w * dnn_price + (1-final_w) * lgb_test_preds
final_pred = np.maximum(final_pred, 50)

submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit_name = 'sota_dnn_fusion_submit.csv'
submit.to_csv(BASE_PATH + submit_name, index=False)
print(f"\n✅ 提交文件: {submit_name}")
print(f"📊 范围: [{final_pred.min():.0f}, {final_pred.max():.0f}], 均值: {final_pred.mean():.0f}")
