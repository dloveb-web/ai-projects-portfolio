# -*- coding: utf-8 -*-
"""
二手车价格预测 - 多项式特征 SFS 轻量筛选
策略：在基线特征基础上，逐个评估 v 多项式特征的增量贡献
筛选标准：1-fold LightGBM MAE 改善 > 0.3
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("多项式特征 SFS 快速筛选")
print("="*70)

# ========== 数据加载 & 基线特征工程 ==========
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)
print(f"训练集: {len(train)}, 测试集: {len(test)}")

# 基线特征（与 sota_correct_stacking 完全一致）
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

# 缺失值处理
numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=['object']).columns
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())
for col in categorical_cols:
    mode_val = data[col].mode()[0] if not data[col].mode().empty else 'unknown'
    data[col] = data[col].fillna(mode_val)
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
if 'price' in test_data.columns:
    test_data = test_data.drop(columns=['price'])

# Target编码
def target_encode(train_df, test_df, col, target='price', n_folds=5):
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[f'{col}_te'] = 0.0
    test_df[f'{col}_te'] = 0.0
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(train_df):
        target_mean = train_df.iloc[train_idx].groupby(col)[target].mean()
        train_df.loc[val_idx, f'{col}_te'] = train_df.loc[val_idx, col].map(target_mean)
    target_mean = train_df.groupby(col)[target].mean()
    test_df[f'{col}_te'] = test_df[col].map(target_mean).fillna(train_df[target].mean())
    return train_df[f'{col}_te'], test_df[f'{col}_te']

for col in ['brand', 'model', 'regionCode']:
    train_te, test_te = target_encode(train_data, test_data, col)
    train_data[f'{col}_te'] = train_te
    test_data[f'{col}_te'] = test_te

X_base = train_data.drop(columns=['price'])
y = train_data['price'].values

# 标准化
scaler = StandardScaler()
numeric_cols_std = X_base.select_dtypes(include=[np.number]).columns
X_base[numeric_cols_std] = scaler.fit_transform(X_base[numeric_cols_std])

print(f"基线特征数: {X_base.shape[1]}")

# ========== 候选多项式特征 ==========

# 基于数据验证结果，选择高潜力特征
candidates = {}

# 平方项（从数据验证：v_14² 从 corr 0.04→0.21 增长最大）
v_cols_interest = ['v_0', 'v_3', 'v_8', 'v_12', 'v_14']
for v in v_cols_interest:
    candidates[f'{v}_sq'] = data[v] ** 2

# 乘法交互（top-4 强相关 v 特征的组合）
# v_0, v_3, v_8, v_12 是相关性最高的4个
strong_v = ['v_0', 'v_3', 'v_8', 'v_12', 'v_14']
for i in range(len(strong_v)):
    for j in range(i+1, len(strong_v)):
        v1, v2 = strong_v[i], strong_v[j]
        name = f'{v1}_{v2}'
        # 跳过基线已含的
        if name in ['v_0_v_3', 'v_0_v_2', 'v_0_v_12', 'v_3_v_12']:
            continue
        candidates[name] = data[v1] * data[v2]

# 高相关 v 对之间的交互（v_0×v_5, v_3×v_8 等）
# 从v间相关性>0.7或<-0.7中选未在基线的
high_corr_pairs = [
    ('v_0', 'v_5'), ('v_1', 'v_10'), ('v_2', 'v_5'), ('v_2', 'v_7'),
    ('v_4', 'v_9'), ('v_4', 'v_13'), ('v_5', 'v_7'), ('v_5', 'v_11'),
    ('v_7', 'v_11'), ('v_9', 'v_13')
]
for v1, v2 in high_corr_pairs:
    name = f'{v1}_{v2}'
    if name not in candidates:
        candidates[name] = data[v1] * data[v2]

print(f"候选多项式特征数: {len(candidates)}")
for name in sorted(candidates.keys()):
    print(f"  {name}")

# ========== SFS 筛选 ==========
# 使用 1-fold LightGBM 快速评估
# 取 80% train, 20% val

SPLIT = 0.8
n_train = int(len(X_base) * SPLIT)
print(f"\nX_base 行数: {len(X_base)}, n_train: {n_train}, n_val: {len(X_base) - n_train}")

X_train = X_base.iloc[:n_train].reset_index(drop=True)
X_val = X_base.iloc[n_train:].reset_index(drop=True)
y_train = y[:n_train]
y_val = y[n_train:]

# val 集的多项式特征
# data 有 {len(data)} 行, 前 150k 是训练集
candidates_val = {}
for name, series in candidates.items():
    # 每个 series 来自 data → 200k 行, 只取训练部分
    train_series = series.iloc[:150000].values  # 训练部分
    candidates_val[name] = train_series[n_train:].copy()  # val 部分 (30k)
    candidates[name] = train_series[:n_train].copy()  # train 部分 (120k)

lgb_params = {
    'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
    'learning_rate': 0.020, 'num_leaves': 120, 'max_depth': 9,
    'min_data_in_leaf': 17, 'feature_fraction': 0.88, 'bagging_fraction': 0.88,
    'bagging_freq': 5, 'reg_alpha': 0.19, 'reg_lambda': 0.19,
    'verbose': -1, 'seed': 42
}

# 基线评估
print("\n评估基线 MAE...")
train_ds = lgb.Dataset(X_train, label=y_train)
val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
baseline_model = lgb.train(lgb_params, train_ds, num_boost_round=1500,
                            valid_sets=[val_ds],
                            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
baseline_pred = baseline_model.predict(X_val)
baseline_mae = mean_absolute_error(y_val, baseline_pred)
print(f"基线 MAE: {baseline_mae:.2f}")

# 逐个评估候选特征
results = []
for name, feat_array in candidates.items():
    # 标准化候选特征
    feat = feat_array.reshape(-1, 1)
    feat_val = candidates_val[name].reshape(-1, 1)
    
    print(f"  调试: name={name}, feat={feat.shape}, feat_val={feat_val.shape}, X_train={X_train.shape}, X_val={X_val.shape}", flush=True)

    scaler_c = StandardScaler()
    feat_scaled = scaler_c.fit_transform(feat)
    feat_val_scaled = scaler_c.transform(feat_val)

    # 加到基线上
    X_aug = np.hstack([X_train.values, feat_scaled])
    X_val_aug = np.hstack([X_val.values, feat_val_scaled])

    train_ds = lgb.Dataset(X_aug, label=y_train)
    val_ds = lgb.Dataset(X_val_aug, label=y_val, reference=train_ds)
    model = lgb.train(lgb_params, train_ds, num_boost_round=1500,
                       valid_sets=[val_ds],
                       callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    pred = model.predict(X_val_aug)
    mae = mean_absolute_error(y_val, pred)

    improvement = baseline_mae - mae
    results.append((name, mae, improvement))
    print(f"  {name:12s}: MAE={mae:.2f}, 改善={improvement:+.2f}")

# ========== 排序输出 ==========
results.sort(key=lambda x: x[2], reverse=True)

print(f"\n{'='*70}")
print("SFS 结果排名（按改善幅度降序）")
print(f"{'='*70}")
print(f"{'特征名':15s} {'MAE':>10s} {'改善':>10s} {'推荐':>8s}")
print("-"*45)

selected = []
for name, mae, imp in results:
    recommend = "✅" if imp > 0.3 else ("?" if imp > 0 else "❌")
    print(f"{name:15s} {mae:>10.2f} {imp:>+10.2f} {recommend:>8s}")
    if imp > 0.3:
        selected.append(name)

print(f"\n{'='*70}")
print(f"选中特征（改善>0.3）：{len(selected)} 个")
for name in selected:
    print(f"  + {name}")
print(f"{'='*70}")
