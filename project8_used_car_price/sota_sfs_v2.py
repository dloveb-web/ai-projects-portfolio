# -*- coding: utf-8 -*-
"""
二手车价格预测 - 多项式特征 SFS 快速筛选（v2 简化版）
在基线特征基础上逐个评估 v 多项式特征增量贡献
方法：1-fold LightGBM, 直接从原数据计算候选特征
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("多项式特征 SFS 快速筛选 v2")
print("="*70)

# ===== 加载数据 =====
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

# ==== 先构建基线特征矩阵（完全对齐 baseline）====
test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)
n_train = len(train)

# 基线特征
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
numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=['object']).columns
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].fillna('unknown').astype(str))

# 分离 train/test 并做 target encoding
train_data = data.iloc[:n_train].reset_index(drop=True)
test_data = data.iloc[n_train:].reset_index(drop=True)

def target_encode(train_df, test_df, col, target='price', n_folds=5, seed=42):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_enc = np.zeros(len(train_df))
    for tr, va in kf.split(train_df):
        m = train_df.iloc[tr].groupby(col)[target].mean()
        train_enc[va] = train_df.iloc[va][col].map(m).fillna(train_df[target].mean())
    g = train_df.groupby(col)[target].mean()
    test_enc = test_df[col].map(g).fillna(train_df[target].mean()).values
    return train_enc, test_enc

for col in ['brand', 'model', 'regionCode']:
    tr, te = target_encode(train_data, test_data, col)
    train_data[f'{col}_te'] = tr
    test_data[f'{col}_te'] = te

X_base = train_data.drop(columns=['price'])
y = train_data['price'].values

# StandardScaler
scaler = StandardScaler()
sc = X_base.select_dtypes(include=[np.number]).columns
X_base[sc] = scaler.fit_transform(X_base[sc])

print(f"基线特征数: {X_base.shape[1]}, 训练样本: {len(X_base)}")

# ===== 候选多项式特征（直接从原始 train data 计算）=====
raw_train = train.copy()

candidates = {}

# 平方项
for v in ['v_0', 'v_3', 'v_8', 'v_12', 'v_14']:
    candidates[f'{v}_sq'] = raw_train[[v]].values.flatten()

# 乘法交互
pairs = [
    ('v_0','v_8'), ('v_0','v_14'), ('v_3','v_8'), ('v_3','v_14'),
    ('v_8','v_12'), ('v_8','v_14'), ('v_12','v_14'), ('v_0','v_5'),
    ('v_3','v_3'), ('v_14','v_14'),
    ('v_4','v_9'), ('v_4','v_13'),
    ('v_5','v_7'), ('v_5','v_11'), ('v_7','v_11'), ('v_9','v_13'),
]
for v1, v2 in pairs:
    name = f'{v1}_x_{v2}'
    candidates[name] = (raw_train[v1] * raw_train[v2]).values

print(f"候选特征数: {len(candidates)}")
print("候选列表:", list(candidates.keys()))

# ===== 1-fold split =====
X_tr, X_va, y_tr, y_va = train_test_split(
    X_base.values, y, test_size=0.2, random_state=42
)
print(f"训练: {len(X_tr)}, 验证: {len(X_va)}")

lgb_p = {
    'objective': 'regression', 'metric': 'mae',
    'learning_rate': 0.020, 'num_leaves': 120, 'max_depth': 9,
    'min_data_in_leaf': 17, 'feature_fraction': 0.88, 'bagging_fraction': 0.88,
    'bagging_freq': 5, 'reg_alpha': 0.19, 'reg_lambda': 0.19,
    'verbose': -1, 'seed': 42
}

# 基线
bl_ds = lgb.Dataset(X_tr, label=y_tr)
bl_va = lgb.Dataset(X_va, label=y_va, reference=bl_ds)
bl_m = lgb.train(lgb_p, bl_ds, valid_sets=[bl_va], num_boost_round=1500,
                 callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
bl_mae = mean_absolute_error(y_va, bl_m.predict(X_va))
print(f"\n基线 MAE: {bl_mae:.2f}")

# 每个候选评估
# 需要对齐行：X_tr/X_va 是 train_test_split 的结果
# 所以候选特征也从相同索引取值
_, va_idx = train_test_split(np.arange(n_train), test_size=0.2, random_state=42)
tr_mask = np.ones(n_train, dtype=bool)
tr_mask[va_idx] = False

results = []
for name, feat in candidates.items():
    feat_tr = feat[tr_mask].reshape(-1, 1)
    feat_va = feat[va_idx].reshape(-1, 1)

    s = StandardScaler()
    feat_tr_s = s.fit_transform(feat_tr)
    feat_va_s = s.transform(feat_va)

    X_tr_aug = np.hstack([X_tr, feat_tr_s])
    X_va_aug = np.hstack([X_va, feat_va_s])

    ds = lgb.Dataset(X_tr_aug, label=y_tr)
    val = lgb.Dataset(X_va_aug, label=y_va, reference=ds)
    m = lgb.train(lgb_p, ds, valid_sets=[val], num_boost_round=1500,
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    p = m.predict(X_va_aug)
    mae = mean_absolute_error(y_va, p)
    imp = bl_mae - mae
    results.append((name, mae, imp))
    print(f"  {name:12s}: MAE={mae:.2f}, 改善={imp:+.2f}")

# ===== 排序输出 =====
results.sort(key=lambda x: x[2], reverse=True)
print(f"\n{'='*70}")
print("SFS 排名（按改善降序）")
print(f"{'='*70}")
print(f"{'特征':15s} {'MAE':>8s} {'改善':>8s} {'推荐':>6s}")
print("-"*40)

selected = []
for name, mae, imp in results:
    tag = "✅" if imp > 0.5 else ("?" if imp > 0 else "❌")
    print(f"{name:15s} {mae:>8.2f} {imp:>+8.2f} {tag:>6s}")
    if imp > 0.5:
        selected.append(name)

print(f"\n选中（改善>0.5）：{len(selected)} 个")
for n in selected:
    print(f"  + {n}")
