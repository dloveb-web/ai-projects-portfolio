# -*- coding: utf-8 -*-
"""
二手车价格预测 - SOTA优化精简版v2
目标: MAE < 450
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("="*70)
print("SOTA优化精简版v2 - 目标 MAE < 450")
print("="*70)

# 数据加载
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

# 特征工程
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)

data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0.0')
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)

# v特征
v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)
data['v_max'] = data[v_cols].max(axis=1)
data['v_min'] = data[v_cols].min(axis=1)

# 交互特征
data['v_0_v_3'] = data['v_0'] * data['v_3']
data['v_0_v_2'] = data['v_0'] * data['v_2']
data['v_0_v_12'] = data['v_0'] * data['v_12']
data['v_3_v_12'] = data['v_3'] * data['v_12']

# 业务特征
data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']
data['power_age'] = data['power'] * data['car_age']
data['usage_intensity'] = data['kilometer'] / (data['car_age'] + 1)

# 分组特征
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

# Label Encoding
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# 分离数据
train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
test_data = test_data.drop(columns=['price'])

X = train_data.drop(columns=['price'])
y = train_data['price'].values

# Target编码
def target_encode(train_df, test_df, col, target='price', n_folds=5):
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[f'{col}_te'] = 0.0
    test_df[f'{col}_te'] = 0.0

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for train_idx, val_idx in kf.split(train_df):
        target_mean = train_df.iloc[train_idx].groupby(col)[target].mean()
        train_df.loc[val_idx, f'{col}_te'] = train_df.loc[val_idx, col].map(target_mean)

    target_mean = train_df.groupby(col)[target].mean()
    test_df[f'{col}_te'] = test_df[col].map(target_mean).fillna(train_df[target].mean())

    return train_df[f'{col}_te'], test_df[f'{col}_te']

# 先分离数据再Target编码
X = train_data.drop(columns=['price'])
y = train_data['price'].values

for col in ['brand', 'model', 'regionCode']:
    train_te, test_te = target_encode(train_data, test_data, col)
    train_data[f'{col}_te'] = train_te
    test_data[f'{col}_te'] = test_te

# 分离数据（在编码之后）
X = train_data.drop(columns=['price'])
test_data = test_data.drop(columns=['price'])

# 标准化
scaler = StandardScaler()
numeric_cols_std = X.select_dtypes(include=[np.number]).columns
X[numeric_cols_std] = scaler.fit_transform(X[numeric_cols_std])
test_data[numeric_cols_std] = scaler.transform(test_data[numeric_cols_std])

print(f"特征数: {X.shape[1]}")

# 5折CV
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

cat_train_preds = np.zeros(len(X))
lgb_train_preds = np.zeros(len(X))
cat_test_preds = np.zeros(len(test_data))
lgb_test_preds = np.zeros(len(test_data))

fold_maes_cat = []
fold_maes_lgb = []
fold_maes_stack = []

# 参数
cat_params = {
    'iterations': 4500, 'learning_rate': 0.020, 'depth': 8,
    'l2_leaf_reg': 6, 'random_strength': 0.7, 'bagging_temperature': 0.6,
    'loss_function': 'MAE', 'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 140
}

lgb_params = {
    'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
    'learning_rate': 0.020, 'num_leaves': 120, 'max_depth': 9,
    'min_data_in_leaf': 17, 'feature_fraction': 0.88, 'bagging_fraction': 0.88,
    'bagging_freq': 5, 'reg_alpha': 0.19, 'reg_lambda': 0.19,
    'verbose': -1, 'seed': SEED
}

print("开始训练...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # CatBoost
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    cat_pred = cat_model.predict(X_val)
    cat_mae = mean_absolute_error(y_val, cat_pred)
    cat_train_preds[val_idx] = cat_pred
    cat_test_preds += cat_model.predict(test_data) / 5
    fold_maes_cat.append(cat_mae)
    print(f"  CatBoost: {cat_mae:.2f}")

    # LightGBM
    train_data_lgb = lgb.Dataset(X_train, label=y_train)
    val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
    lgb_model = lgb.train(lgb_params, train_data_lgb, num_boost_round=4500,
                           valid_sets=[val_data_lgb],
                           callbacks=[lgb.early_stopping(140), lgb.log_evaluation(0)])
    lgb_pred = lgb_model.predict(X_val)
    lgb_mae = mean_absolute_error(y_val, lgb_pred)
    lgb_train_preds[val_idx] = lgb_pred
    lgb_test_preds += lgb_model.predict(test_data) / 5
    fold_maes_lgb.append(lgb_mae)
    print(f"  LightGBM: {lgb_mae:.2f}")

    # Stacking
    stack_X_train = np.column_stack([cat_train_preds[train_idx], lgb_train_preds[train_idx]])
    stack_X_val = np.column_stack([cat_train_preds[val_idx], lgb_train_preds[val_idx]])
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(stack_X_train, y_train)
    stack_pred = meta_model.predict(stack_X_val)
    stack_mae = mean_absolute_error(y_val, stack_pred)
    fold_maes_stack.append(stack_mae)
    print(f"  Stacking: {stack_mae:.2f} (Cat={meta_model.coef_[0]:.3f}, LGB={meta_model.coef_[1]:.3f})")

# 结果
print(f"\n{'='*70}")
print("最终结果")
print(f"{'='*70}")
print(f"CatBoost 平均 MAE: {np.mean(fold_maes_cat):.2f}")
print(f"LightGBM 平均 MAE: {np.mean(fold_maes_lgb):.2f}")
print(f"Stacking 平均 MAE: {np.mean(fold_maes_stack):.2f}")

final_mae = np.mean(fold_maes_stack)
print(f"\n最终MAE: {final_mae:.2f}")
print(f"目标: 450")
print(f"差距: {final_mae - 450:.2f}")

if final_mae < 450:
    print(f"\n🎉 成功达到目标！MAE: {final_mae:.2f} < 450")
else:
    print(f"\n⚠️ 距离目标: {final_mae - 450:.2f}")

# 最终预测
final_stack_X = np.column_stack([cat_train_preds, lgb_train_preds])
final_meta_model = Ridge(alpha=1.0)
final_meta_model.fit(final_stack_X, y)

test_stack_X = np.column_stack([cat_test_preds, lgb_test_preds])
final_pred = final_meta_model.predict(test_stack_X)
final_pred = np.maximum(final_pred, 50)

submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit.to_csv('sota_simple_v2_submit.csv', index=False)
print(f"\n结果已保存: sota_simple_v2_submit.csv")
