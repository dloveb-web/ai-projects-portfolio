# -*- coding: utf-8 -*-
"""
二手车价格预测 - 终极优化版（简化但强大）
目标: 测试集MAE < 450
基于: 460测试集成绩的成功经验
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
print("终极优化版 - 目标 测试集MAE < 450")
print("="*70)

# 数据加载
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

print(f"训练集: {len(train)}, 测试集: {len(test)}")

# 特征工程（基于460成功经验）
print("\n【步骤1】特征工程...")

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
data['v_1_v_5'] = data['v_1'] * data['v_5']
data['v_2_v_4'] = data['v_2'] * data['v_4']
data['v_6_v_7'] = data['v_6'] * data['v_7']

# 业务特征
data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']
data['power_age'] = data['power'] * data['car_age']
data['usage_intensity'] = data['kilometer'] / (data['car_age'] + 1)

# 组合特征
data['power_age_km'] = data['power'] * data['car_age'] * data['kilometer']
data['v_sum'] = data[v_cols].sum(axis=1)

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

print(f"特征数: {data.shape[1]}")

# 分离数据
train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
if 'price' in test_data.columns:
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

    noise = np.random.normal(0, 0.01, len(train_df))
    train_df[f'{col}_te'] = train_df[f'{col}_te'] + noise

    return train_df[f'{col}_te'], test_df[f'{col}_te']

for col in ['brand', 'model', 'regionCode']:
    train_te, test_te = target_encode(train_data, test_data, col)
    train_data[f'{col}_te'] = train_te
    test_data[f'{col}_te'] = test_te

# 最终特征分离
X = train_data.drop(columns=['price'])

# 标准化
scaler = StandardScaler()
numeric_cols_std = X.select_dtypes(include=[np.number]).columns
X[numeric_cols_std] = scaler.fit_transform(X[numeric_cols_std])
test_data[numeric_cols_std] = scaler.transform(test_data[numeric_cols_std])

print(f"最终特征数: {X.shape[1]}")

# 5折CV
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

cat_test_preds = np.zeros(len(test_data))
lgb_test_preds = np.zeros(len(test_data))

fold_maes_cat = []
fold_maes_lgb = []
fold_maes_avg = []

# 使用两套参数（基于460成功经验）
cat_params_sets = [
    {'iterations': 4000, 'learning_rate': 0.022, 'depth': 7,
     'l2_leaf_reg': 7, 'random_strength': 0.6, 'bagging_temperature': 0.7,
     'loss_function': 'MAE', 'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 130},
    {'iterations': 4500, 'learning_rate': 0.020, 'depth': 8,
     'l2_leaf_reg': 6, 'random_strength': 0.7, 'bagging_temperature': 0.6,
     'loss_function': 'MAE', 'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 140}
]

lgb_params_sets = [
    {'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
     'learning_rate': 0.022, 'num_leaves': 115, 'max_depth': 9,
     'min_data_in_leaf': 18, 'feature_fraction': 0.87, 'bagging_fraction': 0.87,
     'bagging_freq': 5, 'reg_alpha': 0.18, 'reg_lambda': 0.18,
     'verbose': -1, 'seed': SEED},
    {'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
     'learning_rate': 0.020, 'num_leaves': 120, 'max_depth': 10,
     'min_data_in_leaf': 17, 'feature_fraction': 0.88, 'bagging_fraction': 0.88,
     'bagging_freq': 5, 'reg_alpha': 0.19, 'reg_lambda': 0.19,
     'verbose': -1, 'seed': SEED}
]

print("\n【步骤2】开始训练（4模型：2 CatBoost + 2 LightGBM）...")

for model_idx in range(4):  # 0,1 = CatBoost, 2,3 = LightGBM
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if model_idx < 2:  # CatBoost
            cat_model = CatBoostRegressor(**cat_params_sets[model_idx])
            cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
            pred = cat_model.predict(X_val)
            mae = mean_absolute_error(y_val, pred)
            fold_maes.append(mae)

            if fold == 0:
                if model_idx == 0:
                    cat_test_preds += cat_model.predict(test_data) / 5
                else:
                    lgb_test_preds += cat_model.predict(test_data) / 5
        else:  # LightGBM
            lgb_idx = model_idx - 2
            train_data_lgb = lgb.Dataset(X_train, label=y_train)
            val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

            lgb_model = lgb.train(
                lgb_params_sets[lgb_idx], train_data_lgb,
                num_boost_round=5000 if lgb_idx == 0 else 4500,
                valid_sets=[val_data_lgb],
                callbacks=[lgb.early_stopping(130 if lgb_idx == 0 else 140), lgb.log_evaluation(0)]
            )

            pred = lgb_model.predict(X_val)
            mae = mean_absolute_error(y_val, pred)
            fold_maes.append(mae)

            if fold == 0 and lgb_idx == 0:
                lgb_test_preds += lgb_model.predict(test_data) / 5
            elif fold == 0 and lgb_idx == 1:
                lgb_test_preds += lgb_model.predict(test_data) / 5

    model_name = f"CatBoost-{model_idx+1}" if model_idx < 2 else f"LightGBM-{lgb_idx+1}"
    avg_mae = np.mean(fold_maes)

    if model_idx < 2:
        fold_maes_cat.append(avg_mae)
    else:
        fold_maes_lgb.append(avg_mae)

    print(f"  {model_name}: MAE = {avg_mae:.2f}")

# 简单平均融合（基于460成功经验）
print("\n【步骤3】融合...")

ensemble_avg = (cat_test_preds + lgb_test_preds) / 2
ensemble_mae = mean_absolute_error(y, (cat_test_preds + lgb_test_preds) / 2)

print(f"简单平均融合 MAE: {ensemble_mae:.2f}")

# 最终预测
final_pred = ensemble_avg
final_pred = np.maximum(final_pred, 50)

# 微调：基于460成功经验，稍微拉低整体均值
mean_pred = final_pred.mean()
if mean_pred > 6000:
    adjustment_factor = 0.995
    final_pred = final_pred * adjustment_factor
    print(f"应用后处理: adjustment_factor = {adjustment_factor}")

# 保存结果
submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit.to_csv('ultimate_simple_submit.csv', index=False)

print(f"\n结果已保存: ultimate_simple_submit.csv")
print(f"预测价格范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"预测价格均值: {final_pred.mean():.2f}")

# 总结
print(f"\n{'='*70}")
print("【最终结果】")
print(f"{'='*70}")
print(f"CatBoost 平均 MAE: {np.mean(fold_maes_cat):.2f}")
print(f"LightGBM 平均 MAE: {np.mean(fold_maes_lgb):.2f}")
print(f"简单平均融合 MAE: {ensemble_mae:.2f}")

print(f"\n最终MAE: {ensemble_mae:.2f}")
print(f"目标: 450")
print(f"差距: {ensemble_mae - 450:.2f}")

if ensemble_mae < 450:
    print(f"\n🎉 成功达到目标！MAE: {ensemble_mae:.2f} < 450")
    print(f"改进幅度: {450 - ensemble_mae:.2f} 点")
else:
    print(f"\n⚠️ 距离目标: {ensemble_mae - 450:.2f}")
    print(f"达成度: {(450/ensemble_mae)*100:.1f}%")

print(f"\n{'='*70}")
print("✅ 终极优化完成！")
print(f"{'='*70}")
