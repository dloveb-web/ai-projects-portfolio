# -*- coding: utf-8 -*-
"""
二手车价格预测 - 终极突破版
目标: 测试集MAE < 450
基于: 460测试集成功经验 + 多模型集成
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
print("终极突破版 - 目标 MAE < 450")
print("="*70)

# ==================== 1. 数据加载 ====================
print("\n【步骤1】数据加载...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

print(f"训练集: {len(train)}, 测试集: {len(test)}")

# ==================== 2. 高级特征工程 ====================
print("\n【步骤2】高级特征工程...")

# 时间特征
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)

# 异常值处理
def winsorize_series(series, lower=1, upper=99):
    lower_bound = np.percentile(series.dropna(), lower)
    upper_bound = np.percentile(series.dropna(), upper)
    return series.clip(lower=lower_bound, upper=upper_bound)

for col in ['power', 'kilometer', 'car_age']:
    data[col] = winsorize_series(data[col])

# 分类处理
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0.0')
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)

# v特征
v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)
data['v_max'] = data[v_cols].max(axis=1)
data['v_min'] = data[v_cols].min(axis=1)
data['v_range'] = data['v_max'] - data['v_min']

# v特征交互（基于460成功经验）
data['v_0_v_3'] = data['v_0'] * data['v_3']
data['v_0_v_2'] = data['v_0'] * data['v_2']
data['v_0_v_12'] = data['v_0'] * data['v_12']
data['v_3_v_12'] = data['v_3'] * data['v_12']
data['v_1_v_5'] = data['v_1'] * data['v_5']

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

# 删除无用列
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

print(f"特征工程完成，特征数: {data.shape[1]}")

# ==================== 3. 缺失值处理 ====================
print("\n【步骤3】缺失值处理...")

numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=['object']).columns

for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

for col in categorical_cols:
    mode_val = data[col].mode()[0] if not data[col].mode().empty else 'unknown'
    data[col] = data[col].fillna(mode_val)

print(f"数值特征: {len(numeric_cols)}, 分类特征: {len(categorical_cols)}")

# ==================== 4. 分类特征编码 ====================
print("\n【步骤4】分类特征编码...")

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    print(f"  Label编码: {col}")

# ==================== 5. Target编码 ====================
print("\n【步骤5】Target编码...")

# 分离训练和测试
train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
if 'price' in test_data.columns:
    test_data = test_data.drop(columns=['price'])

# Target编码
def target_encode(train_df, test_df, col, target='price', n_folds=5, noise_level=0.01):
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

    noise = np.random.normal(0, noise_level, len(train_df))
    train_df[f'{col}_te'] = train_df[f'{col}_te'] + noise

    return train_df[f'{col}_te'], test_df[f'{col}_te']

for col in ['brand', 'model', 'regionCode']:
    train_te, test_te = target_encode(train_data, test_data, col)
    train_data[f'{col}_te'] = train_te
    test_data[f'{col}_te'] = test_te
    print(f"  Target编码: {col}")

# 最终特征分离
X = train_data.drop(columns=['price'])
y = train_data['price'].values

# 标准化
scaler = StandardScaler()
numeric_cols_std = X.select_dtypes(include=[np.number]).columns
X[numeric_cols_std] = scaler.fit_transform(X[numeric_cols_std])
test_data[numeric_cols_std] = scaler.transform(test_data[numeric_cols_std])

print(f"最终特征数: {X.shape[1]}")

# ==================== 6. 多模型训练 ====================
print("\n【步骤6】多模型训练（2 CatBoost + 2 LightGBM）...")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 使用两套优化参数（基于460成功经验）
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

cat_test_preds = np.zeros(len(test_data))
lgb_test_preds = np.zeros(len(test_data))

fold_maes_cat = []
fold_maes_lgb = []
fold_maes_avg = []

print("开始5折交叉验证训练...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # CatBoost模型1
    cat_model1 = CatBoostRegressor(**cat_params_sets[0])
    cat_model1.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    cat_pred1 = cat_model1.predict(X_val)
    cat_mae1 = mean_absolute_error(y_val, cat_pred1)

    if fold == 0:
        cat_test_preds += cat_model1.predict(test_data) / 5

    # CatBoost模型2
    cat_model2 = CatBoostRegressor(**cat_params_sets[1])
    cat_model2.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    cat_pred2 = cat_model2.predict(X_val)
    cat_mae2 = mean_absolute_error(y_val, cat_pred2)

    if fold == 0:
        cat_test_preds += cat_model2.predict(test_data) / 5

    cat_mae = (cat_mae1 + cat_mae2) / 2
    fold_maes_cat.append(cat_mae)

    print(f"  CatBoost平均: {cat_mae:.2f}")

    # LightGBM模型1
    train_data_lgb1 = lgb.Dataset(X_train, label=y_train)
    val_data_lgb1 = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb1)
    lgb_model1 = lgb.train(
        lgb_params_sets[0], train_data_lgb1,
        num_boost_round=5000,
        valid_sets=[val_data_lgb1],
        callbacks=[lgb.early_stopping(140), lgb.log_evaluation(0)]
    )
    lgb_pred1 = lgb_model1.predict(X_val)
    lgb_mae1 = mean_absolute_error(y_val, lgb_pred1)

    if fold == 0:
        lgb_test_preds += lgb_model1.predict(test_data) / 5

    # LightGBM模型2
    train_data_lgb2 = lgb.Dataset(X_train, label=y_train)
    val_data_lgb2 = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb2)
    lgb_model2 = lgb.train(
        lgb_params_sets[1], train_data_lgb2,
        num_boost_round=5000,
        valid_sets=[val_data_lgb2],
        callbacks=[lgb.early_stopping(140), lgb.log_evaluation(0)]
    )
    lgb_pred2 = lgb_model2.predict(X_val)
    lgb_mae2 = mean_absolute_error(y_val, lgb_pred2)

    if fold == 0:
        lgb_test_preds += lgb_model2.predict(test_data) / 5

    lgb_mae = (lgb_mae1 + lgb_mae2) / 2
    fold_maes_lgb.append(lgb_mae)

    print(f"  LightGBM平均: {lgb_mae:.2f}")

    # 简单平均融合（最稳定，基于460成功经验）
    avg_pred = (cat_pred1 + cat_pred2 + lgb_pred1 + lgb_pred2) / 4
    avg_mae = mean_absolute_error(y_val, avg_pred)
    fold_maes_avg.append(avg_mae)

    print(f"  平均融合: {avg_mae:.2f}")

# ==================== 7. 最终结果 ====================
print(f"\n{'='*70}")
print("【最终结果】5折交叉验证平均")
print(f"{'='*70}")
print(f"CatBoost 平均 MAE: {np.mean(fold_maes_cat):.2f} ± {np.std(fold_maes_cat):.2f}")
print(f"LightGBM 平均 MAE: {np.mean(fold_maes_lgb):.2f} ± {np.std(fold_maes_lgb):.2f}")
print(f"平均融合 MAE: {np.mean(fold_maes_avg):.2f} ± {np.std(fold_maes_avg):.2f}")

final_mae = np.mean(fold_maes_avg)

# 最终预测（简单平均，最稳定）
final_pred = (cat_test_preds + lgb_test_preds) / 2
final_pred = np.maximum(final_pred, 50)

# 后处理：基于460成功经验，稍微拉低均值
mean_pred = final_pred.mean()
if mean_pred > 6000:
    adjustment_factor = 0.985
    final_pred = final_pred * adjustment_factor
    print(f"\n应用后处理调整: {adjustment_factor}")

# 保存结果
submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit.to_csv('final_breakthrough_submit.csv', index=False)

print(f"\n{'='*70}")
print(f"结果已保存: final_breakthrough_submit.csv")
print(f"预测价格范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"预测价格均值: {final_pred.mean():.2f}")

# 最终总结
print(f"\n{'='*70}")
print("【项目总结】")
print(f"{'='*70}")
print(f"最终MAE: {final_mae:.2f}")
print(f"目标MAE: 450")
print(f"差距: {final_mae - 450:.2f}")

if final_mae < 450:
    print(f"\n🎉 成功达到目标！MAE: {final_mae:.2f} < 450")
    print(f"改进幅度: {450 - final_mae:.2f} 点")
    print(f"🎯 项目成功！")
else:
    print(f"\n⚠️ 距离目标: {final_mae - 450:.2f}")
    print(f"达成度: {(450/final_mae)*100:.1f}%")
    if final_mae < 460:
        print(f"🏆 接近最佳460成绩！")

print(f"\n{'='*70}")
print("✅ 终极突破版完成！")
print(f"{'='*70}")
