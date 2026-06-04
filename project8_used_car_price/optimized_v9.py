# -*- coding: utf-8 -*-
"""
二手车价格预测 - 优化v9版（基于ultra_simple_final优化）
目标: 测试集MAE < 450
基础: ultra_simple_final CV 484.22 (未过拟合)
策略: 基于可靠基线 + 轻度优化
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("="*70)
print("优化v9版 - 基于可靠基线")
print("基础: ultra_simple_final CV 484.22")
print("策略: 基于可靠基线 + 轻度优化")
print("="*70)

# ==================== 1. 数据加载 ====================
print("\n【步骤1】数据加载...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

print(f"训练集: {len(train)}, 测试集: {len(test)}")

# ==================== 2. 特征工程（基于ultra_simple_final）====================
print("\n【步骤2】特征工程（基于可靠基线）...")

# 时间特征
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)

# 异常值处理
def mild_winsorize(series, lower=2, upper=98):
    lower_bound = np.percentile(series.dropna(), lower)
    upper_bound = np.percentile(series.dropna(), upper)
    return series.clip(lower=lower_bound, upper=upper_bound)

for col in ['power', 'kilometer', 'car_age']:
    data[col] = mild_winsorize(data[col])

# 分类处理
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0.0')
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)

# v特征（基础统计）
v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)
data['v_max'] = data[v_cols].max(axis=1)
data['v_min'] = data[v_cols].min(axis=1)

# v特征交叉（只保留最核心的3个，避免过拟合）
data['v_0_v_3'] = data['v_0'] * data['v_3']
data['v_0_v_12'] = data['v_0'] * data['v_12']
data['v_3_v_12'] = data['v_3'] * data['v_12']

# 业务特征（核心的4个）
data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']
data['power_age'] = data['power'] * data['car_age']
data['usage_intensity'] = data['kilometer'] / (data['car_age'] + 1)

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

# ==================== 4. Label Encoding ====================
print("\n【步骤4】分类特征编码...")

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# 分离数据
train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)

if 'price' in test_data.columns:
    test_data = test_data.drop(columns=['price'])

X = train_data.drop(columns=['price'])
y = train_data['price'].values

# 标准化
scaler = StandardScaler()
numeric_cols_std = X.select_dtypes(include=[np.number]).columns
X[numeric_cols_std] = scaler.fit_transform(X[numeric_cols_std])
test_data[numeric_cols_std] = scaler.transform(test_data[numeric_cols_std])

print(f"最终特征数: {X.shape[1]}")

# ==================== 5. 训练 ====================
print("\n【步骤5】开始训练（基于可靠参数）...")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 基于ultra_simple_final的成功参数微调
cat_params = {
    'iterations': 3500,
    'learning_rate': 0.022,
    'depth': 7,
    'l2_leaf_reg': 7,
    'random_strength': 0.6,
    'bagging_temperature': 0.7,
    'loss_function': 'MAE',
    'random_seed': SEED,
    'verbose': 0,
    'early_stopping_rounds': 130
}

lgb_params = {
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
    'verbose': -1,
    'seed': SEED
}

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

    # CatBoost
    print("  训练CatBoost...")
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)

    cat_pred_val = cat_model.predict(X_val)
    cat_mae = mean_absolute_error(y_val, cat_pred_val)
    fold_maes_cat.append(cat_mae)

    if fold == 0:
        cat_test_preds += cat_model.predict(test_data) / 5

    print(f"  CatBoost MAE: {cat_mae:.2f}")

    # LightGBM
    print("  训练LightGBM...")
    train_data_lgb = lgb.Dataset(X_train, label=y_train)
    val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

    lgb_model = lgb.train(
        lgb_params, train_data_lgb,
        num_boost_round=3500,
        valid_sets=[val_data_lgb],
        callbacks=[lgb.early_stopping(130), lgb.log_evaluation(0)]
    )

    lgb_pred_val = lgb_model.predict(X_val)
    lgb_mae = mean_absolute_error(y_val, lgb_pred_val)
    fold_maes_lgb.append(lgb_mae)

    if fold == 0:
        lgb_test_preds += lgb_model.predict(test_data) / 5

    print(f"  LightGBM MAE: {lgb_mae:.2f}")

    # 简单平均
    avg_pred_val = (cat_pred_val + lgb_pred_val) / 2
    avg_mae = mean_absolute_error(y_val, avg_pred_val)
    fold_maes_avg.append(avg_mae)

    print(f"  平均融合 MAE: {avg_mae:.2f}")

# ==================== 6. 最终结果 ====================
print(f"\n{'='*70}")
print("【最终结果】5折交叉验证平均")
print(f"{'='*70}")
print(f"CatBoost 平均 MAE: {np.mean(fold_maes_cat):.2f} ± {np.std(fold_maes_cat):.2f}")
print(f"LightGBM 平均 MAE: {np.mean(fold_maes_lgb):.2f} ± {np.std(fold_maes_lgb):.2f}")
print(f"平均融合 MAE: {np.mean(fold_maes_avg):.2f} ± {np.std(fold_maes_avg):.2f}")

final_mae = np.mean(fold_maes_avg)

print(f"\n{'='*70}")
print(f"最终CV MAE: {final_mae:.2f}")
print(f"目标MAE: 450")
print(f"差距: {final_mae - 450:.2f}")

print(f"\n📊 基于ultra_simple_final (CV 484.22)的可靠基线")
print(f"预期测试集成绩应接近CV成绩，不会严重过拟合")

# ==================== 7. 测试集预测 ====================
print("\n【步骤7】生成测试集预测...")

# 简单平均
final_pred = (cat_test_preds + lgb_test_preds) / 2
final_pred = np.maximum(final_pred, 50)

# 保守后处理
mean_pred = final_pred.mean()
if mean_pred > 6000:
    final_pred *= 0.99
    print(f"应用均值调整: 0.99")

# 保存结果
submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit.to_csv('optimized_v9_submit.csv', index=False)

print(f"\n结果已保存: optimized_v9_submit.csv")
print(f"预测价格范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"预测价格均值: {final_pred.mean():.2f}")

print(f"\n{'='*70}")
print("✅ 优化v9版（基于可靠基线）完成！")
print(f"{'='*70}")
