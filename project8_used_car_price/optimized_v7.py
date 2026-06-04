# -*- coding: utf-8 -*-
"""
二手车价格预测 - 优化v7版（抗过拟合）
目标: 测试集MAE < 450
问题: v6 CV 460 但测试集 729 (严重过拟合)
策略: 极简特征 + 强正则 + 模型简单化
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
print("优化v7版 - 抗过拟合")
print("问题: v6 CV 460, 测试集 729")
print("策略: 极简 + 强正则 + 简单化")
print("="*70)

# ==================== 1. 数据加载 ====================
print("\n【步骤1】数据加载...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

print(f"训练集: {len(train)}, 测试集: {len(test)}")

# ==================== 2. 极简特征工程（避免过拟合）====================
print("\n【步骤2】极简特征工程...")

# 只保留最核心的时间特征
data['car_age'] = (data['creatDate'] // 10000 - data['regDate'] // 10000).clip(lower=0)

# 异常值处理（更保守）
def conservative_winsorize(series, lower=5, upper=95):
    lower_bound = np.percentile(series.dropna(), lower)
    upper_bound = np.percentile(series.dropna(), upper)
    return series.clip(lower=lower_bound, upper=upper_bound)

for col in ['power', 'kilometer', 'car_age']:
    data[col] = conservative_winsorize(data[col])

# 分类处理
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0.0')
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)

# v特征（只保留基础统计，避免过拟合）
v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)

# 只保留1-2个最核心的v特征交叉
data['v_0_v_3'] = data['v_0'] * data['v_3']

# 业务特征（最核心的3个）
data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']

# 删除无用列和原始v特征
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
drop_cols += v_cols  # 删除所有原始v特征
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

# 不使用标准化，让模型自己处理
print(f"最终特征数: {X.shape[1]}")

# ==================== 5. 强正则化训练 ====================
print("\n【步骤5】开始训练（强正则化）...")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 极保守参数（强正则化）
cat_params = {
    'iterations': 2000,  # 减少迭代次数
    'learning_rate': 0.05,  # 提高学习率，快速收敛
    'depth': 5,  # 降低深度
    'l2_leaf_reg': 10,  # 强L2正则
    'random_strength': 1.0,  # 增加随机性
    'bagging_temperature': 1.0,  # 增强bagging
    'loss_function': 'MAE',
    'random_seed': SEED,
    'verbose': 0,
    'early_stopping_rounds': 100
}

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,  # 大幅减少叶子数
    'max_depth': 5,  # 降低深度
    'min_data_in_leaf': 50,  # 增加每叶最小样本数
    'feature_fraction': 0.6,  # 降低特征采样率
    'bagging_fraction': 0.6,  # 降低样本采样率
    'bagging_freq': 5,
    'reg_alpha': 0.5,  # 强L1正则
    'reg_lambda': 0.5,  # 强L2正则
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
        num_boost_round=2000,
        valid_sets=[val_data_lgb],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
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

print(f"\n⚠️ 注意: v6 CV 460 但测试集 729")
print(f"本版本使用强正则化防止过拟合")
print(f"预期测试集成绩应接近CV成绩")

# ==================== 7. 测试集预测 ====================
print("\n【步骤7】生成测试集预测...")

# 简单平均
final_pred = (cat_test_preds + lgb_test_preds) / 2
final_pred = np.maximum(final_pred, 50)

# 保守后处理
mean_pred = final_pred.mean()
if mean_pred > 6000:
    final_pred *= 0.98
    print(f"应用均值调整: 0.98")
elif mean_pred > 5500:
    final_pred *= 0.99
    print(f"应用均值调整: 0.99")

# 保存结果
submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit.to_csv('optimized_v7_submit.csv', index=False)

print(f"\n结果已保存: optimized_v7_submit.csv")
print(f"预测价格范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"预测价格均值: {final_pred.mean():.2f}")

print(f"\n{'='*70}")
print("✅ 优化v7版（抗过拟合）完成！")
print(f"{'='*70}")
