# -*- coding: utf-8 -*-
"""
二手车价格预测 - 终极优化版
目标: 测试集MAE < 450
策略: 结合所有成功经验 + 多模型保守融合
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("="*70)
print("终极优化版 - 目标 MAE < 450")
print("="*70)

# ==================== 1. 数据加载 ====================
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

print(f"训练集: {len(train)}, 测试集: {len(test)}")

# ==================== 2. 高级特征工程 ====================
print("\n【步骤1】高级特征工程...")

# 时间特征
data['reg_year'] = data['regDate'] // 10000
data['reg_month'] = (data['regDate'] % 10000) // 100
data['creat_year'] = data['creatDate'] // 10000
data['creat_month'] = (data['creatDate'] % 10000) // 100
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)
data['car_age_month'] = data['car_age'] * 12 + data['reg_month'] - data['creat_month']

# 异常值处理（Winsorize 1%-99%）
def winsorize_series(series, lower=1, upper=99):
    lower_bound = np.percentile(series.dropna(), lower)
    upper_bound = np.percentile(series.dropna(), upper)
    return series.clip(lower=lower_bound, upper=upper_bound)

for col in ['power', 'kilometer', 'car_age']:
    data[col] = winsorize_series(data[col])

# v特征高级统计
v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)
data['v_max'] = data[v_cols].max(axis=1)
data['v_min'] = data[v_cols].min(axis=1)
data['v_range'] = data['v_max'] - data['v_min']
data['v_skew'] = data[v_cols].skew(axis=1)
data['v_kurt'] = data[v_cols].kurt(axis=1)
data['v_sum'] = data[v_cols].sum(axis=1)

# v特征交叉（更全面）
v_pairs = [(0,3), (0,2), (0,12), (3,12), (1,5), (2,4), (6,7), (8,9), (10,11), (13,14)]
for i, j in v_pairs:
    data[f'v_{i}_v_{j}'] = data[f'v_{i}'] * data[f'v_{j}']

# 业务交叉特征（核心）
data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']
data['power_age'] = data['power'] * data['car_age']
data['usage_intensity'] = data['kilometer'] / (data['car_age'] + 1)
data['power_per_km'] = data['power'] / (data['kilometer'] + 0.1)
data['km_per_year'] = data['kilometer'] / (data['car_age'] + 0.1)
data['power_per_age'] = data['power'] / (data['car_age'] + 0.1)
data['power_sqrt'] = np.sqrt(data['power'])
data['km_sqrt'] = np.sqrt(data['kilometer'])
data['age_sqrt'] = np.sqrt(data['car_age'])

# 组合特征
data['power_age_km'] = data['power'] * data['car_age'] * data['kilometer']
data['power_age_sqrt'] = data['power'] * np.sqrt(data['car_age'])
data['v_mean_sqrt'] = np.sqrt(data['v_mean'])

# 分组统计特征
for col in ['brand', 'model', 'regionCode', 'bodyType', 'fuelType']:
    data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')
    data[f'{col}_power_mean'] = data.groupby(col)['power'].transform('mean')
    data[f'{col}_age_mean'] = data.groupby(col)['car_age'].transform('mean')

# 分类特征处理
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0.0')
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)

# 删除无用列
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

print(f"特征工程完成，特征数: {data.shape[1]}")

# ==================== 3. 缺失值处理 ====================
print("\n【步骤2】缺失值处理...")

numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=['object']).columns

for col in numeric_cols:
    median_val = data[col].median()
    data[col] = data[col].fillna(median_val)

for col in categorical_cols:
    mode_val = data[col].mode()[0] if not data[col].mode().empty else 'unknown'
    data[col] = data[col].fillna(mode_val)

print(f"数值特征: {len(numeric_cols)}, 分类特征: {len(categorical_cols)}")

# ==================== 4. 分类特征编码 ====================
print("\n【步骤3】分类特征编码...")

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    print(f"  Label编码: {col}")

# 分离数据
train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
if 'price' in test_data.columns:
    test_data = test_data.drop(columns=['price'])

# Target Encoding
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

# ==================== 5. 特征标准化 ====================
print("\n【步骤4】特征标准化...")

X = train_data.drop(columns=['price'])
y = train_data['price'].values

scaler = StandardScaler()
numeric_cols_std = X.select_dtypes(include=[np.number]).columns
X[numeric_cols_std] = scaler.fit_transform(X[numeric_cols_std])
test_data[numeric_cols_std] = scaler.transform(test_data[numeric_cols_std])

print(f"最终特征数: {X.shape[1]}")

# ==================== 6. 多模型训练 ====================
print("\n【步骤5】多模型训练...")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 3组不同的参数
cat_params_sets = [
    # Set 1: 平衡型
    {'iterations': 4000, 'learning_rate': 0.022, 'depth': 7,
     'l2_leaf_reg': 7, 'random_strength': 0.6, 'bagging_temperature': 0.7,
     'loss_function': 'MAE', 'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 120},

    # Set 2: 激进型
    {'iterations': 4500, 'learning_rate': 0.020, 'depth': 8,
     'l2_leaf_reg': 6, 'random_strength': 0.8, 'bagging_temperature': 0.6,
     'loss_function': 'MAE', 'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 140},

    # Set 3: 保守型
    {'iterations': 5000, 'learning_rate': 0.018, 'depth': 6,
     'l2_leaf_reg': 8, 'random_strength': 0.5, 'bagging_temperature': 0.8,
     'loss_function': 'MAE', 'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 150}
]

lgb_params_sets = [
    # Set 1: 平衡型
    {'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
     'learning_rate': 0.022, 'num_leaves': 115, 'max_depth': 9,
     'min_data_in_leaf': 18, 'feature_fraction': 0.87, 'bagging_fraction': 0.87,
     'bagging_freq': 5, 'reg_alpha': 0.18, 'reg_lambda': 0.18,
     'min_split_gain': 0.008, 'verbose': -1, 'seed': SEED},

    # Set 2: 激进型
    {'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
     'learning_rate': 0.020, 'num_leaves': 127, 'max_depth': 10,
     'min_data_in_leaf': 15, 'feature_fraction': 0.90, 'bagging_fraction': 0.90,
     'bagging_freq': 4, 'reg_alpha': 0.20, 'reg_lambda': 0.20,
     'min_split_gain': 0.005, 'verbose': -1, 'seed': SEED},

    # Set 3: 保守型
    {'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
     'learning_rate': 0.018, 'num_leaves': 100, 'max_depth': 8,
     'min_data_in_leaf': 20, 'feature_fraction': 0.85, 'bagging_fraction': 0.85,
     'bagging_freq': 5, 'reg_alpha': 0.22, 'reg_lambda': 0.22,
     'min_split_gain': 0.010, 'verbose': -1, 'seed': SEED}
]

# 存储所有模型预测
n_models = 3  # 3个CatBoost + 3个LightGBM = 6个模型
model_predictions = np.zeros((len(X), n_models))
test_predictions = np.zeros((len(test_data), n_models))

all_fold_maes = []

print(f"训练{n_models*2}个模型（3 CatBoost + 3 LightGBM）...")

for model_idx in range(n_models):
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # CatBoost
        cat_model = CatBoostRegressor(**cat_params_sets[model_idx])
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
        pred = cat_model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        fold_maes.append(mae)

        if fold == 0:  # 只在第一个fold保存测试集预测
            test_predictions[:, model_idx] = cat_model.predict(test_data)

        model_predictions[val_idx, model_idx] = pred

    avg_mae = np.mean(fold_maes)
    all_fold_maes.append(avg_mae)
    print(f"  模型{model_idx+1} (CatBoost): MAE = {avg_mae:.2f}")

for model_idx in range(n_models):
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # LightGBM
        train_data_lgb = lgb.Dataset(X_train, label=y_train)
        val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

        lgb_model = lgb.train(
            lgb_params_sets[model_idx], train_data_lgb,
            num_boost_round=5000,
            valid_sets=[val_data_lgb],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
        )

        pred = lgb_model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        fold_maes.append(mae)

        if fold == 0:
            test_predictions[:, model_idx] = lgb_model.predict(test_data)

        model_predictions[val_idx, n_models + model_idx] = pred

    avg_mae = np.mean(fold_maes)
    all_fold_maes.append(avg_mae)
    print(f"  模型{model_idx+n_models+1} (LightGBM): MAE = {avg_mae:.2f}")

# ==================== 7. 多种融合策略 ====================
print("\n【步骤6】融合策略测试...")

# 策略1: 简单平均（最稳定）
ensemble_avg = model_predictions.mean(axis=1)
mae_avg = mean_absolute_error(y, ensemble_avg)
print(f"简单平均融合: MAE = {mae_avg:.2f}")

# 策略2: 最好模型预测
best_model_idx = np.argmin([np.mean([all_fold_maes[i]]) for i in range(n_models*2)])
ensemble_best = model_predictions[:, best_model_idx]
mae_best = mean_absolute_error(y, ensemble_best)
print(f"最好单模型: MAE = {mae_best:.2f}")

# 策略3: 权重平均（基于CV表现）
weights = np.array([1/mae for mae in all_fold_maes])
weights = weights / weights.sum()
ensemble_weighted = (model_predictions * weights).sum(axis=1)
mae_weighted = mean_absolute_error(y, ensemble_weighted)
print(f"加权平均融合: MAE = {mae_weighted:.2f}")

# 策略4: 选择性融合（只使用最好的3个模型）
top3_indices = np.argsort([np.mean([all_fold_maes[i]]) for i in range(n_models*2)])[:3]
ensemble_top3 = model_predictions[:, top3_indices].mean(axis=1)
mae_top3 = mean_absolute_error(y, ensemble_top3)
print(f"Top3平均融合: MAE = {mae_top3:.2f}")

# 策略5: 自适应权重（Ridge回归）
stack_X = model_predictions
stack_model = Ridge(alpha=1.0)
stack_model.fit(stack_X, y)
ensemble_lr = stack_model.predict(stack_X)
mae_lr = mean_absolute_error(y, ensemble_lr)
print(f"线性回归融合: MAE = {mae_lr:.2f}")

# 找到最佳策略
strategies = {
    '简单平均': mae_avg,
    '最好单模型': mae_best,
    '加权平均': mae_weighted,
    'Top3平均': mae_top3,
    '线性回归': mae_lr
}

best_strategy = min(strategies, key=strategies.get)
best_mae = strategies[best_strategy]

print(f"\n最佳策略: {best_strategy}, MAE = {best_mae:.2f}")

# ==================== 8. 测试集预测 ====================
print("\n【步骤7】生成测试集预测...")

if best_strategy == '简单平均':
    final_pred = test_predictions.mean(axis=1)
elif best_strategy == '最好单模型':
    final_pred = test_predictions[:, best_model_idx]
elif best_strategy == '加权平均':
    weights = np.array([1/mae for mae in all_fold_maes])
    weights = weights / weights.sum()
    final_pred = (test_predictions * weights).sum(axis=1)
elif best_strategy == 'Top3平均':
    top3_indices = np.argsort([np.mean([all_fold_maes[i]]) for i in range(n_models*2)])[:3]
    final_pred = test_predictions[:, top3_indices].mean(axis=1)
else:  # 线性回归
    final_pred = stack_model.predict(test_predictions)

# 价格裁剪和后处理
final_pred = np.maximum(final_pred, 50)

# 可选：基于分布的后处理
mean_pred = final_pred.mean()
std_pred = final_pred.std()

# 微调：稍微拉低高预测值，减少整体MAE
if mean_pred > 5000:
    adjustment_factor = 0.995
    final_pred = final_pred * adjustment_factor
    print(f"应用后处理调整: {adjustment_factor}")

# 保存结果
submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit.to_csv('ultimate_optimization_submit.csv', index=False)
print(f"\n结果已保存: ultimate_optimization_submit.csv")
print(f"预测价格范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"预测价格均值: {final_pred.mean():.2f}")

# ==================== 9. 最终总结 ====================
print(f"\n{'='*70}")
print("【最终结果】")
print(f"{'='*70}")
print(f"最佳策略: {best_strategy}")
print(f"最终MAE: {best_mae:.2f}")
print(f"目标: 450")
print(f"差距: {best_mae - 450:.2f}")

if best_mae < 450:
    print(f"\n🎉 成功达到目标！MAE: {best_mae:.2f} < 450")
    print(f"改进幅度: {450 - best_mae:.2f} 点")
else:
    print(f"\n⚠️ 距离目标: {best_mae - 450:.2f}")
    print(f"达成度: {(450/best_mae)*100:.1f}%")

# 单模型性能
print(f"\n单模型性能（CV平均）:")
for i in range(n_models*2):
    model_type = "CatBoost" if i < n_models else "LightGBM"
    model_num = i + 1 if i < n_models else i - n_models + 1
    print(f"  {model_type}-{model_num}: MAE = {all_fold_maes[i]:.2f}")

print(f"\n{'='*70}")
print("✅ 终极优化完成！")
print(f"{'='*70}")
