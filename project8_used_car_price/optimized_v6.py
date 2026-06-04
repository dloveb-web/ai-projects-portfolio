# -*- coding: utf-8 -*-
"""
二手车价格预测 - 优化v6版（终极突破）
目标: 测试集MAE < 450
基于: optimized_v5 (MAE 462.13) 最后冲刺
策略: 基于v5最优配置微调参数
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
print("优化v6版 - 目标 MAE < 450 (终极突破)")
print("基于: optimized_v5 (MAE 462.13)")
print("="*70)

# ==================== 1. 数据加载 ====================
print("\n【步骤1】数据加载...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

print(f"训练集: {len(train)}, 测试集: {len(test)}")

# ==================== 2. 特征工程 ====================
print("\n【步骤2】特征工程...")

# 时间特征
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)
data['reg_month'] = (data['regDate'] % 10000) // 100
data['creat_month'] = (data['creatDate'] % 10000) // 100
data['month_diff'] = (data['creat_month'] + data['creat_year']*12) - (data['reg_month'] + data['reg_year']*12)

# 异常值处理（更温和）
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
data['v_sum'] = data[v_cols].sum(axis=1)
data['v_skew'] = data[v_cols].skew(axis=1)
data['v_kurt'] = data[v_cols].kurt(axis=1)

# v特征交叉（精选核心）
data['v_0_v_3'] = data['v_0'] * data['v_3']
data['v_0_v_2'] = data['v_0'] * data['v_2']
data['v_0_v_12'] = data['v_0'] * data['v_12']
data['v_3_v_12'] = data['v_3'] * data['v_12']
data['v_1_v_5'] = data['v_1'] * data['v_5']
data['v_2_v_4'] = data['v_2'] * data['v_4']

# v特征与业务交叉（精选）
for col in ['v_0', 'v_1', 'v_2', 'v_3', 'v_5', 'v_12']:
    data[f'{col}_power'] = data[col] * data['power']
    data[f'{col}_km'] = data[col] * data['kilometer']

# 业务特征
data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']
data['power_age'] = data['power'] * data['car_age']
data['usage_intensity'] = data['kilometer'] / (data['car_age'] + 1)
data['power_per_age'] = data['power'] / (data['car_age'] + 1)

# v与业务聚合
data['v_mean_power'] = data['v_mean'] * data['power']
data['v_mean_km'] = data['v_mean'] * data['kilometer']
data['v_mean_age'] = data['v_mean'] * data['car_age']

# 分组统计
data['brand_count'] = data.groupby('brand')['SaleID'].transform('count')
data['model_count'] = data.groupby('model')['SaleID'].transform('count')

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

# ==================== 5. 多模型训练 ====================
print("\n【步骤5】开始多模型训练...")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 基于v5最优结果微调参数（462.13 MAE）
cat_params_sets = [
    # 保守
    {'iterations': 5000, 'learning_rate': 0.028, 'depth': 8, 'l2_leaf_reg': 4,
     'random_strength': 0.35, 'bagging_temperature': 0.55, 'loss_function': 'MAE',
     'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 90},
    # 平衡（v5最优）
    {'iterations': 5500, 'learning_rate': 0.032, 'depth': 9, 'l2_leaf_reg': 3,
     'random_strength': 0.3, 'bagging_temperature': 0.45, 'loss_function': 'MAE',
     'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 80}
]

lgb_params_sets = [
    # 保守
    {'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
     'learning_rate': 0.028, 'num_leaves': 150, 'max_depth': 9,
     'min_data_in_leaf': 10, 'feature_fraction': 0.94, 'bagging_fraction': 0.94,
     'bagging_freq': 3, 'reg_alpha': 0.10, 'reg_lambda': 0.10, 'verbose': -1, 'seed': SEED},
    # 平衡（v5最优）
    {'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
     'learning_rate': 0.032, 'num_leaves': 170, 'max_depth': 10,
     'min_data_in_leaf': 9, 'feature_fraction': 0.97, 'bagging_fraction': 0.97,
     'bagging_freq': 2, 'reg_alpha': 0.08, 'reg_lambda': 0.08, 'verbose': -1, 'seed': SEED}
]

print(f"准备训练4个模型...")

# 存储所有模型的预测
stack_train = np.zeros((len(X), 4))
stack_test = np.zeros((len(test_data), 4))
fold_maes = []

for model_idx in range(4):
    print(f"\n训练模型{model_idx+1}/4...")
    model_fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if model_idx < 2:  # CatBoost
            cat_model = CatBoostRegressor(**cat_params_sets[model_idx])
            cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
            pred = cat_model.predict(X_val)
            mae = mean_absolute_error(y_val, pred)
            model_fold_maes.append(mae)

            stack_train[val_idx, model_idx] = pred
            if fold == 0:
                stack_test[:, model_idx] = cat_model.predict(test_data)

            print(f"  Fold {fold+1}/5 Cat MAE: {mae:.2f}")

        else:  # LightGBM
            lgb_idx = model_idx - 2
            train_data_lgb = lgb.Dataset(X_train, label=y_train)
            val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

            lgb_model = lgb.train(
                lgb_params_sets[lgb_idx], train_data_lgb,
                num_boost_round=5500,
                valid_sets=[val_data_lgb],
                callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)]
            )

            pred = lgb_model.predict(X_val)
            mae = mean_absolute_error(y_val, pred)
            model_fold_maes.append(mae)

            stack_train[val_idx, model_idx] = pred
            if fold == 0:
                stack_test[:, model_idx] = lgb_model.predict(test_data)

            print(f"  Fold {fold+1}/5 Lgb MAE: {mae:.2f}")

    fold_maes.append(model_fold_maes)
    print(f"  模型{model_idx+1} 平均 MAE: {np.mean(model_fold_maes):.2f}")

# ==================== 6. 多种融合策略 ====================
print("\n【步骤6】融合策略...")

# 策略1: 简单平均
avg_pred = stack_train.mean(axis=1)
mae_avg = mean_absolute_error(y, avg_pred)
print(f"简单平均: MAE = {mae_avg:.2f}")

# 策略2: 加权平均（基于CV）
cv_scores = [np.mean(m) for m in fold_maes]
inv_cv = np.array([1/s for s in cv_scores])
weights = inv_cv / inv_cv.sum()
weighted_pred = (stack_train * weights).sum(axis=1)
mae_weighted = mean_absolute_error(y, weighted_pred)
print(f"加权平均: MAE = {mae_weighted:.2f}")
print(f"  权重: {weights}")

# 策略3: Ridge Stacking
ridge = Ridge(alpha=1.0)
ridge.fit(stack_train, y)
ridge_pred = ridge.predict(stack_train)
mae_ridge = mean_absolute_error(y, ridge_pred)
print(f"Ridge Stacking: MAE = {mae_ridge:.2f}")

# 策略4: Top3平均（跳过最差的）
top3_idx = np.argsort(cv_scores)[:3]
top3_pred = stack_train[:, top3_idx].mean(axis=1)
mae_top3 = mean_absolute_error(y, top3_pred)
print(f"Top3平均: MAE = {mae_top3:.2f}")

# 策略5: 手动权重（基于v5经验）
manual_weights = np.array([0.25, 0.25, 0.25, 0.25])
manual_pred = (stack_train * manual_weights).sum(axis=1)
mae_manual = mean_absolute_error(y, manual_pred)
print(f"手动平均(25:25:25:25): MAE = {mae_manual:.2f}")

# 策略6: CatBoost权重略高（Cat更好）
manual_weights2 = np.array([0.35, 0.35, 0.15, 0.15])
manual_pred2 = (stack_train * manual_weights2).sum(axis=1)
mae_manual2 = mean_absolute_error(y, manual_pred2)
print(f"手动平均(35:35:15:15): MAE = {mae_manual2:.2f}")

# ==================== 7. 选择最佳策略 ====================
print("\n【步骤7】选择最佳策略...")

strategies = {
    '简单平均': mae_avg,
    '加权平均': mae_weighted,
    'Ridge Stacking': mae_ridge,
    'Top3平均': mae_top3,
    '手动25:25:25:25': mae_manual,
    '手动35:35:15:15': mae_manual2
}

best_strategy = min(strategies, key=strategies.get)
best_mae = strategies[best_strategy]

print(f"\n最佳策略: {best_strategy}")
print(f"CV MAE: {best_mae:.2f}")
print(f"目标: 450")
print(f"差距: {best_mae - 450:.2f}")

if best_mae < 450:
    print(f"\n🎉 成功达到目标！MAE: {best_mae:.2f} < 450")
    print(f"🏆 项目重大突破！")

# ==================== 8. 生成测试集预测 ====================
print("\n【步骤8】生成测试集预测...")

if best_strategy == '简单平均':
    final_pred = stack_test.mean(axis=1)
elif best_strategy == '加权平均':
    final_pred = (stack_test * weights).sum(axis=1)
elif best_strategy == 'Ridge Stacking':
    final_pred = ridge.predict(stack_test)
elif best_strategy == 'Top3平均':
    final_pred = stack_test[:, top3_idx].mean(axis=1)
elif best_strategy == '手动25:25:25:25':
    final_pred = (stack_test * manual_weights).sum(axis=1)
else:  # 手动35:35:15:15
    final_pred = (stack_test * manual_weights2).sum(axis=1)

# 后处理
final_pred = np.maximum(final_pred, 50)
mean_pred = final_pred.mean()
std_pred = final_pred.std()

# 均值调整
if mean_pred > 6000:
    final_pred *= 0.97
    print(f"应用均值调整: 0.97")
elif mean_pred > 5500:
    final_pred *= 0.985
    print(f"应用均值调整: 0.985")

# 2.5σ平滑
final_pred = np.clip(final_pred, None, mean_pred + 2.5 * std_pred)
print(f"应用2.5σ平滑")

# 保存结果
submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit.to_csv('optimized_v6_submit.csv', index=False)

print(f"\n{'='*70}")
print("【最终结果】")
print(f"{'='*70}")
print(f"最佳策略: {best_strategy}")
print(f"最终CV MAE: {best_mae:.2f}")
print(f"目标MAE: 450")
print(f"差距: {best_mae - 450:.2f}")

if best_mae < 450:
    print(f"\n🎉 成功达到目标！MAE: {best_mae:.2f} < 450")
    print(f"🏆 项目重大突破！")
else:
    print(f"\n⚠️ 距离目标: {best_mae - 450:.2f}")
    print(f"达成度: {(450/best_mae)*100:.1f}%")

print(f"\n结果已保存: optimized_v6_submit.csv")
print(f"预测价格范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"预测价格均值: {final_pred.mean():.2f}")

print(f"\n{'='*70}")
print("✅ 优化v6版（终极突破）完成！")
print(f"{'='*70}")
