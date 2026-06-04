# -*- coding: utf-8 -*-
"""
二手车价格预测 - 深度优化v1
目标: 测试集MAE < 450
策略: 多模型集成 + 高级特征工程 + 后处理优化
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import StackingRegressor
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("="*70)
print("深度优化v1 - 目标 MAE < 450")
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
data['reg_month'] = (data['regDate'] % 10000) // 100
data['creat_year'] = data['creatDate'] // 10000
data['creat_month'] = (data['creatDate'] % 10000) // 100
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)
data['car_age_squared'] = data['car_age'] ** 2

# 异常值处理（多层）
def winsorize_series(series, lower=0.5, upper=99.5):
    lower_bound = np.percentile(series.dropna(), lower)
    upper_bound = np.percentile(series.dropna(), upper)
    return series.clip(lower=lower_bound, upper=upper_bound)

for col in ['power', 'kilometer', 'car_age']:
    data[col] = winsorize_series(data[col])

# 分类处理
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0.0')
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)

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
data['v_zeros'] = (data[v_cols] == 0).sum(axis=1)

# v特征全面交互（14个组合）
v_pairs = [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10), (0,11), (0,12), (0,13), (0,14),
           (1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (3,4), (3,5), (4,5), (6,7), (7,8), (8,9), (9,10)]
for i, j in v_pairs:
    data[f'v_{i}_v_{j}'] = data[f'v_{i}'] * data[f'v_{j}']

# 业务交叉特征（扩展）
data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']
data['power_age'] = data['power'] * data['car_age']
data['usage_intensity'] = data['kilometer'] / (data['car_age'] + 1)
data['power_per_km'] = data['power'] / (data['kilometer'] + 0.1)
data['km_per_year'] = data['kilometer'] / (data['car_age'] + 0.1)
data['power_per_age'] = data['power'] / (data['car_age'] + 0.1)

# 变换特征
data['power_log'] = np.log1p(data['power'] + 1)
data['km_log'] = np.log1p(data['kilometer'] + 1)
data['age_log'] = np.log1p(data['car_age'] + 1)
data['power_sqrt'] = np.sqrt(data['power'])
data['km_sqrt'] = np.sqrt(data['kilometer'])
data['age_sqrt'] = np.sqrt(data['car_age'])

# 组合特征
data['power_age_km'] = data['power'] * data['car_age'] * data['kilometer']
data['power_age_log'] = data['power'] * np.log1p(data['car_age'] + 1)
data['power_km_log'] = data['power'] * np.log1p(data['kilometer'] + 1)
data['v_mean_sqrt'] = np.sqrt(data['v_mean'])
data['v_sum_sqrt'] = np.sqrt(data['v_sum'])

# 分组统计特征（多维度）
for col in ['brand', 'model', 'regionCode', 'bodyType', 'fuelType']:
    data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')
    data[f'{col}_power_mean'] = data.groupby(col)['power'].transform('mean')
    data[f'{col}_age_mean'] = data.groupby(col)['car_age'].transform('mean')
    data[f'{col}_km_mean'] = data.groupby(col)['kilometer'].transform('mean')

# 价格分桶特征（基于训练数据）
train_data_temp = data[data['price'] != -1]
price_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
for q in price_quantiles:
    quantile_val = train_data_temp['price'].quantile(q)
    data[f'price_q{int(q*100)}'] = (data['price'] >= quantile_val).astype(int)

# 删除无用列
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

print(f"特征工程完成，总特征数: {data.shape[1]}")

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

# 分离训练和测试
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
    print(f"  Target编码: {col}")

# ==================== 5. 特征选择 ====================
print("\n【步骤5】特征选择...")

X = train_data.drop(columns=['price'])
y = train_data['price'].values

# 确保没有NaN（在特征选择前检查）
print(f"检查NaN: X中有{X.isnull().sum().sum()}个NaN")
X = X.fillna(X.median())

# 特征选择（选择最重要的50个特征）
selector = SelectKBest(score_func=f_regression, k=min(50, X.shape[1]))
selector.fit(X, y)
selected_feature_names = X.columns[selector.get_support()]

X_selected = selector.transform(X)
test_selected = selector.transform(test_data)

print(f"原始特征数: {X.shape[1]}, 选择后: {X_selected.shape[1]}")

# ==================== 6. 多种标准化 ====================
print("\n【步骤6】多种标准化...")

# 使用不同的标准化方法
scaler_standard = StandardScaler()
scaler_robust = RobustScaler()
scaler_minmax = MinMaxScaler()

# 尝试不同的标准化组合
scaler_methods = [
    ('Standard', scaler_standard),
    ('Robust', scaler_robust),
    ('MinMax', scaler_minmax)
]

# 我们将测试哪种标准化效果最好（基于CV）
X_train_scaled = {}
X_val_scaled = {}
test_scaled = {}

for name, scaler in scaler_methods:
    X_train_scaled[name] = scaler.fit_transform(X_selected)
    X_val_scaled[name] = scaler.transform(X_selected)
    test_scaled[name] = scaler.transform(test_selected)

# ==================== 7. 多模型训练 ====================
print("\n【步骤7】多模型训练（CatBoost + LightGBM + XGBoost）...")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 模型列表
models = []

# CatBoost参数集
cat_params_sets = [
    {'iterations': 3500, 'learning_rate': 0.025, 'depth': 7,
     'l2_leaf_reg': 6, 'random_strength': 0.7, 'bagging_temperature': 0.65,
     'loss_function': 'MAE', 'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 120},
    {'iterations': 4000, 'learning_rate': 0.022, 'depth': 8,
     'l2_leaf_reg': 7, 'random_strength': 0.6, 'bagging_temperature': 0.7,
     'loss_function': 'MAE', 'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 130},
    {'iterations': 4500, 'learning_rate': 0.020, 'depth': 6,
     'l2_leaf_reg': 8, 'random_strength': 0.8, 'bagging_temperature': 0.6,
     'loss_function': 'MAE', 'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 140}
]

# LightGBM参数集
lgb_params_sets = [
    {'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
     'learning_rate': 0.025, 'num_leaves': 100, 'max_depth': 8,
     'min_data_in_leaf': 20, 'feature_fraction': 0.85, 'bagging_fraction': 0.85,
     'bagging_freq': 5, 'reg_alpha': 0.15, 'reg_lambda': 0.15,
     'verbose': -1, 'seed': SEED},
    {'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
     'learning_rate': 0.022, 'num_leaves': 120, 'max_depth': 9,
     'min_data_in_leaf': 17, 'feature_fraction': 0.88, 'bagging_fraction': 0.88,
     'bagging_freq': 5, 'reg_alpha': 0.18, 'reg_lambda': 0.18,
     'verbose': -1, 'seed': SEED},
    {'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
     'learning_rate': 0.020, 'num_leaves': 90, 'max_depth': 7,
     'min_data_in_leaf': 25, 'feature_fraction': 0.82, 'bagging_fraction': 0.82,
     'bagging_freq': 6, 'reg_alpha': 0.20, 'reg_lambda': 0.20,
     'verbose': -1, 'seed': SEED}
]

# XGBoost参数集
xgb_params_sets = [
    {'objective': 'reg:squarederror', 'eval_metric': 'mae', 'booster': 'gbtree',
     'learning_rate': 0.025, 'max_depth': 8, 'min_child_weight': 1,
     'subsample': 0.85, 'colsample_bytree': 0.85,
     'reg_alpha': 0.1, 'reg_lambda': 0.1, 'random_state': SEED,
     'n_estimators': 3500, 'early_stopping_rounds': 120},
    {'objective': 'reg:squarederror', 'eval_metric': 'mae', 'booster': 'gbtree',
     'learning_rate': 0.022, 'max_depth': 9, 'min_child_weight': 1,
     'subsample': 0.88, 'colsample_bytree': 0.88,
     'reg_alpha': 0.15, 'reg_lambda': 0.15, 'random_state': SEED,
     'n_estimators': 4000, 'early_stopping_rounds': 130},
    {'objective': 'reg:squarederror', 'eval_metric': 'mae', 'booster': 'gbtree',
     'learning_rate': 0.020, 'max_depth': 7, 'min_child_weight': 1,
     'subsample': 0.82, 'colsample_bytree': 0.82,
     'reg_alpha': 0.2, 'reg_lambda': 0.2, 'random_state': SEED,
     'n_estimators': 4500, 'early_stopping_rounds': 140}
]

# 训练CatBoost
print("训练CatBoost模型...")
for i, params in enumerate(cat_params_sets):
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled['Standard'])):
        X_tr = X_train_scaled['Standard'][train_idx]
        X_val = X_train_scaled['Standard'][val_idx]
        y_tr = y[train_idx]
        y_val = y[val_idx]

        cat_model = CatBoostRegressor(**params)
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)

        pred = cat_model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        fold_maes.append(mae)

    model = CatBoostRegressor(**params)
    model.fit(X_train_scaled['Standard'], y, verbose=0)
    models.append(('CatBoost', i+1, np.mean(fold_maes), model, 'cat'))

    print(f"  CatBoost-{i+1}: CV MAE = {np.mean(fold_maes):.2f}")

# 训练LightGBM
print("训练LightGBM模型...")
for i, params in enumerate(lgb_params_sets):
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled['Standard'])):
        X_tr = X_train_scaled['Standard'][train_idx]
        X_val = X_train_scaled['Standard'][val_idx]
        y_tr = y[train_idx]
        y_val = y[val_idx]

        train_lgb = lgb.Dataset(X_tr, label=y_tr)
        val_lgb = lgb.Dataset(X_val, label=y_val, reference=train_lgb)

        lgb_model = lgb.train(
            params, train_lgb,
            num_boost_round=3500 if i == 0 else 4500,
            valid_sets=[val_lgb],
            callbacks=[lgb.early_stopping(120 if i == 0 else 140), lgb.log_evaluation(0)]
        )

        pred = lgb_model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        fold_maes.append(mae)

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train_scaled['Standard'], y, verbose=-1)
    models.append(('LightGBM', i+1, np.mean(fold_maes), model, 'lgb'))

    print(f"  LightGBM-{i+1}: CV MAE = {np.mean(fold_maes):.2f}")

# 训练XGBoost
print("训练XGBoost模型...")
for i, params in enumerate(xgb_params_sets):
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled['Standard'])):
        X_tr = X_train_scaled['Standard'][train_idx]
        X_val = X_train_scaled['Standard'][val_idx]
        y_tr = y[train_idx]
        y_val = y[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                 early_stopping_rounds=120 if i == 0 else 140, verbose=False)

        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        fold_maes.append(mae)

    model = xgb.XGBRegressor(**params)
    model.fit(X_train_scaled['Standard'], y, verbose=False)
    models.append(('XGBoost', i+1, np.mean(fold_maes), model, 'xgb'))

    print(f"  XGBoost-{i+1}: CV MAE = {np.mean(fold_maes):.2f}")

print(f"\n总计训练了 {len(models)} 个模型（3 CatBoost + 3 LightGBM + 3 XGBoost）")

# ==================== 8. 模型集成 ====================
print("\n【步骤8】模型集成...")

# 获取所有模型的预测
n_models = len(models)
model_predictions = np.zeros((len(y), n_models))
test_predictions = np.zeros((len(test_data), n_models))

for i, (_, _, model, model_type) in enumerate(models):
    # CV预测
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled['Standard'])):
        X_val = X_train_scaled['Standard'][val_idx]
        pred = model.predict(X_val)
        model_predictions[val_idx, i] = pred

    # 测试集预测
    if i < 3:  # CatBoost
        test_predictions[:, i] = model.predict(test_scaled['Standard'])
    elif i < 6:  # LightGBM
        test_predictions[:, i] = model.predict(test_scaled['Standard'])
    else:  # XGBoost
        test_predictions[:, i] = model.predict(test_scaled['Standard'])

# 找到最佳融合策略
print("测试多种融合策略...")

# 策略1: Top3平均
top3_indices = np.argsort([m[1] for m in models])[:3]
ensemble_top3 = model_predictions[:, top3_indices].mean(axis=1)
mae_top3 = mean_absolute_error(y, ensemble_top3)
print(f"  Top3平均融合: MAE = {mae_top3:.2f}")

# 策略2: Top5平均
top5_indices = np.argsort([m[1] for m in models])[:5]
ensemble_top5 = model_predictions[:, top5_indices].mean(axis=1)
mae_top5 = mean_absolute_error(y, ensemble_top5)
print(f"  Top5平均融合: MAE = {mae_top5:.2f}")

# 策略3: 加权平均（基于CV表现）
weights = np.array([1/m[1] for m in models])
weights = weights / weights.sum()
ensemble_weighted = (model_predictions * weights).sum(axis=1)
mae_weighted = mean_absolute_error(y, ensemble_weighted)
print(f"  加权平均融合: MAE = {mae_weighted:.2f}")

# 策略4: Stacking（元学习器）
print("  Stacking融合（元学习器）...")
stack_X = model_predictions
stacking_models = [
    Ridge(alpha=0.1),
    Ridge(alpha=1.0),
    Ridge(alpha=5.0),
    ElasticNet(alpha=0.1, l1_ratio=0.5)
]

stack_maes = []
for meta_model in stacking_models:
    meta_model.fit(stack_X, y)
    ensemble_pred = meta_model.predict(stack_X)
    mae = mean_absolute_error(y, ensemble_pred)
    stack_maes.append(mae)

best_stack_idx = np.argmin(stack_maes)
mae_stacking = stack_maes[best_stack_idx]
print(f"    最佳Stacking MAE = {mae_stacking:.2f}")

# ==================== 9. 后处理优化 ====================
print("\n【步骤9】后处理优化...")

# 选择最佳策略
strategies = {
    'Top3平均': mae_top3,
    'Top5平均': mae_top5,
    '加权平均': mae_weighted,
    'Stacking': mae_stacking
}

best_strategy = min(strategies, key=strategies.get)
best_mae = strategies[best_strategy]

print(f"最佳策略: {best_strategy}, CV MAE = {best_mae:.2f}")

# 根据最佳策略生成最终预测
if best_strategy == 'Top3平均':
    final_pred = test_predictions[:, top3_indices].mean(axis=1)
elif best_strategy == 'Top5平均':
    final_pred = test_predictions[:, top5_indices].mean(axis=1)
elif best_strategy == '加权平均':
    weights = np.array([1/m[1] for m in models])
    weights = weights / weights.sum()
    final_pred = (test_predictions * weights).sum(axis=1)
else:  # Stacking
    best_meta_model = stacking_models[best_stack_idx]
    best_meta_model.fit(model_predictions, y)
    final_pred = best_meta_model.predict(test_predictions)

# 后处理1: 价格裁剪
final_pred = np.maximum(final_pred, 50)

# 后处理2: 基于分布调整
mean_pred = final_pred.mean()
std_pred = final_pred.std()

# 如果均值过高，适当调整
if mean_pred > 6000:
    adjustment_factor = 0.985
    final_pred = final_pred * adjustment_factor
    print(f"应用均值调整: {adjustment_factor}")

# 后处理3: 异常值平滑（3σ规则）
pred_mean = final_pred.mean()
pred_std = final_pred.std()
upper_bound = pred_mean + 3 * pred_std
lower_bound = pred_mean - 3 * pred_std
final_pred = np.clip(final_pred, lower_bound, upper_bound)
print(f"应用3σ平滑")

# ==================== 10. 保存结果 ====================
print("\n【步骤10】保存结果...")

submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit.to_csv('deep_optimization_v1_submit.csv', index=False)

print(f"\n{'='*70}")
print("【最终结果】")
print(f"{'='*70}")
print(f"训练模型数: {n_models}")
print(f"最佳策略: {best_strategy}")
print(f"最终CV MAE: {best_mae:.2f}")
print(f"目标: 450")
print(f"差距: {best_mae - 450:.2f}")

if best_mae < 450:
    print(f"\n🎉 成功达到目标！MAE: {best_mae:.2f} < 450")
    print(f"改进幅度: {450 - best_mae:.2f} 点")
else:
    print(f"\n⚠️ 距离目标: {best_mae - 450:.2f}")
    print(f"达成度: {(450/best_mae)*100:.1f}%")

print(f"\n结果已保存: deep_optimization_v1_submit.csv")
print(f"预测价格范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"预测价格均值: {final_pred.mean():.2f}")

print(f"\n{'='*70}")
print("✅ 深度优化v1完成！")
print(f"{'='*70}")
