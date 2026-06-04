# -*- coding: utf-8 -*-
"""
二手车价格预测 - 突破450版
目标: 测试集MAE < 450
策略: 极简特征 + 超强正则 + 最优集成
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import StackingRegressor
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("="*70)
print("突破450版 - 目标 MAE < 450")
print("="*70)

# ==================== 1. 数据加载 ====================
print("\n【步骤1】数据加载...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

print(f"训练集: {len(train)}, 测试集: {len(test)}")

# ==================== 2. 极简特征工程（防过拟合）====================
print("\n【步骤2】极简特征工程（防过拟合）...")

# 只使用最核心特征，避免过拟合
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)

# 异常值处理（更保守的缩尾）
def conservative_winsorize(series, lower=2, upper=98):
    lower_bound = np.percentile(series.dropna(), lower)
    upper_bound = np.percentile(series.dropna(), upper)
    return series.clip(lower=lower_bound, upper=upper_bound)

for col in ['power', 'kilometer', 'car_age']:
    data[col] = conservative_winsorize(data[col])

# 分类处理
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0.0')
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)

# v特征（只保留最核心的）
v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)
data['v_sum'] = data[v_cols].sum(axis=1)

# 只保留核心业务特征（基于460成功经验）
data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']
data['power_age'] = data['power'] * data['car_age']

# 删除无用列
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

print(f"特征工程完成，特征数: {data.shape[1]}")

# ==================== 3. 缺失值处理（保守）====================
print("\n【步骤3】缺失值处理...")

numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# 数值用中位数填充
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

# 分类用众数填充
for col in categorical_cols:
    mode_val = data[col].mode()[0] if not data[col].mode().empty else 'unknown'
    data[col] = data[col].fillna(mode_val)

print(f"数值特征: {len(numeric_cols)}, 分类特征: {len(categorical_cols)}")

# ==================== 4. 分类特征编码（极简）====================
print("\n【步骤4】分类特征编码...")

# 只使用Label Encoding，完全避免Target编码的过拟合
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# 分离数据
train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
if 'price' in test_data.columns:
    test_data = test_data.drop(columns=['price'])

# 最终特征分离
X = train_data.drop(columns=['price'])
y = train_data['price'].values

# 标准化（更保守）
scaler = StandardScaler()
numeric_cols_std = X.select_dtypes(include=[np.number]).columns
X[numeric_cols_std] = scaler.fit_transform(X[numeric_cols_std])
test_data[numeric_cols_std] = scaler.transform(test_data[numeric_cols_std])

print(f"最终特征数: {X.shape[1]}")

# ==================== 5. 超强正则化训练 ====================
print("\n【步骤5】超强正则化训练...")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 极保守的参数配置（超强正则化）
cat_params_sets = [
    # Set 1: 超保守
    {'iterations': 3000, 'learning_rate': 0.015, 'depth': 6,
     'l2_leaf_reg': 10, 'random_strength': 0.5, 'bagging_temperature': 1.0,
     'loss_function': 'MAE', 'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 150},

    # Set 2: 平衡保守
    {'iterations': 3500, 'learning_rate': 0.018, 'depth': 7,
     'l2_leaf_reg': 8, 'random_strength': 0.6, 'bagging_temperature': 0.8,
     'loss_function': 'MAE', 'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 130},

    # Set 3: 略激进（基于CV）
    {'iterations': 4000, 'learning_rate': 0.020, 'depth': 8,
     'l2_leaf_reg': 6, 'random_strength': 0.7, 'bagging_temperature': 0.7,
     'loss_function': 'MAE', 'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 120}
]

lgb_params_sets = [
    # Set 1: 超保守
    {'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
     'learning_rate': 0.015, 'num_leaves': 63, 'max_depth': 6,
     'min_data_in_leaf': 30, 'feature_fraction': 0.75, 'bagging_fraction': 0.75,
     'bagging_freq': 6, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
     'min_split_gain': 0.02, 'verbose': -1, 'seed': SEED},

    # Set 2: 平衡保守
    {'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
     'learning_rate': 0.018, 'num_leaves': 95, 'max_depth': 7,
     'min_data_in_leaf': 25, 'feature_fraction': 0.80, 'bagging_fraction': 0.80,
     'bagging_freq': 5, 'reg_alpha': 0.4, 'reg_lambda': 0.4,
     'min_split_gain': 0.015, 'verbose': -1, 'seed': SEED},

    # Set 3: 略激进
    {'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
     'learning_rate': 0.020, 'num_leaves': 80, 'max_depth': 8,
     'min_data_in_leaf': 20, 'feature_fraction': 0.85, 'bagging_fraction': 0.85,
     'bagging_freq': 5, 'reg_alpha': 0.3, 'reg_lambda': 0.3,
     'min_split_gain': 0.010, 'verbose': -1, 'seed': SEED}
]

print(f"准备训练{len(cat_params_sets)*2}个模型...")

# 存储所有模型预测
cat_test_preds = np.zeros(len(test_data))
lgb_test_preds = np.zeros(len(test_data))

fold_maes_cat = []
fold_maes_lgb = []
fold_maes_avg = []

# ==================== 6. 训练所有模型 ====================
print("\n开始5折交叉验证训练...")

for model_idx in range(6):  # 0,1,2 = CatBoost, 3,4,5 = LightGBM
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold+1}/5 - 模型{model_idx+1}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if model_idx < 3:  # CatBoost
            cat_model = CatBoostRegressor(**cat_params_sets[model_idx])
            cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
            pred = cat_model.predict(X_val)
            mae = mean_absolute_error(y_val, pred)
            fold_maes.append(mae)

            if fold == 0:
                cat_test_preds += cat_model.predict(test_data) / 5

            print(f"  CV MAE: {mae:.2f}")

        else:  # LightGBM
            lgb_idx = model_idx - 3
            train_data_lgb = lgb.Dataset(X_train, label=y_train)
            val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

            lgb_model = lgb.train(
                lgb_params_sets[lgb_idx], train_data_lgb,
                num_boost_round=4000,
                valid_sets=[val_data_lgb],
                callbacks=[lgb.early_stopping(150 if lgb_idx == 0 else 130), lgb.log_evaluation(0)]
            )

            pred = lgb_model.predict(X_val)
            mae = mean_absolute_error(y_val, pred)
            fold_maes.append(mae)

            if fold == 0:
                lgb_test_preds += lgb_model.predict(test_data) / 5

            print(f"  CV MAE: {mae:.2f}")

    # 简单平均融合
    X_val_cat = X_val[:, :3]
    X_val_lgb = X_val[:, 3:]
    cat_pred = X_val_cat.mean(axis=1) if fold == 0 else cat_test_preds
    lgb_pred = X_val_lgb.mean(axis=1) if fold == 0 else lgb_test_preds
    avg_pred = (cat_pred + lgb_pred) / 2

    avg_mae = mean_absolute_error(y_val, avg_pred)
    fold_maes_avg.append(avg_mae)

    print(f"  平均融合 MAE: {avg_mae:.2f}")

    if model_idx < 3:
        fold_maes_cat.append(avg_mae)
    else:
        fold_maes_lgb.append(avg_mae)

# ==================== 7. 多种融合策略 ====================
print("\n【步骤6】测试多种融合策略...")

# 获取所有模型的预测
model_predictions = np.zeros((len(X), 6))
test_predictions = np.zeros((len(test_data), 6))

for model_idx in range(6):
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        if fold == 0:  # 只在第0个fold收集测试集预测
            X_val = X.iloc[val_idx]
            if model_idx < 3:  # CatBoost
                pred = CatBoostRegressor(**cat_params_sets[model_idx]).fit(
                    X.iloc[train_idx], y[train_idx], verbose=0).predict(X_val)
            else:  # LightGBM
                lgb_idx = model_idx - 3
                train_data_lgb = lgb.Dataset(X.iloc[train_idx], label=y[train_idx])
                lgb_model = lgb.train(
                    lgb_params_sets[lgb_idx], train_data_lgb,
                    num_boost_round=4000,
                    valid_sets=[lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)],
                    callbacks=[lgb.early_stopping(150 if lgb_idx == 0 else 130), lgb.log_evaluation(0)]
                )
                pred = lgb_model.predict(X_val)

            model_predictions[val_idx, model_idx] = pred

        if fold == 0:  # 测试集预测
            if model_idx < 3:  # CatBoost
                test_predictions[:, model_idx] = CatBoostRegressor(**cat_params_sets[model_idx]).fit(
                    X, y, verbose=0).predict(test_data)
            else:  # LightGBM
                lgb_idx = model_idx - 3
                train_data_lgb = lgb.Dataset(X, label=y)
                lgb_model = lgb.train(
                    lgb_params_sets[lgb_idx], train_data_lgb,
                    num_boost_round=4000,
                    callbacks=[lgb.early_stopping(150 if lgb_idx == 0 else 130), lgb.log_evaluation(0)]
                )
                test_predictions[:, model_idx] = lgb_model.predict(test_data)

# 策略1: 简单平均
ensemble_avg = model_predictions.mean(axis=1)
mae_avg = mean_absolute_error(y, ensemble_avg)
print(f"简单平均融合: MAE = {mae_avg:.2f}")

# 策略2: 最佳单模型
best_single_idx = np.argmin([np.mean([fold_maes_cat[i] if i < 3 else fold_maes_lgb[i-3]])
                                  for i in range(6)])
ensemble_best_single = model_predictions[:, best_single_idx]
mae_best_single = mean_absolute_error(y, ensemble_best_single)
print(f"最佳单模型: MAE = {mae_best_single:.2f}")

# 策略3: Top3平均
top3_indices = np.argsort([np.mean([fold_maes_cat[i] if i < 3 else fold_maes_lgb[i-3]])
                                  for i in range(6)])[:3]
ensemble_top3 = model_predictions[:, top3_indices].mean(axis=1)
mae_top3 = mean_absolute_error(y, ensemble_top3)
print(f"Top3平均融合: MAE = {mae_top3:.2f}")

# 策略4: 加权平均（基于CV）
cat_cv = np.array([np.mean(fold_maes_cat) for i in range(3)])
lgb_cv = np.array([np.mean(fold_maes_lgb) for i in range(3)])
all_cv = np.concatenate([cat_cv, lgb_cv])

inv_cv = 1 / all_cv
weights = inv_cv / inv_cv.sum()

ensemble_weighted = (model_predictions * weights).sum(axis=1)
mae_weighted = mean_absolute_error(y, ensemble_weighted)
print(f"加权平均融合: MAE = {mae_weighted:.2f}")
print(f"  CatBoost权重: {weights[:3]}, LightGBM权重: {weights[3:]}")

# 策略5: Stacking（Ridge）
stack_X = model_predictions
stacking_model = Ridge(alpha=0.1)
stacking_model.fit(stack_X, y)
ensemble_stacking = stacking_model.predict(stack_X)
mae_stacking = mean_absolute_error(y, ensemble_stacking)
print(f"Stacking(Ridge)融合: MAE = {mae_stacking:.2f}")

# 策略6: Stacking（ElasticNet）
stacking_model_en = ElasticNet(alpha=0.1, l1_ratio=0.5)
stacking_model_en.fit(stack_X, y)
ensemble_stacking_en = stacking_model_en.predict(stack_X)
mae_stacking_en = mean_absolute_error(y, ensemble_stacking_en)
print(f"Stacking(ElasticNet)融合: MAE = {mae_stacking_en:.2f}")

# ==================== 8. 选择最佳策略 ====================
print("\n【步骤7】选择最佳策略...")

strategies = {
    '简单平均': mae_avg,
    '最佳单模型': mae_best_single,
    'Top3平均': mae_top3,
    '加权平均': mae_weighted,
    'Stacking(Ridge)': mae_stacking,
    'Stacking(ElasticNet)': mae_stacking_en
}

best_strategy = min(strategies, key=strategies.get)
best_mae = strategies[best_strategy]

print(f"最佳策略: {best_strategy}")
print(f"CV MAE: {best_mae:.2f}")
print(f"目标: 450")
print(f"差距: {best_mae - 450:.2f}")

# ==================== 9. 生成测试集预测 ====================
print("\n【步骤8】生成测试集预测...")

if best_strategy == '简单平均':
    final_pred = ensemble_avg
elif best_strategy == '最佳单模型':
    final_pred = ensemble_best_single
elif best_strategy == 'Top3平均':
    top3_indices = np.argsort([np.mean([fold_maes_cat[i] if i < 3 else fold_maes_lgb[i-3]])
                                  for i in range(6)])[:3]
    final_pred = test_predictions[:, top3_indices].mean(axis=1)
elif best_strategy == '加权平均':
    final_pred = (test_predictions * weights).sum(axis=1)
elif best_strategy == 'Stacking(Ridge)':
    final_pred = stacking_model.predict(test_predictions)
else:  # Stacking(ElasticNet)
    final_pred = stacking_model_en.predict(test_predictions)

# 后处理1: 价格裁剪
final_pred = np.maximum(final_pred, 50)

# 后处理2: 基于分布调整（基于460成功经验）
mean_pred = final_pred.mean()
std_pred = final_pred.std()

if mean_pred > 6000:
    adjustment_factor = 0.975
    final_pred = final_pred * adjustment_factor
    print(f"应用均值调整: {adjustment_factor}")

# 后处理3: 异常值处理（3σ规则）
upper_bound = mean_pred + 2.5 * std_pred
final_pred = np.clip(final_pred, None, upper_bound)
print(f"应用3σ平滑")

# 保存结果
submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit.to_csv('breakthrough_450_submit.csv', index=False)

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
    print(f"改进幅度: {450 - best_mae:.2f} 点")
else:
    print(f"\n⚠️ 距离目标: {best_mae - 450:.2f}")
    print(f"达成度: {(450/best_mae)*100:.1f}%")
    if best_mae < 460:
        print(f"🏆 超过460成绩！")

print(f"\n结果已保存: breakthrough_450_submit.csv")
print(f"预测价格范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"预测价格均值: {final_pred.mean():.2f}")

print(f"\n{'='*70}")
print("【项目总结】")
print(f"{'='*70}")
print(f"训练模型数: 6")
print(f"CatBoost平均MAE: {np.mean(fold_maes_cat):.2f}")
print(f"LightGBM平均MAE: {np.mean(fold_maes_lgb):.2f}")

print(f"\n{'='*70}")
print("✅ 突破450版完成！")
print(f"{'='*70}")
