# -*- coding: utf-8 -*-
"""
二手车价格预测 - 极限优化版
基于成功480.64策略 + 极限优化
目标：MAE <= 450
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("极限优化版 - 基于480.64成功策略")
print("目标: MAE <= 450")
print("="*60)

# ==================== 数据处理 ====================
print("\n数据处理...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

# 与480.64成功版本完全一致的特征工程
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)

data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', np.nan)
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
data['v_1_v_5'] = data['v_1'] * data['v_5']
data['v_2_v_4'] = data['v_2'] * data['v_4']

data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']
data['power_age'] = data['power'] * data['car_age']

# 目标编码
for col in ['brand', 'model', 'regionCode']:
    data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')

drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
test_data = test_data.drop(columns=['price'])

X = train_data.drop(columns=['price'])
y = train_data['price']

X = X.fillna(X.median())
test_data = test_data.fillna(X.median())

common_cols = list(set(X.columns) & set(test_data.columns))
X = X[common_cols]
test_data = test_data[common_cols]

print(f"特征数: {X.shape[1]}")

# ==================== 极限优化训练 ====================
print("\n极限优化训练 (5折交叉验证)...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cat_maes = []
lgb_maes = []
ensemble_maes = []

test_preds_cat = np.zeros(len(test_data))
test_preds_lgb = np.zeros(len(test_data))

# 极限参数空间（在480.64基础上更激进）
extreme_params_catboost = [
    # 更保守
    {'iterations': 3800, 'learning_rate': 0.018, 'depth': 6, 'l2_leaf_reg': 8},
    # 标准优化
    {'iterations': 4200, 'learning_rate': 0.022, 'depth': 7, 'l2_leaf_reg': 7},
    # 更激进
    {'iterations': 4500, 'learning_rate': 0.025, 'depth': 8, 'l2_leaf_reg': 6},
    # 极限
    {'iterations': 5000, 'learning_rate': 0.028, 'depth': 9, 'l2_leaf_reg': 5}
]

extreme_params_lightgbm = [
    # 更保守
    {'n_estimators': 3800, 'learning_rate': 0.018, 'num_leaves': 105, 'max_depth': 7},
    # 标准优化
    {'n_estimators': 4200, 'learning_rate': 0.022, 'num_leaves': 115, 'max_depth': 8},
    # 更激进
    {'n_estimators': 4500, 'learning_rate': 0.025, 'num_leaves': 127, 'max_depth': 9},
    # 极限
    {'n_estimators': 5000, 'learning_rate': 0.028, 'num_leaves': 143, 'max_depth': 10}
]

best_overall_mae = float('inf')
best_overall_params = None
best_overall_models = None

print("参数搜索空间: 4组 × 2模型 = 8种配置")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    fold_cat_preds = []
    fold_lgb_preds = []
    fold_maes = []

    # 训练所有参数组合
    for i, params_cat in enumerate(extreme_params_catboost):
        print(f"   CatBoost 配置{i+1}/{len(extreme_params_catboost)}")

        cat_model = CatBoostRegressor(
            **params_cat,
            loss_function='MAE',
            random_seed=42,
            verbose=0,
            early_stopping_rounds=100
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
        cat_pred = cat_model.predict(X_val)
        cat_mae = mean_absolute_error(y_val, cat_pred)
        fold_cat_preds.append(cat_pred)
        fold_maes.append(('cat', i, cat_mae))
        print(f"     MAE: {cat_mae:.2f}")

    for i, params_lgb in enumerate(extreme_params_lightgbm):
        print(f"   LightGBM 配置{i+1}/{len(extreme_params_lightgbm)}")

        train_data_lgb = lgb.Dataset(X_train, label=y_train)
        val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

        lgb_model = lgb.train(
            {**params_lgb, 'verbose': -1, 'seed': 42},
            train_data_lgb, num_boost_round=params_lgb['n_estimators'],
            valid_sets=[val_data_lgb],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        lgb_pred = lgb_model.predict(X_val)
        lgb_mae = mean_absolute_error(y_val, lgb_pred)
        fold_lgb_preds.append(lgb_pred)
        fold_maes.append(('lgb', i, lgb_mae))
        print(f"     MAE: {lgb_mae:.2f}")

    # 找到fold最佳配置
    fold_df = pd.DataFrame(fold_maes, columns=['model', 'config_idx', 'mae'])
    best_in_fold = fold_df.loc[fold_df['mae'].idxmin()]

    print(f"   Fold最佳: {best_in_fold['model']} 配置{best_in_fold['config_idx']}, MAE={best_in_fold['mae']:.2f}")

    # 使用fold最佳配置预测测试集
    if best_in_fold['model'] == 'cat':
        best_cat_idx = best_in_fold['config_idx']
        best_cat_model = CatBoostRegressor(**extreme_params_catboost[best_cat_idx],
                                           loss_function='MAE',
                                           random_seed=42, verbose=0)
        best_cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
        test_preds_cat += best_cat_model.predict(test_data) / 5
    else:
        best_lgb_idx = best_in_fold['config_idx']
        best_lgb_model = lgb.train(
            {**extreme_params_lightgbm[best_lgb_idx], 'verbose': -1, 'seed': 42},
            lgb.Dataset(X_train, label=y_train),
            num_boost_round=extreme_params_lightgbm[best_lgb_idx]['n_estimators'],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        test_preds_lgb += best_lgb_model.predict(test_data) / 5

    # 使用最佳预测进行ensemble
    best_cat_pred = fold_cat_preds[best_cat_idx] if best_in_fold['model'] == 'cat' else fold_lgb_preds[best_lgb_idx]
    best_lgb_pred = fold_lgb_preds[best_lgb_idx] if best_in_fold['model'] == 'lgb' else fold_cat_preds[best_lgb_idx]

    ensemble_pred = (best_cat_pred + best_lgb_pred) / 2
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    ensemble_maes.append(ensemble_mae)
    cat_maes.append(fold_maes[fold_maes['model'] == 'cat']['mae'].iloc[0])
    lgb_maes.append(fold_maes[fold_maes['model'] == 'lgb']['mae'].iloc[0])

    print(f"   Ensemble MAE: {ensemble_mae:.2f}")

    # 更新全局最佳
    if ensemble_mae < best_overall_mae:
        best_overall_mae = ensemble_mae
        best_overall_params = best_in_fold

# 最终结果
print(f"\n{'='*60}")
print("最终结果 (5折平均)")
print(f"{'='*60}")
print(f"CatBoost 平均 MAE: {np.mean(cat_maes):.2f}")
print(f"LightGBM 平均 MAE: {np.mean(lgb_maes):.2f}")
print(f"Ensemble 平均 MAE: {np.mean(ensemble_maes):.2f}")

print(f"\n全局最佳: {best_overall_params['model']} 配置{best_overall_params['config_idx']}, MAE={best_overall_mae:.2f}")

# 最终预测（使用全局最佳策略）
final_mae = np.mean(ensemble_maes)
if best_overall_params['model'] == 'cat':
    # 使用CatBoost主导
    final_pred = 0.6 * test_preds_cat + 0.4 * test_preds_lgb
else:
    # 使用LightGBM主导
    final_pred = 0.4 * test_preds_cat + 0.6 * test_preds_lgb

final_pred = np.maximum(final_pred, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': sale_ids,
    'price': final_pred
})
submit.to_csv('extreme_optimization_submit.csv', index=False)
print(f"\n结果已保存到 extreme_optimization_submit.csv")

# 最终总结
print(f"\n{'='*60}")
print(f"最终MAE: {final_mae:.2f}")
print(f"目标MAE: 450")
print(f"差距: {final_mae - 450:.2f}")

if final_mae <= 450:
    print(f"🎉 成功达到目标MAE: {final_mae:.2f} <= 450")
    print(f"比之前480.64改进: {480.64 - final_mae:.2f}")
    print(f"比之前477改进: {477 - final_mae:.2f}")
else:
    print(f"⚠️ 距离目标: {final_mae - 450:.2f}")
    print(f"当前MAE: {final_mae:.2f}")
    if final_mae < 480.64:
        print(f"比480.64改进: {480.64 - final_mae:.2f}")
        print(f"比477改进: {477 - final_mae:.2f}")
    else:
        print(f"比480.64差距: {final_mae - 480.64:.2f}")

print(f"\n关键特性:")
print(f"✅ 4组参数空间 × 2模型 = 8种配置")
print(f"✅ 每折自动选择最佳配置")
print(f"✅ 基于已知成功480.64策略")
print(f"✅ 极限优化探索")

print(f"{'='*60}")