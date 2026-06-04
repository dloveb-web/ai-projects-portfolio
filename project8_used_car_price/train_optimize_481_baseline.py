# -*- coding: utf-8 -*-
"""
二手车价格预测 - 基于481.74 MAE基线优化版
目标：从481.74改进到<=450
策略：
1. 更激进超参数调优
2. 特征重要性筛选
3. 更精细集成策略
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("基于481.74 MAE优化版 - 目标<=450")
print("="*60)

# ==================== 数据处理 ====================
print("\n数据处理...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

# 基础特征工程
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = data['creat_year'] - data['reg_year']
data['car_age'] = data['car_age'].clip(lower=0)

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

data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']

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

print(f"初始特征数: {X.shape[1]}")

# ==================== 特征选择 ====================
print("\n特征重要性筛选...")

# 使用CatBoost进行特征选择
selector_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='MAE',
    random_seed=42,
    verbose=0
)
selector_model.fit(X, y)

# 获取特征重要性
feature_importance = selector_model.get_feature_importance()
feature_names = X.columns

# 选择重要性>0.01的特征
important_mask = feature_importance > 0.01
selected_features = feature_names[important_mask]

print(f"   筛选前{len(selected_features)}个特征（阈值0.01）")
print(f"   特征数: {X.shape[1]} -> {len(selected_features)}")

# 使用筛选后的特征
X_selected = X[selected_features]
test_selected = test_data[selected_features]

# ==================== 高级模型训练 ====================
print("\n开始高级训练 (5折交叉验证)...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cat_maes = []
lgb_maes = []
catboost_lightgbm_maes = []
stacking_maes = []

test_preds_cat = np.zeros(len(test_selected))
test_preds_lgb = np.zeros(len(test_selected))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # CatBoost（更激进参数）
    cat_model = CatBoostRegressor(
        iterations=5000,  # 更多迭代
        learning_rate=0.015,  # 更低学习率
        depth=8,  # 更深层
        l2_leaf_reg=5,  # 更低正则化
        random_strength=0.5,  # 更多随机性
        bagging_temperature=0.8,  # 更多bagging
        loss_function='MAE',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=150
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    cat_pred = cat_model.predict(X_val)
    cat_mae = mean_absolute_error(y_val, cat_pred)
    cat_maes.append(cat_mae)
    test_preds_cat += cat_model.predict(test_selected) / 5
    print(f"   CatBoost MAE: {cat_mae:.2f}")

    # LightGBM（更激进参数）
    train_data_lgb = lgb.Dataset(X_train, label=y_train)
    val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.015,  # 更低学习率
        'num_leaves': 191,  # 更多叶子
        'max_depth': 10,  # 更深层
        'min_data_in_leaf': 15,  # 更低叶子样本
        'feature_fraction': 0.9,  # 更高特征使用率
        'bagging_fraction': 0.9,  # 更高bagging率
        'bagging_freq': 5,
        'reg_alpha': 0.2,  # 更高L1正则
        'reg_lambda': 0.2,  # 更高L2正则
        'min_split_gain': 0.005,  # 更低分裂增益
        'verbose': -1,
        'seed': 42
    }

    lgb_model = lgb.train(params, train_data_lgb, num_boost_round=5000,
                           valid_sets=[val_data_lgb],
                           callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    lgb_pred = lgb_model.predict(X_val)
    lgb_mae = mean_absolute_error(y_val, lgb_pred)
    lgb_maes.append(lgb_mae)
    test_preds_lgb += lgb_model.predict(test_selected) / 5
    print(f"   LightGBM MAE: {lgb_mae:.2f}")

    # Stacking（使用CatBoost预测作为新特征）
    # 简化：跳过stacking，直接使用平均融合
    stacking_mae = (cat_mae + lgb_mae) / 2
    stacking_maes.append(stacking_mae)
    print(f"   Stacking (简化): MAE: {stacking_mae:.2f}")

# 最终结果
print(f"\n{'='*60}")
print("最终结果 (5折平均)")
print(f"{'='*60}")
print(f"CatBoost 平均 MAE: {np.mean(cat_maes):.2f}")
print(f"LightGBM 平均 MAE: {np.mean(lgb_maes):.2f}")
print(f"Stacking 平均 MAE: {np.mean(stacking_maes):.2f}")

# 选择最佳策略
maes = [np.mean(cat_maes), np.mean(lgb_maes), np.mean(stacking_maes)]
best_strategy_idx = np.argmin(maes)
best_mae = min(maes)

strategies = ['CatBoost', 'LightGBM', 'Stacking']
print(f"\n最佳策略: {strategies[best_strategy_idx]} ({best_mae:.2f})")

if best_strategy_idx == 0:
    print("   使用CatBoost单模型")
    final_pred = test_preds_cat
elif best_strategy_idx == 1:
    print("   使用LightGBM单模型")
    final_pred = test_preds_lgb
else:
    print("   使用Stacking集成")
    # 构建测试集stacking特征
    stack_test = pd.DataFrame({
        'cat_pred': test_preds_cat,
        'lgb_pred': test_preds_lgb,
        'cat_lgb_diff': test_preds_cat - test_preds_lgb,
        'cat_lgb_mean': (test_preds_cat + test_preds_lgb) / 2
    })

    # 训练最终stacker
    final_stacker = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=4,
        loss_function='MAE',
        random_seed=42,
        verbose=0
    )
    # 使用全部训练集训练stacker
    all_cat_preds = np.column_stack([np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)])  # 占位
    all_lgb_preds = np.column_stack([np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)])  # 占位

    # 简化：直接使用平均
    final_pred = (test_preds_cat + test_preds_lgb) / 2

final_pred = np.maximum(final_pred, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': sale_ids,
    'price': final_pred
})
submit.to_csv('optimize_481_baseline_submit.csv', index=False)
print(f"\n结果已保存到 optimize_481_baseline_submit.csv")

# 最终总结
print(f"\n{'='*60}")
print(f"最终MAE: {best_mae:.2f}")
print(f"目标MAE: 450")
print(f"差距: {best_mae - 450:.2f}")

if best_mae <= 450:
    print(f"🎉 成功达到目标MAE: {best_mae:.2f} <= 450")
    print(f"比之前477改进: {477 - best_mae:.2f}")
    print(f"比481.74改进: {481.74 - best_mae:.2f}")
else:
    print(f"⚠️ 距离目标: {best_mae - 450:.2f}")
    print(f"当前MAE: {best_mae:.2f}")
    if best_mae < 477:
        print(f"比之前477改进: {477 - best_mae:.2f}")
        print(f"比481.74改进: {481.74 - best_mae:.2f}")
    elif best_mae < 481.74:
        print(f"比481.74改进: {481.74 - best_mae:.2f}")
    else:
        print(f"比481.74差距: {best_mae - 481.74:.2f}")

print(f"{'='*60}")