# -*- coding: utf-8 -*-
"""
二手车价格预测 - 优化版V5（平衡版）
平衡策略：
1. 同时保留原始类别列和TE列（不丢失信息）
2. 适度正则化（不过强）
3. 分组特征严格防泄露
4. Target Encoding在CV外计算（与原代码一致，保持信息量）
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

SEEDS = [42, 123, 2024]

print("="*70)
print("二手车价格预测 - 优化版V5（平衡版）")
print("="*70)

# 数据加载
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

print(f"训练集: {len(train)}, 测试集: {len(test)}")

# ==================== 特征工程 ====================
print("\n【特征工程】处理中...")

# 时间特征
train['reg_year'] = train['regDate'] // 10000
train['creat_year'] = train['creatDate'] // 10000
train['car_age'] = (train['creat_year'] - train['reg_year']).clip(lower=0)

test['reg_year'] = test['regDate'] // 10000
test['creat_year'] = test['creatDate'] // 10000
test['car_age'] = (test['creat_year'] - test['reg_year']).clip(lower=0)

# 处理notRepairedDamage
for df in [train, test]:
    df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', '0.0')
    df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce')

# v特征统计
v_cols = [f'v_{i}' for i in range(15)]
for df in [train, test]:
    df['v_mean'] = df[v_cols].mean(axis=1)
    df['v_std'] = df[v_cols].std(axis=1)
    df['v_max'] = df[v_cols].max(axis=1)
    df['v_min'] = df[v_cols].min(axis=1)

# v特征交互
for df in [train, test]:
    df['v_0_v_3'] = df['v_0'] * df['v_3']
    df['v_0_v_2'] = df['v_0'] * df['v_2']
    df['v_0_v_12'] = df['v_0'] * df['v_12']
    df['v_3_v_12'] = df['v_3'] * df['v_12']

# 业务特征
for df in [train, test]:
    df['power_km'] = df['power'] * df['kilometer']
    df['age_km'] = df['car_age'] * df['kilometer']
    df['power_age'] = df['power'] * df['car_age']
    df['usage_intensity'] = df['kilometer'] / (df['car_age'] + 1)

# ==================== 分组特征（严格防泄露） ====================
print("【分组特征】只在训练集计算...")

# 只在训练集计算count，测试集映射
brand_count_map = train.groupby('brand')['SaleID'].count().to_dict()
model_count_map = train.groupby('model')['SaleID'].count().to_dict()
region_count_map = train.groupby('regionCode')['SaleID'].count().to_dict()

train['brand_count'] = train['brand'].map(brand_count_map)
train['model_count'] = train['model'].map(model_count_map)
train['regionCode_count'] = train['regionCode'].map(region_count_map)

# 测试集映射，未知类别使用1
test['brand_count'] = test['brand'].map(brand_count_map).fillna(1)
test['model_count'] = test['model'].map(model_count_map).fillna(1)
test['regionCode_count'] = test['regionCode'].map(region_count_map).fillna(1)

# ==================== 缺失值填充（只用训练集统计） ====================
print("【缺失值填充】使用训练集统计...")

numeric_cols = train.select_dtypes(include=[np.number]).columns
fill_values = {}
for col in numeric_cols:
    fill_values[col] = train[col].median()
    train[col] = train[col].fillna(fill_values[col])
    if col in test.columns:
        test[col] = test[col].fillna(fill_values[col])

# 类别列处理
categorical_cols = train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    train_mode = train[col].mode()[0] if not train[col].mode().empty else 'unknown'
    train[col] = train[col].fillna(train_mode)
    test[col] = test[col].fillna(train_mode)
    
    # Label Encoding（只用训练集fit）
    le = LabelEncoder()
    le.fit(train[col].astype(str))
    train[col] = le.transform(train[col].astype(str))
    # 测试集未知类别映射为-1再映射为训练集最常见的值
    test[col] = test[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

# ==================== Target Encoding（CV方式，与原代码一致） ====================
print("【Target Encoding】CV方式计算...")

def target_encode_cv(train_df, test_df, col, target='price', n_folds=5, seed=42):
    """
    CV方式计算Target Encoding
    训练集：每个fold使用其他folds的均值
    测试集：使用全量训练集均值
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[f'{col}_te'] = 0.0
    test_df[f'{col}_te'] = 0.0
    
    global_mean = train_df[target].mean()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # 训练集：CV方式计算（每个fold使用其他folds的均值）
    for train_idx, val_idx in kf.split(train_df):
        target_mean = train_df.iloc[train_idx].groupby(col)[target].mean()
        train_df.loc[train_df.index[val_idx], f'{col}_te'] = train_df.iloc[val_idx][col].map(target_mean).fillna(global_mean)
    
    # 测试集：使用全量训练集均值
    target_mean = train_df.groupby(col)[target].mean()
    test_df[f'{col}_te'] = test_df[col].map(target_mean).fillna(global_mean)
    
    return train_df[f'{col}_te'].values, test_df[f'{col}_te'].values

# 对关键类别特征做Target Encoding
te_cols = ['brand', 'model', 'regionCode']
for col in te_cols:
    train_te, test_te = target_encode_cv(train, test, col, seed=SEEDS[0])
    train[f'{col}_te'] = train_te
    test[f'{col}_te'] = test_te

# ==================== 准备数据 ====================
# 删除无用列
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType', 'price']
X = train.drop(columns=[c for c in drop_cols if c in train.columns])
y = train['price'].values
test_features = test.drop(columns=[c for c in drop_cols if c in test.columns], errors='ignore')

print(f"特征数: {X.shape[1]}")
print(f"特征列表: {list(X.columns)}")

# ==================== 模型参数配置（适度正则化） ====================
print("\n【模型参数】适度正则化...")

# CatBoost参数
cat_params = {
    'iterations': 4500,
    'learning_rate': 0.018,
    'depth': 7,
    'l2_leaf_reg': 6,
    'random_strength': 0.6,
    'bagging_temperature': 0.7,
    'loss_function': 'MAE',
    'verbose': 0,
    'early_stopping_rounds': 140
}

# LightGBM参数
lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.018,
    'num_leaves': 110,
    'max_depth': 9,
    'min_data_in_leaf': 18,
    'feature_fraction': 0.87,
    'bagging_fraction': 0.87,
    'bagging_freq': 5,
    'reg_alpha': 0.18,
    'reg_lambda': 0.18,
    'verbose': -1
}

# XGBoost参数
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.018,
    'max_depth': 7,
    'min_child_weight': 5,
    'subsample': 0.87,
    'colsample_bytree': 0.87,
    'reg_alpha': 0.18,
    'reg_lambda': 0.18,
    'tree_method': 'hist',
    'verbosity': 0
}

# ==================== 多Seed训练 ====================
print("\n" + "="*70)
print("【开始训练】多Seed集成 + 三模型融合")
print("="*70)

all_seed_preds = {'cat': [], 'lgb': [], 'xgb': [], 'weighted': []}
all_cv_scores = {'cat': [], 'lgb': [], 'xgb': []}

for seed_idx, SEED in enumerate(SEEDS):
    print(f"\n{'='*70}")
    print(f"Seed {seed_idx+1}/{len(SEEDS)}: {SEED}")
    print(f"{'='*70}")

    np.random.seed(SEED)
    cat_params['random_seed'] = SEED
    lgb_params['seed'] = SEED
    xgb_params['seed'] = SEED

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    cat_test_preds = np.zeros(len(test_features))
    lgb_test_preds = np.zeros(len(test_features))
    xgb_test_preds = np.zeros(len(test_features))

    fold_maes_cat = []
    fold_maes_lgb = []
    fold_maes_xgb = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"--- Fold {fold+1}/5 ---", end=" ")

        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # CatBoost
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
        cat_pred_val = cat_model.predict(X_va)
        cat_pred_test = cat_model.predict(test_features)
        cat_mae = mean_absolute_error(y_va, cat_pred_val)
        fold_maes_cat.append(cat_mae)
        cat_test_preds += cat_pred_test / 5

        # LightGBM
        train_data_lgb = lgb.Dataset(X_tr, label=y_tr)
        val_data_lgb = lgb.Dataset(X_va, label=y_va, reference=train_data_lgb)
        lgb_model = lgb.train(
            lgb_params, train_data_lgb, num_boost_round=4500,
            valid_sets=[val_data_lgb],
            callbacks=[lgb.early_stopping(140), lgb.log_evaluation(0)]
        )
        lgb_pred_val = lgb_model.predict(X_va)
        lgb_pred_test = lgb_model.predict(test_features)
        lgb_mae = mean_absolute_error(y_va, lgb_pred_val)
        fold_maes_lgb.append(lgb_mae)
        lgb_test_preds += lgb_pred_test / 5

        # XGBoost
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=4500, early_stopping_rounds=140)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)
        xgb_pred_val = xgb_model.predict(X_va)
        xgb_pred_test = xgb_model.predict(test_features)
        xgb_mae = mean_absolute_error(y_va, xgb_pred_val)
        fold_maes_xgb.append(xgb_mae)
        xgb_test_preds += xgb_pred_test / 5

        # 融合MAE
        avg_pred = (cat_pred_val + lgb_pred_val + xgb_pred_val) / 3
        avg_mae = mean_absolute_error(y_va, avg_pred)

        print(f"Cat: {cat_mae:.2f} | LGB: {lgb_mae:.2f} | XGB: {xgb_mae:.2f} | 融合: {avg_mae:.2f}")

    # 该seed的CV结果
    cat_cv = np.mean(fold_maes_cat)
    lgb_cv = np.mean(fold_maes_lgb)
    xgb_cv = np.mean(fold_maes_xgb)

    all_cv_scores['cat'].append(cat_cv)
    all_cv_scores['lgb'].append(lgb_cv)
    all_cv_scores['xgb'].append(xgb_cv)

    print(f"\n[Seed {SEED}] CV MAE -> Cat: {cat_cv:.2f} | LGB: {lgb_cv:.2f} | XGB: {xgb_cv:.2f}")

    # 加权融合
    weights = np.array([1/cat_cv, 1/lgb_cv, 1/xgb_cv])
    weights = weights / weights.sum()

    weighted_pred = (weights[0] * cat_test_preds +
                     weights[1] * lgb_test_preds +
                     weights[2] * xgb_test_preds)

    all_seed_preds['cat'].append(cat_test_preds)
    all_seed_preds['lgb'].append(lgb_test_preds)
    all_seed_preds['xgb'].append(xgb_test_preds)
    all_seed_preds['weighted'].append(weighted_pred)

# ==================== 最终融合 ====================
print("\n" + "="*70)
print("【最终融合】多Seed平均")
print("="*70)

print("\n【CV MAE 汇总】")
print(f"CatBoost : {np.mean(all_cv_scores['cat']):.2f} ± {np.std(all_cv_scores['cat']):.2f}")
print(f"LightGBM : {np.mean(all_cv_scores['lgb']):.2f} ± {np.std(all_cv_scores['lgb']):.2f}")
print(f"XGBoost  : {np.mean(all_cv_scores['xgb']):.2f} ± {np.std(all_cv_scores['xgb']):.2f}")

# 多Seed平均
final_cat_pred = np.maximum(np.mean(all_seed_preds['cat'], axis=0), 50)
final_lgb_pred = np.maximum(np.mean(all_seed_preds['lgb'], axis=0), 50)
final_xgb_pred = np.maximum(np.mean(all_seed_preds['xgb'], axis=0), 50)
final_weighted_pred = np.maximum(np.mean(all_seed_preds['weighted'], axis=0), 50)

# 保存预测结果
print("\n【保存预测结果】")

submit1 = pd.DataFrame({'SaleID': sale_ids, 'price': final_weighted_pred})
submit1.to_csv('optimized_v5_weighted.csv', index=False)
print(f"✅ optimized_v5_weighted.csv - 加权融合版本")

final_avg_pred = (final_cat_pred + final_lgb_pred + final_xgb_pred) / 3
submit2 = pd.DataFrame({'SaleID': sale_ids, 'price': final_avg_pred})
submit2.to_csv('optimized_v5_avg.csv', index=False)
print(f"✅ optimized_v5_avg.csv - 简单平均版本")

final_cl_pred = (final_cat_pred + final_lgb_pred) / 2
submit3 = pd.DataFrame({'SaleID': sale_ids, 'price': final_cl_pred})
submit3.to_csv('optimized_v5_cat_lgb.csv', index=False)
print(f"✅ optimized_v5_cat_lgb.csv - CatBoost+LightGBM版本")

print(f"\n【预测统计】")
print(f"加权融合 -> 范围: [{final_weighted_pred.min():.2f}, {final_weighted_pred.max():.2f}], 均值: {final_weighted_pred.mean():.2f}")
print(f"简单平均 -> 范围: [{final_avg_pred.min():.2f}, {final_avg_pred.max():.2f}], 均值: {final_avg_pred.mean():.2f}")

print(f"\n{'='*70}")
print("✅ 优化版V5训练完成！")
print(f"{'='*70}")
print("\n【平衡策略】")
print("1. 同时保留原始类别列和TE列（不丢失信息）")
print("2. 参数接近原代码最佳配置")
print("3. 分组特征严格防泄露")
print("4. 缺失值填充使用训练集统计")
print("5. 多Seed集成降低方差")
