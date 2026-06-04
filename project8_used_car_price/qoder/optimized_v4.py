# -*- coding: utf-8 -*-
"""
二手车价格预测 - 优化版V4（防过拟合版）
修复问题：
1. 分组特征在训练集上计算，避免泄露
2. 缺失值填充在训练集上计算
3. Target Encoding噪声随机化
4. 移除特征冗余
5. 增加正则化强度
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
print("二手车价格预测 - 优化版V4（防过拟合版）")
print("="*70)

# 数据加载
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

print(f"训练集: {len(train)}, 测试集: {len(test)}")

# ==================== 特征工程（严格分离train/test） ====================
print("\n【特征工程】处理中...")

def create_features(df, is_train=True, train_stats=None):
    """
    创建特征，训练集和测试集严格分离
    is_train: 是否为训练集
    train_stats: 训练集统计信息（用于测试集）
    """
    df = df.copy()

    # 时间特征（无需统计信息）
    df['reg_year'] = df['regDate'] // 10000
    df['creat_year'] = df['creatDate'] // 10000
    df['car_age'] = (df['creat_year'] - df['reg_year']).clip(lower=0)

    # 处理notRepairedDamage
    df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', '0.0')
    df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce')

    # v特征统计（行内计算，无泄露风险）
    v_cols = [f'v_{i}' for i in range(15)]
    df['v_mean'] = df[v_cols].mean(axis=1)
    df['v_std'] = df[v_cols].std(axis=1)
    df['v_max'] = df[v_cols].max(axis=1)
    df['v_min'] = df[v_cols].min(axis=1)

    # v特征交互
    df['v_0_v_3'] = df['v_0'] * df['v_3']
    df['v_0_v_2'] = df['v_0'] * df['v_2']
    df['v_0_v_12'] = df['v_0'] * df['v_12']
    df['v_3_v_12'] = df['v_3'] * df['v_12']

    # 业务特征
    df['power_km'] = df['power'] * df['kilometer']
    df['age_km'] = df['car_age'] * df['kilometer']
    df['power_age'] = df['power'] * df['car_age']
    df['usage_intensity'] = df['kilometer'] / (df['car_age'] + 1)

    if is_train:
        # 训练集：计算统计信息
        stats = {}

        # 缺失值填充统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats['fill_median'] = {col: df[col].median() for col in numeric_cols}

        # 分组统计（只在训练集计算）
        for col in ['brand', 'model', 'regionCode']:
            stats[f'{col}_count'] = df.groupby(col)['SaleID'].count().to_dict()

        return df, stats
    else:
        # 测试集：使用训练集统计信息
        # 填充缺失值
        for col, median_val in train_stats['fill_median'].items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)

        # 分组特征（使用训练集统计映射）
        for col in ['brand', 'model', 'regionCode']:
            count_map = train_stats[f'{col}_count']
            # 映射count，未知类别使用1（避免过拟合）
            df[f'{col}_count'] = df[col].map(count_map).fillna(1)

        return df

# 处理训练集
train_processed, train_stats = create_features(train, is_train=True)

# 处理测试集
test['price'] = -1  # 临时标记
test_processed = create_features(test, is_train=False, train_stats=train_stats)

# 填充训练集缺失值
for col, median_val in train_stats['fill_median'].items():
    if col in train_processed.columns:
        train_processed[col] = train_processed[col].fillna(median_val)

# 添加分组特征到训练集
for col in ['brand', 'model', 'regionCode']:
    count_map = train_stats[f'{col}_count']
    train_processed[f'{col}_count'] = train_processed[col].map(count_map)

# 类别列处理
categorical_cols = train_processed.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    # 合并train+test的类别值，确保编码一致
    all_values = pd.concat([train_processed[col], test_processed[col]]).astype(str).fillna('unknown')
    le.fit(all_values)
    train_processed[col] = le.transform(train_processed[col].astype(str).fillna('unknown'))
    test_processed[col] = le.transform(test_processed[col].astype(str).fillna('unknown'))

# 分离特征和标签
if 'price' in test_processed.columns:
    test_processed = test_processed.drop(columns=['price'])

X = train_processed.drop(columns=['price'])
y = train_processed['price'].values

# ==================== Target Encoding（CV内计算，严格防泄露） ====================
print("【Target Encoding】CV内严格防泄露计算...")

def target_encode_cv(X_train, y_train, X_val, X_test, col, smoothing=10, noise_std=3):
    """
    在CV循环内计算Target Encoding，严格防止泄露
    只使用训练折的数据计算TE
    """
    train_df = X_train.copy()
    train_df['_target_'] = y_train

    global_mean = y_train.mean()

    # 只使用训练折计算统计
    stats = train_df.groupby(col)['_target_'].agg(['mean', 'count'])
    smooth_mean = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)

    # 映射到验证集和测试集
    val_te = X_val[col].map(smooth_mean).fillna(global_mean)
    test_te = X_test[col].map(smooth_mean).fillna(global_mean)

    # 添加噪声（仅训练集，每次随机）
    train_te = train_df[col].map(smooth_mean).fillna(global_mean)
    train_te += np.random.randn(len(train_te)) * noise_std

    return train_te.values, val_te.values, test_te.values

# TE目标列
te_cols = ['brand', 'model', 'regionCode']

# 最终特征列（排除原始类别列）
feature_cols = [c for c in X.columns if c not in te_cols]

# ==================== 模型参数配置（增强正则化） ====================
print("\n【模型参数】增强正则化配置...")

# CatBoost参数（增强正则化）
cat_params = {
    'iterations': 5000,
    'learning_rate': 0.018,
    'depth': 6,              # 从7降低到6
    'l2_leaf_reg': 8,        # 从6增加到8
    'random_strength': 0.8,  # 从0.6增加到0.8
    'bagging_temperature': 0.8,
    'loss_function': 'MAE',
    'verbose': 0,
    'early_stopping_rounds': 150
}

# LightGBM参数（增强正则化）
lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.018,
    'num_leaves': 80,        # 从100降低到80
    'max_depth': 7,          # 从8降低到7
    'min_data_in_leaf': 25,  # 从18增加到25
    'feature_fraction': 0.8, # 从0.85降低到0.8
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.25,       # 从0.18增加到0.25
    'reg_lambda': 0.25,
    'verbose': -1
}

# XGBoost参数（增强正则化）
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.018,
    'max_depth': 6,          # 从7降低到6
    'min_child_weight': 8,   # 从5增加到8
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.25,
    'reg_lambda': 0.25,
    'tree_method': 'hist',
    'verbosity': 0
}

# ==================== 多Seed训练 ====================
print("\n" + "="*70)
print("【开始训练】多Seed集成 + 三模型融合 + 严格防泄露")
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

    cat_test_preds = np.zeros(len(test_processed))
    lgb_test_preds = np.zeros(len(test_processed))
    xgb_test_preds = np.zeros(len(test_processed))

    fold_maes_cat = []
    fold_maes_lgb = []
    fold_maes_xgb = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"--- Fold {fold+1}/5 ---", end=" ")

        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # ========== 在CV内计算Target Encoding（严格防泄露） ==========
        te_train = pd.DataFrame(index=X_tr.index)
        te_val = pd.DataFrame(index=X_va.index)
        te_test = pd.DataFrame(index=test_processed.index)

        for col in te_cols:
            tr_te, va_te, te_te = target_encode_cv(
                X_tr, y_tr, X_va, test_processed, col,
                smoothing=10, noise_std=3
            )
            te_train[f'{col}_te'] = tr_te
            te_val[f'{col}_te'] = va_te
            te_test[f'{col}_te'] = te_te

        # 合并特征：基础特征 + TE特征
        X_tr_final = pd.concat([X_tr[feature_cols].reset_index(drop=True), te_train.reset_index(drop=True)], axis=1)
        X_va_final = pd.concat([X_va[feature_cols].reset_index(drop=True), te_val.reset_index(drop=True)], axis=1)
        X_test_final = pd.concat([test_processed[feature_cols].reset_index(drop=True), te_test.reset_index(drop=True)], axis=1)

        # CatBoost
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(X_tr_final, y_tr, eval_set=(X_va_final, y_va), verbose=0)
        cat_pred_val = cat_model.predict(X_va_final)
        cat_pred_test = cat_model.predict(X_test_final)
        cat_mae = mean_absolute_error(y_va, cat_pred_val)
        fold_maes_cat.append(cat_mae)
        cat_test_preds += cat_pred_test / 5

        # LightGBM
        train_data_lgb = lgb.Dataset(X_tr_final, label=y_tr)
        val_data_lgb = lgb.Dataset(X_va_final, label=y_va, reference=train_data_lgb)
        lgb_model = lgb.train(
            lgb_params, train_data_lgb, num_boost_round=5000,
            valid_sets=[val_data_lgb],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
        )
        lgb_pred_val = lgb_model.predict(X_va_final)
        lgb_pred_test = lgb_model.predict(X_test_final)
        lgb_mae = mean_absolute_error(y_va, lgb_pred_val)
        fold_maes_lgb.append(lgb_mae)
        lgb_test_preds += lgb_pred_test / 5

        # XGBoost
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=5000, early_stopping_rounds=150)
        xgb_model.fit(X_tr_final, y_tr, eval_set=[(X_va_final, y_va)], verbose=0)
        xgb_pred_val = xgb_model.predict(X_va_final)
        xgb_pred_test = xgb_model.predict(X_test_final)
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
submit1.to_csv('optimized_v4_weighted.csv', index=False)
print(f"✅ optimized_v4_weighted.csv - 加权融合版本")

final_avg_pred = (final_cat_pred + final_lgb_pred + final_xgb_pred) / 3
submit2 = pd.DataFrame({'SaleID': sale_ids, 'price': final_avg_pred})
submit2.to_csv('optimized_v4_avg.csv', index=False)
print(f"✅ optimized_v4_avg.csv - 简单平均版本")

final_cl_pred = (final_cat_pred + final_lgb_pred) / 2
submit3 = pd.DataFrame({'SaleID': sale_ids, 'price': final_cl_pred})
submit3.to_csv('optimized_v4_cat_lgb.csv', index=False)
print(f"✅ optimized_v4_cat_lgb.csv - CatBoost+LightGBM版本")

print(f"\n【预测统计】")
print(f"加权融合 -> 范围: [{final_weighted_pred.min():.2f}, {final_weighted_pred.max():.2f}], 均值: {final_weighted_pred.mean():.2f}")
print(f"简单平均 -> 范围: [{final_avg_pred.min():.2f}, {final_avg_pred.max():.2f}], 均值: {final_avg_pred.mean():.2f}")

print(f"\n{'='*70}")
print("✅ 优化版V4训练完成！")
print(f"{'='*70}")
print("\n【防过拟合措施】")
print("1. 分组特征只在训练集计算，测试集映射")
print("2. 缺失值填充使用训练集统计")
print("3. Target Encoding在CV循环内计算")
print("4. 移除特征冗余（原始类别列）")
print("5. 增强正则化参数")
