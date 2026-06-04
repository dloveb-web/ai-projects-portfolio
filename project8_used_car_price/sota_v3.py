# -*- coding: utf-8 -*-
"""
二手车价格预测 - SOTA V3 优化版
策略：CatBoost cat_features + 新特征 + 反泄露 + 品牌价格统计 + 频率编码
目标: Test MAE < 450
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ========== 配置 ==========
# Phase 1（快速验证）：用缩减参数
PHASE = 2  # 2=完整训练
USE_MULTI_SEED = (PHASE == 2)
SEEDS = [42, 123, 2024] if USE_MULTI_SEED else [42]

# Phase 2 参数（与基线一致）
CAT_ITER = 4500
CAT_DEPTH = 8
LGB_LEAVES = 120
LGB_ROUND = 4500
EARLY_STOP = 140

print(f"Phase {PHASE}: Cat iter={CAT_ITER}, depth={CAT_DEPTH}, LGB leaves={LGB_LEAVES}, seeds={SEEDS}")

# ========== 数据加载 ==========
BASE_PATH = ''
train = pd.read_csv(BASE_PATH + 'used_car_train_20200313.csv', sep=' ')
test = pd.read_csv(BASE_PATH + 'used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

print(f"训练集: {len(train)}, 测试集: {len(test)}")

y_original = train['price'].values.copy()  # 保存原始price（绝不截断）

# ========== 数据清洗（分别处理，防泄露）==========

# 1. Power=0 修复（12829条，8.5%）：用品牌中位数填充，train-only统计
brand_median_power = train[train['power'] > 0].groupby('brand')['power'].median()
global_median_power = train[train['power'] > 0]['power'].median()

for df in [train, test]:
    mask = df['power'] == 0
    df.loc[mask, 'power'] = df.loc[mask, 'brand'].map(brand_median_power).fillna(global_median_power).astype(df['power'].dtype)

# 2. Power 异常值截断 + 标记
POWER_UPPER = 600
for df in [train, test]:
    df['power_outlier'] = (df['power'] > POWER_UPPER).astype(int)
    df['power'] = df['power'].clip(upper=POWER_UPPER)

# 3. notRepairedDamage：'-' 转为 NaN 再填充
for df in [train, test]:
    df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
    df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce')

# ========== 特征工程（分别处理每个df）==========

# CatBoost cat_features：只包含低基数分类列（brand=40, bodyType=8, fuelType=7, gearbox=2, notRepairedDamage, age_segment=5）
# model(248)和brand_model(333)基数太高，作为数值处理
CAT_COLS = ['brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'age_segment']
# regionCode(7905唯一值)不做分类特征


def feature_engineer(df, is_train=False, train_ref=None):
    """特征工程，每个df独立处理"""
    # 时间特征
    df['reg_year'] = df['regDate'] // 10000
    df['creat_year'] = df['creatDate'] // 10000
    df['car_age'] = (df['creat_year'] - df['reg_year']).clip(lower=0)

    # 车龄分段
    bins = [0, 2, 5, 10, 15, 999]
    labels = [0, 1, 2, 3, 4]
    df['age_segment'] = pd.cut(df['car_age'], bins=bins, labels=labels, right=True).astype('float')

    # 新车标记
    df['is_new_car'] = (df['car_age'] <= 1).astype(int)

    # v特征统计
    v_cols = [f'v_{i}' for i in range(15)]
    df['v_mean'] = df[v_cols].mean(axis=1)
    df['v_std'] = df[v_cols].std(axis=1)
    df['v_max'] = df[v_cols].max(axis=1)
    df['v_min'] = df[v_cols].min(axis=1)

    # v交互特征（保留基线4个 + 新增2个）
    df['v_0_v_3'] = df['v_0'] * df['v_3']
    df['v_0_v_2'] = df['v_0'] * df['v_2']
    df['v_0_v_12'] = df['v_0'] * df['v_12']
    df['v_3_v_12'] = df['v_3'] * df['v_12']
    df['v_0_v_8'] = df['v_0'] * df['v_8']
    df['v_3_v_8'] = df['v_3'] * df['v_8']

    # 业务特征
    df['power_km'] = df['power'] * df['kilometer']
    df['age_km'] = df['car_age'] * df['kilometer']
    df['power_age'] = df['power'] * df['car_age']
    df['usage_intensity'] = df['kilometer'] / (df['car_age'] + 1)

    # 品牌×车型组合
    if is_train:
        model_fill = df['model'].fillna(-1).astype(int)
    else:
        model_fill = df['model'].fillna(-1).astype(int)
    df['brand_model'] = df['brand'].astype(str) + '_' + model_fill.astype(str)

    return df


train = feature_engineer(train, is_train=True)
test = feature_engineer(test, is_train=False)

# ========== 频率编码（train-only统计，安全）==========

def add_freq_encoding(train_df, test_df, cols):
    """频率编码，基于train的频率映射到test"""
    for col in cols:
        freq_map = train_df[col].value_counts() / len(train_df)
        train_df[f'{col}_freq'] = train_df[col].map(freq_map)
        test_df[f'{col}_freq'] = test_df[col].map(freq_map).fillna(0)

add_freq_encoding(train, test, ['brand', 'model', 'brand_model'])

# ========== 分组计数（count 特征，train-only统计）==========

def add_count_features(train_df, test_df, cols):
    """分组计数，基于train的统计映射到test"""
    for col in cols:
        count_map = train_df.groupby(col)['SaleID'].count().to_dict()
        train_df[f'{col}_count'] = train_df[col].map(count_map)
        test_df[f'{col}_count'] = test_df[col].map(count_map).fillna(1)

add_count_features(train, test, ['brand', 'model'])

# ========== 缺失值处理（train-only统计，防泄露）==========

# 数值列
numeric_cols_candidates = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5',
                           'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14',
                           'bodyType', 'fuelType', 'gearbox', 'model']
for col in numeric_cols_candidates:
    if col in train.columns and train[col].isnull().sum() > 0:
        med_val = train[col].median()
        train[col] = train[col].fillna(med_val)
        if col in test.columns:
            test[col] = test[col].fillna(med_val)

# 分类列填 'unknown'
for col in ['bodyType', 'fuelType', 'gearbox', 'model', 'notRepairedDamage']:
    train[col] = train[col].fillna('unknown').astype(str)
    test[col] = test[col].fillna('unknown').astype(str)

# age_segment 缺失处理
train['age_segment'] = train['age_segment'].fillna(-1)
test['age_segment'] = test['age_segment'].fillna(-1)

# brand_model 缺失
train['brand_model'] = train['brand_model'].fillna('unknown')
test['brand_model'] = test['brand_model'].fillna('unknown')

# ========== Target 编码（CV-fold，train-only）==========

def target_encode_cv(train_df, test_df, col, y_values, n_folds=5, seed=42):
    """CV-based target encoding，完全防泄露"""
    train_result = np.zeros(len(train_df), dtype=np.float64)
    test_result = np.zeros(len(test_df), dtype=np.float64)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in kf.split(train_df):
        fold_train_y = y_values[train_idx]
        fold_train_df = train_df.iloc[train_idx]
        fold_val_df = train_df.iloc[val_idx]

        target_mean = pd.Series(fold_train_y, index=fold_train_df.index).groupby(
            fold_train_df[col].values
        ).mean()

        val_map = fold_val_df[col].map(target_mean)
        train_result[val_idx] = val_map.fillna(y_values[train_idx].mean()).values

    # test：全量train mean
    global_target_mean = pd.Series(y_values, index=train_df.index).groupby(
        train_df[col].values
    ).mean()
    test_result = test_df[col].map(global_target_mean).fillna(y_values.mean()).values

    return train_result, test_result

# RegionCode做目标编码（高基数特征，不做分类处理）
# brand和model的目标编码
y_values = y_original

# 先临时创建编码需要的目标列
for col in ['brand', 'model', 'regionCode']:
    train_te, test_te = target_encode_cv(train, test, col, y_values)
    train[f'{col}_te'] = train_te
    test[f'{col}_te'] = test_te

# ========== 品牌价格统计特征（CV-fold防泄露）==========

def brand_price_stats_cv(train_df, test_df, col, y_values, n_folds=5, seed=42):
    """品牌价格统计特征：mean, std，CV-fold防泄露"""
    train_mean = np.zeros(len(train_df), dtype=np.float64)
    train_std = np.zeros(len(train_df), dtype=np.float64)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in kf.split(train_df):
        fold_train_y = y_values[train_idx]
        fold_train_df = train_df.iloc[train_idx]
        fold_val_df = train_df.iloc[val_idx]

        group_stats = pd.DataFrame({
            'y': fold_train_y,
            'group': fold_train_df[col].values
        }).groupby('group')['y'].agg(['mean', 'std'])

        train_mean[val_idx] = fold_val_df[col].map(group_stats['mean']).fillna(y_values.mean()).values
        train_std[val_idx] = fold_val_df[col].map(group_stats['std']).fillna(y_values.std()).values

    # test：全量train stats
    global_stats = pd.DataFrame({
        'y': y_values,
        'group': train_df[col].values
    }).groupby('group')['y'].agg(['mean', 'std'])

    test_mean = test_df[col].map(global_stats['mean']).fillna(y_values.mean()).values
    test_std = test_df[col].map(global_stats['std']).fillna(y_values.std()).values

    return train_mean, train_std, test_mean, test_std

train_brand_mean, train_brand_std, test_brand_mean, test_brand_std = \
    brand_price_stats_cv(train, test, 'brand', y_values)
train['brand_price_mean'] = train_brand_mean
train['brand_price_std'] = train_brand_std
test['brand_price_mean'] = test_brand_mean
test['brand_price_std'] = test_brand_std

# ========== 删除无用列 ==========
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType', 'price']
for col in drop_cols:
    if col in train.columns:
        train = train.drop(columns=[col])
    if col in test.columns:
        test = test.drop(columns=[col])

# brand_model保留字符形式给CatBoost，同时创建整数版本给LightGBM
# 注意：我们保留原始字符串给CatBoost，LightGBM需要额外编码

print(f"特征工程完成，特征数: {train.shape[1]}")

# ========== 对所有分类列做Label Encoding（fit train-only，防泄露）==========
cat_columns = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox',
               'notRepairedDamage', 'age_segment', 'brand_model', 'regionCode']

label_encoders = {}
for col in cat_columns:
    if col in train.columns:
        le = LabelEncoder()
        le.fit(train[col].astype(str))
        train[col] = le.transform(train[col].astype(str))
        test[col] = test[col].astype(str).apply(
            lambda x, c=col: le.transform([x])[0] if x in le.classes_ else 0
        )
        label_encoders[col] = le

X = train.copy()
X_test = test.copy()
print(f"训练特征矩阵: {X.shape}, 测试特征矩阵: {X_test.shape}")

# ========== 模型训练 ==========

def train_single_seed(seed, X_df, X_test_df, y):
    """单seed训练：5-fold CV with CatBoost + LightGBM"""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    cat_test_preds = np.zeros(len(X_test_df))
    lgb_test_preds = np.zeros(len(X_test_df))

    fold_maes_cat = []
    fold_maes_lgb = []
    fold_maes_avg = []

    cat_params = {
        'iterations': CAT_ITER, 'learning_rate': 0.020, 'depth': CAT_DEPTH,
        'l2_leaf_reg': 6, 'random_strength': 0.7, 'bagging_temperature': 0.6,
        'loss_function': 'MAE', 'random_seed': seed, 'verbose': 0,
        'early_stopping_rounds': EARLY_STOP
    }

    lgb_params = {
        'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
        'learning_rate': 0.020, 'num_leaves': LGB_LEAVES, 'max_depth': 9,
        'min_data_in_leaf': 17, 'feature_fraction': 0.88, 'bagging_fraction': 0.88,
        'bagging_freq': 5, 'reg_alpha': 0.19, 'reg_lambda': 0.19,
        'verbose': -1, 'seed': seed
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_df)):
        print(f"\n  Fold {fold+1}/5")

        X_train_fold, X_val_fold = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # CatBoost（全部数值特征）
        print("  训练CatBoost...")
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(X_train_fold, y_train, eval_set=(X_val_fold, y_val), verbose=0)
        cat_pred_val = cat_model.predict(X_val_fold)
        cat_pred_test = cat_model.predict(X_test_df)

        cat_mae = mean_absolute_error(y_val, cat_pred_val)
        fold_maes_cat.append(cat_mae)
        cat_test_preds += cat_pred_test / 5
        print(f"  CatBoost MAE: {cat_mae:.2f}")

        # LightGBM（全部数值特征）
        print("  训练LightGBM...")
        train_ds = lgb.Dataset(X_train_fold, label=y_train)
        val_ds = lgb.Dataset(X_val_fold, label=y_val, reference=train_ds)
        lgb_model = lgb.train(
            lgb_params, train_ds, num_boost_round=LGB_ROUND,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(0)]
        )
        lgb_pred_val = lgb_model.predict(X_val_fold)
        lgb_pred_test = lgb_model.predict(X_test_df)

        lgb_mae = mean_absolute_error(y_val, lgb_pred_val)
        fold_maes_lgb.append(lgb_mae)
        lgb_test_preds += lgb_pred_test / 5
        print(f"  LightGBM MAE: {lgb_mae:.2f}")

        # 简单平均融合
        avg_pred = (cat_pred_val + lgb_pred_val) / 2
        avg_mae = mean_absolute_error(y_val, avg_pred)
        fold_maes_avg.append(avg_mae)
        print(f"  平均融合 MAE: {avg_mae:.2f}")

    return fold_maes_cat, fold_maes_lgb, fold_maes_avg, cat_test_preds, lgb_test_preds


# ========== 多 Seed 集成 ==========

seed_maes_cat = []
seed_maes_lgb = []
seed_maes_avg = []
seed_cat_test = []
seed_lgb_test = []

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*70}")
    print(f"Seed {seed_idx+1}/{len(SEEDS)}: {seed}")
    print(f"{'='*70}")

    np.random.seed(seed)

    cat_maes, lgb_maes, avg_maes, cat_test, lgb_test = \
        train_single_seed(seed, X, X_test, y_original)

    seed_maes_cat.append(np.mean(cat_maes))
    seed_maes_lgb.append(np.mean(lgb_maes))
    seed_maes_avg.append(np.mean(avg_maes))
    seed_cat_test.append(cat_test)
    seed_lgb_test.append(lgb_test)

    print(f"\nSeed {seed} 结果:")
    print(f"  CatBoost: {np.mean(cat_maes):.2f} ± {np.std(cat_maes):.2f}")
    print(f"  LightGBM: {np.mean(lgb_maes):.2f} ± {np.std(lgb_maes):.2f}")
    print(f"  平均融合: {np.mean(avg_maes):.2f} ± {np.std(avg_maes):.2f}")

# ========== 结果汇总 ==========

print(f"\n{'='*70}")
print("【最终结果 - 多Seed集成】")
print(f"{'='*70}")
print(f"CatBoost: {np.mean(seed_maes_cat):.2f} ± {np.std(seed_maes_cat):.2f}")
print(f"LightGBM: {np.mean(seed_maes_lgb):.2f} ± {np.std(seed_maes_lgb):.2f}")
print(f"平均融合: {np.mean(seed_maes_avg):.2f} ± {np.std(seed_maes_avg):.2f}")

final_mae = np.mean(seed_maes_avg)
print(f"\n最终 OOF MAE: {final_mae:.2f}")
print(f"目标: 450")
print(f"差距: {final_mae - 450:.2f}")

if final_mae < 450:
    print(f"\n🎉 达到目标！MAE: {final_mae:.2f} < 450")
else:
    print(f"\n⚠️ 距离目标: {final_mae - 450:.2f}")
    print(f"📊 达成度: {(450/final_mae)*100:.1f}%")

# ========== 测试集预测 ==========

# 多seed平均test预测
cat_final_test = np.mean(seed_cat_test, axis=0)
lgb_final_test = np.mean(seed_lgb_test, axis=0)
final_pred = (cat_final_test + lgb_final_test) / 2

# 预测值健全性检查
pred_min = final_pred.min()
pred_max = final_pred.max()
print(f"\n预测值范围: [{pred_min:.2f}, {pred_max:.2f}]")
print(f"预测均值: {final_pred.mean():.2f}")

if pred_min < 11:
    print("⚠️ 预测最小值异常（<11），检查是否截断")
if pred_max < 5000:
    print("⚠️ 预测最大值异常（<5000），模型可能严重欠拟合")

final_pred = np.maximum(final_pred, 50)

# ========== 保存提交 ==========
submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit_name = 'sota_v3_submit.csv'
submit.to_csv(BASE_PATH + submit_name, index=False)
print(f"\n✅ 提交文件已保存: {submit_name}")
print(f"📊 提交价格范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"💰 提交价格均值: {final_pred.mean():.2f}")

print(f"\n{'='*70}")
print(f"✅ SOTA V3 优化完成！（Phase {PHASE}）")
print(f"{'='*70}")
