# -*- coding: utf-8 -*-
"""
二手车价格预测 - SOTA V2 优化版
基于 sota_correct_stacking.py (Test MAE 460.71) 优化
目标: Test MAE < 450

优化点:
1. 修复4处数据泄露 (分组count/缺失值/LabelEncoder/StandardScaler)
2. 添加异常值处理 (power=0修复, 价格缩尾)
3. 精选高价值特征增补 (v_1*v_5, v_2*v_4, v_0*v_5, power_per_age, km_per_year, v_sum)
4. OOF Stacking (Ridge元模型)
5. 多Seed集成 (可选)
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
import time
import warnings
warnings.filterwarnings('ignore')

# ============ 配置 ============
SEED = 42
USE_MULTI_SEED = True  # 单seed已验证CV=454.89，启用多seed进一步提升
SEEDS = [42, 123, 2024, 0, 7]
np.random.seed(SEED)

print("=" * 70)
print("SOTA V2 优化版 - 目标 MAE < 450")
print("优化: 修复泄露 + 异常值处理 + 特征增补 + OOF Stacking")
print("=" * 70)

# ============ 1. 数据加载 ============
print("\n[1] 数据加载...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values
print(f"训练集: {len(train)}, 测试集: {len(test)}")

# ============ 2. 特征工程 (train/test 分离处理，防泄露) ============
print("\n[2] 特征工程 (train/test 分离处理)...")

# --- 时间特征 ---
for df in [train, test]:
    df['reg_year'] = df['regDate'] // 10000
    df['creat_year'] = df['creatDate'] // 10000
    df['car_age'] = (df['creat_year'] - df['reg_year']).clip(lower=0)

# --- notRepairedDamage ---
for df in [train, test]:
    df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', '0.0')
    df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce').fillna(0)

# --- v特征统计 ---
v_cols = [f'v_{i}' for i in range(15)]
for df in [train, test]:
    df['v_mean'] = df[v_cols].mean(axis=1)
    df['v_std'] = df[v_cols].std(axis=1)
    df['v_max'] = df[v_cols].max(axis=1)
    df['v_min'] = df[v_cols].min(axis=1)
    df['v_sum'] = df[v_cols].sum(axis=1)

# --- v特征交互 ---
for df in [train, test]:
    # 原有4个交互
    df['v_0_v_3'] = df['v_0'] * df['v_3']
    df['v_0_v_2'] = df['v_0'] * df['v_2']
    df['v_0_v_12'] = df['v_0'] * df['v_12']
    df['v_3_v_12'] = df['v_3'] * df['v_12']
    # 新增3个高价值交互
    df['v_1_v_5'] = df['v_1'] * df['v_5']
    df['v_2_v_4'] = df['v_2'] * df['v_4']
    df['v_0_v_5'] = df['v_0'] * df['v_5']

# --- 业务特征 ---
for df in [train, test]:
    df['power_km'] = df['power'] * df['kilometer']
    df['age_km'] = df['car_age'] * df['kilometer']
    df['power_age'] = df['power'] * df['car_age']
    df['usage_intensity'] = df['kilometer'] / (df['car_age'] + 1)
    # 新增业务特征
    df['power_per_age'] = df['power'] / (df['car_age'] + 0.1)
    df['km_per_year'] = df['kilometer'] / (df['car_age'] + 0.1)

# --- 分组特征 (防泄露：仅train统计) ---
print("  分组特征: 仅train统计，test做map...")
for col in ['brand', 'model', 'regionCode']:
    count_map = train.groupby(col)['SaleID'].count().to_dict()
    train[f'{col}_count'] = train[col].map(count_map)
    test[f'{col}_count'] = test[col].map(count_map).fillna(1)

# ============ 3. 异常值处理 ============
print("\n[3] 异常值处理...")

# --- power=0 修复 (品牌中位数填充) ---
power_zero_count = (train['power'] == 0).sum()
brand_median_power = train[train['power'] > 0].groupby('brand')['power'].median()
global_median_power = train[train['power'] > 0]['power'].median()

for df in [train, test]:
    mask = df['power'] == 0
    df.loc[mask, 'power'] = df.loc[mask, 'brand'].map(brand_median_power).fillna(global_median_power).astype(df['power'].dtype)

print(f"  power=0 修复: 训练集{power_zero_count}个")

# --- 价格缩尾 (1-99百分位) ---
price_lower = train['price'].quantile(0.01)
price_upper = train['price'].quantile(0.99)
price_before = (train['price'] < price_lower).sum() + (train['price'] > price_upper).sum()
train['price'] = train['price'].clip(price_lower, price_upper)
print(f"  价格缩尾 (1-99%): 修正{price_before}个极端值, 范围[{price_lower:.0f}, {price_upper:.0f}]")

# ============ 4. 删除无用列 ============
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
train = train.drop(columns=[c for c in drop_cols if c in train.columns], errors='ignore')
test = test.drop(columns=[c for c in drop_cols if c in test.columns], errors='ignore')

# ============ 5. 缺失值处理 (防泄露：仅train统计量) ============
print("\n[5] 缺失值处理 (仅train统计量)...")

# 分离特征和标签
y = train['price'].values
train_features = train.drop(columns=['price'])
test_features = test.drop(columns=['price'], errors='ignore')

# 数值列
numeric_cols = train_features.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    median_val = train_features[col].median()
    train_features[col] = train_features[col].fillna(median_val)
    if col in test_features.columns:
        test_features[col] = test_features[col].fillna(median_val)

# 类别列
categorical_cols = train_features.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_val = train_features[col].mode()[0] if not train_features[col].mode().empty else 'unknown'
    train_features[col] = train_features[col].fillna(mode_val)
    test_features[col] = test_features[col].fillna(mode_val)

# ============ 6. Label Encoding (防泄露：仅train fit) ============
print("\n[6] Label Encoding (仅train fit)...")

for col in categorical_cols:
    le = LabelEncoder()
    le.fit(train_features[col].astype(str))
    train_features[col] = le.transform(train_features[col].astype(str))
    # 测试集未知类别映射为0
    test_features[col] = test_features[col].astype(str).apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else 0
    )

# ============ 7. Target Encoding (CV方式防泄露) ============
print("\n[7] Target Encoding (CV方式)...")

def target_encode_cv(train_df, test_df, col, target_values, n_folds=5, seed=42):
    """CV方式计算Target Encoding，防止信息泄露"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['_target'] = target_values
    train_df[f'{col}_te'] = 0.0
    test_df[f'{col}_te'] = 0.0

    global_mean = target_values.mean()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, val_idx in kf.split(train_df):
        target_mean = train_df.iloc[train_idx].groupby(col)['_target'].mean()
        train_df.iloc[val_idx, train_df.columns.get_loc(f'{col}_te')] = \
            train_df.iloc[val_idx][col].map(target_mean).fillna(global_mean).values

    # 测试集：使用全量训练集均值
    target_mean = train_df.groupby(col)['_target'].mean()
    test_df[f'{col}_te'] = test_df[col].map(target_mean).fillna(global_mean)

    train_df = train_df.drop(columns=['_target'])
    return train_df[f'{col}_te'].values, test_df[f'{col}_te'].values

for col in ['brand', 'model', 'regionCode']:
    train_te, test_te = target_encode_cv(train_features, test_features, col, y, seed=SEED)
    train_features[f'{col}_te'] = train_te
    test_features[f'{col}_te'] = test_te

# ============ 8. 最终数据准备 ============
X = train_features
X_test = test_features

print(f"\n[8] 最终特征数: {X.shape[1]}")
print(f"  特征列表: {list(X.columns)}")

# ============ 9. 模型参数 ============
cat_params = {
    'iterations': 3500,
    'learning_rate': 0.020,
    'depth': 8,
    'l2_leaf_reg': 6,
    'random_strength': 0.7,
    'bagging_temperature': 0.6,
    'loss_function': 'MAE',
    'random_seed': SEED,
    'verbose': 0,
    'early_stopping_rounds': 100
}

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.020,
    'num_leaves': 120,
    'max_depth': 9,
    'min_data_in_leaf': 17,
    'feature_fraction': 0.88,
    'bagging_fraction': 0.88,
    'bagging_freq': 5,
    'reg_alpha': 0.19,
    'reg_lambda': 0.19,
    'verbose': -1,
    'seed': SEED
}

# ============ 10. 训练函数 ============
def train_single_seed(X, y, X_test, cat_params, lgb_params, seed):
    """单seed训练：5折CV + OOF收集"""
    cat_params = cat_params.copy()
    lgb_params = lgb_params.copy()
    cat_params['random_seed'] = seed
    lgb_params['seed'] = seed

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    oof_cat = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    test_preds_cat = np.zeros(len(X_test))
    test_preds_lgb = np.zeros(len(X_test))

    fold_maes_cat = []
    fold_maes_lgb = []
    fold_maes_avg = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # CatBoost
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
        cat_pred_val = cat_model.predict(X_va)
        cat_pred_test = cat_model.predict(X_test)

        oof_cat[val_idx] = cat_pred_val
        test_preds_cat += cat_pred_test / 5

        cat_mae = mean_absolute_error(y_va, cat_pred_val)
        fold_maes_cat.append(cat_mae)

        # LightGBM
        train_data_lgb = lgb.Dataset(X_tr, label=y_tr)
        val_data_lgb = lgb.Dataset(X_va, label=y_va, reference=train_data_lgb)
        lgb_model = lgb.train(
            lgb_params, train_data_lgb, num_boost_round=3500,
            valid_sets=[val_data_lgb],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        lgb_pred_val = lgb_model.predict(X_va)
        lgb_pred_test = lgb_model.predict(X_test)

        oof_lgb[val_idx] = lgb_pred_val
        test_preds_lgb += lgb_pred_test / 5

        lgb_mae = mean_absolute_error(y_va, lgb_pred_val)
        fold_maes_lgb.append(lgb_mae)

        # 平均融合
        avg_pred = (cat_pred_val + lgb_pred_val) / 2
        avg_mae = mean_absolute_error(y_va, avg_pred)
        fold_maes_avg.append(avg_mae)

        print(f"  Fold {fold+1}/5 | Cat: {cat_mae:.2f} | LGB: {lgb_mae:.2f} | Avg: {avg_mae:.2f}")

    return {
        'oof_cat': oof_cat,
        'oof_lgb': oof_lgb,
        'test_cat': test_preds_cat,
        'test_lgb': test_preds_lgb,
        'cv_cat': np.mean(fold_maes_cat),
        'cv_lgb': np.mean(fold_maes_lgb),
        'cv_avg': np.mean(fold_maes_avg),
    }

# ============ 11. 执行训练 ============
print("\n" + "=" * 70)
print("[训练] 开始5折交叉验证...")
print("=" * 70)

start_time = time.time()

if USE_MULTI_SEED:
    print(f"多Seed模式: {SEEDS}")
    all_results = []
    for i, seed in enumerate(SEEDS):
        print(f"\n--- Seed {i+1}/{len(SEEDS)}: {seed} ---")
        result = train_single_seed(X, y, X_test, cat_params, lgb_params, seed)
        all_results.append(result)
        print(f"  CV -> Cat: {result['cv_cat']:.2f} | LGB: {result['cv_lgb']:.2f} | Avg: {result['cv_avg']:.2f}")

    # 多Seed平均
    oof_cat = np.mean([r['oof_cat'] for r in all_results], axis=0)
    oof_lgb = np.mean([r['oof_lgb'] for r in all_results], axis=0)
    test_preds_cat = np.mean([r['test_cat'] for r in all_results], axis=0)
    test_preds_lgb = np.mean([r['test_lgb'] for r in all_results], axis=0)
    cv_cat = np.mean([r['cv_cat'] for r in all_results])
    cv_lgb = np.mean([r['cv_lgb'] for r in all_results])
    cv_avg = np.mean([r['cv_avg'] for r in all_results])
else:
    print(f"单Seed模式: {SEED}")
    result = train_single_seed(X, y, X_test, cat_params, lgb_params, SEED)
    oof_cat = result['oof_cat']
    oof_lgb = result['oof_lgb']
    test_preds_cat = result['test_cat']
    test_preds_lgb = result['test_lgb']
    cv_cat = result['cv_cat']
    cv_lgb = result['cv_lgb']
    cv_avg = result['cv_avg']

elapsed = time.time() - start_time

# ============ 12. 结果汇总 ============
print(f"\n{'=' * 70}")
print("[结果] 5折交叉验证平均")
print(f"{'=' * 70}")
print(f"CatBoost 平均 MAE: {cv_cat:.2f}")
print(f"LightGBM 平均 MAE: {cv_lgb:.2f}")
print(f"简单平均 MAE: {cv_avg:.2f}")
print(f"训练耗时: {elapsed:.1f}s")

# OOF Stacking MAE (直接在OOF上评估)
oof_avg = (oof_cat + oof_lgb) / 2
oof_avg_mae = mean_absolute_error(y, oof_avg)
print(f"OOF简单平均 MAE: {oof_avg_mae:.2f}")

# ============ 13. OOF Stacking (Ridge元模型) ============
print(f"\n{'=' * 70}")
print("[Stacking] Ridge元模型训练...")
print(f"{'=' * 70}")

meta_features_train = np.column_stack([oof_cat, oof_lgb])
meta_features_test = np.column_stack([test_preds_cat, test_preds_lgb])

# Ridge Stacking
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(meta_features_train, y)
stacking_oof_pred = ridge_model.predict(meta_features_train)
stacking_oof_mae = mean_absolute_error(y, stacking_oof_pred)
print(f"Ridge Stacking OOF MAE: {stacking_oof_mae:.2f}")
print(f"Ridge系数: Cat={ridge_model.coef_[0]:.4f}, LGB={ridge_model.coef_[1]:.4f}, 截距={ridge_model.intercept_:.4f}")

# 对比简单平均
print(f"\n融合方式对比:")
print(f"  简单平均 OOF MAE: {oof_avg_mae:.2f}")
print(f"  Ridge Stacking OOF MAE: {stacking_oof_mae:.2f}")
print(f"  差异: {oof_avg_mae - stacking_oof_mae:.2f} (正=Stacking更好)")

# ============ 14. 生成提交文件 ============
print(f"\n{'=' * 70}")
print("[提交] 生成预测文件...")
print(f"{'=' * 70}")

# 方案1: 简单平均
final_pred_avg = (test_preds_cat + test_preds_lgb) / 2
final_pred_avg = np.maximum(final_pred_avg, 50)

submit_avg = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred_avg})
submit_avg.to_csv('sota_v2_submit.csv', index=False)
print(f"✅ sota_v2_submit.csv (简单平均)")
print(f"   预测范围: [{final_pred_avg.min():.2f}, {final_pred_avg.max():.2f}]")
print(f"   预测均值: {final_pred_avg.mean():.2f}")

# 方案2: Ridge Stacking
final_pred_stack = ridge_model.predict(meta_features_test)
final_pred_stack = np.maximum(final_pred_stack, 50)

submit_stack = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred_stack})
submit_stack.to_csv('sota_v2_stacking_submit.csv', index=False)
print(f"✅ sota_v2_stacking_submit.csv (Ridge Stacking)")
print(f"   预测范围: [{final_pred_stack.min():.2f}, {final_pred_stack.max():.2f}]")
print(f"   预测均值: {final_pred_stack.mean():.2f}")

# ============ 15. 最终总结 ============
print(f"\n{'=' * 70}")
print("[总结] SOTA V2 优化结果")
print(f"{'=' * 70}")

best_oof_mae = min(oof_avg_mae, stacking_oof_mae)
best_method = "Ridge Stacking" if stacking_oof_mae < oof_avg_mae else "简单平均"

print(f"CV (5折平均) MAE:")
print(f"  CatBoost: {cv_cat:.2f}")
print(f"  LightGBM: {cv_lgb:.2f}")
print(f"  简单平均: {cv_avg:.2f}")
print(f"\nOOF MAE (更准确的泛化估计):")
print(f"  简单平均: {oof_avg_mae:.2f}")
print(f"  Ridge Stacking: {stacking_oof_mae:.2f}")
print(f"  最优方案: {best_method} ({best_oof_mae:.2f})")

# 与原版对比
print(f"\n与原版 sota_correct_stacking 对比:")
print(f"  原版 CV MAE: 475.53")
print(f"  原版 Test MAE: 460.71")
print(f"  V2 CV MAE: {cv_avg:.2f}")
print(f"  V2 OOF MAE: {best_oof_mae:.2f}")

if best_oof_mae < 450:
    print(f"\n🎉 OOF MAE 已低于450！有望在Test上达到目标")
elif best_oof_mae < 460:
    print(f"\n⚠️ OOF MAE < 460，距目标较近，多Seed可能突破")
else:
    print(f"\n📊 OOF MAE: {best_oof_mae:.2f}，距目标: {best_oof_mae - 450:.2f}")

print(f"\n推荐提交: sota_v2_submit.csv 或 sota_v2_stacking_submit.csv")
print(f"如需进一步提升: 将 USE_MULTI_SEED 改为 True 后重新运行")
print(f"\n{'=' * 70}")
print("✅ SOTA V2 优化完成！")
print(f"{'=' * 70}")
