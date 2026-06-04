# -*- coding: utf-8 -*-
"""
二手车价格预测 - SOTA V3-B（基于concat+新特征）
策略：沿用baseline的train+test concat + V3新特征
目标: Test MAE < 450
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 配置
SEEDS = [42]
N_FOLDS = 5

# 数据加载
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values
print(f"训练集: {len(train)}, 测试集: {len(test)}")

y_train_orig = train['price'].values.copy()  # 保存原始price，绝不截断

# 先处理 power=0（在concat前，用train统计）
brand_median_power = train[train['power'] > 0].groupby('brand')['power'].median()
global_median_power = train[train['power'] > 0]['power'].median()
for df in [train, test]:
    mask = df['power'] == 0
    df.loc[mask, 'power'] = df.loc[mask, 'brand'].map(brand_median_power).fillna(global_median_power).astype(df['power'].dtype)
    df['power_outlier'] = (df['power'] > 600).astype(int)
    df['power'] = df['power'].clip(upper=600)

# ========== 特征工程（基于concat）==========
test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

# 时间特征
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)

# 车龄分段
bins = [0, 2, 5, 10, 15, 999]
labels = [0, 1, 2, 3, 4]
data['age_segment'] = pd.cut(data['car_age'], bins=bins, labels=labels, right=True).astype('float')

# 新车标记
data['is_new_car'] = (data['car_age'] <= 1).astype(int)

# notRepairedDamage
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0.0')
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)

# v特征统计
v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)
data['v_max'] = data[v_cols].max(axis=1)
data['v_min'] = data[v_cols].min(axis=1)

# v交互（基线4个 + 新增2个）
data['v_0_v_3'] = data['v_0'] * data['v_3']
data['v_0_v_2'] = data['v_0'] * data['v_2']
data['v_0_v_12'] = data['v_0'] * data['v_12']
data['v_3_v_12'] = data['v_3'] * data['v_12']
data['v_0_v_8'] = data['v_0'] * data['v_8']
data['v_3_v_8'] = data['v_3'] * data['v_8']

# 业务特征
data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']
data['power_age'] = data['power'] * data['car_age']
data['usage_intensity'] = data['kilometer'] / (data['car_age'] + 1)

# 品牌×车型组合
model_fill = data['model'].fillna(-1).astype(int)
data['brand_model'] = data['brand'].astype(str) + '_' + model_fill.astype(str)

# 分组计数
for col in ['brand', 'model', 'regionCode']:
    data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')

# 频率编码（用全量数据，无监督安全）
for col in ['brand', 'model', 'brand_model']:
    freq_map = data[col].value_counts() / len(data)
    data[f'{col}_freq'] = data[col].map(freq_map)

# 品牌价格统计（用全量数据，无泄露风险 — 但注意这里price=-1的test也被算入）
# 修正：只用train统计
train_mask = data['price'] != -1
for stat_col in ['brand']:
    group_stats = data.loc[train_mask].groupby(stat_col)['price'].agg(['mean', 'std'])
    data[f'{stat_col}_price_mean'] = data[stat_col].map(group_stats['mean'])
    data[f'{stat_col}_price_std'] = data[stat_col].map(group_stats['std'])

# 删除无用列
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

# 缺失值处理
numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=['object']).columns

for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

for col in categorical_cols:
    mode_val = data[col].mode()[0] if not data[col].mode().empty else 'unknown'
    data[col] = data[col].fillna(mode_val)

# age_segment 填中位数
data['age_segment'] = data['age_segment'].fillna(data['age_segment'].median())

# Label Encoding
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# ========== 分离 train/test ==========
train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
if 'price' in test_data.columns:
    test_data = test_data.drop(columns=['price'])

X = train_data.drop(columns=['price'])
y = train_data['price'].values

# ========== Target编码 ==========
def target_encode(train_df, test_df, col, target='price', n_folds=N_FOLDS):
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[f'{col}_te'] = 0.0
    test_df[f'{col}_te'] = 0.0

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(train_df):
        target_mean = train_df.iloc[train_idx].groupby(col)[target].mean()
        train_df.loc[val_idx, f'{col}_te'] = train_df.loc[val_idx, col].map(target_mean)

    target_mean = train_df.groupby(col)[target].mean()
    test_df[f'{col}_te'] = test_df[col].map(target_mean).fillna(train_df[target].mean())

    return train_df[f'{col}_te'], test_df[f'{col}_te']

for col in ['brand', 'model', 'regionCode', 'brand_model']:
    train_te, test_te = target_encode(train_data, test_data, col)
    train_data[f'{col}_te'] = train_te
    test_data[f'{col}_te'] = test_te

X = train_data.drop(columns=['price'])

# 标准化（提升树模型稳定性）
scaler = StandardScaler()
numeric_cols_std = X.select_dtypes(include=[np.number]).columns
X[numeric_cols_std] = scaler.fit_transform(X[numeric_cols_std])
test_data[numeric_cols_std] = scaler.transform(test_data[numeric_cols_std])

print(f"特征数: {X.shape[1]}")

# ========== 训练 ==========
cat_params = {
    'iterations': 4500, 'learning_rate': 0.020, 'depth': 8,
    'l2_leaf_reg': 6, 'random_strength': 0.7, 'bagging_temperature': 0.6,
    'loss_function': 'MAE', 'random_seed': 42, 'verbose': 0, 'early_stopping_rounds': 140
}

lgb_params = {
    'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
    'learning_rate': 0.020, 'num_leaves': 120, 'max_depth': 9,
    'min_data_in_leaf': 17, 'feature_fraction': 0.88, 'bagging_fraction': 0.88,
    'bagging_freq': 5, 'reg_alpha': 0.19, 'reg_lambda': 0.19,
    'verbose': -1, 'seed': 42
}

all_cat_maes = []
all_lgb_maes = []
all_avg_maes = []
all_cat_test = []
all_lgb_test = []

for seed_idx, SEED in enumerate(SEEDS):
    print(f"\n{'='*70}")
    print(f"Seed {seed_idx+1}/{len(SEEDS)}: {SEED}")
    print(f"{'='*70}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cat_test_preds = np.zeros(len(test_data))
    lgb_test_preds = np.zeros(len(test_data))
    fold_maes_cat = []
    fold_maes_lgb = []
    fold_maes_avg = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # CatBoost
        print("  训练CatBoost...")
        cat_params['random_seed'] = SEED
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
        cat_pred_val = cat_model.predict(X_val)
        cat_pred_test = cat_model.predict(test_data)

        cat_mae = mean_absolute_error(y_val, cat_pred_val)
        fold_maes_cat.append(cat_mae)
        cat_test_preds += cat_pred_test / N_FOLDS
        print(f"  CatBoost MAE: {cat_mae:.2f}")

        # LightGBM
        print("  训练LightGBM...")
        lgb_params['seed'] = SEED
        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
        lgb_model = lgb.train(lgb_params, train_ds, num_boost_round=4500,
                               valid_sets=[val_ds],
                               callbacks=[lgb.early_stopping(140), lgb.log_evaluation(0)])
        lgb_pred_val = lgb_model.predict(X_val)
        lgb_pred_test = lgb_model.predict(test_data)

        lgb_mae = mean_absolute_error(y_val, lgb_pred_val)
        fold_maes_lgb.append(lgb_mae)
        lgb_test_preds += lgb_pred_test / N_FOLDS
        print(f"  LightGBM MAE: {lgb_mae:.2f}")

        # 简单平均
        avg_pred = (cat_pred_val + lgb_pred_val) / 2
        avg_mae = mean_absolute_error(y_val, avg_pred)
        fold_maes_avg.append(avg_mae)
        print(f"  平均融合 MAE: {avg_mae:.2f}")

    all_cat_maes.append(np.mean(fold_maes_cat))
    all_lgb_maes.append(np.mean(fold_maes_lgb))
    all_avg_maes.append(np.mean(fold_maes_avg))
    all_cat_test.append(cat_test_preds)
    all_lgb_test.append(lgb_test_preds)

    print(f"\n  Seed {SEED} 结果:")
    print(f"  CatBoost: {np.mean(fold_maes_cat):.2f} ± {np.std(fold_maes_cat):.2f}")
    print(f"  LightGBM: {np.mean(fold_maes_lgb):.2f} ± {np.std(fold_maes_lgb):.2f}")
    print(f"  平均融合: {np.mean(fold_maes_avg):.2f} ± {np.std(fold_maes_avg):.2f}")

# ========== 结果汇总 ==========
print(f"\n{'='*70}")
print("【最终结果】")
print(f"{'='*70}")
print(f"CatBoost: {np.mean(all_cat_maes):.2f} ± {np.std(all_cat_maes):.2f}")
print(f"LightGBM: {np.mean(all_lgb_maes):.2f} ± {np.std(all_lgb_maes):.2f}")
print(f"平均融合: {np.mean(all_avg_maes):.2f} ± {np.std(all_avg_maes):.2f}")

final_mae = np.mean(all_avg_maes)
print(f"\n最终 OOF MAE: {final_mae:.2f}")
print(f"目标: 450")
print(f"差距: {final_mae - 450:.2f}")

if final_mae < 450:
    print(f"\n🎉 达到目标！MAE: {final_mae:.2f} < 450")
else:
    print(f"\n⚠️ 距离目标: {final_mae - 450:.2f}")
    print(f"📊 达成度: {(450/final_mae)*100:.1f}%")

# ========== 测试集预测 ==========
cat_final = np.mean(all_cat_test, axis=0)
lgb_final = np.mean(all_lgb_test, axis=0)
final_pred = (cat_final + lgb_final) / 2

pred_min = final_pred.min()
pred_max = final_pred.max()
print(f"\n预测值范围: [{pred_min:.2f}, {pred_max:.2f}]")
print(f"预测均值: {final_pred.mean():.2f}")

if pred_min < 11:
    print("⚠️ 预测最小值异常")
if pred_max < 5000:
    print("⚠️ 预测最大值异常，模型可能欠拟合")

final_pred = np.maximum(final_pred, 50)

submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit.to_csv('sota_v3b_submit.csv', index=False)
print(f"\n✅ 提交文件: sota_v3b_submit.csv")
print(f"📊 价格范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"💰 价格均值: {final_pred.mean():.2f}")

print(f"\n{'='*70}")
print("✅ SOTA V3-B 优化完成！")
print(f"{'='*70}")
