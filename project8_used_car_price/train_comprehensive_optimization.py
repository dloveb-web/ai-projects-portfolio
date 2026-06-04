# -*- coding: utf-8 -*-
"""二手车价格预测 - 综合优化版
目标：MAE <= 450

优化策略（11项）：
1. 缺失值填充：数值型用中位数，类别型用众数
2. Power 处理：截断到[0,600]（不删除）
3. Price 处理：删除≤0的样本（仅训练集）
4. 车龄处理：删除负值样本（仅训练集）
5. Kilometer 处理：IQR截断+0值替换
6. 匿名特征：IQR截断
7. 特征工程：构造car_age等4个新特征
8. 修正明显的业务异常值（power截断到[0,600]）
9. 删除严重异常的样本（price≤0，车龄为负）
10. 使用稳健的统计方法（中位数、IQR）
11. 使用神经网络进行优化
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("综合优化版 - 11项全面优化")
print("="*60)

# ==================== 数据加载 ====================
print("\n加载数据...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"原始训练集: {train.shape}")
print(f"原始测试集: {test.shape}")

# ==================== 缺失值处理（第1、10项） ====================
print("\n1. 缺失值填充...")
print("   - 数值型：用中位数填充")
print("   - 类别型：用众数填充")

def smart_fillna(df, train_median_dict=None, train_mode_dict=None):
    """智能填充缺失值"""
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            # 数值型用中位数
            if train_median_dict and col in train_median_dict:
                median_val = train_median_dict[col]
            else:
                median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"   {col}: 用中位数 {median_val:.2f} 填充 {missing_count} 个缺失")
        else:
            # 类别型用众数
            if train_mode_dict and col in train_mode_dict:
                mode_val = train_mode_dict[col]
            else:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown'
            df[col].fillna(mode_val, inplace=True)
            print(f"   {col}: 用众数 '{mode_val}' 填充 {missing_count} 个缺失")
    return df

# 处理notRepairedDamage
for df in [train, test]:
    if 'notRepairedDamage' in df.columns and df['notRepairedDamage'].dtype == object:
        df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', '0.0')
        df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce').fillna(0)

# 计算训练集的统计信息
train_median_dict = {}
train_mode_dict = {}

for col in train.columns:
    if pd.api.types.is_numeric_dtype(train[col]):
        train_median_dict[col] = train[col].median()
    else:
        if len(train[col].mode()) > 0:
            train_mode_dict[col] = train[col].mode()[0]

# 应用智能填充
train = smart_fillna(train, train_median_dict, train_mode_dict)
test = smart_fillna(test, train_median_dict, train_mode_dict)

# ==================== Price 处理（第3项） ====================
print("\n2. Price 处理 - 删除≤0的样本（仅训练集）")
price_before = len(train)
train = train[train['price'] > 0].copy()
price_after = len(train)
print(f"   删除price≤0的样本: {price_before - price_after} 个")

# 对数变换
print("\n3. 对数变换...")
train['log_price'] = np.log1p(train['price'])

# ==================== 基础特征工程（第7项） ====================
print("\n4. 特征工程 - 构造car_age等特征...")
for df in [train, test]:
    # 车龄计算
    df['reg_year'] = df['regDate'] // 10000
    df['creat_year'] = df['creatDate'] // 10000
    df['car_age'] = (df['creat_year'] - df['reg_year']).clip(lower=0)

    # 车龄相关特征
    df['car_age_squared'] = df['car_age'] ** 2
    df['car_age_cubed'] = df['car_age'] ** 3
    df['car_age_sqrt'] = np.sqrt(df['car_age'])
    df['car_age_log'] = np.log1p(df['car_age'])

# ==================== 车龄处理（第4、9项） ====================
print("\n5. 车龄处理 - 删除负值样本（仅训练集）")
age_before = len(train)
train = train[train['car_age'] >= 0].copy()
age_after = len(train)
print(f"   删除车龄为负的样本: {age_before - age_after} 个")

# ==================== Power 处理（第2、8项） ====================
print("\n6. Power 处理 - 截断到[0,600]（不删除）")
power_min_before = train['power'].min()
power_max_before = train['power'].max()

for df in [train, test]:
    df['power'] = df['power'].clip(lower=0, upper=600)

print(f"   Power范围截断: [{power_min_before:.2f}, {power_max_before:.2f}] -> [0, 600]")
print(f"   训练集Power异常值: {(train['power'] < 0).sum() + (train['power'] > 600).sum()} 个")
print(f"   测试集Power异常值: {(test['power'] < 0).sum() + (test['power'] > 600).sum()} 个")

# Power相关特征
for df in [train, test]:
    df['log_power'] = np.log1p(df['power'])
    df['power_squared'] = df['power'] ** 2
    df['power_cubed'] = df['power'] ** 3
    df['power_sqrt'] = np.sqrt(df['power'])

# ==================== Kilometer 处理（第5、10项 - IQR） ====================
print("\n7. Kilometer 处理 - IQR截断+0值替换")

def robust_fill_iqr(series, name):
    """使用IQR方法稳健填充0值"""
    non_zero = series[series > 0]
    if len(non_zero) == 0:
        return series

    q1 = non_zero.quantile(0.25)
    q3 = non_zero.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    median_val = non_zero.median()

    # 填充0值为中位数
    series_filled = series.copy()
    zero_mask = series == 0
    series_filled[zero_mask] = median_val

    # 截断异常值
    series_filled = series_filled.clip(lower=lower_bound, upper=upper_bound)

    print(f"   {name}:")
    print(f"     0值数量: {zero_mask.sum()}, 用中位数 {median_val:.2f} 替换")
    print(f"     IQR范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"     截断异常值: {((series_filled < lower_bound) | (series_filled > upper_bound)).sum()} 个")

    return series_filled

train['kilometer'] = robust_fill_iqr(train['kilometer'], '训练集Kilometer')
test['kilometer'] = robust_fill_iqr(test['kilometer'], '测试集Kilometer')

# Kilometer相关特征
for df in [train, test]:
    df['log_kilometer'] = np.log1p(df['kilometer'])
    df['kilometer_squared'] = df['kilometer'] ** 2

# ==================== 匿名特征处理（第6、10项 - IQR） ====================
print("\n8. 匿名特征处理 - IQR截断")

v_cols = [f'v_{i}' for i in range(15)]
available_v_cols = [col for col in v_cols if col in train.columns]

for col in available_v_cols:
    train[col] = robust_fill_iqr(train[col], f'训练集{col}')
    test[col] = robust_fill_iqr(test[col], f'测试集{col}')

# v特征交互特征
if len(available_v_cols) >= 3:
    print("   添加v特征交互...")
    for df in [train, test]:
        df['v_mean'] = df[available_v_cols].mean(axis=1)
        df['v_std'] = df[available_v_cols].std(axis=1)
        df['v_0_v_3'] = df['v_0'] * df['v_3']
        df['v_1_v_2'] = df['v_1'] * df['v_2']
        df['v_4_v_5'] = df['v_4'] * df['v_5']

# 组合特征
for df in [train, test]:
    df['power_km'] = df['power'] * df['kilometer']
    df['power_per_year'] = df['power'] / (df['car_age'] + 1)
    df['km_per_year'] = df['kilometer'] / (df['car_age'] + 1)

# ==================== 数据准备 ====================
print("\n9. 数据准备...")
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType', 'notRepairedDamage', 'price', 'log_price']
X = train.drop(columns=[c for c in drop_cols if c in train.columns])
X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])
y = train['log_price']
y_orig = train['price']

numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
X = X[numeric_cols]
X_test = X_test[[col for col in numeric_cols if col in X_test.columns]]

print(f"   特征数: {X.shape[1]}")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ==================== 模型训练 ====================
print("\n10. 模型训练 (5折交叉验证)...")
print("="*60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

all_maes = {
    'cat': [], 'lgb': [], 'xgb': [], 'ensemble': [],
    'cat_orig': [], 'lgb_orig': [], 'xgb_orig': [], 'ensemble_orig': []
}
test_preds_cat = np.zeros(len(test))
test_preds_lgb = np.zeros(len(test))
test_preds_xgb = np.zeros(len(test))

# ==================== 神经网络模型（第11项） ====================
# TensorFlow未安装，暂时跳过神经网络，使用三模型融合
print("\n   神经网络: TensorFlow未安装，使用三模型融合（CatBoost+LightGBM+XGBoost）")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    y_val_orig = y_orig.iloc[val_idx]

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=4000,
        learning_rate=0.01,
        depth=8,
        l2_leaf_reg=5,
        loss_function='MAE',
        random_seed=42,
        verbose=200,
        early_stopping_rounds=150
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    cat_pred = cat_model.predict(X_val)
    cat_mae = mean_absolute_error(y_val, cat_pred)
    all_maes['cat'].append(cat_mae)
    cat_pred_orig = np.exp(cat_pred)
    cat_mae_orig = mean_absolute_error(y_val_orig, cat_pred_orig)
    all_maes['cat_orig'].append(cat_mae_orig)
    test_preds_cat += cat_model.predict(X_test_scaled) / 5
    print(f"   CatBoost MAE (log空间): {cat_mae:.4f}, 真实价格: {cat_mae_orig:.2f}")

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        num_leaves=127,
        max_depth=8,
        learning_rate=0.01,
        n_estimators=4000,
        min_data_in_leaf=20,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    lgb_pred = lgb_model.predict(X_val)
    lgb_mae = mean_absolute_error(y_val, lgb_pred)
    all_maes['lgb'].append(lgb_mae)
    lgb_pred_orig = np.exp(lgb_pred)
    lgb_mae_orig = mean_absolute_error(y_val_orig, lgb_pred_orig)
    all_maes['lgb_orig'].append(lgb_mae_orig)
    test_preds_lgb += lgb_model.predict(X_test_scaled) / 5
    print(f"   LightGBM MAE (log空间): {lgb_mae:.4f}, 真实价格: {lgb_mae_orig:.2f}")

    # XGBoost
    xgb_model = XGBRegressor(
        n_estimators=4000,
        learning_rate=0.01,
        max_depth=8,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1,
        objective='reg:absoluteerror',
        random_state=42,
        n_jobs=-1,
        eval_metric='mae',
        early_stopping_rounds=150,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_pred = xgb_model.predict(X_val)
    xgb_mae = mean_absolute_error(y_val, xgb_pred)
    all_maes['xgb'].append(xgb_mae)
    xgb_pred_orig = np.exp(xgb_pred)
    xgb_mae_orig = mean_absolute_error(y_val_orig, xgb_pred_orig)
    all_maes['xgb_orig'].append(xgb_mae_orig)
    test_preds_xgb += xgb_model.predict(X_test_scaled) / 5
    print(f"   XGBoost MAE (log空间): {xgb_mae:.4f}, 真实价格: {xgb_mae_orig:.2f}")

    # 三模型融合
    ensemble_pred = 0.4 * cat_pred + 0.35 * lgb_pred + 0.25 * xgb_pred
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    all_maes['ensemble'].append(ensemble_mae)
    ensemble_pred_orig = np.exp(ensemble_pred)
    ensemble_mae_orig = mean_absolute_error(y_val_orig, ensemble_pred_orig)
    all_maes['ensemble_orig'].append(ensemble_mae_orig)
    print(f"   Ensemble MAE (log空间): {ensemble_mae:.4f}, 真实价格: {ensemble_mae_orig:.2f}")

# ==================== 最终结果 ====================
print("\n" + "="*60)
print("最终结果 (5折平均)")
print("="*60)
print(f"CatBoost MAE (log空间): {np.mean(all_maes['cat']):.4f}")
print(f"LightGBM MAE (log空间): {np.mean(all_maes['lgb']):.4f}")
print(f"XGBoost MAE (log空间): {np.mean(all_maes['xgb']):.4f}")
print(f"Ensemble MAE (log空间): {np.mean(all_maes['ensemble']):.4f}")
print("")
print(f"CatBoost MAE (真实价格): {np.mean(all_maes['cat_orig']):.2f}")
print(f"LightGBM MAE (真实价格): {np.mean(all_maes['lgb_orig']):.2f}")
print(f"XGBoost MAE (真实价格): {np.mean(all_maes['xgb_orig']):.2f}")
print(f"Ensemble MAE (真实价格): {np.mean(all_maes['ensemble_orig']):.2f}")

# 最终预测（三模型融合）
print(f"\n融合权重: CatBoost=0.40, LightGBM=0.35, XGBoost=0.25")
test_preds_final = (0.4 * test_preds_cat +
                  0.35 * test_preds_lgb +
                  0.25 * test_preds_xgb)
final_pred = np.exp(test_preds_final)
final_pred = np.maximum(final_pred, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': test['SaleID'].values,
    'price': final_pred
})
submit.to_csv('comprehensive_optimized_submit.csv', index=False)
print(f"\n11. 结果已保存到: comprehensive_optimized_submit.csv")

final_mae_log = np.mean(all_maes['ensemble'])
final_mae_orig = np.mean(all_maes['ensemble_orig'])
print(f"\n" + "="*60)
if final_mae_orig <= 450:
    print(f"🎉 成功！真实价格MAE: {final_mae_orig:.2f} <= 450")
    print(f"Log空间MAE: {final_mae_log:.4f}")
    print(f"比之前改进: {529.98 - final_mae_orig:.2f}")
else:
    print(f"⚠️ 距离目标: {final_mae_orig - 450:.2f}")
    print(f"当前真实价格MAE: {final_mae_orig:.2f}")
    print(f"比之前改进: {529.98 - final_mae_orig:.2f}")
print("="*60)
