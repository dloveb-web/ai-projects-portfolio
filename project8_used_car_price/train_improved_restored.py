# -*- coding: utf-8 -*-
"""二手车价格预测 - 改进版
关键改进：
1. 取消对数变换，直接预测原始价格
2. 保守数据清洗
3. K-Fold目标编码
4. 自适应融合权重
目标：MAE <= 450 (参考原始477 MAE)
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("改进版 - 回归原始价格 + 目标编码 + 自适应权重")
print("="*60)

# ==================== 数据加载 ====================
print("\n加载数据...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"原始训练集: {train.shape}")
print(f"原始测试集: {test.shape}")

# ==================== 保守数据清洗 ====================
print("\n保守数据清洗（只处理明显错误）...")

# 1. notRepairedDamage处理
for df in [train, test]:
    if 'notRepairedDamage' in df.columns and df['notRepairedDamage'].dtype == object:
        df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', '0.0')
        df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce').fillna(0)

# 2. 删除price<=0的明显异常
price_before = len(train)
train = train[train['price'] > 0].copy()
price_after = len(train)
print(f"   删除price<=0的样本: {price_before - price_after} 个")

# 3. Power截断（保守处理）
for df in [train, test]:
    df['power'] = df['power'].clip(lower=0, upper=600)

print(f"   Power截断: [0, 600]（保守）")

# 4. 填充缺失值（保守：数值用中位数，类别用众数）
def conservative_fillna(df, train_dict=None):
    """保守填充缺失值"""
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            # 数值型用中位数
            if train_dict and col in train_dict:
                df[col].fillna(train_dict[col], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        else:
            # 类别型用众数
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                mode_val = mode_val[0]
                df[col].fillna(mode_val, inplace=True)

# 计算训练集的填充值
train_fill_dict = {}
for col in train.select_dtypes(include=[np.number]).columns:
    train_fill_dict[col] = train[col].median()

# 应用填充
conservative_fillna(train)
conservative_fillna(test)

# ==================== 特征工程（参考原始方案） ====================
print("\n特征工程...")

for df in [train, test]:
    # 基础时间特征
    df['reg_year'] = df['regDate'] // 10000
    df['creat_year'] = df['creatDate'] // 10000
    df['car_age'] = (df['creat_year'] - df['reg_year']).clip(lower=0)

    # 车龄特征
    df['car_age_squared'] = df['car_age'] ** 2
    df['car_age_cubed'] = df['car_age'] ** 3

    # v特征交互
    v_cols = [f'v_{i}' for i in range(15)]
    available_v_cols = [col for col in v_cols if col in df.columns]
    if len(available_v_cols) >= 3:
        df['v_mean'] = df[available_v_cols].mean(axis=1)
        df['v_std'] = df[available_v_cols].std(axis=1)
        df['v_0_v_3'] = df['v_0'] * df['v_3']
        df['v_1_v_2'] = df['v_1'] * df['v_2']

    # Power相关
    df['log_power'] = np.log1p(df['power'])
    df['power_squared'] = df['power'] ** 2
    df['power_km'] = df['power'] * df['kilometer']

    # Kilometer相关
    df['log_km'] = np.log1p(df['kilometer'])
    df['km_squared'] = df['kilometer'] ** 2

    # 组合特征
    df['power_per_year'] = df['power'] / (df['car_age'] + 1)
    df['km_per_year'] = df['kilometer'] / (df['car_age'] + 1)
    df['power_per_km'] = df['power'] / (df['kilometer'] + 1)

print(f"   特征数: {train.shape[1]-1}")

# ==================== K-Fold目标编码 ====================
print("\nK-Fold目标编码...")

def kfold_target_encode(train_col, y, test_col, n_splits=5, smoothing=1.0):
    """K-Fold目标编码，避免数据泄露"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_encoded = np.zeros(len(train_col))
    target_mean = y.mean()

    for train_idx, val_idx in kf.split(train_col):
        fold_train = train_col.iloc[train_idx]
        fold_y = y.iloc[train_idx]

        # 计算fold内的类别均值
        category_means = fold_y.groupby(fold_train).mean()
        category_counts = fold_train.groupby(fold_train).count()

        # 平滑
        smoothed_mean = (category_means * category_counts + target_mean * smoothing) / \
                     (category_counts + smoothing)

        # 对验证集编码
        val_data = train_col.iloc[val_idx]
        train_encoded[val_idx] = val_data.map(smoothed_mean).fillna(target_mean)

    # 对整个训练集编码测试集
    category_means = y.groupby(train_col).mean()
    category_counts = train_col.groupby(train_col).count()
    smoothed_mean = (category_means * category_counts + target_mean * smoothing) / \
                 (category_counts + smoothing)

    test_encoded = test_col.map(smoothed_mean).fillna(target_mean)

    return train_encoded, test_encoded

# 对brand进行目标编码
if 'brand' in train.columns:
    print("   对brand进行K-Fold目标编码...")
    train_brand_te, test_brand_te = kfold_target_encode(
        train['brand'], train['price'], test['brand'], n_splits=5, smoothing=5.0
    )
    train['brand_te'] = train_brand_te
    test['brand_te'] = test_brand_te

# 对model进行目标编码
if 'model' in train.columns:
    print("   对model进行K-Fold目标编码...")
    train_model_te, test_model_te = kfold_target_encode(
        train['model'], train['price'], test['model'], n_splits=5, smoothing=5.0
    )
    train['model_te'] = train_model_te
    test['model_te'] = test_model_te

# 对bodyType进行目标编码
if 'bodyType' in train.columns:
    print("   对bodyType进行K-Fold目标编码...")
    train_body_te, test_body_te = kfold_target_encode(
        train['bodyType'], train['price'], test['bodyType'], n_splits=5, smoothing=5.0
    )
    train['bodyType_te'] = train_body_te
    test['bodyType_te'] = test_body_te

# ==================== 数据准备 ====================
print("\n数据准备...")

# 分类特征
categorical_cols = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox']
for col in categorical_cols:
    if col in train.columns:
        le = LabelEncoder()
        train[col + '_encoded'] = le.fit_transform(train[col].astype(str))
        test[col + '_encoded'] = le.transform(test[col].astype(str))

# 删除不使用的列
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType', 'notRepairedDamage', 'price']
# 保留编码后的分类列
drop_cols.extend(['brand', 'model', 'bodyType', 'fuelType', 'gearbox'])

X = train.drop(columns=[c for c in drop_cols if c in train.columns])
X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])
y = train['price']  # 直接用原始价格，不对数变换

numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
X = X[numeric_cols]
X_test = X_test[[col for col in numeric_cols if col in X_test.columns]]

# 填充缺失值（如果还有）
for col in X.columns:
    median_val = X[col].median()
    X[col].fillna(median_val, inplace=True)
    if col in X_test.columns:
        X_test[col].fillna(median_val, inplace=True)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print(f"   准备完成，特征数: {X_scaled.shape[1]}")

# ==================== 模型训练（5折交叉验证） ====================
print("\n模型训练 (5折交叉验证)...")
print("="*60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

all_maes = {'cat': [], 'lgb': [], 'xgb': [], 'ensemble': []}
test_preds_cat = np.zeros(len(test))
test_preds_lgb = np.zeros(len(test))
test_preds_xgb = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # CatBoost（参考原始超参数）
    cat_model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=10,
        loss_function='MAE',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    cat_pred = cat_model.predict(X_val)
    cat_mae = mean_absolute_error(y_val, cat_pred)
    all_maes['cat'].append(cat_mae)
    test_preds_cat += cat_model.predict(X_test_scaled) / 5
    print(f"   CatBoost MAE: {cat_mae:.2f}")

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        num_leaves=63,
        max_depth=6,
        learning_rate=0.03,
        n_estimators=2500,
        min_data_in_leaf=30,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    lgb_pred = lgb_model.predict(X_val)
    lgb_mae = mean_absolute_error(y_val, lgb_pred)
    all_maes['lgb'].append(lgb_mae)
    test_preds_lgb += lgb_model.predict(X_test_scaled) / 5
    print(f"   LightGBM MAE: {lgb_mae:.2f}")

    # XGBoost
    xgb_model = XGBRegressor(
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        objective='reg:absoluteerror',
        random_state=42,
        n_jobs=-1,
        eval_metric='mae',
        early_stopping_rounds=100,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_pred = xgb_model.predict(X_val)
    xgb_mae = mean_absolute_error(y_val, xgb_pred)
    all_maes['xgb'].append(xgb_mae)
    test_preds_xgb += xgb_model.predict(X_test_scaled) / 5
    print(f"   XGBoost MAE: {xgb_mae:.2f}")

    # 自适应权重融合（基于验证MAE计算权重）
    maes = [cat_mae, lgb_mae, xgb_mae]
    # 反比例权重（MAE越低权重越高）
    inv_maes = [1 / max(m, 0.01) for m in maes]
    total = sum(inv_maes)
    weights = [w / total for w in inv_maes]

    ensemble_pred = (weights[0] * cat_pred +
                   weights[1] * lgb_pred +
                   weights[2] * xgb_pred)
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    all_maes['ensemble'].append(ensemble_mae)

    print(f"   自适应权重: Cat={weights[0]:.3f}, LGB={weights[1]:.3f}, XGB={weights[2]:.3f}")
    print(f"   Ensemble MAE: {ensemble_mae:.2f}")

# ==================== 最终结果 ====================
print("\n" + "="*60)
print("最终结果 (5折平均)")
print("="*60)
print(f"CatBoost MAE: {np.mean(all_maes['cat']):.2f}")
print(f"LightGBM MAE: {np.mean(all_maes['lgb']):.2f}")
print(f"XGBoost MAE: {np.mean(all_maes['xgb']):.2f}")
print(f"Ensemble MAE: {np.mean(all_maes['ensemble']):.2f}")

# 最终预测（自适应权重）
test_preds_final = (test_preds_cat +
                   test_preds_lgb +
                   test_preds_xgb) / 3

final_pred = np.maximum(test_preds_final, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': test['SaleID'].values,
    'price': final_pred
})
submit.to_csv('improved_restored_submit.csv', index=False)
print(f"\n结果已保存到: improved_restored_submit.csv")

final_mae = np.mean(all_maes['ensemble'])
print(f"\n" + "="*60)
if final_mae <= 450:
    print(f"🎉 成功！MAE: {final_mae:.2f} <= 450")
    print(f"比原始477改进: {477 - final_mae:.2f}")
else:
    print(f"⚠️ 距离目标: {final_mae - 450:.2f}")
    print(f"当前MAE: {final_mae:.2f}")
    print(f"比之前513.96改进: {513.96 - final_mae:.2f}")
    print(f"距离目标: {final_mae - 450:.2f}")
print("="*60)
