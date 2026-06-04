# -*- coding: utf-8 -*-
"""二手车价格预测 - 稳定简化版
策略：基模型 + 稳健特征工程 + 保守融合
目标：MAE <= 450
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
print("稳定简化版 - 基模型 + 稳健特征工程")
print("="*60)

# ==================== 数据加载 ====================
print("\n加载数据...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"原始训练集: {train.shape}")
print(f"原始测试集: {test.shape}")

# ==================== 数据清洗 ====================
print("\n数据清洗...")

# notRepairedDamage处理
for df in [train, test]:
    if 'notRepairedDamage' in df.columns and df['notRepairedDamage'].dtype == object:
        df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', '0.0')
        df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce').fillna(0)

# 删除price<=0的异常
price_before = len(train)
train = train[train['price'] > 0].copy()
price_after = len(train)
print(f"   删除price<=0的样本: {price_before - price_after} 个")

# Power保守截断
for df in [train, test]:
    df['power'] = df['power'].clip(lower=0, upper=600)

# 智能缺失值填充
def smart_fillna(df, fill_dict=None):
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if fill_dict and col in fill_dict:
                df[col].fillna(fill_dict[col], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    return df

# 计算训练集填充值
train_fill_dict = {}
for col in train.select_dtypes(include=[np.number]).columns:
    train_fill_dict[col] = train[col].median()

smart_fillna(train, train_fill_dict)
smart_fillna(test, train_fill_dict)

print("   数据清洗完成")

# ==================== 特征工程 ====================
print("\n特征工程...")

for df in [train, test]:
    # 车龄特征
    df['reg_year'] = df['regDate'] // 10000
    df['creat_year'] = df['creatDate'] // 10000
    df['car_age'] = (df['creat_year'] - df['reg_year']).clip(lower=0)

    # v特征（只使用最关键的交互）
    v_cols = ['v_0', 'v_3', 'v_5']
    available_v_cols = [col for col in v_cols if col in df.columns]
    if len(available_v_cols) >= 2:
        df['v_0_v_3'] = df['v_0'] * df['v_3']
        df['v_5_power'] = df['v_5'] * df['power']

    # Power特征
    df['log_power'] = np.log1p(df['power'])
    df['power_km'] = df['power'] * df['kilometer']

    # Kilometer特征
    df['log_km'] = np.log1p(df['kilometer'])

    # 分类特征编码
    categorical_cols = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))

print(f"   特征数: {train.shape[1]-1}")

# ==================== 数据准备 ====================
print("\n数据准备...")

# 删除不使用的列
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType', 'notRepairedDamage', 'price']
# 保留编码后的分类列
drop_cols.extend(['brand', 'model', 'bodyType', 'fuelType', 'gearbox'])

X = train.drop(columns=[c for c in drop_cols if c in train.columns])
X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])
y = train['price']  # 直接预测原始价格

numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
X = X[numeric_cols]
X_test = X_test[[col for col in numeric_cols if col in X_test.columns]]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print(f"   准备完成，特征数: {X_scaled.shape[1]}")

# ==================== 模型训练（5折交叉验证） ====================
print("\n模型训练 (5折交叉验证)...")
print("="*60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 固定超参数（基于之前最优结果）
model_params = {
    'cat': {
        'iterations': 2500,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 10,
        'loss_function': 'MAE',
        'random_seed': 42,
        'verbose': False
    },
    'lgb': {
        'num_leaves': 63,
        'max_depth': 6,
        'learning_rate': 0.03,
        'n_estimators': 2500,
        'min_data_in_leaf': 30,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    },
    'xgb': {
        'n_estimators': 2500,
        'learning_rate': 0.03,
        'max_depth': 6,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'objective': 'reg:absoluteerror',
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mae',
        'verbosity': 0
    }
}

# 存储预测
all_maes = {'cat': [], 'lgb': [], 'xgb': [], 'ensemble': []}
test_preds_cat = np.zeros(len(test))
test_preds_lgb = np.zeros(len(test))
test_preds_xgb = np.zeros(len(test))

print("\n开始训练...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # CatBoost
    cat_model = CatBoostRegressor(**model_params['cat'])
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict(X_val)
    cat_mae = mean_absolute_error(y_val, cat_pred)
    all_maes['cat'].append(cat_mae)
    test_preds_cat += cat_model.predict(X_test_scaled) / 5
    print(f"   CatBoost MAE: {cat_mae:.2f}")

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**model_params['lgb'])
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_val)
    lgb_mae = mean_absolute_error(y_val, lgb_pred)
    all_maes['lgb'].append(lgb_mae)
    test_preds_lgb += lgb_model.predict(X_test_scaled) / 5
    print(f"   LightGBM MAE: {lgb_mae:.2f}")

    # XGBoost
    xgb_model = XGBRegressor(**model_params['xgb'])
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_pred = xgb_model.predict(X_val)
    xgb_mae = mean_absolute_error(y_val, xgb_pred)
    all_maes['xgb'].append(xgb_mae)
    test_preds_xgb += xgb_model.predict(X_test_scaled) / 5
    print(f"   XGBoost MAE: {xgb_mae:.2f}")

    # 融合
    ensemble_pred = (cat_pred + lgb_pred + xgb_pred) / 3
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    all_maes['ensemble'].append(ensemble_mae)
    print(f"   Ensemble MAE: {ensemble_mae:.2f}")

# ==================== 最终结果 ====================
print("\n最终结果 (5折平均)")
print("="*60)

print(f"CatBoost MAE: {np.mean(all_maes['cat']):.2f}")
print(f"LightGBM MAE: {np.mean(all_maes['lgb']):.2f}")
print(f"XGBoost MAE: {np.mean(all_maes['xgb']):.2f}")
print(f"Ensemble MAE: {np.mean(all_maes['ensemble']):.2f}")

# 最终预测
final_pred = (test_preds_cat + test_preds_lgb + test_preds_xgb) / 3
final_pred = np.maximum(final_pred, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': test['SaleID'].values,
    'price': final_pred
})
submit.to_csv('stable_simplified_submit.csv', index=False)
print(f"\n结果已保存到: stable_simplified_submit.csv")

# 最终总结
final_mae = np.mean(all_maes['ensemble'])
print(f"\n" + "="*60)
print(f"最终MAE: {final_mae:.2f}")
print(f"目标MAE: 450")
print(f"距离目标: {final_mae - 450:.2f}")

if final_mae <= 450:
    print(f"✅ 成功！MAE: {final_mae:.2f} <= 450")
else:
    print(f"⚠️ 距离目标: {final_mae - 450:.2f}")

print("="*60)
