# -*- coding: utf-8 -*-
"""
二手车价格预测 - SOTA工业级优化完整版（快速）
目标: 测试集MAE < 450
策略: LightGBM + CatBoost + Stacking融合
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
SEED = 42
np.random.seed(SEED)

print("="*70)
print("SOTA工业级优化完整版 - 二手车价格预测")
print("目标: MAE < 450")
print("="*70)

# ==================== 1. 数据预处理 ====================
print("\n【步骤1】数据预处理...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

print(f"训练集大小: {len(train)}, 测试集大小: {len(test)}")

# ==================== 2. 特征工程 ====================
print("\n【步骤2】特征工程...")

# 2.1 基础时间特征
data['reg_year'] = data['regDate'] // 10000
data['reg_month'] = (data['regDate'] % 10000) // 100
data['creat_year'] = data['creatDate'] // 10000
data['creat_month'] = (data['creatDate'] % 10000) // 100
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)

# 2.2 异常值处理（Winsorize缩尾）
def winsorize_series(series, lower=1, upper=99):
    """缩尾处理：将异常值限制在指定分位数范围内"""
    lower_bound = np.percentile(series.dropna(), lower)
    upper_bound = np.percentile(series.dropna(), upper)
    return series.clip(lower=lower_bound, upper=upper_bound)

for col in ['power', 'kilometer', 'car_age']:
    data[col] = winsorize_series(data[col])

# 2.3 业务交叉特征（精度核心）
data['power_km'] = data['power'] * data['kilometer']  # 功率×里程
data['age_km'] = data['car_age'] * data['kilometer']  # 车龄×里程
data['power_age'] = data['power'] * data['car_age']  # 功率×车龄
data['usage_intensity'] = data['kilometer'] / (data['car_age'] + 1)  # 使用强度（年均里程）
data['power_per_km'] = data['power'] / (data['kilometer'] + 1)  # 每公里功率
data['km_per_year'] = data['kilometer'] / (data['car_age'] + 1)  # 年均里程

# 2.4 v特征统计和交互
v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)
data['v_max'] = data[v_cols].max(axis=1)
data['v_min'] = data[v_cols].min(axis=1)
data['v_range'] = data['v_max'] - data['v_min']
data['v_skew'] = data[v_cols].skew(axis=1)
data['v_kurt'] = data[v_cols].kurt(axis=1)

# v特征交叉（高阶特征）
data['v_0_v_3'] = data['v_0'] * data['v_3']
data['v_0_v_2'] = data['v_0'] * data['v_2']
data['v_0_v_12'] = data['v_0'] * data['v_12']
data['v_3_v_12'] = data['v_3'] * data['v_12']
data['v_1_v_5'] = data['v_1'] * data['v_5']
data['v_2_v_4'] = data['v_2'] * data['v_4']
data['v_6_v_7'] = data['v_6'] * data['v_7']

# 2.5 分类特征处理
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0.0')
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)

# 2.6 分组统计特征
for col in ['brand', 'model', 'regionCode', 'bodyType', 'fuelType']:
    data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')
    data[f'{col}_price_mean'] = data.groupby(col)['price'].transform('mean')

# 删除无用列
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

print(f"特征工程完成，总特征数: {data.shape[1]}")

# ==================== 3. 缺失值处理 ====================
print("\n【步骤3】缺失值处理...")

# 数值特征用中位数填充，分类特征用众数填充
numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=['object']).columns

for col in numeric_cols:
    median_val = data[col].median()
    data[col] = data[col].fillna(median_val)

for col in categorical_cols:
    mode_val = data[col].mode()[0] if not data[col].mode().empty else 'unknown'
    data[col] = data[col].fillna(mode_val)

print(f"数值特征: {len(numeric_cols)}, 分类特征: {len(categorical_cols)}")

# ==================== 4. 分类特征编码 ====================
print("\n【步骤4】分类特征编码...")

# Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le
    print(f"  Label编码完成: {col}")

# Target Encoding for high cardinality features (on train data only)
def target_encode(train_df, test_df, col, target='price', n_folds=5, noise_level=0.01):
    """Target编码：使用交叉验证避免过拟合"""
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[f'{col}_te'] = 0.0
    test_df[f'{col}_te'] = 0.0

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    for train_idx, val_idx in kf.split(train_df):
        target_mean = train_df.iloc[train_idx].groupby(col)[target].mean()
        train_df.loc[val_idx, f'{col}_te'] = train_df.loc[val_idx, col].map(target_mean)

    # 对测试集使用全量训练数据的统计
    target_mean = train_df.groupby(col)[target].mean()
    test_df[f'{col}_te'] = test_df[col].map(target_mean).fillna(train_df[target].mean())

    # 添加噪声避免过拟合
    noise = np.random.normal(0, noise_level, len(train_df))
    train_df[f'{col}_te'] = train_df[f'{col}_te'] + noise

    return train_df[f'{col}_te'], test_df[f'{col}_te']

# 分离训练和测试
train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
test_data = test_data.drop(columns=['price'])

# 对高基数分类特征进行Target编码
high_cardinality_cols = ['brand', 'model', 'regionCode']
for col in high_cardinality_cols:
    if col in train_data.columns:
        train_te, test_te = target_encode(train_data, test_data, col)
        train_data[f'{col}_te'] = train_te
        test_data[f'{col}_te'] = test_te
        print(f"  Target编码完成: {col}")

# ==================== 5. 特征标准化 ====================
print("\n【步骤5】特征标准化...")

X = train_data.drop(columns=['price'])
y = train_data['price'].values

# 数值特征标准化
scaler = StandardScaler()
numeric_cols_for_std = X.select_dtypes(include=[np.number]).columns
X[numeric_cols_for_std] = scaler.fit_transform(X[numeric_cols_for_std])
test_data[numeric_cols_for_std] = scaler.transform(test_data[numeric_cols_for_std])

print(f"标准化完成，特征数: {X.shape[1]}")

# ==================== 6. 模型参数（SOTA配置）====================
print("\n【步骤6】SOTA模型配置...")

# CatBoost最优参数
cat_params = {
    'iterations': 5000,
    'learning_rate': 0.018,
    'depth': 8,
    'l2_leaf_reg': 5,
    'random_strength': 0.8,
    'bagging_temperature': 0.6,
    'loss_function': 'MAE',
    'random_seed': SEED,
    'verbose': 0,
    'early_stopping_rounds': 150
}

# LightGBM最优参数
lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.018,
    'num_leaves': 127,
    'max_depth': 10,
    'min_data_in_leaf': 15,
    'feature_fraction': 0.90,
    'bagging_fraction': 0.90,
    'bagging_freq': 4,
    'reg_alpha': 0.20,
    'reg_lambda': 0.20,
    'min_split_gain': 0.005,
    'verbose': -1,
    'seed': SEED
}

print("CatBoost参数:", cat_params)
print("LightGBM参数:", lgb_params)

# ==================== 7. Stacking训练 ====================
print("\n【步骤7】Stacking融合训练...")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 存储基模型预测
cat_train_preds = np.zeros(len(X))
lgb_train_preds = np.zeros(len(X))
cat_test_preds = np.zeros(len(test_data))
lgb_test_preds = np.zeros(len(test_data))

fold_maes_cat = []
fold_maes_lgb = []
fold_maes_stack = []

print("开始5折交叉验证...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n{'='*70}")
    print(f"Fold {fold+1}/5")
    print(f"{'='*70}")

    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    # --- CatBoost ---
    print("训练CatBoost...")
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), verbose=0)

    cat_pred_val = cat_model.predict(X_val_fold)
    cat_pred_test = cat_model.predict(test_data)

    cat_mae = mean_absolute_error(y_val_fold, cat_pred_val)
    fold_maes_cat.append(cat_mae)
    cat_train_preds[val_idx] = cat_pred_val
    cat_test_preds += cat_pred_test / 5

    print(f"CatBoost MAE: {cat_mae:.2f}")

    # --- LightGBM ---
    print("训练LightGBM...")
    train_data_lgb = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data_lgb = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data_lgb)

    lgb_model = lgb.train(
        lgb_params, train_data_lgb,
        num_boost_round=5000,
        valid_sets=[val_data_lgb],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
    )

    lgb_pred_val = lgb_model.predict(X_val_fold)
    lgb_pred_test = lgb_model.predict(test_data)

    lgb_mae = mean_absolute_error(y_val_fold, lgb_pred_val)
    fold_maes_lgb.append(lgb_mae)
    lgb_train_preds[val_idx] = lgb_pred_val
    lgb_test_preds += lgb_pred_test / 5

    print(f"LightGBM MAE: {lgb_mae:.2f}")

    # --- Stacking（第二层：线性回归）---
    print("Stacking融合...")
    stack_X_train = np.column_stack([cat_train_preds[train_idx], lgb_train_preds[train_idx]])
    stack_X_val = np.column_stack([cat_train_preds[val_idx], lgb_train_preds[val_idx]])

    # 简单平均作为基线
    stack_pred_avg = (cat_pred_val + lgb_pred_val) / 2
    stack_mae_avg = mean_absolute_error(y_val_fold, stack_pred_avg)

    # 线性回归作为元学习器
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(stack_X_train, y_train_fold)
    stack_pred_lr = meta_model.predict(stack_X_val)
    stack_mae_lr = mean_absolute_error(y_val_fold, stack_pred_lr)
    fold_maes_stack.append(stack_mae_lr)

    print(f"Stacking (平均) MAE: {stack_mae_avg:.2f}")
    print(f"Stacking (LR) MAE: {stack_mae_lr:.2f}")
    print(f"最终权重: CatBoost={meta_model.coef_[0]:.4f}, LightGBM={meta_model.coef_[1]:.4f}")

# ==================== 8. 最终结果 ====================
print(f"\n{'='*70}")
print("【最终结果】5折交叉验证平均")
print(f"{'='*70}")
print(f"CatBoost 平均 MAE: {np.mean(fold_maes_cat):.2f} ± {np.std(fold_maes_cat):.2f}")
print(f"LightGBM 平均 MAE: {np.mean(fold_maes_lgb):.2f} ± {np.std(fold_maes_lgb):.2f}")
print(f"Stacking (线性回归) 平均 MAE: {np.mean(fold_maes_stack):.2f} ± {np.std(fold_maes_stack):.2f}")

# 计算RMSE和R²
final_stack_preds = (cat_train_preds + lgb_train_preds) / 2
rmse = np.sqrt(mean_squared_error(y, final_stack_preds))
r2 = r2_score(y, final_stack_preds)

print(f"\n附加指标:")
print(f"  RMSE: {rmse:.2f}")
print(f"  R² Score: {r2:.4f}")

# ==================== 9. 测试集预测 ====================
print("\n【步骤8】生成测试集预测...")

# 训练最终元学习器
final_stack_X = np.column_stack([cat_train_preds, lgb_train_preds])
final_meta_model = Ridge(alpha=1.0)
final_meta_model.fit(final_stack_X, y)

# 测试集预测
test_stack_X = np.column_stack([cat_test_preds, lgb_test_preds])
final_pred = final_meta_model.predict(test_stack_X)

# 价格裁剪（确保合理范围）
final_pred = np.maximum(final_pred, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': sale_ids,
    'price': final_pred
})
submit.to_csv('sota_fast_complete_submit.csv', index=False)
print(f"\n结果已保存到: sota_fast_complete_submit.csv")
print(f"预测价格范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"预测价格均值: {final_pred.mean():.2f}")

# ==================== 总结 ====================
print(f"\n{'='*70}")
print("【项目总结】")
print(f"{'='*70}")
final_mae = np.mean(fold_maes_stack)
print(f"最终MAE: {final_mae:.2f}")
print(f"目标MAE: 450")
print(f"差距: {final_mae - 450:.2f}")

if final_mae < 450:
    print(f"\n🎉 成功达到目标！MAE: {final_mae:.2f} < 450")
    print(f"改进幅度: {450 - final_mae:.2f} 点")
else:
    print(f"\n⚠️ 距离目标: {final_mae - 450:.2f}")
    print(f"达成度: {(450/final_mae)*100:.1f}%")

print(f"{'='*70}")
print("优化完成！")
