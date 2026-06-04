# -*- coding: utf-8 -*-
"""
二手车价格预测 - 优化v11版（修复预测偏差）
目标: 测试集MAE < 450
问题: 全部模型严重过拟合 + 预测偏低5.5倍
策略: 修复预测分布 + 极简特征 + 不标准化
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("="*70)
print("优化v11版 - 修复预测偏差")
print("问题: 预测偏低5.5倍 (5900 vs 1100)")
print("策略: 修复分布 + 极简特征 + 不标准化")
print("="*70)

# ==================== 1. 数据加载 ====================
print("\n【步骤1】数据加载...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"训练集: {len(train)}, 测试集: {len(test)}")

# 计算训练集价格统计
train_price_mean = train['price'].mean()
train_price_median = train['price'].median()

print(f"\n训练集价格统计:")
print(f"  Mean: {train_price_mean:.2f}")
print(f"  Median: {train_price_median:.2f}")

# ==================== 2. 极简特征工程（不标准化）====================
print("\n【步骤2】极简特征工程（不标准化）...")

sale_ids = test['SaleID'].values
test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

# 只保留核心特征
data['car_age'] = (data['creatDate'] // 10000 - data['regDate'] // 10000).clip(lower=0)

# 分类处理（确保是字符串）
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0')
# 先转换为数值，再转字符串分类
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0).astype(int).astype(str)

# 删除无用列和v特征（可能导致问题）
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
drop_cols += [f'v_{i}' for i in range(15)]

# 保留核心特征
keep_cols = ['price', 'power', 'kilometer', 'car_age',
             'brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']

data = data[keep_cols]

print(f"特征工程完成，特征数: {data.shape[1]}")
print(f"保留特征: {list(data.columns)}")

# ==================== 3. 缺失值处理 ====================
print("\n【步骤3】缺失值处理...")

numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=['object']).columns

for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

for col in categorical_cols:
    mode_val = data[col].mode()[0] if not data[col].mode().empty else 'unknown'
    data[col] = data[col].fillna(mode_val)

print(f"数值特征: {len(numeric_cols)}, 分类特征: {len(categorical_cols)}")

# ==================== 4. 不做Label Encoding（CatBoost自动处理）====================
print("\n【步骤4】不做编码（让CatBoost自动处理分类特征）...")

# 分离数据
train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)

if 'price' in test_data.columns:
    test_data = test_data.drop(columns=['price'])

X = train_data.drop(columns=['price'])
y = train_data['price'].values

# 标记分类列
cat_features = [col for col in X.columns if X[col].dtype == 'object' or X[col].nunique() < 50]

print(f"最终特征数: {X.shape[1]}")
print(f"分类特征: {cat_features}")

# ==================== 5. 训练 ====================
print("\n【步骤5】训练（CatBoost自动处理分类）...")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

cat_params = {
    'iterations': 1500,
    'learning_rate': 0.08,
    'depth': 6,
    'l2_leaf_reg': 8,
    'random_strength': 0.7,
    'bagging_temperature': 0.7,
    'loss_function': 'MAE',
    'random_seed': SEED,
    'verbose': 0,
    'early_stopping_rounds': 80
    # 不指定cat_features，让CatBoost自动处理
}

cat_test_preds = np.zeros(len(test_data))
fold_maes = []

print("开始5折交叉验证训练...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # CatBoost
    print("  训练CatBoost...")
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)

    cat_pred_val = cat_model.predict(X_val)
    cat_mae = mean_absolute_error(y_val, cat_pred_val)
    fold_maes.append(cat_mae)

    if fold == 0:
        cat_test_preds += cat_model.predict(test_data) / 5

    print(f"  CatBoost MAE: {cat_mae:.2f}")
    print(f"  调试 - 预测均值: {cat_pred_val.mean():.2f}, 真实均值: {y_val.mean():.2f}")

# ==================== 6. 最终结果 ====================
print(f"\n{'='*70}")
print("【最终结果】5折交叉验证平均")
print(f"{'='*70}")
print(f"CatBoost 平均 MAE: {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f}")

final_mae = np.mean(fold_maes)

print(f"\n{'='*70}")
print(f"最终CV MAE: {final_mae:.2f}")
print(f"目标MAE: 450")
print(f"差距: {final_mae - 450:.2f}")

# ==================== 7. 生成测试集预测 ====================
print("\n【步骤7】生成测试集预测...")

final_pred = cat_test_preds

# 调试：检查预测分布
print(f"\n预测分布诊断:")
print(f"  预测均值: {final_pred.mean():.2f}")
print(f"  训练集均值: {train_price_mean:.2f}")
print(f"  比例: {final_pred.mean()/train_price_mean:.2f}")

# 修复：调整到训练集分布
scale_factor = train_price_mean / final_pred.mean()
print(f"\n应用分布调整系数: {scale_factor:.2f}")
final_pred_adjusted = final_pred * scale_factor

# 确保不低于50
final_pred_adjusted = np.maximum(final_pred_adjusted, 50)

print(f"调整后预测均值: {final_pred_adjusted.mean():.2f}")

# 保存结果
submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred_adjusted})
submit.to_csv('optimized_v11_submit.csv', index=False)

print(f"\n结果已保存: optimized_v11_submit.csv")
print(f"预测价格范围: [{final_pred_adjusted.min():.2f}, {final_pred_adjusted.max():.2f}]")
print(f"预测价格均值: {final_pred_adjusted.mean():.2f}")

print(f"\n{'='*70}")
print("✅ 优化v11版（修复预测偏差）完成！")
print(f"{'='*70}")
