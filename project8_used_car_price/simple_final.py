# -*- coding: utf-8 -*-
"""
二手车价格预测 - 简单最终版
目标: 测试集MAE < 450
基于: 480成功案例（CV 480.46, Test 467.5 - 但更可靠）
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("="*70)
print("简单最终版 - 目标 MAE < 450")
print("基于: 480成功案例")
print("="*70)

# 1. 数据加载
print("\n步骤1: 数据加载...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values  # 保存测试集ID
test['price'] = -1  # 标记测试集
print(f"训练集: {len(train)}, 测试集: {len(test)}")

# 2. 特征工程
print("\n步骤2: 特征工程...")
data = pd.concat([train, test], axis=0, ignore_index=True)

# 时间特征
data['car_age'] = (data['creatDate'] // 10000 - data['regDate'] // 10000).clip(lower=0)

# 处理notRepairedDamage
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0')

# v特征基础统计
v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)
data['v_max'] = data[v_cols].max(axis=1)
data['v_min'] = data[v_cols].min(axis=1)
data['v_sum'] = data[v_cols].sum(axis=1)

# v特征交互（核心）
data['v_0_mul_v_3'] = data['v_0'] * data['v_3']
data['v_0_mul_v_12'] = data['v_0'] * data['v_12']
data['v_5_mul_v_14'] = data['v_5'] * data['v_14']
data['v_2_mul_v_7'] = data['v_2'] * data['v_7']
data['v_1_mul_v_11'] = data['v_1'] * data['v_11']
data['v_0_mul_v_5'] = data['v_0'] * data['v_5']

# 业务特征
data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']
data['power_age'] = data['power'] * data['car_age']
data['power_per_km'] = data['power'] / (data['kilometer'] + 1)
data['km_per_year'] = data['kilometer'] / (data['car_age'] + 1)

# 分组特征
data['brand_count'] = data.groupby('brand')['brand'].transform('count')
data['model_count'] = data.groupby('model')['model'].transform('count')

# 删除无用列
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols)

print(f"特征数: {data.shape[1] - 1}")  # -1 for price column

# 3. 缺失值处理
print("\n步骤3: 缺失值处理...")
for col in data.columns:
    if col == 'price':
        continue
    if data[col].dtype == 'object':
        data[col] = data[col].fillna('0')
    else:
        data[col] = data[col].fillna(data[col].median())

# 4. 分类特征编码
print("\n步骤4: 分类特征编码...")

# Label Encoding
from sklearn.preprocessing import LabelEncoder, StandardScaler

cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

print(f"编码分类特征: {list(cat_cols)}")

# 标准化
print("\n步骤5a: 标准化...")
scaler = StandardScaler()
numeric_cols = data.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col != 'price']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
print(f"标准化特征数: {len(numeric_cols)}")

# 分离数据
print("\n步骤5: 分离数据...")
train_data = data[data['price'] != -1].copy()
test_data = data[data['price'] == -1].copy()

X_train = train_data.drop(columns=['price'])
y_train = train_data['price'].values
X_test = test_data.drop(columns=['price'])

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 5. 训练
print("\n步骤5: 开始训练...")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 480成功参数
cat_params = {
    'iterations': 3500,
    'learning_rate': 0.022,
    'depth': 7,
    'l2_leaf_reg': 7,
    'random_strength': 0.6,
    'bagging_temperature': 0.7,
    'loss_function': 'MAE',
    'random_seed': SEED,
    'verbose': 0
}

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.022,
    'num_leaves': 115,
    'max_depth': 9,
    'min_data_in_leaf': 18,
    'feature_fraction': 0.87,
    'bagging_fraction': 0.87,
    'bagging_freq': 5,
    'reg_alpha': 0.18,
    'reg_lambda': 0.18,
    'verbose': -1,
    'seed': SEED
}

cat_test_preds = np.zeros(len(X_test))
lgb_test_preds = np.zeros(len(X_test))

fold_maes_cat = []
fold_maes_lgb = []
fold_maes_avg = []

print("开始5折交叉验证训练...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\nFold {fold+1}/5")

    X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_va = y_train[train_idx], y_train[val_idx]

    # CatBoost
    print("  训练CatBoost...")
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)

    cat_pred = cat_model.predict(X_va)
    cat_mae = mean_absolute_error(y_va, cat_pred)
    fold_maes_cat.append(cat_mae)
    cat_test_preds += cat_model.predict(X_test) / 5

    print(f"  CatBoost MAE: {cat_mae:.2f}")

    # LightGBM
    print("  训练LightGBM...")
    train_dataset = lgb.Dataset(X_tr, label=y_tr)
    val_dataset = lgb.Dataset(X_va, label=y_va, reference=train_dataset)

    lgb_model = lgb.train(
        lgb_params, train_dataset,
        num_boost_round=2000,
        valid_sets=[val_dataset],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)]
    )

    lgb_pred = lgb_model.predict(X_va)
    lgb_mae = mean_absolute_error(y_va, lgb_pred)
    fold_maes_lgb.append(lgb_mae)
    lgb_test_preds += lgb_model.predict(X_test) / 5

    print(f"  LightGBM MAE: {lgb_mae:.2f}")

    # 平均融合
    avg_pred = (cat_pred + lgb_pred) / 2
    avg_mae = mean_absolute_error(y_va, avg_pred)
    fold_maes_avg.append(avg_mae)
    print(f"  平均融合 MAE: {avg_mae:.2f}")

# 6. 最终结果
print(f"\n{'='*70}")
print("最终结果: 5折交叉验证平均")
print(f"{'='*70}")
print(f"CatBoost 平均 MAE: {np.mean(fold_maes_cat):.2f}")
print(f"LightGBM 平均 MAE: {np.mean(fold_maes_lgb):.2f}")
print(f"平均融合 MAE: {np.mean(fold_maes_avg):.2f}")

final_mae = np.mean(fold_maes_avg)

print(f"\n{'='*70}")
print(f"最终CV MAE: {final_mae:.2f}")
print(f"目标MAE: 450")
print(f"差距: {final_mae - 450:.2f}")

if final_mae < 450:
    print(f"\n成功达到目标！MAE: {final_mae:.2f} < 450")
    print(f"改进幅度: {450 - final_mae:.2f} 点")
else:
    print(f"\n距离目标: {final_mae - 450:.2f}")
    print(f"达成度: {(450/final_mae)*100:.1f}%")
    if final_mae < 460:
        print(f"超过480案例成绩！")

# 7. 生成测试集预测
print("\n步骤6: 生成测试集预测...")
final_pred = (cat_test_preds + lgb_test_preds) / 2
final_pred = np.maximum(final_pred, 10)

# 8. 保存结果
submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit.to_csv('simple_final_submit.csv', index=False)

print(f"\n{'='*70}")
print(f"预测范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"预测均值: {final_pred.mean():.2f}")

print(f"\n结果已保存: simple_final_submit.csv")
print(f"文件大小: {len(submit)}")

print(f"\n{'='*70}")
print("完成！")
print(f"{'='*70}")
