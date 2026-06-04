# -*- coding: utf-8 -*-
"""
极简版伪标签 - 仅CatBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("极简版伪标签训练 (仅CatBoost)")
print("=" * 50)

# 加载数据
print("\n加载数据...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

# 预处理
train['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([train, test], ignore_index=True)
combined['power'] = combined['power'].clip(0, 600)
combined['v_0_v_3'] = combined['v_0'] * combined['v_3']
combined['v_0_sq'] = combined['v_0'] ** 2

train_data = combined[combined['is_train'] == 1]
test_data = combined[combined['is_train'] == 0]

# 数值特征
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'model', 'price', 'is_train']
feature_cols = [c for c in train_data.columns if c not in drop_cols]
num_cols = train_data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

X_train = train_data[num_cols].values.astype(np.float64)
y_train = train_data['price'].values.astype(np.float64)
X_test = test_data[num_cols].values.astype(np.float64)
sale_ids = test_data['SaleID'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 第一轮：基础训练
print("\n[第1轮] 基础CatBoost训练...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof1 = np.zeros(len(X_train))
pred1 = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"  Fold {fold+1}/5...", end=' ')
    model = cb.CatBoostRegressor(
        iterations=1000, learning_rate=0.1, depth=8,
        l2_leaf_reg=3, random_seed=42, verbose=0,
        early_stopping_rounds=50
    )
    model.fit(X_train[tr_idx], y_train[tr_idx], 
              eval_set=(X_train[val_idx], y_train[val_idx]), verbose=0)
    oof1[val_idx] = model.predict(X_train[val_idx])
    pred1 += model.predict(X_test) / 5
    print(f"MAE: {mean_absolute_error(y_train[val_idx], oof1[val_idx]):.2f}")

mae1 = mean_absolute_error(y_train, oof1)
print(f"\n基础MAE: {mae1:.2f}")

# 第二轮：伪标签增强
print("\n[第2轮] 伪标签增强...")
# 选择预测稳定的样本
median = np.median(pred1)
dist = np.abs(pred1 - median)
selected = np.argsort(dist)[:10000]  # 最稳定的10000个

X_aug = np.vstack([X_train, X_test[selected]])
y_aug = np.concatenate([y_train, pred1[selected]])

print(f"增强数据集: {X_aug.shape}")

oof2 = np.zeros(len(X_train))
pred2 = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"  Fold {fold+1}/5...", end=' ')
    model = cb.CatBoostRegressor(
        iterations=1000, learning_rate=0.1, depth=8,
        l2_leaf_reg=3, random_seed=42, verbose=0,
        early_stopping_rounds=50
    )
    model.fit(X_aug[tr_idx], y_aug[tr_idx],
              eval_set=(X_train[val_idx], y_train[val_idx]), verbose=0)
    oof2[val_idx] = model.predict(X_train[val_idx])
    pred2 += model.predict(X_test) / 5
    print(f"MAE: {mean_absolute_error(y_train[val_idx], oof2[val_idx]):.2f}")

mae2 = mean_absolute_error(y_train, oof2)
print(f"\n伪标签MAE: {mae2:.2f}")

# 选择最佳
if mae2 < mae1:
    final_pred, final_mae = pred2, mae2
    print(f"\n改进: {mae1 - mae2:.2f}")
else:
    final_pred, final_mae = pred1, mae1
    print("\n未改进，使用基础模型")

print("\n" + "=" * 50)
print(f"最终MAE: {final_mae:.2f}")

if final_mae < 400:
    print("🎉 目标达成！MAE < 400")
else:
    print(f"距离目标: {final_mae - 400:.2f}")

# 保存
final_pred = np.clip(final_pred, y_train.min() * 0.9, y_train.max() * 1.1)
pd.DataFrame({'SaleID': sale_ids, 'price': final_pred}).to_csv('pseudo_simple_submit.csv', index=False)
print(f"\n保存: pseudo_simple_submit.csv")