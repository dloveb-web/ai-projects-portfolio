# -*- coding: utf-8 -*-
"""
极速版伪标签训练 - 最小化配置
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("极速版伪标签训练")
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

print(f"训练集: {X_train.shape}")

# 5折交叉验证训练
def train_ensemble(X_tr, y_tr, X_te):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cat_pred = np.zeros(len(X_te))
    lgb_pred = np.zeros(len(X_te))
    xgb_pred = np.zeros(len(X_te))
    oof_cat = np.zeros(len(X_tr))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
        print(f"Fold {fold+1}/5...", end=' ')
        X_t, X_v = X_tr[tr_idx], X_tr[val_idx]
        y_t, y_v = y_tr[tr_idx], y_tr[val_idx]
        
        # CatBoost (主要模型)
        cat = cb.CatBoostRegressor(iterations=1500, learning_rate=0.08, depth=8,
                                   l2_leaf_reg=3, random_seed=42, verbose=0,
                                   early_stopping_rounds=80)
        cat.fit(X_t, y_t, eval_set=(X_v, y_v), verbose=0)
        oof_cat[val_idx] = cat.predict(X_v)
        cat_pred += cat.predict(X_te) / 5
        
        # LightGBM
        lgb_train = lgb.Dataset(X_t, y_t)
        lgb_val = lgb.Dataset(X_v, y_v)
        lgb_model = lgb.train(
            {'objective': 'regression_l1', 'verbosity': -1, 'learning_rate': 0.08, 'num_leaves': 63},
            lgb_train, num_boost_round=1500, valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)]
        )
        lgb_pred += lgb_model.predict(X_te) / 5
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.08, max_depth=8,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42,
                                     objective='reg:absoluteerror', early_stopping_rounds=80, verbosity=0)
        xgb_model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        xgb_pred += xgb_model.predict(X_te) / 5
        
        print(f"Fold MAE: {mean_absolute_error(y_v, cat.predict(X_v)):.2f}")
    
    mae = mean_absolute_error(y_tr, oof_cat)
    ensemble = 0.5 * cat_pred + 0.3 * lgb_pred + 0.2 * xgb_pred
    return ensemble, mae

# 第一轮：基础训练
print("\n[第1轮] 基础训练...")
pred1, mae1 = train_ensemble(X_train, y_train, X_test)
print(f"基础MAE: {mae1:.2f}")

# 第二轮：伪标签
print("\n[第2轮] 伪标签增强...")
# 选择预测值稳定区域的样本
median_p = np.median(pred1)
distances = np.abs(pred1 - median_p)
selected = np.argsort(distances)[:8000]  # 选择最稳定的8000个样本

X_aug = np.vstack([X_train, X_test[selected]])
y_aug = np.concatenate([y_train, pred1[selected]])

pred2, mae2 = train_ensemble(X_aug, y_aug, X_test)
print(f"伪标签MAE: {mae2:.2f}")

# 选择最佳
if mae2 < mae1:
    print(f"\n改进: {mae1 - mae2:.2f}")
    final_pred = pred2
    final_mae = mae2
else:
    print(f"\n未改进，使用基础模型")
    final_pred = pred1
    final_mae = mae1

print("\n" + "=" * 50)
print(f"最终MAE: {final_mae:.2f}")

if final_mae < 400:
    print("🎉 目标达成！MAE < 400")
else:
    print(f"距离目标: {final_mae - 400:.2f}")

# 保存
final_pred = np.clip(final_pred, y_train.min() * 0.9, y_train.max() * 1.1)
pd.DataFrame({'SaleID': sale_ids, 'price': final_pred}).to_csv('pseudo_quick_submit.csv', index=False)
print(f"保存: pseudo_quick_submit.csv")
