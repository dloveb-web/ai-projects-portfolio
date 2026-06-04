# -*- coding: utf-8 -*-
"""
伪标签训练 - 基于最优参数配置
使用之前调优过的最佳参数
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 之前调优的最佳参数
BEST_CATBOOST_PARAMS = {
    'learning_rate': 0.022,
    'depth': 6,
    'l2_leaf_reg': 4,
    'min_data_in_leaf': 98,
    'rsm': 0.606,
    'random_strength': 0.545,
    'bagging_temperature': 0.183,
}

print("=" * 60)
print("伪标签训练 - 最优参数版")
print("=" * 60)

# ==================== 数据加载 ====================
print("\n加载数据...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

# ==================== 数据预处理 ====================
print("数据预处理...")

# 合并处理
train['is_train'] = 1
test['is_train'] = 0
test['price'] = -1

combined = pd.concat([train, test], axis=0, ignore_index=True)

# 处理异常值
combined['power'] = combined['power'].clip(0, 600)
combined['kilometer'] = combined['kilometer'].clip(0, 50)

# 删除噪声特征
noise_cols = ['seller', 'offerType']
for col in noise_cols:
    if col in combined.columns:
        combined = combined.drop(columns=[col])

# ==================== 特征工程 ====================
print("特征工程...")

# v特征列表
v_cols = ['v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_11', 'v_14']

# 统计特征
combined['v_mean'] = combined[v_cols].mean(axis=1)
combined['v_std'] = combined[v_cols].std(axis=1)
combined['v_max'] = combined[v_cols].max(axis=1)
combined['v_min'] = combined[v_cols].min(axis=1)
combined['v_range'] = combined['v_max'] - combined['v_min']

# 关键特征交互
combined['v_0_sq'] = combined['v_0'] ** 2
combined['v_3_sq'] = combined['v_3'] ** 2
combined['v_0_v_3'] = combined['v_0'] * combined['v_3']
combined['v_0_v_2'] = combined['v_0'] * combined['v_2']
combined['v_0_v_2_v_3'] = combined['v_0'] * combined['v_2'] * combined['v_3']

# 与power/kilometer交叉
combined['v_0_power'] = combined['v_0'] * combined['power']
combined['v_3_power'] = combined['v_3'] * combined['power']
combined['v_0_kilometer'] = combined['v_0'] * combined['kilometer']

# 使用年限特征 - 从regDate和creatDate提取
combined['regYear'] = (combined['regDate'] // 10000).astype(int)
combined['regMonth'] = ((combined['regDate'] % 10000) // 100).astype(int)
combined['creatYear'] = (combined['creatDate'] // 10000).astype(int)
combined['car_age'] = combined['creatYear'] - combined['regYear'] + combined['regMonth'] / 12
combined['used_days'] = (combined['creatDate'] - combined['regDate']) / 10000

# ==================== K-Fold目标编码 ====================
print("K-Fold目标编码...")

train_mask = combined['is_train'] == 1
test_mask = combined['is_train'] == 0
train_idx = combined[train_mask].index.tolist()
test_idx = combined[test_mask].index.tolist()

# brand目标编码
kf = KFold(n_splits=5, shuffle=True, random_state=42)
brand_te = np.zeros(len(combined))
y_all = combined.loc[train_idx, 'price'].values

for fold_tr_idx, fold_val_idx in kf.split(train_idx):
    tr_indices = [train_idx[i] for i in fold_tr_idx]
    val_indices = [train_idx[i] for i in fold_val_idx]
    
    brand_mean = combined.loc[tr_indices].groupby('brand')['price'].mean()
    brand_te[val_indices] = combined.loc[val_indices, 'brand'].map(brand_mean).values

# 测试集使用全局均值
global_brand_mean = combined.loc[train_idx].groupby('brand')['price'].mean()
brand_te[test_idx] = combined.loc[test_idx, 'brand'].map(global_brand_mean).values
combined['brand_te'] = brand_te

# model目标编码
model_te = np.zeros(len(combined))
for fold_tr_idx, fold_val_idx in kf.split(train_idx):
    tr_indices = [train_idx[i] for i in fold_tr_idx]
    val_indices = [train_idx[i] for i in fold_val_idx]
    
    model_mean = combined.loc[tr_indices].groupby('model')['price'].mean()
    model_te[val_indices] = combined.loc[val_indices, 'model'].map(model_mean).values

global_model_mean = combined.loc[train_idx].groupby('model')['price'].mean()
model_te[test_idx] = combined.loc[test_idx, 'model'].map(global_model_mean).values
combined['model_te'] = model_te

# ==================== 准备训练数据 ====================
print("准备训练数据...")

train_data = combined[train_mask].copy()
test_data = combined[test_mask].copy()

# 特征列
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'model', 'price', 'is_train', 'regYear', 'regMonth', 'creatYear']
feature_cols = [col for col in train_data.columns if col not in drop_cols]

# 只保留数值特征
numeric_cols = train_data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print(f"特征数量: {len(numeric_cols)}")

X_train_full = train_data[numeric_cols].values.astype(np.float64)
y_train_full = train_data['price'].values.astype(np.float64)
X_test = test_data[numeric_cols].values.astype(np.float64)
sale_ids = test_data['SaleID'].values

# 处理无穷值和缺失值
X_train_full = np.nan_to_num(X_train_full, nan=np.nanmedian(X_train_full))
X_test = np.nan_to_num(X_test, nan=np.nanmedian(X_test))

# 标准化
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

print(f"训练集: {X_train_full.shape}, 测试集: {X_test.shape}")

# ==================== 模型训练函数 ====================
def train_ensemble(X_tr, y_tr, X_te, params_cat=None):
    """训练三模型融合"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cat_oof = np.zeros(len(X_tr))
    lgb_oof = np.zeros(len(X_tr))
    xgb_oof = np.zeros(len(X_tr))
    cat_test = np.zeros(len(X_te))
    lgb_test = np.zeros(len(X_te))
    xgb_test = np.zeros(len(X_te))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
        print(f"  Fold {fold+1}/5...", end=' ')
        X_t, X_v = X_tr[tr_idx], X_tr[val_idx]
        y_t, y_v = y_tr[tr_idx], y_tr[val_idx]
        
        # CatBoost - 使用最优参数
        cat_params = {
            'iterations': 3000,
            'learning_rate': params_cat.get('learning_rate', 0.022) if params_cat else 0.022,
            'depth': params_cat.get('depth', 6) if params_cat else 6,
            'l2_leaf_reg': params_cat.get('l2_leaf_reg', 4) if params_cat else 4,
            'min_data_in_leaf': params_cat.get('min_data_in_leaf', 98) if params_cat else 98,
            'rsm': params_cat.get('rsm', 0.606) if params_cat else 0.606,
            'random_seed': 42,
            'od_type': 'Iter',
            'od_wait': 50,
            'verbose': 0,
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
        }
        
        cat = cb.CatBoostRegressor(**cat_params)
        cat.fit(X_t, y_t, eval_set=(X_v, y_v), verbose=0)
        cat_oof[val_idx] = cat.predict(X_v)
        cat_test += cat.predict(X_te) / 5
        
        # LightGBM
        lgb_train = lgb.Dataset(X_t, y_t)
        lgb_val = lgb.Dataset(X_v, y_v, reference=lgb_train)
        lgb_model = lgb.train(
            {'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1,
             'learning_rate': 0.02, 'num_leaves': 31, 'feature_fraction': 0.7,
             'bagging_fraction': 0.7, 'bagging_freq': 5, 'seed': 42,
             'min_data_in_leaf': 50},
            lgb_train, num_boost_round=3000, valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        lgb_oof[val_idx] = lgb_model.predict(X_v)
        lgb_test += lgb_model.predict(X_te) / 5
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.02, max_depth=6,
            subsample=0.7, colsample_bytree=0.7, random_state=42,
            objective='reg:absoluteerror', eval_metric='mae',
            early_stopping_rounds=50, verbosity=0, min_child_weight=50
        )
        xgb_model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        xgb_oof[val_idx] = xgb_model.predict(X_v)
        xgb_test += xgb_model.predict(X_te) / 5
        
        # fold MAE
        fold_mae = mean_absolute_error(y_v, cat.predict(X_v))
        print(f"MAE: {fold_mae:.2f}")
    
    cat_mae = mean_absolute_error(y_tr, cat_oof)
    lgb_mae = mean_absolute_error(y_tr, lgb_oof)
    xgb_mae = mean_absolute_error(y_tr, xgb_oof)
    
    print(f"\n  CatBoost: {cat_mae:.2f}, LightGBM: {lgb_mae:.2f}, XGBoost: {xgb_mae:.2f}")
    
    # 加权平均
    ensemble_oof = 0.5 * cat_oof + 0.3 * lgb_oof + 0.2 * xgb_oof
    ensemble_test = 0.5 * cat_test + 0.3 * lgb_test + 0.2 * xgb_test
    ensemble_mae = mean_absolute_error(y_tr, ensemble_oof)
    
    return ensemble_oof, ensemble_test, ensemble_mae

# ==================== 第一轮：基础训练 ====================
print("\n" + "="*60)
print("[第1轮] 基础模型训练...")
print("="*60)

oof1, pred1, mae1 = train_ensemble(X_train_full, y_train_full, X_test, BEST_CATBOOST_PARAMS)
print(f"\n基础模型MAE: {mae1:.2f}")

# ==================== 第二轮：伪标签增强 ====================
print("\n" + "="*60)
print("[第2轮] 伪标签增强...")
print("="*60)

# 选择预测稳定的样本（接近中位数的样本）
median_pred = np.median(pred1)
distances = np.abs(pred1 - median_pred)
selected_idx = np.argsort(distances)[:15000]  # 选择最稳定的15000个样本

print(f"选择 {len(selected_idx)} 个高置信度样本")

# 创建增强数据集
X_aug = np.vstack([X_train_full, X_test[selected_idx]])
y_aug = np.concatenate([y_train_full, pred1[selected_idx]])

print(f"增强数据集: {X_aug.shape}")

oof2, pred2, mae2 = train_ensemble(X_aug, y_aug, X_test, BEST_CATBOOST_PARAMS)
print(f"\n伪标签MAE: {mae2:.2f}")

# ==================== 第三轮：更多伪标签 ====================
print("\n" + "="*60)
print("[第3轮] 更多伪标签...")
print("="*60)

# 使用第二轮预测选择更多样本
median_pred2 = np.median(pred2)
distances2 = np.abs(pred2 - median_pred2)
selected_idx2 = np.argsort(distances2)[:20000]  # 扩展到20000

X_aug2 = np.vstack([X_train_full, X_test[selected_idx2]])
y_aug2 = np.concatenate([y_train_full, pred2[selected_idx2]])

print(f"增强数据集: {X_aug2.shape}")

oof3, pred3, mae3 = train_ensemble(X_aug2, y_aug2, X_test, BEST_CATBOOST_PARAMS)
print(f"\n伪标签2 MAE: {mae3:.2f}")

# ==================== 选择最佳结果 ====================
print("\n" + "="*60)
print("结果汇总")
print("="*60)

results = [
    ("基础模型", mae1, pred1),
    ("伪标签1", mae2, pred2),
    ("伪标签2", mae3, pred3),
]

for name, mae, _ in results:
    print(f"  {name}: MAE = {mae:.2f}")

best_idx = np.argmin([r[1] for r in results])
best_name, best_mae, best_pred = results[best_idx]

print(f"\n最佳结果: {best_name}, MAE = {best_mae:.2f}")

if best_mae < 400:
    print("🎉 目标达成！MAE < 400")
else:
    print(f"距离目标: {best_mae - 400:.2f}")

# ==================== 保存结果 ====================
best_pred = np.clip(best_pred, y_train_full.min() * 0.9, y_train_full.max() * 1.1)
submission = pd.DataFrame({'SaleID': sale_ids, 'price': best_pred})
submission.to_csv('pseudo_labeling_optimized_submit.csv', index=False)
print(f"\n结果已保存: pseudo_labeling_optimized_submit.csv")
