# -*- coding: utf-8 -*-
"""
快速版伪标签(Pseudo Labeling)训练
使用更简单的配置加速训练
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

def load_and_preprocess():
    """快速数据加载和预处理"""
    print("加载数据...")
    train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    
    # 合并处理
    train['is_train'] = 1
    test['is_train'] = 0
    test['price'] = -1
    
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    
    # 处理异常值
    combined['power'] = combined['power'].clip(0, 600)
    
    # 关键特征工程
    combined['v_0_sq'] = combined['v_0'] ** 2
    combined['v_3_sq'] = combined['v_3'] ** 2
    combined['v_0_v_3'] = combined['v_0'] * combined['v_3']
    
    # 分离数据
    train_data = combined[combined['is_train'] == 1].copy()
    test_data = combined[combined['is_train'] == 0].copy()
    
    # 只使用数值特征
    drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'model', 'price', 'is_train']
    feature_cols = [col for col in train_data.columns if col not in drop_cols]
    numeric_cols = train_data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X_train = train_data[numeric_cols].values.astype(np.float64)
    y_train = train_data['price'].values.astype(np.float64)
    X_test = test_data[numeric_cols].values.astype(np.float64)
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    return X_train, y_train, X_test, test_data['SaleID'].values

def train_model(X_train, y_train, X_test, model_type='catboost'):
    """训练单个模型"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        if model_type == 'catboost':
            model = cb.CatBoostRegressor(
                iterations=2000, learning_rate=0.05, depth=8,
                l2_leaf_reg=3, random_seed=42, verbose=0,
                early_stopping_rounds=100
            )
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        elif model_type == 'lightgbm':
            lgb_train = lgb.Dataset(X_tr, y_tr)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            model = lgb.train(
                {'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1,
                 'learning_rate': 0.05, 'num_leaves': 63},
                lgb_train, num_boost_round=2000, valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
        else:  # xgboost
            model = xgb.XGBRegressor(
                n_estimators=2000, learning_rate=0.05, max_depth=8,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                objective='reg:absoluteerror', early_stopping_rounds=100, verbosity=0
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        if model_type == 'lightgbm':
            oof_pred[val_idx] = model.predict(X_val)
            test_pred += model.predict(X_test) / 5
        else:
            oof_pred[val_idx] = model.predict(X_val)
            test_pred += model.predict(X_test) / 5
        
        print(f"  {model_type} Fold {fold+1} done")
    
    mae = mean_absolute_error(y_train, oof_pred)
    return oof_pred, test_pred, mae

def pseudo_label_iter(X_train, y_train, X_test, pseudo_labels, n_samples=5000):
    """伪标签迭代 - 选择最可靠的样本"""
    # 选择预测值最接近中位数的样本（更稳定）
    median_pred = np.median(pseudo_labels)
    distances = np.abs(pseudo_labels - median_pred)
    selected_idx = np.argsort(distances)[:n_samples]
    
    print(f"  选择 {len(selected_idx)} 个高置信度样本")
    
    # 创建增强数据集
    X_aug = np.vstack([X_train, X_test[selected_idx]])
    y_aug = np.concatenate([y_train, pseudo_labels[selected_idx]])
    
    return X_aug, y_aug

def main():
    print("=" * 60)
    print("快速版伪标签训练")
    print("=" * 60)
    
    X_train, y_train, X_test, sale_ids = load_and_preprocess()
    
    # ===== 第一阶段：基础模型 =====
    print("\n[阶段1] 训练基础模型...")
    
    cat_oof, cat_test, cat_mae = train_model(X_train, y_train, X_test, 'catboost')
    print(f"CatBoost MAE: {cat_mae:.2f}")
    
    lgb_oof, lgb_test, lgb_mae = train_model(X_train, y_train, X_test, 'lightgbm')
    print(f"LightGBM MAE: {lgb_mae:.2f}")
    
    xgb_oof, xgb_test, xgb_mae = train_model(X_train, y_train, X_test, 'xgboost')
    print(f"XGBoost MAE: {xgb_mae:.2f}")
    
    # 加权平均
    base_ensemble = 0.4 * cat_test + 0.35 * lgb_test + 0.25 * xgb_test
    base_oof = 0.4 * cat_oof + 0.35 * lgb_oof + 0.25 * xgb_oof
    base_mae = mean_absolute_error(y_train, base_oof)
    print(f"\n基础融合 MAE: {base_mae:.2f}")
    
    # ===== 第二阶段：伪标签迭代 =====
    print("\n[阶段2] 伪标签迭代...")
    
    best_mae = base_mae
    best_pred = base_ensemble.copy()
    
    for iteration in range(3):
        print(f"\n迭代 {iteration + 1}:")
        
        # 使用上一轮预测作为伪标签
        pseudo_labels = best_pred.copy()
        
        # 创建增强数据集
        n_samples = min(10000, len(X_test) // 5)  # 每轮添加样本数
        X_aug, y_aug = pseudo_label_iter(X_train, y_train, X_test, pseudo_labels, n_samples)
        
        # 重新训练模型
        cat_oof2, cat_test2, cat_mae2 = train_model(X_aug, y_aug, X_test, 'catboost')
        lgb_oof2, lgb_test2, lgb_mae2 = train_model(X_aug, y_aug, X_test, 'lightgbm')
        xgb_oof2, xgb_test2, xgb_mae2 = train_model(X_aug, y_aug, X_test, 'xgboost')
        
        # 融合
        iter_pred = 0.4 * cat_test2 + 0.35 * lgb_test2 + 0.25 * xgb_test2
        
        # 在原始训练集上评估（使用原始y_train）
        iter_oof = 0.4 * cat_oof2[:len(X_train)] + 0.35 * lgb_oof2[:len(X_train)] + 0.25 * xgb_oof2[:len(X_train)]
        iter_mae = mean_absolute_error(y_train, iter_oof)
        
        print(f"迭代 {iteration + 1} MAE: {iter_mae:.2f}")
        
        if iter_mae < best_mae:
            best_mae = iter_mae
            best_pred = iter_pred
            print(f"  -> 改进! 更新最佳预测")
        else:
            print(f"  -> 未改进，保持之前结果")
    
    # ===== 输出结果 =====
    print("\n" + "=" * 60)
    print(f"最终 MAE: {best_mae:.2f}")
    print(f"相比基础模型改进: {base_mae - best_mae:.2f}")
    
    if best_mae < 400:
        print("🎉 目标达成！MAE < 400")
    else:
        print(f"距离目标还差: {best_mae - 400:.2f}")
    
    # 保存预测
    best_pred = np.clip(best_pred, y_train.min() * 0.9, y_train.max() * 1.1)
    submission = pd.DataFrame({'SaleID': sale_ids, 'price': best_pred})
    submission.to_csv('pseudo_labeling_quick_submit.csv', index=False)
    print(f"\n结果已保存: pseudo_labeling_quick_submit.csv")

if __name__ == '__main__':
    main()
