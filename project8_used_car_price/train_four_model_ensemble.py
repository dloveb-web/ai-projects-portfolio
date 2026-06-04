# -*- coding: utf-8 -*-
"""
二手车价格预测 - 四模型融合 (CatBoost + LightGBM + XGBoost + PyTorch深度残差网络)
目标：MAE < 450
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import joblib
import warnings
warnings.filterwarnings('ignore')


# ==================== 数据加载 ====================
def load_processed_data():
    """加载预处理后的数据"""
    print("正在加载预处理后的数据...")
    X_train = joblib.load('processed_data/X_train.joblib')
    X_val = joblib.load('processed_data/X_val.joblib')
    y_train = joblib.load('processed_data/y_train.joblib')
    y_val = joblib.load('processed_data/y_val.joblib')
    test_data = joblib.load('processed_data/test_data.joblib')
    sale_ids = joblib.load('processed_data/sale_ids.joblib')
    return X_train, X_val, y_train, y_val, test_data, sale_ids


# ==================== 噪声特征删除 ====================
NOISE_FEATURES = ['seller', 'offerType', 'SaleID', 'name', 
                   'creat_day', 'creat_year', 'creat_month', 'reg_day', 'reg_month']


# ==================== K-Fold目标编码 ====================
def kfold_target_encode(X_train, y_train, X_val, col, n_splits=5, smoothing=10.0):
    """K-Fold目标编码，避免数据泄露"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, index=X_train.index)
    
    train_encoded = np.zeros(len(X_train))
    
    for train_idx, val_idx in kf.split(X_train):
        fold_train_col = X_train.iloc[train_idx][col]
        fold_y = y_train.iloc[train_idx]
        target_mean = fold_y.mean()
        category_means = fold_y.groupby(fold_train_col).mean()
        category_counts = fold_train_col.groupby(fold_train_col).count()
        
        smoothed_mean = (category_means * category_counts + target_mean * smoothing) / (category_counts + smoothing)
        val_data = X_train.iloc[val_idx][col]
        train_encoded[val_idx] = val_data.map(smoothed_mean).fillna(target_mean)
    
    target_mean = y_train.mean()
    category_means = y_train.groupby(X_train[col]).mean()
    category_counts = X_train[col].groupby(X_train[col]).count()
    smoothed_mean = (category_means * category_counts + target_mean * smoothing) / (category_counts + smoothing)
    val_encoded = X_val[col].map(smoothed_mean).fillna(target_mean)
    
    return train_encoded, val_encoded


def kfold_group_stats(X_train, y_train, X_val, group_col, n_splits=5):
    """K-Fold分组统计，避免数据泄露"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, index=X_train.index)
    
    train_mean = np.zeros(len(X_train))
    train_median = np.zeros(len(X_train))
    train_std = np.zeros(len(X_train))
    
    for train_idx, val_idx in kf.split(X_train):
        fold_group = X_train.iloc[train_idx][group_col]
        fold_y = y_train.iloc[train_idx]
        
        group_stats = fold_y.groupby(fold_group).agg(['mean', 'median', 'std'])
        
        val_groups = X_train.iloc[val_idx][group_col]
        train_mean[val_idx] = val_groups.map(group_stats['mean']).fillna(fold_y.mean())
        train_median[val_idx] = val_groups.map(group_stats['median']).fillna(fold_y.median())
        train_std[val_idx] = val_groups.map(group_stats['std']).fillna(0)
    
    group_stats = y_train.groupby(X_train[group_col]).agg(['mean', 'median', 'std'])
    val_mean = X_val[group_col].map(group_stats['mean']).fillna(y_train.mean())
    val_median = X_val[group_col].map(group_stats['median']).fillna(y_train.median())
    val_std = X_val[group_col].map(group_stats['std']).fillna(0)
    
    return (train_mean, train_median, train_std), (val_mean, val_median, val_std)


# ==================== 特征工程 ====================
def feature_engineering(X_train, y_train, X_val, test_data):
    """特征工程"""
    print("\n" + "="*50)
    print("开始特征工程...")
    print("="*50)
    
    # 1. 删除噪声特征
    noise_cols = [col for col in NOISE_FEATURES if col in X_train.columns]
    if noise_cols:
        print(f"删除噪声特征: {noise_cols}")
        X_train = X_train.drop(columns=noise_cols)
        X_val = X_val.drop(columns=noise_cols)
        test_data = test_data.drop(columns=noise_cols)
    
    # 2. 删除高相关特征
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = X_train[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.85)]
        if high_corr_features:
            print(f"删除高相关特征: {high_corr_features}")
            X_train = X_train.drop(columns=high_corr_features)
            X_val = X_val.drop(columns=high_corr_features)
            test_data = test_data.drop(columns=high_corr_features)
    
    # 3. v特征统计
    v_cols = [col for col in X_train.columns if col.startswith('v_') and col not in ['v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_12', 'v_13']]
    
    if len(v_cols) >= 3:
        X_train['v_mean'] = X_train[v_cols].mean(axis=1)
        X_val['v_mean'] = X_val[v_cols].mean(axis=1)
        test_data['v_mean'] = test_data[v_cols].mean(axis=1)
        
        X_train['v_std'] = X_train[v_cols].std(axis=1)
        X_val['v_std'] = X_val[v_cols].std(axis=1)
        test_data['v_std'] = test_data[v_cols].std(axis=1)
        
        X_train['v_max'] = X_train[v_cols].max(axis=1)
        X_val['v_max'] = X_val[v_cols].max(axis=1)
        test_data['v_max'] = test_data[v_cols].max(axis=1)
        
        X_train['v_min'] = X_train[v_cols].min(axis=1)
        X_val['v_min'] = X_val[v_cols].min(axis=1)
        test_data['v_min'] = test_data[v_cols].min(axis=1)
        
        X_train['v_range'] = X_train['v_max'] - X_train['v_min']
        X_val['v_range'] = X_val['v_max'] - X_val['v_min']
        test_data['v_range'] = test_data['v_max'] - test_data['v_min']
    
    # 4. v_0和v_3深度特征
    if 'v_0' in X_train.columns and 'v_3' in X_train.columns:
        X_train['v_0_v_3_mul'] = X_train['v_0'] * X_train['v_3']
        X_val['v_0_v_3_mul'] = X_val['v_0'] * X_val['v_3']
        test_data['v_0_v_3_mul'] = test_data['v_0'] * test_data['v_3']
        
        X_train['v_0_sq'] = X_train['v_0'] ** 2
        X_val['v_0_sq'] = X_val['v_0'] ** 2
        test_data['v_0_sq'] = test_data['v_0'] ** 2
        
        X_train['v_3_sq'] = X_train['v_3'] ** 2
        X_val['v_3_sq'] = X_val['v_3'] ** 2
        test_data['v_3_sq'] = test_data['v_3'] ** 2
        
        X_train['v_0_v_3_diff'] = X_train['v_0'] - X_train['v_3']
        X_val['v_0_v_3_diff'] = X_val['v_0'] - X_val['v_3']
        test_data['v_0_v_3_diff'] = test_data['v_0'] - test_data['v_3']
    
    # 5. 对数变换
    if 'kilometer' in X_train.columns:
        X_train['log_km'] = np.log1p(X_train['kilometer'])
        X_val['log_km'] = np.log1p(X_val['kilometer'])
        test_data['log_km'] = np.log1p(test_data['kilometer'])
    
    if 'power' in X_train.columns:
        X_train['log_power'] = np.log1p(X_train['power'])
        X_val['log_power'] = np.log1p(X_val['power'])
        test_data['log_power'] = np.log1p(test_data['power'])
    
    # 6. K-Fold目标编码
    if 'brand' in X_train.columns:
        print("\n添加brand目标编码...")
        train_encoded, val_encoded = kfold_target_encode(X_train, y_train, X_val, 'brand', n_splits=5, smoothing=15.0)
        X_train['brand_te'] = train_encoded
        X_val['brand_te'] = val_encoded
        
        target_mean = y_train.mean()
        category_means = y_train.groupby(X_train['brand']).mean()
        category_counts = X_train['brand'].groupby(X_train['brand']).count()
        smoothed_mean = (category_means * category_counts + target_mean * 15.0) / (category_counts + 15.0)
        test_data['brand_te'] = test_data['brand'].map(smoothed_mean).fillna(target_mean)
    
    # 7. 处理无穷值和缺失值
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    test_data = test_data.replace([np.inf, -np.inf], np.nan)
    
    for col in X_train.columns:
        if X_train[col].dtype in [np.float64, np.int64]:
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_val[col].fillna(median_val, inplace=True)
            test_data[col].fillna(median_val, inplace=True)
    
    print(f"\n特征工程完成，最终特征数: {X_train.shape[1]}")
    
    return X_train, X_val, test_data


# ==================== 模型训练 ====================
def train_catboost(X_train, X_val, y_train, y_val):
    """训练CatBoost模型"""
    print("\n" + "="*50)
    print("训练CatBoost模型...")
    print("="*50)
    
    params = {
        'iterations': 3000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 10,
        'min_data_in_leaf': 30,
        'rsm': 0.8,
        'random_seed': 42,
        'od_type': 'Iter',
        'od_wait': 50,
        'verbose': 500,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'task_type': 'CPU',
        'thread_count': -1,
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=500)
    
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"CatBoost MAE: {mae:.2f}")
    
    return model, y_pred


def train_lightgbm(X_train, X_val, y_train, y_val):
    """训练LightGBM模型"""
    print("\n" + "="*50)
    print("训练LightGBM模型...")
    print("="*50)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 63,
        'max_depth': 8,
        'min_data_in_leaf': 30,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'verbose': -1,
        'seed': 42,
    }
    
    callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=500)]
    
    model = lgb.train(params, train_data, num_boost_round=3000, valid_sets=[train_data, val_data],
                      valid_names=['train', 'valid'], callbacks=callbacks)
    
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"LightGBM MAE: {mae:.2f}")
    
    return model, y_pred


def train_xgboost(X_train, X_val, y_train, y_val):
    """训练XGBoost模型"""
    print("\n" + "="*50)
    print("训练XGBoost模型...")
    print("="*50)
    
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.02,
        'max_depth': 8,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 3000,
        'random_state': 42,
        'eval_metric': 'mae',
        'early_stopping_rounds': 50,
        'n_jobs': -1,
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=500)
    
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"XGBoost MAE: {mae:.2f}")
    
    return model, y_pred


# ==================== 加权融合 ====================
def weighted_ensemble(predictions_list, weights, y_val=None):
    """加权融合"""
    ensemble_pred = np.zeros(len(predictions_list[0]))
    for pred, weight in zip(predictions_list, weights):
        ensemble_pred += pred * weight
    
    if y_val is not None:
        mae = mean_absolute_error(y_val, ensemble_pred)
        print(f"\n融合模型 MAE: {mae:.2f}")
    
    return ensemble_pred


def optimize_weights(predictions_list, y_val, n_iter=100):
    """优化融合权重"""
    from scipy.optimize import minimize
    
    def objective(weights):
        # 归一化权重
        weights = np.abs(weights) / np.sum(np.abs(weights))
        pred = np.zeros(len(y_val))
        for p, w in zip(predictions_list, weights):
            pred += p * w
        return mean_absolute_error(y_val, pred)
    
    n_models = len(predictions_list)
    initial_weights = np.ones(n_models) / n_models
    
    result = minimize(objective, initial_weights, method='Nelder-Mead', 
                     options={'maxiter': n_iter, 'xatol': 0.001})
    
    optimal_weights = np.abs(result.x) / np.sum(np.abs(result.x))
    return optimal_weights


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("四模型融合训练 (CatBoost + LightGBM + XGBoost + DeepNN)")
    print("=" * 60)
    
    # 1. 加载数据
    X_train, X_val, y_train, y_val, test_data, sale_ids = load_processed_data()
    
    # 2. 特征工程
    X_train_eng, X_val_eng, test_data_eng = feature_engineering(X_train, y_train, X_val, test_data)
    
    # 确保列一致
    common_cols = list(set(X_train_eng.columns) & set(X_val_eng.columns) & set(test_data_eng.columns))
    X_train_eng = X_train_eng[common_cols].reset_index(drop=True)
    X_val_eng = X_val_eng[common_cols].reset_index(drop=True)
    test_data_eng = test_data_eng[common_cols].reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    
    # 3. 训练三个树模型
    cat_model, cat_pred = train_catboost(X_train_eng, X_val_eng, y_train, y_val)
    lgb_model, lgb_pred = train_lightgbm(X_train_eng, X_val_eng, y_train, y_val)
    xgb_model, xgb_pred = train_xgboost(X_train_eng, X_val_eng, y_train, y_val)
    
    # 4. 尝试加载深度模型预测
    try:
        deep_oof = np.load('deep_oof_preds.npy')
        deep_test = np.load('deep_test_preds.npy')
        
        # 获取验证集对应的深度模型预测
        # 假设深度模型是全量训练后对验证集的预测
        deep_val_pred = deep_oof  # 这里需要根据实际情况调整
        
        print("\n成功加载深度模型预测结果")
        predictions_list = [cat_pred, lgb_pred, xgb_pred, deep_val_pred]
        model_names = ['CatBoost', 'LightGBM', 'XGBoost', 'DeepNN']
        use_deep = True
    except FileNotFoundError:
        print("\n未找到深度模型预测结果，仅使用三模型融合")
        predictions_list = [cat_pred, lgb_pred, xgb_pred]
        model_names = ['CatBoost', 'LightGBM', 'XGBoost']
        use_deep = False
    
    # 5. 优化融合权重
    print("\n" + "="*50)
    print("优化融合权重...")
    print("="*50)
    
    optimal_weights = optimize_weights(predictions_list, y_val)
    
    print("\n最优权重:")
    for name, w in zip(model_names, optimal_weights):
        print(f"  {name}: {w:.4f}")
    
    # 6. 加权融合
    ensemble_pred = weighted_ensemble(predictions_list, optimal_weights, y_val)
    
    # 7. 评估
    final_mae = mean_absolute_error(y_val, ensemble_pred)
    final_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    final_r2 = r2_score(y_val, ensemble_pred)
    
    print("\n" + "="*60)
    print("最终融合结果")
    print("="*60)
    print(f"MAE: {final_mae:.2f}")
    print(f"RMSE: {final_rmse:.2f}")
    print(f"R²: {final_r2:.4f}")
    
    # 8. 测试集预测
    cat_test = cat_model.predict(test_data_eng)
    lgb_test = lgb_model.predict(test_data_eng)
    xgb_test = xgb_model.predict(test_data_eng)
    
    if use_deep:
        test_predictions = cat_test * optimal_weights[0] + lgb_test * optimal_weights[1] + \
                          xgb_test * optimal_weights[2] + deep_test * optimal_weights[3]
    else:
        test_predictions = cat_test * optimal_weights[0] + lgb_test * optimal_weights[1] + \
                          xgb_test * optimal_weights[2]
    
    # 9. 保存结果
    submit_data = pd.DataFrame({
        'SaleID': sale_ids,
        'price': test_predictions
    })
    submit_data.to_csv('final_ensemble_submit.csv', index=False)
    
    print(f"\n预测结果已保存到 final_ensemble_submit.csv")
    
    if final_mae < 450:
        print(f"\n🎯 恭喜！MAE={final_mae:.0f} 已达到目标 (<450)")
    else:
        print(f"\n还需继续优化，距离目标还差 {final_mae - 450:.0f}")
    
    return final_mae


if __name__ == "__main__":
    main()
