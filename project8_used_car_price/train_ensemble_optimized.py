# -*- coding: utf-8 -*-
"""
二手车价格预测 - 深度特征工程 + 三模型融合优化版
目标：MAE < 400
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


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
    
    # 初始化结果
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
    
    # 验证集使用全量训练数据
    group_stats = y_train.groupby(X_train[group_col]).agg(['mean', 'median', 'std'])
    val_mean = X_val[group_col].map(group_stats['mean']).fillna(y_train.mean())
    val_median = X_val[group_col].map(group_stats['median']).fillna(y_train.median())
    val_std = X_val[group_col].map(group_stats['std']).fillna(0)
    
    return (train_mean, train_median, train_std), (val_mean, val_median, val_std)


# ==================== 深度特征工程 ====================
def deep_feature_engineering(X_train, y_train, X_val, test_data):
    """
    深度特征工程：重点挖掘v_0~v_14匿名特征
    """
    print("\n" + "="*50)
    print("开始深度特征工程...")
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
    
    # ==================== 核心特征：深入挖掘v_0~v_14 ====================
    v_cols = [col for col in X_train.columns if col.startswith('v_') and col not in ['v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_12', 'v_13']]
    print(f"\n发现的v特征: {v_cols}")
    
    # 3. v特征的统计特征
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
        
        print("添加v特征统计: v_mean, v_std, v_max, v_min, v_range")
    
    # 4. v特征交互（重要特征v_0, v_3与其他特征交互）
    important_v = ['v_0', 'v_3', 'v_2', 'v_1']
    for v in important_v:
        if v in X_train.columns:
            for other_v in v_cols:
                if other_v != v and other_v in X_train.columns:
                    # 加法交互
                    X_train[f'{v}_{other_v}_add'] = X_train[v] + X_train[other_v]
                    X_val[f'{v}_{other_v}_add'] = X_val[v] + X_val[other_v]
                    test_data[f'{v}_{other_v}_add'] = test_data[v] + test_data[other_v]
    print(f"添加v特征交互特征")
    
    # 5. v_0和v_3的特殊处理（这两个最重要）- 深度挖掘
    if 'v_0' in X_train.columns and 'v_3' in X_train.columns:
        # 基础交互
        X_train['v_0_v_3_mul'] = X_train['v_0'] * X_train['v_3']
        X_val['v_0_v_3_mul'] = X_val['v_0'] * X_val['v_3']
        test_data['v_0_v_3_mul'] = test_data['v_0'] * test_data['v_3']
        
        X_train['v_0_v_3_ratio'] = X_train['v_0'] / (X_train['v_3'] + 1e-5)
        X_val['v_0_v_3_ratio'] = X_val['v_0'] / (X_val['v_3'] + 1e-5)
        test_data['v_0_v_3_ratio'] = test_data['v_0'] / (test_data['v_3'] + 1e-5)
        
        # 高阶多项式特征
        X_train['v_0_sq'] = X_train['v_0'] ** 2
        X_val['v_0_sq'] = X_val['v_0'] ** 2
        test_data['v_0_sq'] = test_data['v_0'] ** 2
        
        X_train['v_3_sq'] = X_train['v_3'] ** 2
        X_val['v_3_sq'] = X_val['v_3'] ** 2
        test_data['v_3_sq'] = test_data['v_3'] ** 2
        
        X_train['v_0_cube'] = X_train['v_0'] ** 3
        X_val['v_0_cube'] = X_val['v_0'] ** 3
        test_data['v_0_cube'] = test_data['v_0'] ** 3
        
        X_train['v_3_cube'] = X_train['v_3'] ** 3
        X_val['v_3_cube'] = X_val['v_3'] ** 3
        test_data['v_3_cube'] = test_data['v_3'] ** 3
        
        # 平方根特征
        X_train['v_0_sqrt'] = np.sqrt(np.abs(X_train['v_0']))
        X_val['v_0_sqrt'] = np.sqrt(np.abs(X_val['v_0']))
        test_data['v_0_sqrt'] = np.sqrt(np.abs(test_data['v_0']))
        
        X_train['v_3_sqrt'] = np.sqrt(np.abs(X_train['v_3']))
        X_val['v_3_sqrt'] = np.sqrt(np.abs(X_val['v_3']))
        test_data['v_3_sqrt'] = np.sqrt(np.abs(test_data['v_3']))
        
        # 高阶交互
        X_train['v_0_sq_v_3'] = (X_train['v_0'] ** 2) * X_train['v_3']
        X_val['v_0_sq_v_3'] = (X_val['v_0'] ** 2) * X_val['v_3']
        test_data['v_0_sq_v_3'] = (test_data['v_0'] ** 2) * test_data['v_3']
        
        X_train['v_0_v_3_sq'] = X_train['v_0'] * (X_train['v_3'] ** 2)
        X_val['v_0_v_3_sq'] = X_val['v_0'] * (X_val['v_3'] ** 2)
        test_data['v_0_v_3_sq'] = test_data['v_0'] * (test_data['v_3'] ** 2)
        
        # 差值特征
        X_train['v_0_v_3_diff'] = X_train['v_0'] - X_train['v_3']
        X_val['v_0_v_3_diff'] = X_val['v_0'] - X_val['v_3']
        test_data['v_0_v_3_diff'] = test_data['v_0'] - test_data['v_3']
        
        X_train['v_0_v_3_diff_sq'] = (X_train['v_0'] - X_train['v_3']) ** 2
        X_val['v_0_v_3_diff_sq'] = (X_val['v_0'] - X_val['v_3']) ** 2
        test_data['v_0_v_3_diff_sq'] = (test_data['v_0'] - test_data['v_3']) ** 2
        
        # 分段特征
        X_train['v_0_v_3_dominant'] = (X_train['v_0'] > X_train['v_3']).astype(int)
        X_val['v_0_v_3_dominant'] = (X_val['v_0'] > X_val['v_3']).astype(int)
        test_data['v_0_v_3_dominant'] = (test_data['v_0'] > test_data['v_3']).astype(int)
        
        print("添加v_0和v_3深度特征（多项式、高阶交互、分段）")
    
    # 5.5 三阶交互特征
    if 'v_0' in X_train.columns and 'v_3' in X_train.columns and 'v_2' in X_train.columns:
        X_train['v_0_v_2_v_3'] = X_train['v_0'] * X_train['v_2'] * X_train['v_3']
        X_val['v_0_v_2_v_3'] = X_val['v_0'] * X_val['v_2'] * X_val['v_3']
        test_data['v_0_v_2_v_3'] = test_data['v_0'] * test_data['v_2'] * test_data['v_3']
        print("添加三阶交互特征 v_0*v_2*v_3")
    
    # 6. v特征分箱
    for v in ['v_0', 'v_3', 'v_2']:
        if v in X_train.columns:
            X_train[f'{v}_bin'] = pd.qcut(X_train[v], q=10, labels=False, duplicates='drop')
            X_val[f'{v}_bin'] = pd.qcut(X_val[v], q=10, labels=False, duplicates='drop')
            test_data[f'{v}_bin'] = pd.qcut(test_data[v], q=10, labels=False, duplicates='drop')
    print("添加v特征分箱")
    
    # ==================== 其他特征 ====================
    
    # 7. 对数变换
    if 'kilometer' in X_train.columns:
        X_train['log_km'] = np.log1p(X_train['kilometer'])
        X_val['log_km'] = np.log1p(X_val['kilometer'])
        test_data['log_km'] = np.log1p(test_data['kilometer'])
        
        # 里程分段
        X_train['km_bracket'] = pd.cut(X_train['kilometer'], bins=[0, 2, 5, 10, 15, 100], labels=False)
        X_val['km_bracket'] = pd.cut(X_val['kilometer'], bins=[0, 2, 5, 10, 15, 100], labels=False)
        test_data['km_bracket'] = pd.cut(test_data['kilometer'], bins=[0, 2, 5, 10, 15, 100], labels=False)
    
    if 'power' in X_train.columns:
        X_train['log_power'] = np.log1p(X_train['power'])
        X_val['log_power'] = np.log1p(X_val['power'])
        test_data['log_power'] = np.log1p(test_data['power'])
        
        # 功率分段
        X_train['power_bracket'] = pd.qcut(X_train['power'], q=5, labels=False, duplicates='drop')
        X_val['power_bracket'] = pd.qcut(X_val['power'], q=5, labels=False, duplicates='drop')
        test_data['power_bracket'] = pd.qcut(test_data['power'], q=5, labels=False, duplicates='drop')
    
    # 7.5 与v_0/v_3交叉的特征
    if 'power' in X_train.columns and 'v_0' in X_train.columns:
        X_train['power_v_0'] = X_train['power'] * X_train['v_0']
        X_val['power_v_0'] = X_val['power'] * X_val['v_0']
        test_data['power_v_0'] = test_data['power'] * test_data['v_0']
        
        X_train['power_v_3'] = X_train['power'] * X_train['v_3']
        X_val['power_v_3'] = X_val['power'] * X_val['v_3']
        test_data['power_v_3'] = test_data['power'] * test_data['v_3']
        print("添加power与v_0/v_3交互特征")
    
    if 'kilometer' in X_train.columns and 'v_0' in X_train.columns:
        X_train['km_v_0'] = X_train['kilometer'] * X_train['v_0']
        X_val['km_v_0'] = X_val['kilometer'] * X_val['v_0']
        test_data['km_v_0'] = test_data['kilometer'] * test_data['v_0']
        
        X_train['km_v_3'] = X_train['kilometer'] * X_train['v_3']
        X_val['km_v_3'] = X_val['kilometer'] * X_val['v_3']
        test_data['km_v_3'] = test_data['kilometer'] * test_data['v_3']
        print("添加kilometer与v_0/v_3交互特征")
    
    # 8. K-Fold目标编码
    if 'brand' in X_train.columns:
        print("\n添加brand目标编码...")
        train_encoded, val_encoded = kfold_target_encode(X_train, y_train, X_val, 'brand', n_splits=5, smoothing=15.0)
        X_train['brand_te'] = train_encoded
        X_val['brand_te'] = val_encoded
        # 测试集使用全量训练数据编码
        target_mean = y_train.mean()
        category_means = y_train.groupby(X_train['brand']).mean()
        category_counts = X_train['brand'].groupby(X_train['brand']).count()
        smoothed_mean = (category_means * category_counts + target_mean * 15.0) / (category_counts + 15.0)
        test_data['brand_te'] = test_data['brand'].map(smoothed_mean).fillna(target_mean)
    
    # 9. K-Fold分组统计
    if 'brand' in X_train.columns:
        print("添加brand分组统计...")
        train_stats, val_stats = kfold_group_stats(X_train, y_train, X_val, 'brand', n_splits=5)
        X_train['brand_mean'] = train_stats[0]
        X_train['brand_median'] = train_stats[1]
        X_train['brand_std'] = train_stats[2]
        X_val['brand_mean'] = val_stats[0]
        X_val['brand_median'] = val_stats[1]
        X_val['brand_std'] = val_stats[2]
        
        # 测试集
        group_stats = y_train.groupby(X_train['brand']).agg(['mean', 'median', 'std'])
        test_data['brand_mean'] = test_data['brand'].map(group_stats['mean']).fillna(y_train.mean())
        test_data['brand_median'] = test_data['brand'].map(group_stats['median']).fillna(y_train.median())
        test_data['brand_std'] = test_data['brand'].map(group_stats['std']).fillna(0)
    
    # 10. model分组统计
    if 'model' in X_train.columns:
        print("添加model分组统计...")
        train_stats, val_stats = kfold_group_stats(X_train, y_train, X_val, 'model', n_splits=5)
        X_train['model_mean'] = train_stats[0]
        X_train['model_median'] = train_stats[1]
        X_val['model_mean'] = val_stats[0]
        X_val['model_median'] = val_stats[1]
        
        group_stats = y_train.groupby(X_train['model']).agg(['mean', 'median'])
        test_data['model_mean'] = test_data['model'].map(group_stats['mean']).fillna(y_train.mean())
        test_data['model_median'] = test_data['model'].map(group_stats['median']).fillna(y_train.median())
    
    # 11. PCA压缩v特征
    remaining_v = [col for col in X_train.columns if col.startswith('v_') and not col.endswith('_bin') and not '_' in col[3:]]
    if len(remaining_v) >= 3:
        print(f"\n对v特征进行PCA压缩: {remaining_v}")
        pca = PCA(n_components=min(5, len(remaining_v)))
        X_train_v_pca = pca.fit_transform(X_train[remaining_v].fillna(0))
        X_val_v_pca = pca.transform(X_val[remaining_v].fillna(0))
        test_v_pca = pca.transform(test_data[remaining_v].fillna(0))
        
        for i in range(min(5, len(remaining_v))):
            X_train[f'v_pca_{i}'] = X_train_v_pca[:, i]
            X_val[f'v_pca_{i}'] = X_val_v_pca[:, i]
            test_data[f'v_pca_{i}'] = test_v_pca[:, i]
        print(f"PCA保留方差: {pca.explained_variance_ratio_.sum():.2%}")
    
    # 12. 处理无穷值和缺失值
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    test_data = test_data.replace([np.inf, -np.inf], np.nan)
    
    # 填充缺失值
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
        'iterations': 5000,
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
        'use_best_model': True,
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, verbose=500)
    
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
    
    model = lgb.train(params, train_data, num_boost_round=5000, valid_sets=[train_data, val_data],
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
        'n_estimators': 5000,
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


# ==================== 模型融合 ====================
def ensemble_predictions(predictions_list, weights=None, y_val=None):
    """
    模型融合：加权平均 + 权重优化
    """
    print("\n" + "="*50)
    print("模型融合（加权平均）...")
    print("="*50)
    
    if weights is None:
        # 基于验证集MAE自动计算权重（MAE越低权重越高）
        maes = [mean_absolute_error(y_val, pred) for pred in predictions_list]
        # 反比例权重
        inv_maes = [1/m for m in maes]
        total = sum(inv_maes)
        weights = [w/total for w in inv_maes]
        print(f"自动计算权重: {[f'{w:.3f}' for w in weights]} (基于MAE反比)")
    
    # 加权平均
    ensemble_pred = np.zeros(len(predictions_list[0]))
    for pred, weight in zip(predictions_list, weights):
        ensemble_pred += pred * weight
    
    if y_val is not None:
        mae = mean_absolute_error(y_val, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        r2 = r2_score(y_val, ensemble_pred)
        print(f"\n融合模型 MAE: {mae:.2f}")
        print(f"融合模型 RMSE: {rmse:.2f}")
        print(f"融合模型 R²: {r2:.4f}")
    
    return ensemble_pred, weights


def stacking_ensemble(X_train, y_train, X_val, y_val, test_data, n_folds=5):
    """
    Stacking融合：使用K-Fold生成元特征，Ridge作为元学习器
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler as SS
    
    print("\n" + "="*50)
    print("Stacking融合训练...")
    print("="*50)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 存储元特征
    train_meta = np.zeros((len(X_train), 3))  # 3个基模型
    val_meta = np.zeros((len(X_val), 3))
    test_meta = np.zeros((len(test_data), 3))
    
    # K-Fold训练基模型
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # CatBoost
        cat = CatBoostRegressor(
            iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=10,
            min_data_in_leaf=30, random_seed=42, verbose=0, loss_function='MAE'
        )
        cat.fit(X_tr, y_tr, verbose=0)
        train_meta[val_idx, 0] = cat.predict(X_va)
        val_meta[:, 0] += cat.predict(X_val) / n_folds
        test_meta[:, 0] += cat.predict(test_data) / n_folds
        
        # LightGBM
        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'learning_rate': 0.02,
             'num_leaves': 63, 'max_depth': 8, 'verbose': -1, 'seed': 42},
            lgb_train, num_boost_round=3000
        )
        train_meta[val_idx, 1] = lgb_model.predict(X_va)
        val_meta[:, 1] += lgb_model.predict(X_val) / n_folds
        test_meta[:, 1] += lgb_model.predict(test_data) / n_folds
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.02, max_depth=8,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )
        xgb_model.fit(X_tr, y_tr, verbose=False)
        train_meta[val_idx, 2] = xgb_model.predict(X_va)
        val_meta[:, 2] += xgb_model.predict(X_val) / n_folds
        test_meta[:, 2] += xgb_model.predict(test_data) / n_folds
        
        fold_mae = mean_absolute_error(y_va, train_meta[val_idx, 0])
        print(f"Fold {fold+1} CatBoost MAE: {fold_mae:.2f}")
    
    # 训练元学习器（Ridge）
    print("\n训练元学习器（Ridge）...")
    scaler = SS()
    train_meta_scaled = scaler.fit_transform(train_meta)
    val_meta_scaled = scaler.transform(val_meta)
    test_meta_scaled = scaler.transform(test_meta)
    
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(train_meta_scaled, y_train)
    
    # 预测
    val_pred = meta_model.predict(val_meta_scaled)
    test_pred = meta_model.predict(test_meta_scaled)
    
    # 评估
    mae = mean_absolute_error(y_val, val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    r2 = r2_score(y_val, val_pred)
    
    print(f"\nStacking模型 MAE: {mae:.2f}")
    print(f"Stacking模型 RMSE: {rmse:.2f}")
    print(f"Stacking模型 R²: {r2:.4f}")
    
    # 也计算加权平均作为对比
    weights = meta_model.coef_
    weights = weights / weights.sum()  # 归一化
    print(f"学习到的权重: CatBoost={weights[0]:.3f}, LightGBM={weights[1]:.3f}, XGBoost={weights[2]:.3f}")
    
    return val_pred, test_pred, mae


# ==================== 主函数 ====================
def main():
    # 加载数据
    X_train, X_val, y_train, y_val, test_data, sale_ids = load_processed_data()
    
    # 深度特征工程
    X_train_eng, X_val_eng, test_data_eng = deep_feature_engineering(X_train, y_train, X_val, test_data)
    
    # 确保列一致
    common_cols = list(set(X_train_eng.columns) & set(X_val_eng.columns) & set(test_data_eng.columns))
    X_train_eng = X_train_eng[common_cols].reset_index(drop=True)
    X_val_eng = X_val_eng[common_cols].reset_index(drop=True)
    test_data_eng = test_data_eng[common_cols].reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    
    # 使用Stacking融合
    val_pred, test_predictions, stacking_mae = stacking_ensemble(
        X_train_eng, y_train, X_val_eng, y_val, test_data_eng, n_folds=5
    )
    
    # 保存结果
    submit_data = pd.DataFrame({
        'SaleID': sale_ids,
        'price': test_predictions
    })
    submit_data.to_csv('ensemble_submit_result.csv', index=False)
    
    print(f"\n预测结果已保存到 ensemble_submit_result.csv")
    
    # 绘制预测对比图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, val_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('Stacking融合模型 - 预测价格 vs 实际价格')
    plt.tight_layout()
    plt.savefig('ensemble_prediction_vs_actual.png')
    plt.close()
    
    # 最终评估
    final_mae = mean_absolute_error(y_val, val_pred)
    final_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    final_r2 = r2_score(y_val, val_pred)
    
    print("\n" + "="*50)
    print("最终结果")
    print("="*50)
    print(f"MAE: {final_mae:.2f}")
    print(f"RMSE: {final_rmse:.2f}")
    print(f"R²: {final_r2:.4f}")
    
    if final_mae < 400:
        print("\n🎉 恭喜！MAE已达到目标 (<400)")
    else:
        print(f"\n还需继续优化，距离目标还差 {final_mae - 400:.2f}")
    
    return final_mae


if __name__ == "__main__":
    main()
