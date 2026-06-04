# -*- coding: utf-8 -*-
"""
二手车价格预测 - CatBoost模型（全面优化版）
优化要点：
1. K-Fold目标编码防止数据泄露
2. 删除噪声特征
3. 分组统计特征（K-Fold方式）
4. Optuna超参数调优
5. K-Fold交叉验证评估
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 噪声特征列表（根据特征重要性分析）
NOISE_FEATURES = ['seller', 'offerType', 'SaleID', 'name', 'creat_day', 
                  'creat_year', 'creat_month', 'reg_day', 'reg_month']


# ==================== 数据加载 ====================
def load_processed_data():
    """加载预处理后的数据"""
    print("正在加载预处理后的数据...")
    X_train = joblib.load('processed_data/X_train.joblib')
    X_val = joblib.load('processed_data/X_val.joblib')
    y_train = joblib.load('processed_data/y_train.joblib')
    y_val = joblib.load('processed_data/y_val.joblib')
    return X_train, X_val, y_train, y_val


# ==================== K-Fold目标编码（防止数据泄露）====================
def kfold_target_encode(X_train, y_train, X_val, col, n_splits=5, smoothing=10.0):
    """
    K-Fold目标编码，避免数据泄露
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, index=X_train.index)
    
    train_encoded = np.zeros(len(X_train))
    
    # K-Fold编码训练集
    for train_idx, val_idx in kf.split(X_train):
        fold_train_col = X_train.iloc[train_idx][col]
        fold_y = y_train.iloc[train_idx]
        target_mean = fold_y.mean()
        
        category_means = fold_y.groupby(fold_train_col).mean()
        category_counts = fold_train_col.groupby(fold_train_col).count()
        
        smoothed_mean = (
            category_means * category_counts + target_mean * smoothing
        ) / (category_counts + smoothing)
        
        val_data = X_train.iloc[val_idx][col]
        train_encoded[val_idx] = val_data.map(smoothed_mean).fillna(target_mean)
    
    # 编码验证集
    target_mean = y_train.mean()
    category_means = y_train.groupby(X_train[col]).mean()
    category_counts = X_train[col].groupby(X_train[col]).count()
    
    smoothed_mean = (
        category_means * category_counts + target_mean * smoothing
    ) / (category_counts + smoothing)
    
    val_encoded = X_val[col].map(smoothed_mean).fillna(target_mean)
    
    return train_encoded, val_encoded


# ==================== K-Fold分组统计（防止数据泄露）====================
def kfold_group_stats(X_train, y_train, X_val, group_col, n_splits=5):
    """
    K-Fold分组统计特征，避免数据泄露
    计算：均值、中位数、标准差、计数
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, index=X_train.index)
    
    # 存储结果
    train_mean = np.zeros(len(X_train))
    train_median = np.zeros(len(X_train))
    train_std = np.zeros(len(X_train))
    train_count = np.zeros(len(X_train))
    
    # K-Fold计算
    for train_idx, val_idx in kf.split(X_train):
        fold_group = X_train.iloc[train_idx][group_col]
        fold_y = y_train.iloc[train_idx]
        
        # 计算分组统计
        group_stats = fold_y.groupby(fold_group).agg(['mean', 'median', 'std', 'count'])
        group_stats.columns = ['mean', 'median', 'std', 'count']
        global_mean = fold_y.mean()
        global_std = fold_y.std()
        
        # 映射到验证折
        val_group = X_train.iloc[val_idx][group_col]
        train_mean[val_idx] = val_group.map(group_stats['mean']).fillna(global_mean)
        train_median[val_idx] = val_group.map(group_stats['median']).fillna(global_mean)
        train_std[val_idx] = val_group.map(group_stats['std']).fillna(global_std)
        train_count[val_idx] = val_group.map(group_stats['count']).fillna(1)
    
    # 验证集统计
    group_stats = y_train.groupby(X_train[group_col]).agg(['mean', 'median', 'std', 'count'])
    group_stats.columns = ['mean', 'median', 'std', 'count']
    global_mean = y_train.mean()
    global_std = y_train.std()
    
    val_group = X_val[group_col]
    val_mean = val_group.map(group_stats['mean']).fillna(global_mean)
    val_median = val_group.map(group_stats['median']).fillna(global_mean)
    val_std = val_group.map(group_stats['std']).fillna(global_std)
    val_count = val_group.map(group_stats['count']).fillna(1)
    
    return (train_mean, train_median, train_std, train_count,
            val_mean, val_median, val_std, val_count)


# ==================== 数据预处理 ====================
def preprocess_data(X_train, y_train, X_val, y_val):
    """数据预处理：删除噪声特征、缩尾处理、填充缺失值"""
    
    # 1. 删除噪声特征
    noise_cols = [col for col in NOISE_FEATURES if col in X_train.columns]
    if noise_cols:
        print(f"删除噪声特征: {noise_cols}")
        X_train = X_train.drop(columns=noise_cols)
        X_val = X_val.drop(columns=noise_cols)
    
    # 2. 缩尾处理目标变量
    def winsorize_outliers(y, lower_percentile=1, upper_percentile=99):
        lower_bound = np.percentile(y, lower_percentile)
        upper_bound = np.percentile(y, upper_percentile)
        return np.clip(y, lower_bound, upper_bound)
    
    y_train = winsorize_outliers(y_train)
    y_val = winsorize_outliers(y_val)
    
    # 3. 填充缺失值
    X_train.fillna(X_train.mean(numeric_only=True), inplace=True)
    X_val.fillna(X_val.mean(numeric_only=True), inplace=True)
    
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if not X_train[col].mode().empty:
            mode_value = X_train[col].mode().iloc[0]
            X_train[col].fillna(mode_value, inplace=True)
            X_val[col].fillna(mode_value, inplace=True)
    
    return X_train, y_train, X_val, y_val


# ==================== 特征工程 ====================
def engineer_features(X_train, y_train, X_val):
    """
    特征工程：全面优化版
    """
    # 1. 删除高相关特征
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = X_train[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr_features = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > 0.85)
        ]
        if high_corr_features:
            print(f"删除高相关特征: {high_corr_features}")
            X_train = X_train.drop(columns=high_corr_features)
            X_val = X_val.drop(columns=high_corr_features)

    # 2. 交互特征
    if 'kilometer' in X_train.columns:
        X_train['log_km'] = np.log1p(X_train['kilometer'])
        X_val['log_km'] = np.log1p(X_val['kilometer'])
        print("添加特征: log_km")
    
    if 'power' in X_train.columns:
        X_train['log_power'] = np.log1p(X_train['power'])
        X_val['log_power'] = np.log1p(X_val['power'])
        print("添加特征: log_power")

    # 3. K-Fold目标编码（brand）
    if 'brand' in X_train.columns:
        print("添加brand目标编码...")
        train_encoded, val_encoded = kfold_target_encode(
            X_train, y_train, X_val, 'brand', n_splits=5, smoothing=15.0
        )
        X_train['brand_te'] = train_encoded
        X_val['brand_te'] = val_encoded

    # 4. K-Fold分组统计（brand）
    if 'brand' in X_train.columns:
        print("添加brand分组统计...")
        stats = kfold_group_stats(X_train, y_train, X_val, 'brand', n_splits=5)
        X_train['brand_mean'] = stats[0]
        X_train['brand_median'] = stats[1]
        X_train['brand_std'] = stats[2]
        X_val['brand_mean'] = stats[4]
        X_val['brand_median'] = stats[5]
        X_val['brand_std'] = stats[6]
        print("添加特征: brand_mean, brand_median, brand_std")

    # 5. K-Fold分组统计（model）
    if 'model' in X_train.columns:
        print("添加model分组统计...")
        stats = kfold_group_stats(X_train, y_train, X_val, 'model', n_splits=5)
        X_train['model_mean'] = stats[0]
        X_train['model_median'] = stats[1]
        X_val['model_mean'] = stats[4]
        X_val['model_median'] = stats[5]
        print("添加特征: model_mean, model_median")

    # 6. 标准化数值特征
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    
    # 保存scaler
    joblib.dump(scaler, 'processed_data/scaler.joblib')

    print(f"\n特征工程完成，最终特征数: {X_train.shape[1]}")
    return X_train, X_val


# ==================== Optuna超参数调优 ====================
def objective(trial, X_train, y_train, n_splits=5):
    """Optuna优化目标函数"""
    
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'depth': trial.suggest_int('depth', 3, 6),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 5, 30),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 80),
        'rsm': trial.suggest_float('rsm', 0.6, 0.9),
        'random_strength': trial.suggest_float('random_strength', 1.0, 5.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 1.5),
        'bootstrap_type': 'Bayesian',
        'random_seed': 42,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'verbose': 0,
        'task_type': 'CPU',
        'thread_count': -1,
    }
    
    # K-Fold交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=0
        )
        
        y_pred = model.predict(X_va)
        mae = mean_absolute_error(y_va, y_pred)
        mae_scores.append(mae)
    
    return np.mean(mae_scores)


def run_optuna_optimization(X_train, y_train, n_trials=50):
    """运行Optuna优化"""
    print("\n" + "="*50)
    print("开始Optuna超参数优化...")
    print("="*50)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\n最佳MAE: {study.best_value:.2f}")
    print(f"最佳参数: {study.best_params}")
    
    # 保存结果
    joblib.dump(study, 'processed_data/optuna_study.joblib')
    
    return study.best_params


# ==================== 模型训练 ====================
def train_catboost_model(X_train, X_val, y_train, y_val, params=None):
    """训练CatBoost模型"""
    print("\n正在训练CatBoost模型...")
    
    if params is None:
        params = {
            'iterations': 3000,
            'learning_rate': 0.025,
            'depth': 4,
            'l2_leaf_reg': 15,
            'min_data_in_leaf': 40,
            'rsm': 0.7,
            'bootstrap_type': 'Bayesian',
            'bagging_temperature': 0.8,
            'random_strength': 2.5,
            'random_seed': 42,
            'od_type': 'Iter',
            'od_wait': 60,
            'verbose': 100,
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'task_type': 'CPU',
            'thread_count': -1,
        }
    
    model = CatBoostRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=100
    )
    
    # 保存模型
    model.save_model('processed_data/catboost_model.cbm')
    joblib.dump(X_train.columns.tolist(), 'processed_data/train_columns.joblib')
    print("模型已保存到 processed_data/catboost_model.cbm")
    
    return model


# ==================== K-Fold交叉验证评估 ====================
def kfold_evaluate(X_train, y_train, params, n_splits=5):
    """K-Fold交叉验证评估"""
    print("\n" + "="*50)
    print("K-Fold交叉验证评估...")
    print("="*50)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_scores = []
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=0
        )
        
        y_pred = model.predict(X_va)
        mae = mean_absolute_error(y_va, y_pred)
        rmse = np.sqrt(mean_squared_error(y_va, y_pred))
        
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        print(f"Fold {fold+1}: MAE={mae:.2f}, RMSE={rmse:.2f}")
    
    print(f"\n平均 MAE: {np.mean(mae_scores):.2f} (+/- {np.std(mae_scores):.2f})")
    print(f"平均 RMSE: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})")
    
    return np.mean(mae_scores)


# ==================== 模型评估 ====================
def evaluate_model(model, X_val, y_val):
    """评估模型性能"""
    y_pred = model.predict(X_val)
    
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print("\n模型评估结果：")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('CatBoost预测价格 vs 实际价格')
    plt.tight_layout()
    plt.savefig('catboost_prediction_vs_actual.png')
    plt.close()
    
    return rmse, mae, r2


# ==================== 特征重要性 ====================
def plot_feature_importance(model, X_train):
    """绘制特征重要性图"""
    importance = model.get_feature_importance()
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv('catboost_feature_importance.csv', index=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('CatBoost Top 20 特征重要性')
    plt.tight_layout()
    plt.savefig('catboost_feature_importance.png')
    plt.close()
    
    return feature_importance


# ==================== 预测测试集 ====================
def predict_test_data():
    """预测测试集数据"""
    print("\n正在预测测试集...")
    
    test_data = joblib.load('processed_data/test_data.joblib')
    sale_ids = joblib.load('processed_data/sale_ids.joblib')
    model = CatBoostRegressor()
    model.load_model('processed_data/catboost_model.cbm')
    train_columns = joblib.load('processed_data/train_columns.joblib')
    
    # 对齐特征列
    test_data = test_data.reindex(columns=train_columns, fill_value=0)
    
    predictions = model.predict(test_data)
    
    submit_data = pd.DataFrame({
        'SaleID': sale_ids,
        'price': predictions
    })
    
    submit_data.to_csv('catboost_submit_result.csv', index=False)
    print("预测结果已保存到 catboost_submit_result.csv")


# ==================== 主函数 ====================
def main():
    # 1. 加载数据
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = load_processed_data()
    
    # 2. 数据预处理
    X_train_clean, y_train_clean, X_val_clean, y_val_clean = preprocess_data(
        X_train_raw, y_train_raw, X_val_raw, y_val_raw
    )
    
    # 3. 特征工程
    X_train_eng, X_val_eng = engineer_features(
        X_train_clean, y_train_clean, X_val_clean
    )
    
    # 4. Optuna超参数调优（可选，设为0跳过）
    run_optuna = True  # 设为False跳过调优
    n_trials = 30      # 调优次数
    
    if run_optuna:
        best_params = run_optuna_optimization(X_train_eng, y_train_clean, n_trials=n_trials)
        best_params.update({
            'random_seed': 42,
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'verbose': 100,
            'task_type': 'CPU',
            'thread_count': -1,
        })
    else:
        best_params = None
    
    # 5. K-Fold交叉验证
    if best_params:
        kfold_evaluate(X_train_eng, y_train_clean, best_params)
    
    # 6. 训练最终模型
    model = train_catboost_model(
        X_train_eng, X_val_eng, y_train_clean, y_val_clean, 
        params=best_params
    )
    
    # 7. 评估
    rmse, mae, r2 = evaluate_model(model, X_val_eng, y_val_clean)
    
    # 8. 特征重要性
    plot_feature_importance(model, X_train_eng)
    
    # 9. 预测测试集
    predict_test_data()
    
    print("\n" + "="*50)
    print("模型训练、评估和预测完成！")
    print(f"最终 MAE: {mae:.2f}")
    print(f"最终 RMSE: {rmse:.2f}")
    print(f"最终 R²: {r2:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
