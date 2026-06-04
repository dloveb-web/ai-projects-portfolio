# -*- coding: utf-8 -*-
"""
二手车价格预测 - CatBoost模型（过拟合优化版）
优化目标：MAE < 400 & 防止过拟合
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.stats import iqr
from category_encoders import TargetEncoder

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 数据预处理 ====================
def preprocess_data(X_train, y_train, X_val, y_val):
    """
    数据预处理：改进缩尾处理、填充缺失值、标准化数值特征
    优化：更激进的异常值处理 + 特征标准化提升泛化能力
    """
    # 改进版缩尾法（更激进的异常值处理）
    def winsorize_outliers(y, lower_percentile=2, upper_percentile=98):
        """更激进的缩尾处理，去除极端异常值"""
        lower_bound = np.percentile(y, lower_percentile)
        upper_bound = np.percentile(y, upper_percentile)
        return np.clip(y, lower_bound, upper_bound)

    y_train = winsorize_outliers(y_train)
    y_val = winsorize_outliers(y_val)
    
    print(f"缩尾后训练集目标值范围: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"缩尾后验证集目标值范围: [{y_val.min():.2f}, {y_val.max():.2f}]")

    # 填充缺失值（数值型用中位数而非均值，更robust）
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
    X_val[numeric_cols] = X_val[numeric_cols].fillna(X_train[numeric_cols].median())
    
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if not X_train[col].mode().empty:
            mode_value = X_train[col].mode().iloc[0]
            X_train[col].fillna(mode_value, inplace=True)
            X_val[col].fillna(mode_value, inplace=True)

    # 标准化数值特征（防止特征量纲影响）
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    return X_train, y_train, X_val, y_val


# ==================== 特征工程 ====================
def engineer_features(X_train, y_train, X_val):
    """
    特征工程：严格的特征选择 + 有选择性的交互特征
    优化：减少特征维度，只保留高价值特征
    """
    # 删除缺失率过高的特征（缺失率>30%）
    missing_ratio = X_train.isnull().sum() / len(X_train)
    high_missing_cols = missing_ratio[missing_ratio > 0.3].index.tolist()
    if high_missing_cols:
        print(f"删除缺失率>30%的特征: {high_missing_cols}")
        X_train = X_train.drop(columns=high_missing_cols, errors='ignore')
        X_val = X_val.drop(columns=high_missing_cols, errors='ignore')

    # 计算特征之间的相关性矩阵
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    corr_matrix = X_train[numeric_cols].corr().abs()

    # 找出高度相关的特征对（提高阈值到0.95以保留更多信息）
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [
        column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)
    ]

    if high_corr_features:
        print(f"删除高度相关的特征: {high_corr_features}")
        X_train = X_train.drop(columns=high_corr_features, errors='ignore')
        X_val = X_val.drop(columns=high_corr_features, errors='ignore')

    # ⭐ 只添加高价值的交互特征（谨慎添加）
    if 'age' in X_train.columns and 'mileage' in X_train.columns:
        # 车龄和里程通常强相关，交互特征可能过于冗余
        print("保留 age_mileage 交互特征")
        X_train['age_mileage'] = X_train['age'] * X_train['mileage']
        X_val['age_mileage'] = X_val['age'] * X_val['mileage']
    
    # 动力密度特征（通常有预测价值）
    if 'engine_power' in X_train.columns and 'weight' in X_train.columns:
        print("添加 power_to_weight 特征")
        X_train['power_to_weight'] = X_train['engine_power'] / (X_train['weight'] + 1e-5)
        X_val['power_to_weight'] = X_val['engine_power'] / (X_val['weight'] + 1e-5)

    # log变换（可选，仅对右偏分布的特征有效）
    if 'mileage' in X_train.columns:
        mileage_skew = X_train['mileage'].skew()
        if abs(mileage_skew) > 1:  # 只有偏度较大时才添加
            print("添加 log_mileage 特征")
            X_train['log_mileage'] = np.log1p(X_train['mileage'])
            X_val['log_mileage'] = np.log1p(X_val['mileage'])

    # ⭐ 改进的目标编码（防止过拟合的关键）
    if 'brand' in X_train.columns:
        print("对 brand 进行目标编码")
        encoder = TargetEncoder(
            min_samples_leaf=30,      # 提高最小样本数（默认1→30）
            smoothing=15.0,           # 增加平滑系数（防止过拟合）
            handle_unknown='value',
            handle_missing='value'
        )
        X_train['brand_encoded'] = encoder.fit_transform(X_train['brand'], y_train)
        X_val['brand_encoded'] = encoder.transform(X_val['brand'])

    print(f"工程后特征数: {X_train.shape[1]}")
    return X_train, X_val


# ==================== CatBoost 相关函数 ====================
def load_processed_data():
    """加载预处理后的数据"""
    print("正在加载预处理后的数据...")
    X_train = joblib.load('processed_data/X_train.joblib')
    X_val = joblib.load('processed_data/X_val.joblib')
    y_train = joblib.load('processed_data/y_train.joblib')
    y_val = joblib.load('processed_data/y_val.joblib')
    return X_train, X_val, y_train, y_val


class ValidationMAECallback:
    """打印验证集MAE的回调类"""
    def __init__(self):
        self.best_mae = float('inf')
        self.best_iteration = 0
        
    def after_iteration(self, info):
        info_dict = vars(info)
        if 'validation' in info_dict and 'MAE' in info_dict['validation']:
            current_mae = info_dict['validation']['MAE']
            iteration = info_dict.get('iteration', 0)
            print(f"Iteration {iteration}: 验证集 MAE = {current_mae:.2f}", end="")
            
            if current_mae < self.best_mae:
                self.best_mae = current_mae
                self.best_iteration = iteration
                print(" ✓ (更新最佳)")
            else:
                print()
        return True


def train_catboost_model(X_train, X_val, y_train, y_val):
    """
    训练CatBoost模型（过拟合优化版本）
    核心优��：
    1. 降低树深度 (5→4)
    2. 增加正则化 (8→18)
    3. 提前早停 (200→80)
    4. 降低学习率 (0.035→0.025)
    5. 增加样本子采样比例
    """
    print("正在训练CatBoost模型（优化版）...\n")

    params = {
        # ⭐ 核心参数优化
        'iterations': 3000,           # 减少迭代次数
        'learning_rate': 0.025,       # 降低学习率（防止过拟合）
        'depth': 4,                   # 降低树深度（4层而非5层）
        'l2_leaf_reg': 18,            # 大幅增加L2正则化（8→18）
        'min_data_in_leaf': 25,       # 增加叶子最小样本数
        
        # ⭐ 采样参数优化
        'rsm': 0.75,                  # 降低特征子采样比例
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.75,            # 降低样本子采样比例
        'random_strength': 2.0,       # 增加随机性
        
        # ⭐ 早停参数优化
        'od_type': 'Iter',
        'od_wait': 80,                # 提前早停（200→80）
        
        # 基础参数
        'random_seed': 42,
        'verbose': 100,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'task_type': 'CPU',
        'thread_count': -1
    }

    print("📊 模型参数配置:")
    for key, value in params.items():
        if key in ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg', 'od_wait']:
            print(f"  {key}: {value}")
    print()

    # 创建模型
    model = CatBoostRegressor(**params)

    # 创建回调
    callback = ValidationMAECallback()

    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        plot=True,
        callbacks=[callback]
    )

    # 保存模型
    model.save_model('processed_data/catboost_model_optimized.cbm')
    joblib.dump(X_train.columns.tolist(), 'processed_data/train_columns.joblib')
    print("\n✅ 模型已保存到 processed_data/catboost_model_optimized.cbm")

    return model


def evaluate_model(model, X_val, y_val):
    """
    评估模型性能（目标MAE < 400）
    """
    y_pred = model.predict(X_val)

    # 计算评估指标
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print("\n" + "="*50)
    print("📈 模型评估结果：")
    print("="*50)
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对误差 (MAE):  {mae:.2f}", end="")
    if mae < 400:
        print(" ✅ (达成目标!)")
    else:
        print(f" ⚠️  (需要进一步优化, 差距: {mae-400:.2f})")
    print(f"R2分数:           {r2:.4f}")
    print("="*50 + "\n")

    # 分析预测误差
    residuals = y_val - y_pred
    print(f"预测误差统计:")
    print(f"  平均误差 (偏差):    {residuals.mean():.2f}")
    print(f"  误差标准差:        {residuals.std():.2f}")
    print(f"  ���差中位数:        {residuals.median():.2f}")
    print(f"  最大正误差:        {residuals.max():.2f}")
    print(f"  最大负误差:        {residuals.min():.2f}\n")

    # 绘制预测值与实际值的对比图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_val, y_pred, alpha=0.5, s=10)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('预测价格 vs 实际价格')
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('预测误差')
    plt.ylabel('频数')
    plt.title(f'误差分布 (MAE={mae:.2f})')
    plt.axvline(residuals.mean(), color='r', linestyle='--', label=f'均值={residuals.mean():.2f}')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('catboost_evaluation_optimized.png', dpi=100)
    plt.close()

    return rmse, mae, r2


def plot_feature_importance(model, X_train):
    """绘制特征重要性（只显示重要的特征）"""
    importance = model.get_feature_importance()
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    # 保存完整数据
    feature_importance.to_csv('catboost_feature_importance_optimized.csv', index=False)

    # 绘制Top 15特征
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(15)
    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
    plt.title('CatBoost Top 15 特征重要性')
    plt.tight_layout()
    plt.savefig('catboost_feature_importance_optimized.png', dpi=100)
    plt.close()
    
    print("🔍 Top 10 重要特征:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")


def predict_test_data():
    """预测测试集数据"""
    print("\n正在加载测试数据...")
    test_data = joblib.load('processed_data/test_data.joblib')
    sale_ids = joblib.load('processed_data/sale_ids.joblib')
    model = CatBoostRegressor()
    model.load_model('processed_data/catboost_model_optimized.cbm')

    # 加载训练集特征列
    train_columns = joblib.load('processed_data/train_columns.joblib')

    # 对测试数据进行相同的预处理
    scaler = StandardScaler()
    numeric_cols = test_data.select_dtypes(include=[np.number]).columns
    test_data[numeric_cols] = scaler.fit_transform(test_data[numeric_cols])

    # 对齐测试集特征列
    test_data = test_data.reindex(columns=train_columns, fill_value=0)

    # 预测
    print("正在预测测试集...")
    predictions = model.predict(test_data)

    # 创建提交文件
    submit_data = pd.DataFrame({
        'SaleID': sale_ids,
        'price': predictions
    })

    submit_data.to_csv('catboost_submit_result_optimized.csv', index=False)
    print("✅ 预测结果已保存到 catboost_submit_result_optimized.csv")


# ==================== 主函数 ====================
def main():
    print("🚀 开始优化的CatBoost模型训练流程\n")
    
    # 加载原始数据
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = load_processed_data()
    print(f"原始数据形状: X_train={X_train_raw.shape}, X_val={X_val_raw.shape}\n")

    # 数据预处理
    print("【1】数据预处理中...")
    X_train_clean, y_train_clean, X_val_clean, y_val_clean = preprocess_data(
        X_train_raw, y_train_raw, X_val_raw, y_val_raw
    )
    print()

    # 特征工程
    print("【2】特征工程中...")
    X_train_eng, X_val_eng = engineer_features(X_train_clean, y_train_clean, X_val_clean)
    print()

    # 训练模型
    print("【3】模型训练中...")
    model = train_catboost_model(X_train_eng, X_val_eng, y_train_clean, y_val_clean)
    print()

    # 评估模型
    print("【4】模型评估中...")
    rmse, mae, r2 = evaluate_model(model, X_val_eng, y_val_clean)

    # 绘制特征重要性
    print("【5】绘制特征重要性...")
    plot_feature_importance(model, X_train_eng)
    print()

    # 预测测试集
    print("【6】预测测试集...")
    predict_test_data()

    print("✨ 优化的模型训练、评估和预测完成！")


if __name__ == "__main__":
    main()