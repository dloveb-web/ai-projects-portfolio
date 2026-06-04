# -*- coding: utf-8 -*-
"""
二手车价格预测 - CatBoost模型（优化版）
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
    数据预处理：缩尾处理异常值、填充缺失值、标准化数值特征
    """
    # 使用缩尾法处理异常值（避免丢失过多样本）
    def winsorize_outliers(y, lower_percentile=1, upper_percentile=99):
        lower_bound = np.percentile(y, lower_percentile)
        upper_bound = np.percentile(y, upper_percentile)
        return np.clip(y, lower_bound, upper_bound)

    y_train = winsorize_outliers(y_train)
    y_val = winsorize_outliers(y_val)

    # 填充缺失值（数值型用均值，分类型用众数）
    X_train.fillna(X_train.mean(numeric_only=True), inplace=True)
    X_val.fillna(X_val.mean(numeric_only=True), inplace=True)
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if not X_train[col].mode().empty:
            mode_value = X_train[col].mode().iloc[0]
            X_train[col].fillna(mode_value, inplace=True)
            X_val[col].fillna(mode_value, inplace=True)

    # 标准化数值特征
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    return X_train, y_train, X_val, y_val


# ==================== 特征工程 ====================
def engineer_features(X_train, y_train, X_val):
    """
    特征工程：分析相关性、处理高度相关特征，新增交互特征和类别编码
    """
    # 计算特征之间的相关性矩阵
    corr_matrix = X_train.corr().abs()

    # 找出高度相关的特征对（阈值设为0.94）
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [
        column for column in upper_triangle.columns if any(upper_triangle[column] > 0.94)
    ]

    # 删除高度相关的特征
    X_train = X_train.drop(columns=high_corr_features)
    X_val = X_val.drop(columns=high_corr_features)

    # 新增组合特征（示例：车龄 × 里程）
    if 'age' in X_train.columns and 'mileage' in X_train.columns:
        X_train['age_mileage'] = X_train['age'] * X_train['mileage']
        X_val['age_mileage'] = X_val['age'] * X_val['mileage']

    # 新增动力密度特征
    if 'engine_power' in X_train.columns and 'weight' in X_train.columns:
        X_train['power_to_weight'] = X_train['engine_power'] / (X_train['weight'] + 1e-5)
        X_val['power_to_weight'] = X_val['engine_power'] / (X_val['weight'] + 1e-5)

    # 新增对数变换特征
    if 'mileage' in X_train.columns:
        X_train['log_mileage'] = np.log1p(X_train['mileage'])
        X_val['log_mileage'] = np.log1p(X_val['mileage'])

    # 类别特征目标编码（带平滑）
    if 'brand' in X_train.columns:
        encoder = TargetEncoder(
            min_samples_leaf=20,
            smoothing=10.0,
            handle_unknown='value',
            handle_missing='value'
        )
        X_train['brand_encoded'] = encoder.fit_transform(X_train['brand'], y_train)
        X_val['brand_encoded'] = encoder.transform(X_val['brand'])

    return X_train, X_val


# ==================== CatBoost 相关函数 ====================
def load_processed_data():
    """
    加载预处理后的数据
    """
    print("正在加载预处理后的数据...")
    X_train = joblib.load('processed_data/X_train.joblib')
    X_val = joblib.load('processed_data/X_val.joblib')
    y_train = joblib.load('processed_data/y_train.joblib')
    y_val = joblib.load('processed_data/y_val.joblib')
    return X_train, X_val, y_train, y_val


# 自定义回调类
class ValidationMAECallback:
    def after_iteration(self, info):
        # 将 SimpleNamespace 转换为字典
        info_dict = vars(info)
        if 'validation' in info_dict and 'MAE' in info_dict['validation']:
            current_mae = info_dict['validation']['MAE']
            print(f"Iteration {info_dict['iteration']}: 验证集 MAE = {current_mae:.2f}")
        return True  # 返回True继续训练


def train_catboost_model(X_train, X_val, y_train, y_val):
    """
    训练CatBoost模型，并在训练过程中观察验证集MAE
    """
    print("正在训练CatBoost模型...")

    # 设置模型参数（优化后）
    params = {
        'iterations': 5000,           # 迭代次数
        'learning_rate': 0.035,       # 学习率
        'depth': 5,                   # 树的深度
        'l2_leaf_reg': 8,             # L2正则化
        'min_data_in_leaf': 20,       # 叶子最小样本数
        'rsm': 0.8,                   # 特征子采样
        'bootstrap_type': 'Bernoulli',# 采样方式
        'subsample': 0.8,             # 样本子采样
        'random_strength': 1.5,       # 随机性增强
        'random_seed': 42,
        'od_type': 'Iter',            # 早停类型
        'od_wait': 200,               # 早停等待轮数
        'verbose': 100,               # 每100轮打印一次
        'loss_function': 'MAE',       # 损失函数
        'eval_metric': 'MAE',         # 评估指标
        'task_type': 'CPU',           # 使用CPU训练
        'thread_count': -1            # 使用所有CPU核心
    }

    # 创建模型
    model = CatBoostRegressor(**params)

    # 创建回调实例
    callback = ValidationMAECallback()

    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,          # 使用最佳模型
        plot=True,                    # 绘制训练过程
        callbacks=[callback]          # 动态打印验证集MAE
    )

    # 保存模型和训练集特征列
    model.save_model('processed_data/catboost_model.cbm')
    joblib.dump(X_train.columns.tolist(), 'processed_data/train_columns.joblib')
    print("模型已保存到 processed_data/catboost_model.cbm")
    print("训练集特征列已保存到 processed_data/train_columns.joblib")

    return model


def evaluate_model(model, X_val, y_val):
    """
    评估模型性能
    """
    # 预测
    y_pred = model.predict(X_val)

    # 计算评估指标
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print("\n模型评估结果：")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"R2分数: {r2:.4f}")

    # 绘制预测值与实际值的对比图
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


def plot_feature_importance(model, X_train):
    """
    绘制特征重要性图
    """
    # 获取特征重要性
    importance = model.get_feature_importance()
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    # 保存特征重要性到CSV
    feature_importance.to_csv('catboost_feature_importance.csv', index=False)

    # 绘制特征重要性图
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('CatBoost Top 20 特征重要性')
    plt.tight_layout()
    plt.savefig('catboost_feature_importance.png')
    plt.close()


def predict_test_data():
    """
    预测测试集数据
    """
    print("\n正在加载测试数据...")
    # 加载测试数据和模型
    test_data = joblib.load('processed_data/test_data.joblib')
    sale_ids = joblib.load('processed_data/sale_ids.joblib')
    model = CatBoostRegressor()
    model.load_model('processed_data/catboost_model.cbm')

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

    # 保存预测结果
    submit_data.to_csv('catboost_submit_result.csv', index=False)
    print("预测结果已保存到 catboost_submit_result.csv")


# ==================== 主函数 ====================
def main():
    # 加载原始数据
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = load_processed_data()

    # 数据预处理
    X_train_clean, y_train_clean, X_val_clean, y_val_clean = preprocess_data(
        X_train_raw, y_train_raw, X_val_raw, y_val_raw
    )

    # 特征工程
    X_train_eng, X_val_eng = engineer_features(X_train_clean, y_train_clean, X_val_clean)

    # 训练模型
    model = train_catboost_model(X_train_eng, X_val_eng, y_train_clean, y_val_clean)

    # 评估模型
    rmse, mae, r2 = evaluate_model(model, X_val_eng, y_val_clean)

    # 绘制特征重要性
    plot_feature_importance(model, X_train_eng)

    # 预测测试集
    predict_test_data()

    print("\n模型训练、评估和预测完成！")


if __name__ == "__main__":
    main()