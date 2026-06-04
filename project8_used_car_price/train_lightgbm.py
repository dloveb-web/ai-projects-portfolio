# -*- coding: utf-8 -*-
"""
二手车价格预测 - LightGBM模型
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

def train_lightgbm_model(X_train, X_val, y_train, y_val):
    """
    训练LightGBM模型
    """
    print("正在训练LightGBM模型...")
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # 设置模型参数
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 31,  # 叶子节点数
        'max_depth': 6,
        'min_data_in_leaf': 20,  # 每个叶子节点最少样本数
        'feature_fraction': 0.8,  # 相当于XGBoost的colsample_bytree
        'bagging_fraction': 0.8,  # 相当于XGBoost的subsample
        'bagging_freq': 5,  # 每5次迭代执行一次bagging
        'lambda_l1': 0.1,  # L1正则化
        'lambda_l2': 0.1,  # L2正则化
        'verbose': 1
    }
    
    print("\n开始训练...")
    # 训练模型
    callbacks = [
        lgb.early_stopping(stopping_rounds=100),  # 早停
        lgb.log_evaluation(period=100)  # 每100轮打印一次评估结果
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=12000,
        valid_sets=[train_data, val_data],
        valid_names=['训练集', '验证集'],
        callbacks=callbacks
    )
    
    # 保存模型
    model_path = 'processed_data/lightgbm_model.txt'
    model.save_model(model_path)
    print(f"\n模型已保存到 {model_path}")
    
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
    plt.title('预测价格 vs 实际价格')
    plt.tight_layout()
    plt.savefig('lightgbm_prediction_vs_actual.png')
    plt.close()
    
    return rmse, mae, r2

def plot_feature_importance(model, X_train):
    """
    绘制特征重要性图
    """
    # 获取特征重要性
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importance('gain')
    })
    importance = importance.sort_values('importance', ascending=False)
    
    # 绘制前20个最重要的特征
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=importance.head(20))
    plt.title('LightGBM - Top 20 特征重要性')
    plt.tight_layout()
    plt.savefig('lightgbm_feature_importance.png')
    plt.close()
    
    # 保存特征重要性到CSV文件
    importance.to_csv('feature_importance.csv', index=False)
    print("\n特征重要性已保存到 feature_importance.csv")

def predict_test_data():
    """
    预测测试集数据
    """
    print("\n正在加载测试数据...")
    # 加载测试数据和模型
    test_data = joblib.load('processed_data/test_data.joblib')
    sale_ids = joblib.load('processed_data/sale_ids.joblib')
    
    # 加载模型
    model = lgb.Booster(model_file='processed_data/lightgbm_model.txt')
    
    # 预测
    print("正在预测测试集...")
    predictions = model.predict(test_data)
    
    # 创建提交文件
    submit_data = pd.DataFrame({
        'SaleID': sale_ids,
        'price': predictions
    })
    
    # 保存预测结果
    submit_data.to_csv('lightgbm_submit_result.csv', index=False)
    print("预测结果已保存到 lightgbm_submit_result.csv")

def main():
    # 加载预处理后的数据
    X_train, X_val, y_train, y_val = load_processed_data()
    
    # 训练模型
    model = train_lightgbm_model(X_train, X_val, y_train, y_val)
    
    # 评估模型
    rmse, mae, r2 = evaluate_model(model, X_val, y_val)
    
    # 绘制特征重要性
    plot_feature_importance(model, X_train)
    
    # 预测测试集
    predict_test_data()
    
    print("\n模型训练、评估和预测完成！")

if __name__ == "__main__":
    main() 