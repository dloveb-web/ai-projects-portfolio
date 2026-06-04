# -*- coding: utf-8 -*-
"""
二手车价格预测 - CatBoost模型
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
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

def train_catboost_model(X_train, X_val, y_train, y_val):
    """
    训练CatBoost模型
    """
    print("正在训练CatBoost模型...")
    
    # 设置模型参数
    params = {
        'iterations': 10000,  # 迭代次数
        'learning_rate': 0.04,  # 学习率
        'depth': 5,  # 树的深度
        'l2_leaf_reg': 3,  # L2正则化
        'bootstrap_type': 'Bayesian',  # 采样方式
        'random_seed': 42,
        'od_type': 'Iter',  # 早停类型
        'od_wait': 200,  # 早停等待轮数
        'verbose': 100,  # 每100轮打印一次
        'loss_function': 'MAE',  # 损失函数
        'eval_metric': 'MAE',  # 评估指标
        'task_type': 'CPU',  # 使用CPU训练
        'thread_count': -1  # 使用所有CPU核心
    }
    
    # 创建模型
    model = CatBoostRegressor(**params)
    
    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,  # 使用最佳模型
        plot=True  # 绘制训练过程
    )
    
    # 保存模型
    model.save_model('processed_data/catboost_model.cbm')
    print("模型已保存到 processed_data/catboost_model.cbm")
    
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

def main():
    # 加载预处理后的数据
    X_train, X_val, y_train, y_val = load_processed_data()
    
    # 训练模型
    model = train_catboost_model(X_train, X_val, y_train, y_val)
    
    # 评估模型
    rmse, mae, r2 = evaluate_model(model, X_val, y_val)
    
    # 绘制特征重要性
    plot_feature_importance(model, X_train)
    
    # 预测测试集
    predict_test_data()
    
    print("\n模型训练、评估和预测完成！")

if __name__ == "__main__":
    main() 