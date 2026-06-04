# -*- coding: utf-8 -*-
"""
二手车价格预测 - XGBoost模型
"""

import pandas as pd
import numpy as np
import xgboost as xgb
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

def train_xgboost_model(X_train, X_val, y_train, y_val):
    """
    训练XGBoost模型
    """
    print("正在训练XGBoost模型...")
    
    # 设置模型参数
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.001,  # 降低学习率
        'max_depth': 8,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 30000,  # 增加树的数量
        'random_state': 42,
        'eval_metric': 'mae',
        'early_stopping_rounds': 200  # 添加早停机制
    }
    
    # 创建模型
    model = xgb.XGBRegressor(**params)
    
    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100  # 每100轮打印一次评估结果
    )
    
    # 获取最佳迭代次数
    best_iteration = model.best_iteration
    print(f"\n最佳迭代次数: {best_iteration}")
    
    # 保存模型
    joblib.dump(model, 'processed_data/xgboost_model.joblib')
    print("模型已保存到 processed_data/xgboost_model.joblib")
    
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
    plt.savefig('prediction_vs_actual.png')
    plt.close()
    
    return rmse, mae, r2

def plot_feature_importance(model, X_train):
    """
    绘制特征重要性图
    """
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 特征重要性')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def predict_test_data():
    """
    预测测试集数据
    """
    print("\n正在加载测试数据...")
    # 加载测试数据和模型
    test_data = joblib.load('processed_data/test_data.joblib')
    sale_ids = joblib.load('processed_data/sale_ids.joblib')
    model = joblib.load('processed_data/xgboost_model.joblib')
    
    # 预测
    print("正在预测测试集...")
    predictions = model.predict(test_data)
    
    # 创建提交文件
    submit_data = pd.DataFrame({
        'SaleID': sale_ids,
        'price': predictions
    })
    
    # 保存预测结果
    submit_data.to_csv('submit_result-xgboost.csv', index=False)
    print("预测结果已保存到 submit_result-xgboost.csv")

def main():
    # 加载预处理后的数据
    X_train, X_val, y_train, y_val = load_processed_data()
    
    # 训练模型
    model = train_xgboost_model(X_train, X_val, y_train, y_val)
    
    # 评估模型
    rmse, mae, r2 = evaluate_model(model, X_val, y_val)
    
    # 绘制特征重要性
    plot_feature_importance(model, X_train)
    
    # 预测测试集
    predict_test_data()
    
    print("\n模型训练、评估和预测完成！")

if __name__ == "__main__":
    main() 