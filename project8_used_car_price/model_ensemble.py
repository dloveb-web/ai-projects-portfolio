# -*- coding: utf-8 -*-
"""
模型融合 - XGBoost和LightGBM预测结果加权平均
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_predictions():
    """
    加载两个模型的预测结果
    """
    print("正在加载模型预测结果...")
    
    # 加载XGBoost预测结果
    xgb_pred = pd.read_csv('submit_result-xgboost.csv')
    print(f"XGBoost预测结果形状: {xgb_pred.shape}")
    
    # 加载LightGBM预测结果
    lgb_pred = pd.read_csv('lightgbm_submit_result.csv')
    print(f"LightGBM预测结果形状: {lgb_pred.shape}")
    
    # 验证SaleID是否一致
    if not (xgb_pred['SaleID'] == lgb_pred['SaleID']).all():
        raise ValueError("两个模型的SaleID不一致！")
    
    return xgb_pred, lgb_pred

def ensemble_predictions(xgb_pred, lgb_pred, weights=(0.5, 0.5)):
    """
    对两个模型的预测结果进行加权平均
    """
    print("\n开始模型融合...")
    print(f"XGBoost权重: {weights[0]}")
    print(f"LightGBM权重: {weights[1]}")
    
    # 确保权重和为1
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("权重之和必须为1！")
    
    # 计算加权平均
    ensemble_pred = pd.DataFrame({
        'SaleID': xgb_pred['SaleID'],
        'price': weights[0] * xgb_pred['price'] + weights[1] * lgb_pred['price']
    })
    
    return ensemble_pred

def analyze_predictions(xgb_pred, lgb_pred, ensemble_pred):
    """
    分析预测结果
    """
    print("\n预测结果分析：")
    print("-" * 50)
    
    # 基本统计信息
    print("\nXGBoost预测统计：")
    print(xgb_pred['price'].describe())
    print("\nLightGBM预测统计：")
    print(lgb_pred['price'].describe())
    print("\n融合后预测统计：")
    print(ensemble_pred['price'].describe())
    
    # 计算模型间的相关性
    correlation = xgb_pred['price'].corr(lgb_pred['price'])
    print(f"\n两个模型预测结果的相关性: {correlation:.4f}")
    
    # 计算预测差异
    diff = abs(xgb_pred['price'] - lgb_pred['price'])
    print(f"\n预测差异统计：")
    print(diff.describe())
    
    # 绘制预测结果对比图
    plt.figure(figsize=(15, 5))
    
    # 预测值散点图
    plt.subplot(1, 2, 1)
    plt.scatter(xgb_pred['price'], lgb_pred['price'], alpha=0.5)
    plt.plot([xgb_pred['price'].min(), xgb_pred['price'].max()], 
             [xgb_pred['price'].min(), xgb_pred['price'].max()], 
             'r--', lw=2)
    plt.xlabel('XGBoost预测价格')
    plt.ylabel('LightGBM预测价格')
    plt.title('两个模型预测结果对比')
    
    # 预测差异分布图
    plt.subplot(1, 2, 2)
    sns.histplot(diff, bins=50)
    plt.xlabel('预测差异')
    plt.ylabel('频数')
    plt.title('预测差异分布')
    
    plt.tight_layout()
    plt.savefig('ensemble_analysis.png')
    plt.close()

def save_ensemble_result(ensemble_pred):
    """
    保存融合后的预测结果
    """
    # 保存预测结果
    ensemble_pred.to_csv('submit_result.csv', index=False)
    print("\n融合后的预测结果已保存到 submit_result.csv")

def main():
    # 加载预测结果
    xgb_pred, lgb_pred = load_predictions()
    
    # 模型融合（可以调整权重）
    ensemble_pred = ensemble_predictions(xgb_pred, lgb_pred, weights=(0.5, 0.5))
    
    # 分析预测结果
    analyze_predictions(xgb_pred, lgb_pred, ensemble_pred)
    
    # 保存结果
    save_ensemble_result(ensemble_pred)
    
    print("\n模型融合完成！")

if __name__ == "__main__":
    main() 