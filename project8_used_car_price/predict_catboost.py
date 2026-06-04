# -*- coding: utf-8 -*-
"""
基于已训练的CatBoost模型进行二手车价格预测
并解决分类特征中的NaN值问题
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_test_data():
    """
    加载测试数据
    """
    print("正在加载测试数据...")
    
    # 判断是否存在预处理后的测试数据
    if os.path.exists('processed_data/fe_test_data.joblib') and os.path.exists('processed_data/fe_sale_ids.joblib'):
        test_data = joblib.load('processed_data/fe_test_data.joblib')
        test_ids = joblib.load('processed_data/fe_sale_ids.joblib')
        print(f"已加载预处理后的测试数据, 形状: {test_data.shape}")
    else:
        print("未找到预处理后的测试数据，请先运行特征工程代码")
        return None, None
    
    return test_data, test_ids

def load_model_and_cat_features():
    """
    加载模型和分类特征信息
    """
    print("正在加载模型...")
    
    # 加载模型
    if os.path.exists('processed_data/fe_catboost_model.cbm'):
        model = CatBoostRegressor()
        model.load_model('processed_data/fe_catboost_model.cbm')
        print("模型加载成功")
    else:
        print("未找到模型文件，请先训练模型")
        return None, None
    
    # 加载分类特征列表
    if os.path.exists('processed_data/fe_cat_features.joblib'):
        cat_features = joblib.load('processed_data/fe_cat_features.joblib')
        print(f"分类特征加载成功，共 {len(cat_features)} 个特征")
    else:
        # 如果没有保存的分类特征列表，尝试从模型中获取
        try:
            cat_features = model.get_cat_feature_indices()
            print(f"从模型中获取分类特征，共 {len(cat_features)} 个特征")
        except:
            print("无法获取分类特征信息")
            return model, []
    
    return model, cat_features

def preprocess_test_data(test_data, cat_features):
    """
    预处理测试数据，处理分类特征中的NaN值问题
    """
    print("正在预处理测试数据...")
    
    # 处理测试数据中的分类特征
    test_data_clean = test_data.copy()
    
    # 检查category类型列
    category_cols = test_data_clean.select_dtypes(['category']).columns
    print(f"Category类型列: {len(category_cols)}个")
    
    # 处理所有可能的分类特征
    all_cat_features = list(set(list(cat_features) + list(category_cols)))
    
    for col in all_cat_features:
        if col in test_data_clean.columns:
            # 检查是否有缺失值
            null_count = test_data_clean[col].isnull().sum()
            if null_count > 0:
                print(f"列 '{col}' 有 {null_count} 个缺失值，填充为'未知'")
                # 对于category类型需要特殊处理
                if test_data_clean[col].dtype.name == 'category':
                    # 先添加'未知'到类别，然后填充
                    if '未知' not in test_data_clean[col].cat.categories:
                        test_data_clean[col] = test_data_clean[col].cat.add_categories(['未知'])
                    test_data_clean[col] = test_data_clean[col].fillna('未知')
                else:
                    test_data_clean[col] = test_data_clean[col].fillna('未知')
            
            # 确保列是字符串类型
            test_data_clean[col] = test_data_clean[col].astype(str)
    
    return test_data_clean

def predict_prices(model, test_data_clean, test_ids, cat_features):
    """
    使用CatBoost模型预测价格
    """
    print("正在进行预测...")
    
    # 过滤掉不在test_data_clean中的cat_features
    valid_cat_features = [col for col in cat_features if col in test_data_clean.columns]
    print(f"有效分类特征: {len(valid_cat_features)}个")
    
    # 创建测试数据池
    try:
        test_pool = Pool(test_data_clean, cat_features=valid_cat_features)
        print("测试数据池创建成功")
    except Exception as e:
        print(f"创建数据池时出错: {e}")
        
        # 打印所有分类特征的数据类型和示例值
        for col in valid_cat_features:
            print(f"特征 '{col}': 类型={test_data_clean[col].dtype}, 示例值={test_data_clean[col].iloc[0]}")
            print(f"是否有空值: {test_data_clean[col].isnull().any()}")
            
        # 尝试不使用cat_features直接预测
        print("尝试不使用分类特征进行预测...")
        predictions = model.predict(test_data_clean)
        return test_ids, predictions
    
    # 预测
    predictions = model.predict(test_pool)
    print(f"预测完成，共 {len(predictions)} 个预测结果")
    
    return test_ids, predictions

def save_predictions(test_ids, predictions):
    """
    保存预测结果
    """
    # 创建提交文件
    submit_data = pd.DataFrame({
        'SaleID': test_ids,
        'price': predictions
    })
    
    # 保存预测结果
    output_file = 'fe_catboost_submit_result.csv'
    submit_data.to_csv(output_file, index=False)
    print(f"预测结果已保存到 {output_file}")
    
    # 显示预测结果的基本统计信息
    print("\n预测结果统计信息:")
    print(submit_data['price'].describe())

def main():
    # 创建必要的目录
    os.makedirs('processed_data', exist_ok=True)
    
    # 加载测试数据
    test_data, test_ids = load_test_data()
    if test_data is None or test_ids is None:
        return
    
    # 加载模型和分类特征
    model, cat_features = load_model_and_cat_features()
    if model is None:
        return
    
    # 预处理测试数据
    test_data_clean = preprocess_test_data(test_data, cat_features)
    
    # 进行预测
    test_ids, predictions = predict_prices(model, test_data_clean, test_ids, cat_features)
    
    # 保存预测结果
    save_predictions(test_ids, predictions)
    
    print("\n预测完成!")

if __name__ == "__main__":
    main() 