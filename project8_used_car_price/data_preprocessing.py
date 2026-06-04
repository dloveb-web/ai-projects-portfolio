# -*- coding: utf-8 -*-
"""
二手车数据预处理
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime

def process_date(date_str):
    """
    处理日期字符串，提取年、月、日信息，并计算与基准日期的差值
    添加异常值处理
    """
    try:
        date_str = str(date_str)
        # 确保日期字符串长度为8
        if len(date_str) != 8:
            return pd.Series([np.nan, np.nan, np.nan, np.nan], 
                           index=['year', 'month', 'day', 'date'])
        
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        
        # 检查日期是否有效
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return pd.Series([np.nan, np.nan, np.nan, np.nan], 
                           index=['year', 'month', 'day', 'date'])
            
        # 转换为datetime对象
        date = datetime(year, month, day)
        
        return pd.Series([year, month, day, date], 
                        index=['year', 'month', 'day', 'date'])
    except (ValueError, TypeError):
        # 如果转换失败，返回NaN
        return pd.Series([np.nan, np.nan, np.nan, np.nan], 
                        index=['year', 'month', 'day', 'date'])

def analyze_categorical_features(data):
    """
    分析分类特征的唯一值个数
    """
    categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode', 'seller', 'offerType']
    
    print("\n分类特征分析：")
    print("-" * 50)
    for feature in categorical_features:
        if feature in data.columns:
            unique_values = data[feature].nunique()
            print(f"{feature}: {unique_values} 个唯一值")
            if unique_values < 10:  # 如果唯一值较少，显示具体值
                print(f"  具体值: {sorted(data[feature].unique())}")
    print("-" * 50)

def process_data(data, label_encoders=None, is_train=True):
    """
    处理数据，包括日期特征和类别特征
    """
    # 处理日期特征
    print("\n处理日期特征...")
    reg_date_features = data['regDate'].apply(process_date)
    creat_date_features = data['creatDate'].apply(process_date)
    
    # 检查日期处理后的缺失值
    print("\n日期处理后的缺失值统计：")
    print("注册日期缺失值：", reg_date_features['date'].isna().sum())
    print("创建日期缺失值：", creat_date_features['date'].isna().sum())
    
    # 添加日期特征
    data['reg_year'] = reg_date_features['year']
    data['reg_month'] = reg_date_features['month']
    data['reg_day'] = reg_date_features['day']
    data['creat_year'] = creat_date_features['year']
    data['creat_month'] = creat_date_features['month']
    data['creat_day'] = creat_date_features['day']
    
    # 计算与基准日期的差值（使用数据集中最早的有效注册日期作为基准）
    base_date = reg_date_features['date'].min()
    if pd.isna(base_date):
        print("警告：找不到有效的基准日期！")
    else:
        data['reg_date_diff'] = (reg_date_features['date'] - base_date).dt.days
        data['creat_date_diff'] = (creat_date_features['date'] - base_date).dt.days
        
        # 填充日期差值的缺失值
        data['reg_date_diff'].fillna(data['reg_date_diff'].median(), inplace=True)
        data['creat_date_diff'].fillna(data['creat_date_diff'].median(), inplace=True)
    
    # 填充年月日的缺失值
    for col in ['reg_year', 'reg_month', 'reg_day', 'creat_year', 'creat_month', 'creat_day']:
        data[col].fillna(data[col].median(), inplace=True)
    
    # 计算车龄（以年为单位）
    data['car_age'] = data['creat_year'] - data['reg_year']
    
    # 删除原始日期列
    data = data.drop(['regDate', 'creatDate'], axis=1)
    
    # 处理类别特征
    categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode', 'seller', 'offerType']
    
    if is_train:
        label_encoders = {}
        for feature in categorical_features:
            if feature in data.columns:
                label_encoders[feature] = LabelEncoder()
                data[feature] = label_encoders[feature].fit_transform(data[feature].astype(str))
        return data, label_encoders
    else:
        for feature in categorical_features:
            if feature in data.columns and feature in label_encoders:
                # 将未知类别值替换为最常见的类别
                unknown_mask = ~data[feature].astype(str).isin(label_encoders[feature].classes_)
                if unknown_mask.any():
                    most_common = data[feature].mode()[0]
                    data.loc[unknown_mask, feature] = most_common
                data[feature] = label_encoders[feature].transform(data[feature].astype(str))
        return data

def process_and_save_data():
    """
    处理训练数据并保存处理后的数据
    """
    # 创建保存目录
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    # 读取训练数据
    print("正在加载训练数据...")
    train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ', encoding='utf-8')
    
    # 查看日期数据的基本情况
    print("\n日期数据概览：")
    print("注册日期(regDate)唯一值示例：", train_data['regDate'].head().values)
    print("创建日期(creatDate)唯一值示例：", train_data['creatDate'].head().values)
    
    # 分析原始分类特征
    print("\n分析原始训练数据中的分类特征：")
    analyze_categorical_features(train_data)
    
    # 处理数据
    train_data, label_encoders = process_data(train_data, is_train=True)
    
    # 分析处理后的分类特征
    print("\n分析处理后的训练数据中的分类特征：")
    analyze_categorical_features(train_data)
    
    # 分离特征和目标变量
    X = train_data.drop('price', axis=1)
    y = train_data['price']
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 保存处理后的数据
    print("\n保存处理后的数据...")
    joblib.dump(X_train, 'processed_data/X_train.joblib')
    joblib.dump(X_val, 'processed_data/X_val.joblib')
    joblib.dump(y_train, 'processed_data/y_train.joblib')
    joblib.dump(y_val, 'processed_data/y_val.joblib')
    joblib.dump(label_encoders, 'processed_data/label_encoders.joblib')
    
    print("数据预处理完成！处理后的数据已保存到 processed_data 目录")
    
    return X_train, X_val, y_train, y_val, label_encoders

def process_test_data(label_encoders):
    """
    处理测试数据
    """
    print("\n正在处理测试数据...")
    # 读取测试数据
    test_data = pd.read_csv('used_car_testB_20200421.csv', sep=' ', encoding='utf-8')
    
    # 分析测试数据中的分类特征
    print("\n分析测试数据中的分类特征：")
    analyze_categorical_features(test_data)
    
    # 保存SaleID
    sale_ids = test_data['SaleID']
    
    # 处理测试数据
    test_data = process_data(test_data, label_encoders, is_train=False)
    
    # 分析处理后的测试数据中的分类特征
    print("\n分析处理后的测试数据中的分类特征：")
    analyze_categorical_features(test_data)
    
    # 保存处理后的测试数据
    joblib.dump(test_data, 'processed_data/test_data.joblib')
    joblib.dump(sale_ids, 'processed_data/sale_ids.joblib')
    
    print("测试数据处理完成！")
    return test_data, sale_ids

if __name__ == "__main__":
    # 处理训练数据
    X_train, X_val, y_train, y_val, label_encoders = process_and_save_data()
    
    # 处理测试数据
    process_test_data(label_encoders) 