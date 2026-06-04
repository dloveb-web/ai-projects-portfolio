# -*- coding: utf-8 -*-
"""
二手车数据探索性分析(EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 先读取文件的前几行来查看实际内容
print("原始文件内容预览：")
with open('used_car_train_20200313.csv', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 5:  # 只打印前5行
            print(f"第{i+1}行: {line.strip()}")
        else:
            break
    
print("\n尝试不同的分隔符读取：")
# 尝试不同的分隔符
separators = ['\t', ',', ';', '|']
for sep in separators:
    print(f"\n使用分隔符 '{sep}':")
    try:
        df = pd.read_csv('used_car_train_20200313.csv', sep=sep, nrows=5)
        print(f"列数: {len(df.columns)}")
        print("列名:", df.columns.tolist())
        print("\n数据预览:")
        print(df.head(2))
    except Exception as e:
        print(f"使用分隔符 '{sep}' 时出错: {str(e)}")

# 读取训练数据，使用空格分隔符
train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ', encoding='utf-8')

# 1. 数据概览
print("\n1. 数据基本信息：")
print(train_data.info())

print("\n2. 数据前5行：")
print(train_data.head())

print("\n3. 数据列名：")
print(train_data.columns.tolist())

print("\n4. 数据类型：")
print(train_data.dtypes)

print("\n5. 数值型特征的统计描述：")
print(train_data.describe())

# 检查price列是否存在
if 'price' in train_data.columns:
    print("\n6. price列的基本统计：")
    print(train_data['price'].describe())
else:
    print("\n6. 警告：数据中没有price列！")
    print("可用的列名：", train_data.columns.tolist())

# 6. 缺失值分析
missing_data = train_data.isnull().sum()
missing_percentage = (missing_data / len(train_data) * 100).round(2)
missing_info = pd.DataFrame({
    '缺失值数量': missing_data,
    '缺失值比例(%)': missing_percentage
})
print("\n7. 缺失值分析：")
print(missing_info[missing_info['缺失值数量'] > 0])

# 7. 数值型特征分布分析
numeric_features = train_data.select_dtypes(include=[np.number]).columns
print("\n8. 数值型特征：")
print(numeric_features.tolist())

# 绘制价格分布图
if 'price' in train_data.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_data, x='price', bins=50)
    plt.title('二手车价格分布')
    plt.xlabel('价格')
    plt.ylabel('频数')
    plt.savefig('price_distribution.png')
    plt.close()

# 8. 类别型特征分析
categorical_features = train_data.select_dtypes(include=['object']).columns
print("\n9. 类别型特征：")
print(categorical_features.tolist())

# 分析品牌分布
if 'brand' in train_data.columns:
    brand_counts = train_data['brand'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    brand_counts.plot(kind='bar')
    plt.title('Top 10 品牌分布')
    plt.xlabel('品牌')
    plt.ylabel('数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('brand_distribution.png')
    plt.close()

# 9. 相关性分析
numeric_data = train_data.select_dtypes(include=[np.number])
if len(numeric_data.columns) > 1:  # 至少需要两个数值型特征才能计算相关性
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

# 10. 目标变量(price)与其他特征的关系
if 'price' in train_data.columns:
    # 选择几个重要的数值型特征与价格的关系
    important_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2']
    for feature in important_features:
        if feature in train_data.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=train_data, x=feature, y='price')
            plt.title(f'{feature}与价格的关系')
            plt.tight_layout()
            plt.savefig(f'price_vs_{feature}.png')
            plt.close()

print("\nEDA分析完成！") 