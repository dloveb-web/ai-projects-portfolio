# -*- coding: utf-8 -*-
"""
数据诊断脚本 - 检查训练集和测试集分布差异
"""

import pandas as pd
import numpy as np

print("="*70)
print("数据诊断 - 检查训练集vs测试集分布")
print("="*70)

# 加载数据
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"\n训练集: {len(train)}, 测试集: {len(test)}")

# 价格分布（训练集）
print("\n" + "="*70)
print("【训练集价格分布】")
print("="*70)
print(f"Mean: {train['price'].mean():.2f}")
print(f"Median: {train['price'].median():.2f}")
print(f"Std: {train['price'].std():.2f}")
print(f"Min: {train['price'].min():.2f}")
print(f"Max: {train['price'].max():.2f}")
print(f"25%: {train['price'].quantile(0.25):.2f}")
print(f"50%: {train['price'].quantile(0.50):.2f}")
print(f"75%: {train['price'].quantile(0.75):.2f}")
print(f"90%: {train['price'].quantile(0.90):.2f}")

# 关键特征对比
print("\n" + "="*70)
print("【训练集 vs 测试集特征对比】")
print("="*70)

for col in ['power', 'kilometer', 'brand', 'model', 'bodyType', 'fuelType', 'gearbox']:
    print(f"\n{col}:")
    if train[col].dtype == 'object':
        print(f"  训练集: {train[col].nunique()} 类别")
        print(f"  测试集: {test[col].nunique()} 类别")
        print(f"  重合类别: {len(set(train[col].unique()) & set(test[col].unique()))}")

        # 检查是否有新类别
        train_cats = set(train[col].dropna().unique())
        test_cats = set(test[col].dropna().unique())
        new_in_test = test_cats - train_cats
        if new_in_test:
            print(f"  ⚠️ 测试集新类别: {len(new_in_test)}个")
    else:
        print(f"  训练集 - Mean: {train[col].mean():.2f}, Std: {train[col].std():.2f}")
        print(f"  测试集 - Mean: {test[col].mean():.2f}, Std: {test[col].std():.2f}")

# v特征对比
print("\n" + "="*70)
print("【v特征对比】")
print("="*70)

v_cols = [f'v_{i}' for i in range(15)]
v_train_mean = train[v_cols].mean().mean()
v_test_mean = test[v_cols].mean().mean()

print(f"训练集v均值: {v_train_mean:.4f}")
print(f"测试集v均值: {v_test_mean:.4f}")
print(f"差异: {abs(v_train_mean - v_test_mean):.4f}")

# 检查缺失值
print("\n" + "="*70)
print("【缺失值统计】")
print("="*70)

print("\n训练集:")
for col in train.columns:
    missing = train[col].isna().sum()
    if missing > 0:
        print(f"  {col}: {missing} ({missing/len(train)*100:.1f}%)")

print("\n测试集:")
for col in test.columns:
    missing = test[col].isna().sum()
    if missing > 0:
        print(f"  {col}: {missing} ({missing/len(test)*100:.1f}%)")

# 检查异常值
print("\n" + "="*70)
print("【异常值检查】")
print("="*70)

for col in ['power', 'kilometer']:
    print(f"\n{col}:")
    q1 = train[col].quantile(0.25)
    q3 = train[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    train_outliers = ((train[col] < lower) | (train[col] > upper)).sum()
    test_outliers = ((test[col] < lower) | (test[col] > upper)).sum()

    print(f"  训练集异常值: {train_outliers} ({train_outliers/len(train)*100:.1f}%)")
    print(f"  测试集异常值: {test_outliers} ({test_outliers/len(test)*100:.1f}%)")

print(f"\n{'='*70}")
print("✅ 数据诊断完成！")
print(f"{'='*70}")
