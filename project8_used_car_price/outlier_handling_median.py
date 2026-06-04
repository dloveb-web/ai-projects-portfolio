# -*- coding: utf-8 -*-
"""
训练集异常数据处理 - 中位数填充方法
核心策略：识别异常值后，用合理的中位数进行填充
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 异常值检测 ====================
def detect_outliers_iqr(series, multiplier=1.5):
    """IQR方法检测异常值"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (series < lower_bound) | (series > upper_bound), lower_bound, upper_bound


def detect_outliers_percentile(series, lower_pct=0.01, upper_pct=0.99):
    """百分位数方法检测异常值"""
    lower_bound = series.quantile(lower_pct)
    upper_bound = series.quantile(upper_pct)
    return (series < lower_bound) | (series > upper_bound), lower_bound, upper_bound


# ==================== 中位数填充方法 ====================
def fill_price_with_median(train_df, method='iqr', plot=False):
    """
    价格异常值中位数填充

    策略：
    1. 识别异常值
    2. 根据品牌分组计算中位数
    3. 用对应品牌的中位数填充异常值
    """
    print(f"\n{'='*60}")
    print(f"方法1: 价格异常值中位数填充 ({method.upper()})")
    print(f"{'='*60}")

    clean_df = train_df.copy()

    # 检测异常值
    if method == 'iqr':
        outliers_mask, lower_bound, upper_bound = detect_outliers_iqr(clean_df['price'], multiplier=1.5)
    else:  # percentile
        outliers_mask, lower_bound, upper_bound = detect_outliers_percentile(clean_df['price'])

    outlier_count = outliers_mask.sum()

    # 计算各品牌价格中位数
    brand_median_price = clean_df.groupby('brand')['price'].median()

    # 中位数填充函数
    def fill_with_brand_median(row):
        if outliers_mask[row.name]:
            return brand_median_price.get(row['brand'], clean_df['price'].median())
        return row['price']

    # 填充异常值
    original_prices = clean_df['price'].copy()
    clean_df['price'] = clean_df.apply(fill_with_brand_median, axis=1)

    print(f"异常值数量: {outlier_count} ({outlier_count/len(train_df)*100:.2f}%)")
    print(f"异常值范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"原始价格范围: [{original_prices.min():.2f}, {original_prices.max():.2f}]")
    print(f"填充后价格范围: [{clean_df['price'].min():.2f}, {clean_df['price'].max():.2f}]")

    if plot:
        plot_median_fill_result(original_prices, clean_df['price'], outliers_mask, f"价格中位数填充 ({method.upper()})")

    return clean_df


def fill_power_with_median(train_df, plot=False):
    """
    功率异常值中位数填充

    策略：
    1. 识别功率为0的异常值
    2. 根据品牌计算正常功率的中位数
    3. 用中位数填充
    """
    print(f"\n{'='*60}")
    print(f"方法2: 功率异常值中位数填充")
    print(f"{'='*60}")

    clean_df = train_df.copy()

    # 识别功率为0的异常值
    power_outliers = clean_df['power'] == 0
    outlier_count = power_outliers.sum()

    # 计算各品牌正常功率的中位数
    brand_median_power = clean_df[clean_df['power'] > 0].groupby('brand')['power'].median()

    # 中位数填充
    original_power = clean_df['power'].copy()
    clean_df['power'] = clean_df.apply(
        lambda row: brand_median_power.get(row['brand'], clean_df[clean_df['power'] > 0]['power'].median())
        if power_outliers[row.name] else row['power'],
        axis=1
    )

    print(f"功率为0的异常值: {outlier_count} ({outlier_count/len(train_df)*100:.2f}%)")
    print(f"原始功率范围: [{original_power.min():.2f}, {original_power.max():.2f}]")
    print(f"填充后功率范围: [{clean_df['power'].min():.2f}, {clean_df['power'].max():.2f}]")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(original_power, bins=50, ax=axes[0], color='red', alpha=0.7)
        axes[0].set_title('原始功率分布', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('功率', fontsize=10)
        axes[0].grid(alpha=0.3)

        sns.histplot(clean_df['power'], bins=50, ax=axes[1], color='green', alpha=0.7)
        axes[1].set_title('中位数填充后功率分布', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('功率', fontsize=10)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('power_median_fill.png', dpi=150)
        plt.close()
        print(f"✓ 功率分布对比图已保存: power_median_fill.png")

    return clean_df


def fill_km_with_median(train_df, plot=False):
    """
    里程异常值中位数填充

    策略：
    1. 识别异常低里程（< 5万公里）
    2. 根据品牌和车龄分组计算中位数
    3. 用中位数填充
    """
    print(f"\n{'='*60}")
    print(f"方法3: 里程异常值中位数填充")
    print(f"{'='*60}")

    clean_df = train_df.copy()

    # 计算车龄
    clean_df['reg_year'] = clean_df['regDate'] // 10000
    clean_df['creat_year'] = clean_df['creatDate'] // 10000
    clean_df['car_age'] = (clean_df['creat_year'] - clean_df['reg_year']).clip(lower=0)

    # 识别异常低里程
    km_outliers = clean_df['kilometer'] < 5
    outlier_count = km_outliers.sum()

    # 计算各品牌-车龄组合的里程中位数
    brand_age_median_km = clean_df[clean_df['kilometer'] >= 5].groupby(['brand', 'car_age'])['kilometer'].median()

    # 中位数填充
    original_km = clean_df['kilometer'].copy()

    def fill_km_with_group_median(row):
        if km_outliers[row.name]:
            key = (row['brand'], row['car_age'])
            return brand_age_median_km.get(key, clean_df[clean_df['kilometer'] >= 5]['kilometer'].median())
        return row['kilometer']

    clean_df['kilometer'] = clean_df.apply(fill_km_with_group_median, axis=1)

    print(f"异常低里程数量: {outlier_count} ({outlier_count/len(train_df)*100:.2f}%)")
    print(f"原始里程范围: [{original_km.min():.2f}, {original_km.max():.2f}]")
    print(f"填充后里程范围: [{clean_df['kilometer'].min():.2f}, {clean_df['kilometer'].max():.2f}]")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(original_km, bins=50, ax=axes[0], color='red', alpha=0.7)
        axes[0].set_title('原始里程分布', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('里程(万公里)', fontsize=10)
        axes[0].grid(alpha=0.3)

        sns.histplot(clean_df['kilometer'], bins=50, ax=axes[1], color='green', alpha=0.7)
        axes[1].set_title('中位数填充后里程分布', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('里程(万公里)', fontsize=10)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('km_median_fill.png', dpi=150)
        plt.close()
        print(f"✓ 里程分布对比图已保存: km_median_fill.png")

    return clean_df


def fill_v_features_with_median(train_df):
    """
    v特征异常值中位数填充

    策略：
    1. 识别v特征的异常值（IQR方法）
    2. 用中位数填充
    """
    print(f"\n{'='*60}")
    print(f"方法4: v特征异常值中位数填充")
    print(f"{'='*60}")

    clean_df = train_df.copy()
    v_cols = [f'v_{i}' for i in range(15)]

    for col in v_cols:
        if col in clean_df.columns:
            # 检测异常值
            outliers_mask, lower_bound, upper_bound = detect_outliers_iqr(clean_df[col], multiplier=3)  # v特征用3倍IQR
            outlier_count = outliers_mask.sum()

            if outlier_count > 0:
                # 用中位数填充
                col_median = clean_df[col].median()
                clean_df.loc[outliers_mask, col] = col_median

                if outlier_count > 100:  # 只显示重要的
                    print(f"  {col}: {outlier_count} 个异常值已填充")

    print(f"✓ v特征异常值处理完成")

    return clean_df


# ==================== 综合中位数填充方案 ====================
def comprehensive_median_fill(train_df, plot=False):
    """
    综合中位数填充方案
    """
    print(f"\n{'='*60}")
    print(f"综合异常值中位数填充方案")
    print(f"{'='*60}")

    clean_df = train_df.copy()

    # 1. 填充功率异常值
    print("\n步骤1: 填充功率异常值...")
    clean_df = fill_power_with_median(clean_df, plot=plot)

    # 2. 填充里程异常值
    print("\n步骤2: 填充里程异常值...")
    clean_df = fill_km_with_median(clean_df, plot=plot)

    # 3. 填充价格异常值
    print("\n步骤3: 填充价格异常值...")
    clean_df = fill_price_with_median(clean_df, method='percentile', plot=plot)

    # 4. 填充v特征异常值
    print("\n步骤4: 填充v特征异常值...")
    clean_df = fill_v_features_with_median(clean_df)

    print(f"\n✓ 综合中位数填充完成")
    print(f"  处理后样本数: {len(clean_df)} (无样本损失)")

    return clean_df


# ==================== 可视化函数 ====================
def plot_median_fill_result(original, filled, outlier_mask, method_name):
    """绘制中位数填充结果"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 原始数据
    sns.histplot(original, bins=100, ax=axes[0], color='red', alpha=0.7)
    axes[0].set_title('原始价格分布', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('价格', fontsize=10)
    axes[0].set_ylabel('频数', fontsize=10)
    axes[0].grid(alpha=0.3)

    # 填充后数据
    sns.histplot(filled, bins=100, ax=axes[1], color='green', alpha=0.7)
    axes[1].set_title('填充后价格分布', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('价格', fontsize=10)
    axes[1].set_ylabel('频数', fontsize=10)
    axes[1].grid(alpha=0.3)

    # 对比箱线图
    data_to_plot = pd.DataFrame({
        '原始价格': original,
        '填充后价格': filled
    })
    data_to_plot_melted = data_to_plot.melt(var_name='类型', value_name='价格')

    sns.boxplot(x='类型', y='价格', data=data_to_plot_melted, ax=axes[2])
    axes[2].set_title('价格分布对比', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('价格', fontsize=10)
    axes[2].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    filename = method_name.replace(' ', '_') + '.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✓ 分布对比图已保存: {filename}")


# ==================== 主函数 ====================
def main():
    # 加载数据
    train = pd.read_csv('used_car_train_20200313.csv', sep=' ')

    print(f"{'='*60}")
    print(f"训练集异常值中位数填充处理")
    print(f"{'='*60}")
    print(f"原始训练集大小: {train.shape}")

    # 综合中位数填充
    clean_df = comprehensive_median_fill(train, plot=True)

    # 保存处理后的数据
    clean_df.to_csv('train_median_fill.csv', index=False)
    print(f"\n✓ 处理后的数据已保存: train_median_fill.csv")

    # 对比分析
    print(f"\n{'='*60}")
    print(f"处理前后对比")
    print(f"{'='*60}")

    comparison = pd.DataFrame({
        '指标': ['样本数', '价格均值', '价格标准差', '价格中位数',
                 '功率均值', '功率为0数量', '里程均值'],
        '原始数据': [
            len(train),
            train['price'].mean(),
            train['price'].std(),
            train['price'].median(),
            train['power'].mean(),
            (train['power'] == 0).sum(),
            train['kilometer'].mean()
        ],
        '中位数填充后': [
            len(clean_df),
            clean_df['price'].mean(),
            clean_df['price'].std(),
            clean_df['price'].median(),
            clean_df['power'].mean(),
            (clean_df['power'] == 0).sum(),
            clean_df['kilometer'].mean()
        ]
    })

    print(comparison.to_string(index=False))
    comparison.to_csv('median_fill_comparison.csv', index=False)
    print(f"\n✓ 对比结果已保存: median_fill_comparison.csv")


if __name__ == "__main__":
    main()
