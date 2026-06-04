# -*- coding: utf-8 -*-
"""
训练集异常数据处理方案
提供多种异常值处理方法，可根据实际情况选择
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 异常值检测函数 ====================
def detect_price_outliers(train_df, method='iqr'):
    """
    价格异常值检测

    Args:
        method: 'iqr', 'zscore', 'percentile', 'isolation'
    """
    prices = train_df['price'].values

    if method == 'iqr':
        # IQR方法
        Q1 = np.percentile(prices, 25)
        Q3 = np.percentile(prices, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (prices < lower_bound) | (prices > upper_bound)

    elif method == 'zscore':
        # Z-score方法
        z_scores = np.abs(stats.zscore(prices))
        outliers = z_scores > 3

    elif method == 'percentile':
        # 百分位数方法
        lower_bound = np.percentile(prices, 1)
        upper_bound = np.percentile(prices, 99)
        outliers = (prices < lower_bound) | (prices > upper_bound)

    elif method == 'isolation':
        # Isolation Forest
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso_forest.fit_predict(prices.reshape(-1, 1)) == -1

    return outliers


# ==================== 异常值处理方法 ====================
def remove_outliers(train_df, method='iqr', plot=False):
    """
    移除异常值
    """
    print(f"\n{'='*60}")
    print(f"方法1: 移除异常值 ({method.upper()})")
    print(f"{'='*60}")

    outliers = detect_price_outliers(train_df, method=method)
    clean_df = train_df[~outliers].copy()

    print(f"原始样本数: {len(train_df)}")
    print(f"异常值数量: {outliers.sum()} ({outliers.sum()/len(train_df)*100:.2f}%)")
    print(f"处理后样本数: {len(clean_df)}")

    if plot:
        plot_price_distribution(train_df, clean_df, f"移除异常值 ({method.upper()})")

    return clean_df


def cap_outliers(train_df, method='iqr', plot=False):
    """
    截断异常值（替换为边界值）
    """
    print(f"\n{'='*60}")
    print(f"方法2: 截断异常值 ({method.upper()})")
    print(f"{'='*60}")

    clean_df = train_df.copy()

    if method == 'iqr':
        Q1 = np.percentile(clean_df['price'], 25)
        Q3 = np.percentile(clean_df['price'], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        clean_df['price'] = clean_df['price'].clip(lower=lower_bound, upper=upper_bound)

    elif method == 'percentile':
        lower_bound = np.percentile(clean_df['price'], 1)
        upper_bound = np.percentile(clean_df['price'], 99)
        clean_df['price'] = clean_df['price'].clip(lower=lower_bound, upper=upper_bound)

    print(f"价格截断范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"原始价格范围: [{train_df['price'].min():.2f}, {train_df['price'].max():.2f}]")
    print(f"处理后价格范围: [{clean_df['price'].min():.2f}, {clean_df['price'].max():.2f}]")

    if plot:
        plot_price_distribution(train_df, clean_df, f"截断异常值 ({method.upper()})")

    return clean_df


def winsorize_outliers(train_df, limits=(0.05, 0.05), plot=False):
    """
    Winsorize异常值（用分位数替换）
    """
    print(f"\n{'='*60}")
    print(f"方法3: Winsorize异常值 (limits={limits})")
    print(f"{'='*60}")

    clean_df = train_df.copy()
    lower_limit, upper_limit = limits

    # 计算分位数
    lower_bound = np.percentile(clean_df['price'], lower_limit * 100)
    upper_bound = np.percentile(clean_df['price'], 100 - upper_limit * 100)

    # 替换异常值
    clean_df.loc[clean_df['price'] < lower_bound, 'price'] = lower_bound
    clean_df.loc[clean_df['price'] > upper_bound, 'price'] = upper_bound

    print(f"Winsorize范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"原始价格范围: [{train_df['price'].min():.2f}, {train_df['price'].max():.2f}]")
    print(f"处理后价格范围: [{clean_df['price'].min():.2f}, {clean_df['price'].max():.2f}]")

    if plot:
        plot_price_distribution(train_df, clean_df, f"Winsorize异常值")

    return clean_df


def fix_feature_outliers(train_df):
    """
    修复特征异常值（功率、里程等）
    """
    print(f"\n{'='*60}")
    print(f"方法4: 修复特征异常值")
    print(f"{'='*60}")

    clean_df = train_df.copy()

    # 修复功率为0的异常值
    power_zero_count = (clean_df['power'] == 0).sum()
    if power_zero_count > 0:
        # 用品牌的中位数填充
        brand_median_power = clean_df[clean_df['power'] > 0].groupby('brand')['power'].median()

        def fix_power(row):
            if row['power'] == 0:
                return brand_median_power.get(row['brand'], clean_df[clean_df['power'] > 0]['power'].median())
            return row['power']

        clean_df['power'] = clean_df.apply(fix_power, axis=1)
        print(f"修复功率异常值: {power_zero_count} 个样本")

    # 修复里程异常值
    # 里程最小值为0.5可能异常，用品牌-车龄的中位数填充
    km_stats = clean_df[clean_df['kilometer'] >= 5].groupby(['brand', 'regDate'])['kilometer'].median()

    def fix_km(row):
        if row['kilometer'] < 5:
            key = (row['brand'], row['regDate'])
            return km_stats.get(key, clean_df['kilometer'].median())
        return row['kilometer']

        clean_df['kilometer'] = clean_df.apply(fix_km, axis=1)
        print(f"修复里程异常值")

    return clean_df


def robust_regression_outliers(train_df, plot=False):
    """
    使用稳健回归识别并处理异常值
    """
    print(f"\n{'='*60}")
    print(f"方法5: 稳健回归识别异常值")
    print(f"{'='*60}")

    clean_df = train_df.copy()

    # 使用RANSAC回归识别异常值
    from sklearn.linear_model import RANSACRegressor

    # 准备特征
    X = clean_df[['power', 'kilometer', 'car_age']].values
    y = clean_df['price'].values

    # 稳健回归
    ransac = RANSACRegressor(random_state=42, max_trials=100)
    ransac.fit(X, y)

    # 识别异常值
    inlier_mask = ransac.inlier_mask_
    outlier_count = (~inlier_mask).sum()

    print(f"原始样本数: {len(train_df)}")
    print(f"识别异常值: {outlier_count} ({outlier_count/len(train_df)*100:.2f}%)")
    print(f"保留样本数: {inlier_mask.sum()}")

    if plot:
        clean_df['is_outlier'] = ~inlier_mask
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='power', y='price', hue='is_outlier',
                    data=clean_df, alpha=0.6, s=20)
        plt.title('RANSAC稳健回归 - 功率 vs 价格', fontsize=14, fontweight='bold')
        plt.xlabel('功率', fontsize=12)
        plt.ylabel('价格', fontsize=12)
        plt.legend(['正常值', '异常值'])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('ransac_outliers.png', dpi=150)
        plt.close()
        print(f"✓ 异常值可视化已保存: ransac_outliers.png")

    return clean_df[inlier_mask].copy()


# ==================== 可视化函数 ====================
def plot_price_distribution(original_df, clean_df, method_name):
    """绘制价格分布对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 原始数据
    sns.histplot(original_df['price'], bins=100, ax=axes[0], color='red', alpha=0.7)
    axes[0].set_title('原始价格分布', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('价格', fontsize=10)
    axes[0].set_ylabel('频数', fontsize=10)
    axes[0].grid(alpha=0.3)

    # 处理后数据
    sns.histplot(clean_df['price'], bins=100, ax=axes[1], color='green', alpha=0.7)
    axes[1].set_title(f'{method_name} 后价格分布', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('价格', fontsize=10)
    axes[1].set_ylabel('频数', fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    filename = method_name.replace(' ', '_') + '_distribution.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✓ 分布图已保存: {filename}")


# ==================== 综合处理方案 ====================
def comprehensive_outlier_handling(train_df, plot=False):
    """
    综合异常值处理方案
    """
    print(f"\n{'='*60}")
    print(f"综合异常值处理方案")
    print(f"{'='*60}")

    clean_df = train_df.copy()

    # 1. 修复特征异常值
    print("\n步骤1: 修复特征异常值...")
    clean_df = fix_feature_outliers(clean_df)

    # 2. Winsorize价格异常值
    print("\n步骤2: Winsorize价格异常值...")
    clean_df = winsorize_outliers(clean_df, limits=(0.01, 0.01))

    # 3. 识别并标记极端异常值
    print("\n步骤3: 识别极端异常值...")
    extreme_outliers = detect_price_outliers(clean_df, method='percentile')

    # 对极端异常值使用更强的Winsorize
    if extreme_outliers.sum() > 0:
        print(f"发现 {extreme_outliers.sum()} 个极端异常值，进行额外处理...")
        lower_bound = np.percentile(clean_df[~extreme_outliers]['price'], 5)
        upper_bound = np.percentile(clean_df[~extreme_outliers]['price'], 95)
        clean_df['price'] = clean_df['price'].clip(lower=lower_bound, upper=upper_bound)

    print(f"\n✓ 综合处理完成")
    print(f"  原始样本数: {len(train_df)}")
    print(f"  处理后样本数: {len(clean_df)}")
    print(f"  最终价格范围: [{clean_df['price'].min():.2f}, {clean_df['price'].max():.2f}]")

    if plot:
        plot_price_distribution(train_df, clean_df, "综合异常值处理")

    return clean_df


# ==================== 主函数 ====================
def main():
    # 加载数据
    train = pd.read_csv('used_car_train_20200313.csv', sep=' ')

    # 计算车龄
    train['reg_year'] = train['regDate'] // 10000
    train['creat_year'] = train['creatDate'] // 10000
    train['car_age'] = (train['creat_year'] - train['reg_year']).clip(lower=0)

    print(f"{'='*60}")
    print(f"训练集异常数据处理")
    print(f"{'='*60}")
    print(f"原始训练集大小: {train.shape}")

    # 测试各种方法
    # 方法1: 移除异常值 (IQR)
    clean_iqr = remove_outliers(train, method='iqr', plot=True)

    # 方法2: 截断异常值 (百分位数)
    clean_cap = cap_outliers(train, method='percentile', plot=True)

    # 方法3: Winsorize异常值
    clean_win = winsorize_outliers(train, limits=(0.05, 0.05), plot=True)

    # 方法4: 修复特征异常值
    clean_features = fix_feature_outliers(train)

    # 方法5: 综合处理
    clean_comprehensive = comprehensive_outlier_handling(train, plot=True)

    # 保存处理后的数据
    clean_comprehensive.to_csv('train_clean.csv', index=False)
    print(f"\n✓ 处理后的数据已保存: train_clean.csv")

    # 对比不同方法的影响
    print(f"\n{'='*60}")
    print(f"不同方法效果对比")
    print(f"{'='*60}")

    results = {
        '原始数据': train,
        'IQR移除': clean_iqr,
        '百分位数截断': clean_cap,
        'Winsorize': clean_win,
        '综合处理': clean_comprehensive
    }

    comparison = pd.DataFrame({
        method: {
            '样本数': len(df),
            '价格均值': df['price'].mean(),
            '价格标准差': df['price'].std(),
            '价格范围': f"{df['price'].min():.0f} ~ {df['price'].max():.0f}"
        }
        for method, df in results.items()
    }).T

    print(comparison.to_string())
    comparison.to_csv('outlier_handling_comparison.csv')
    print(f"\n✓ 对比结果已保存: outlier_handling_comparison.csv")


if __name__ == "__main__":
    main()
