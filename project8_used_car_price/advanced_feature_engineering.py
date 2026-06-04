# -*- coding: utf-8 -*-
"""
高级特征工程模块 - 针对二手车价格预测的深度优化
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, KBinsDiscretizer
from sklearn.decomposition import PCA
from category_encoders import TargetEncoder, BinaryEncoder, OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineering:
    """高级特征工程类"""
    
    def __init__(self, target_col='price'):
        self.target_col = target_col
        self.feature_stats = {}
        self.scaler = StandardScaler()
        
    # ================== 方案一：基于领域知识的特征 ==================
    def create_domain_features(self, X_train, X_val, y_train):
        """
        基于二手车领域知识的特征创建
        重点：车辆价值衰减规律、性能指标、使用成本等
        """
        print("【方案一】创建领域知识特征...")
        
        # 1️⃣ 价值衰减特征（关键！）
        if 'age' in X_train.columns:
            # 非线性衰减：新车衰减快，老车衰减慢
            X_train['age_squared'] = X_train['age'] ** 2
            X_val['age_squared'] = X_val['age'] ** 2
            
            # 分段衰减：3年、5年、10年为关键时间点
            X_train['age_bracket'] = pd.cut(
                X_train['age'], 
                bins=[0, 3, 5, 10, 20],
                labels=['新车', '准新', '中年', '老车']
            )
            X_val['age_bracket'] = pd.cut(
                X_val['age'],
                bins=[0, 3, 5, 10, 20],
                labels=['新车', '准新', '中年', '老车']
            )
            
            # 对数衰减（模型常用）
            X_train['log_age'] = np.log1p(X_train['age'])
            X_val['log_age'] = np.log1p(X_val['age'])
            
            print("  ✓ 添加: age_squared, age_bracket, log_age")

        # 2️⃣ 里程特征（关键！）
        if 'mileage' in X_train.columns:
            # 平均年里程（使用强度）
            if 'age' in X_train.columns:
                X_train['mileage_per_year'] = X_train['mileage'] / (X_train['age'] + 1)
                X_val['mileage_per_year'] = X_val['mileage'] / (X_val['age'] + 1)
                print("  ✓ 添加: mileage_per_year")
            
            # 里程段（保养成本）
            X_train['mileage_bracket'] = pd.cut(
                X_train['mileage'],
                bins=[0, 50000, 100000, 150000, 200000, 1000000],
                labels=['优', '良', '中', '差', '很差']
            )
            X_val['mileage_bracket'] = pd.cut(
                X_val['mileage'],
                bins=[0, 50000, 100000, 150000, 200000, 1000000],
                labels=['优', '良', '中', '差', '很差']
            )
            print("  ✓ 添加: mileage_bracket")

        # 3️⃣ 发动机特征（动力性）
        if 'engine_power' in X_train.columns:
            # 功率等级分类
            X_train['power_level'] = pd.qcut(
                X_train['engine_power'],
                q=5,
                labels=['低功率', '中低功率', '中功率', '中高功率', '高功率'],
                duplicates='drop'
            )
            X_val['power_level'] = pd.qcut(
                X_val['engine_power'],
                q=5,
                labels=['低功率', '中低功率', '中功率', '中高功率', '高功率'],
                duplicates='drop'
            )
            print("  ✓ 添加: power_level")

        # 4️⃣ 车重特征（安全性、油耗）
        if 'weight' in X_train.columns:
            # 重量等级
            X_train['weight_category'] = pd.cut(
                X_train['weight'],
                bins=[0, 1000, 1500, 2000, 3000],
                labels=['轻型车', '中轻���车', '中型车', '重型车']
            )
            X_val['weight_category'] = pd.cut(
                X_val['weight'],
                bins=[0, 1000, 1500, 2000, 3000],
                labels=['轻型车', '中轻型车', '中型车', '重型车']
            )
            print("  ✓ 添加: weight_category")

        # 5️⃣ 综合性能指标
        if 'engine_power' in X_train.columns and 'weight' in X_train.columns:
            # 动力密度（越高越好）
            X_train['power_density'] = X_train['engine_power'] / (X_train['weight'] + 1e-5)
            X_val['power_density'] = X_val['engine_power'] / (X_val['weight'] + 1e-5)
            
            # 动力密度等级
            X_train['power_density_level'] = pd.qcut(
                X_train['power_density'],
                q=4,
                labels=['低', '中低', '中高', '高'],
                duplicates='drop'
            )
            X_val['power_density_level'] = pd.qcut(
                X_val['power_density'],
                q=4,
                labels=['低', '中低', '中高', '高'],
                duplicates='drop'
            )
            print("  ✓ 添加: power_density, power_density_level")

        # 6️⃣ 油耗成本估算（如果有排量信息）
        if 'displacement' in X_train.columns:
            # 排量等级
            X_train['displacement_level'] = pd.cut(
                X_train['displacement'],
                bins=[0, 1.5, 2.0, 2.5, 10],
                labels=['小排量', '中小排量', '中排量', '大排量']
            )
            X_val['displacement_level'] = pd.cut(
                X_val['displacement'],
                bins=[0, 1.5, 2.0, 2.5, 10],
                labels=['小排量', '中小排量', '中排量', '大排量']
            )
            print("  ✓ 添加: displacement_level")

        return X_train, X_val

    # ================== 方案二：交互特征优化 ==================
    def create_interaction_features(self, X_train, X_val):
        """
        智能交互特征：只创建相关性强的交互项
        避免过多交互特征导致维度爆炸
        """
        print("\n【方案二】创建智能交互特征...")
        
        interaction_count = 0
        
        # 只在关键列存在时才创建交互
        feature_pairs = [
            ('age', 'mileage', '车龄×里程'),
            ('engine_power', 'weight', '功率×重量'),
            ('mileage', 'power_density', '里程×功率密度') if 'power_density' in X_train.columns else None,
            ('age', 'engine_power', '车龄×功率'),
        ]
        
        for pair in feature_pairs:
            if pair is None:
                continue
            
            feat1, feat2, name = pair
            if feat1 in X_train.columns and feat2 in X_train.columns:
                # 加法交互（通常比乘法更稳定）
                X_train[f'{feat1}_{feat2}_add'] = X_train[feat1] + X_train[feat2]
                X_val[f'{feat1}_{feat2}_add'] = X_val[feat1] + X_val[feat2]
                
                # 乘法交互（仅对重要特征）
                if feat1 in ['age', 'engine_power'] or feat2 in ['age', 'engine_power']:
                    X_train[f'{feat1}_{feat2}_mul'] = X_train[feat1] * X_train[feat2]
                    X_val[f'{feat1}_{feat2}_mul'] = X_val[feat1] * X_val[feat2]
                    interaction_count += 2
                else:
                    interaction_count += 1
                
                print(f"  ✓ 添加: {name}")
        
        print(f"  共添加 {interaction_count} 个交互特征")
        return X_train, X_val

    # ================== 方案三：多项式特征（谨慎使用） ==================
    def create_polynomial_features(self, X_train, X_val, degree=2, columns=None):
        """
        对关键特征创建多项式特征
        仅对top特征使用，防止维度爆炸
        """
        print("\n【方案三】创建多项式特征...")
        
        if columns is None:
            # 只对数值型特征中的关键列创建多项式
            columns = [col for col in X_train.select_dtypes(include=[np.number]).columns 
                      if col in ['age', 'mileage', 'engine_power', 'power_density']]
        
        if not columns:
            print("  ⚠️  没有可用的多项式特征列")
            return X_train, X_val
        
        # 只创建 degree=2 的特征（避免过拟合）
        for col in columns:
            X_train[f'{col}_squared'] = X_train[col] ** 2
            X_val[f'{col}_squared'] = X_val[col] ** 2
            
            X_train[f'{col}_sqrt'] = np.sqrt(np.abs(X_train[col]))
            X_val[f'{col}_sqrt'] = np.sqrt(np.abs(X_val[col]))
        
        print(f"  ✓ 为 {columns} 添加平方和平方根特征")
        return X_train, X_val

    # ================== 方案四���分组统计特征 ==================
    def create_group_statistics(self, X_train, X_val, y_train):
        """
        基于分类特征的分组统计（中位数、平均数、分位数）
        关键：防止目标泄露和过拟合
        """
        print("\n【方案四】创建分组统计特征...")
        
        if 'brand' not in X_train.columns:
            print("  ⚠️  没有 brand 特征，跳过分组统计")
            return X_train, X_val
        
        # 获取品牌的统计信息
        brand_stats = X_train.groupby('brand')[y_train.name if y_train.name else 'target'].agg([
            'count',      # 样本量
            'mean',       # 平均价格
            'median',     # 中位数价格
            'std'         # 价格波动
        ]).rename(columns={
            'count': 'brand_count',
            'mean': 'brand_mean_price',
            'median': 'brand_median_price',
            'std': 'brand_price_std'
        })
        
        # 只保留样本量>10的品牌统计（防止噪声）
        brand_stats_filtered = brand_stats[brand_stats['brand_count'] >= 10]
        
        # 映射到训练集和验证集
        for col in brand_stats_filtered.columns:
            X_train[col] = X_train['brand'].map(brand_stats_filtered[col])
            X_val[col] = X_val['brand'].map(brand_stats_filtered[col])
            X_train[col].fillna(X_train[col].median(), inplace=True)
            X_val[col].fillna(X_val[col].median(), inplace=True)
        
        print(f"  ✓ 添加 {len(brand_stats_filtered.columns)} 个品牌分组特征")
        return X_train, X_val

    # ================== 方案五：特征二值化和分桶 ==================
    def create_binned_features(self, X_train, X_val):
        """
        将连续特征分桶处理，捕捉非线性关系
        """
        print("\n【方案五】创建分桶特征...")
        
        bin_features = {
            'age': [0, 2, 5, 8, 15, 100],
            'mileage': [0, 50000, 100000, 150000, 200000, 1000000],
            'engine_power': [0, 100, 150, 200, 300, 500],
        }
        
        for col, bins in bin_features.items():
            if col in X_train.columns:
                X_train[f'{col}_binned'] = pd.cut(X_train[col], bins=bins)
                X_val[f'{col}_binned'] = pd.cut(X_val[col], bins=bins)
                print(f"  ✓ 创建 {col}_binned 特征 ({len(bins)-1} 个桶)")
        
        return X_train, X_val

    # ================== 方案六：编码策略优化 ==================
    def encode_categorical_features(self, X_train, X_val, y_train):
        """
        使用多种编码策略处理分类特征
        1. TargetEncoder：低卡度、与目标强相关
        2. BinaryEncoder：高卡度（如品牌数>50）
        3. OrdinalEncoder：有序分类
        """
        print("\n【方案六】优化分类特征编码...")
        
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            unique_count = X_train[col].nunique()
            
            # 低卡度：使用目标编码
            if unique_count <= 20:
                encoder = TargetEncoder(
                    min_samples_leaf=25,
                    smoothing=15.0,
                    handle_unknown='value',
                    handle_missing='value'
                )
                X_train[f'{col}_target_encoded'] = encoder.fit_transform(
                    X_train[col], y_train
                )
                X_val[f'{col}_target_encoded'] = encoder.transform(X_val[col])
                print(f"  ✓ {col}（卡度{unique_count}）: TargetEncoder")
            
            # 中卡度：使用二进制编码
            elif unique_count <= 100:
                encoder = BinaryEncoder(
                    handle_unknown='value',
                    handle_missing='value'
                )
                encoded = encoder.fit_transform(X_train[col])
                X_train = pd.concat([X_train, encoded.add_prefix(f'{col}_binary_')], axis=1)
                
                encoded_val = encoder.transform(X_val[col])
                X_val = pd.concat([X_val, encoded_val.add_prefix(f'{col}_binary_')], axis=1)
                print(f"  ✓ {col}（卡度{unique_count}）: BinaryEncoder")
            
            # 高卡度：频率编码或直接删除
            else:
                freq_encoding = X_train[col].value_counts().to_dict()
                X_train[f'{col}_freq'] = X_train[col].map(freq_encoding)
                X_val[f'{col}_freq'] = X_val[col].map(freq_encoding).fillna(1)
                print(f"  ✓ {col}（卡度{unique_count}）: 频率编码")
        
        return X_train, X_val

    # ================== 方案七：PCA特征压缩 ==================
    def apply_pca_compression(self, X_train, X_val, n_components=None):
        """
        对高维数值特征进行PCA压缩
        保留90%的方差
        """
        print("\n【方案七】应用PCA特征压缩...")
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 5:
            print(f"  ⚠️  数值特征过少({len(numeric_cols)})，跳过PCA")
            return X_train, X_val
        
        if n_components is None:
            n_components = max(5, int(len(numeric_cols) * 0.7))
        
        pca = PCA(n_components=min(n_components, len(numeric_cols)))
        X_train_numeric = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
        X_val_numeric = X_val[numeric_cols].fillna(X_train[numeric_cols].mean())
        
        X_train_pca = pca.fit_transform(X_train_numeric)
        X_val_pca = pca.transform(X_val_numeric)
        
        # 转换为DataFrame并添加到原数据
        pca_cols = [f'pca_{i}' for i in range(n_components)]
        X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_cols, index=X_train.index)
        X_val_pca_df = pd.DataFrame(X_val_pca, columns=pca_cols, index=X_val.index)
        
        X_train = pd.concat([X_train, X_train_pca_df], axis=1)
        X_val = pd.concat([X_val, X_val_pca_df], axis=1)
        
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"  ✓ PCA 压缩为 {n_components} 维，保留方差: {explained_var:.2%}")
        
        return X_train, X_val

    # ================== 方案八：异常值特征化 ==================
    def create_anomaly_features(self, X_train, X_val):
        """
        识别异常值特征，而不是直接删除
        异常值本身可能包含重要信息
        """
        print("\n【方案八】创建异常值特征...")
        
        anomaly_cols = ['age', 'mileage', 'engine_power']
        
        for col in anomaly_cols:
            if col in X_train.columns:
                Q1 = X_train[col].quantile(0.25)
                Q3 = X_train[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 异常值标记（1表示异常，0表示正常）
                X_train[f'{col}_is_anomaly'] = (
                    (X_train[col] < lower_bound) | (X_train[col] > upper_bound)
                ).astype(int)
                X_val[f'{col}_is_anomaly'] = (
                    (X_val[col] < lower_bound) | (X_val[col] > upper_bound)
                ).astype(int)
                
                print(f"  ✓ {col}_is_anomaly 标记")
        
        return X_train, X_val

    # ================== 综合特征工程管道 ==================
    def execute_all(self, X_train, X_val, y_train, enable_pca=False):
        """
        执行所有特征工程方案（推荐）
        """
        print("="*60)
        print("🚀 启动综合特征工程管道")
        print("="*60 + "\n")
        
        # 方案一：领域知识特征（必须）
        X_train, X_val = self.create_domain_features(X_train, X_val, y_train)
        
        # 方案二：交互特征（推荐）
        X_train, X_val = self.create_interaction_features(X_train, X_val)
        
        # 方案三：多项式特征（可选，仅限关键特征）
        X_train, X_val = self.create_polynomial_features(
            X_train, X_val, 
            columns=['age', 'mileage', 'engine_power']
        )
        
        # 方案四：分组统计特征（推荐）
        X_train, X_val = self.create_group_statistics(X_train, X_val, y_train)
        
        # 方案五：分桶特征（推荐）
        X_train, X_val = self.create_binned_features(X_train, X_val)
        
        # 方案六：分类编码优化（必须）
        X_train, X_val = self.encode_categorical_features(X_train, X_val, y_train)
        
        # 方案七：PCA压缩（可选，仅当特征过多时使用）
        if enable_pca and X_train.select_dtypes(include=[np.number]).shape[1] > 30:
            X_train, X_val = self.apply_pca_compression(X_train, X_val)
        
        # 方案八：异常值特征（推荐）
        X_train, X_val = self.create_anomaly_features(X_train, X_val)
        
        print("\n" + "="*60)
        print(f"✨ 特征工程完成!")
        print(f"   原始特征数: N/A")
        print(f"   工程后特征数: {X_train.shape[1]}")
        print("="*60 + "\n")
        
        return X_train, X_val


# ================== 使用示例 ==================
def advanced_feature_engineering_example():
    """
    在你的主训练脚本中使用方式
    """
    # 加载数据
    X_train = pd.read_csv('X_train.csv')
    X_val = pd.read_csv('X_val.csv')
    y_train = pd.read_csv('y_train.csv').squeeze()
    
    # 初始化特征工程器
    fe = AdvancedFeatureEngineering(target_col='price')
    
    # 执行综合特征工程（推荐方式）
    X_train_eng, X_val_eng = fe.execute_all(X_train, X_val, y_train, enable_pca=False)
    
    # 或者只执行特定方案
    # X_train_eng, X_val_eng = fe.create_domain_features(X_train, X_val, y_train)
    # X_train_eng, X_val_eng = fe.create_group_statistics(X_train_eng, X_val_eng, y_train)
    
    return X_train_eng, X_val_eng


if __name__ == "__main__":
    print("高级特征工程模块已加载，请在主训练脚本中调用")