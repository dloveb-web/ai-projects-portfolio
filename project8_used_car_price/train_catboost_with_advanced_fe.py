# 在你的原始脚本基础上修改 engineer_features 函数

from advanced_feature_engineering import AdvancedFeatureEngineering

def engineer_features(X_train, y_train, X_val):
    """
    改进版特征工程：使用高级特征工程模块
    """
    print("🔧 应用高级特征工程...\n")
    
    # 初始化特征工程器
    fe = AdvancedFeatureEngineering(target_col='price')
    
    # 执行综合特征工程
    X_train, X_val = fe.execute_all(X_train, X_val, y_train, enable_pca=False)
    
    # 删除原始的高相关特征（可选）
    corr_matrix = X_train.select_dtypes(include=[np.number]).corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [
        column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)
    ]
    
    if high_corr_features:
        print(f"删除高度相关特征: {high_corr_features}")
        X_train = X_train.drop(columns=high_corr_features, errors='ignore')
        X_val = X_val.drop(columns=high_corr_features, errors='ignore')
    
    # 删除缺失率过高的特征
    missing_ratio = X_train.isnull().sum() / len(X_train)
    high_missing_cols = missing_ratio[missing_ratio > 0.3].index.tolist()
    if high_missing_cols:
        print(f"删除缺失率>30%的特征: {high_missing_cols}")
        X_train = X_train.drop(columns=high_missing_cols, errors='ignore')
        X_val = X_val.drop(columns=high_missing_cols, errors='ignore')
    
    print(f"\n最终特征数: {X_train.shape[1]}\n")
    
    return X_train, X_val