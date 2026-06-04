# -*- coding: utf-8 -*-
"""
二手车价格预测 - 四模型融合优化版
(CatBoost + LightGBM + XGBoost + MLP深度网络)
目标：MAE < 450
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import joblib
import warnings
warnings.filterwarnings('ignore')


# ==================== 数据加载 ====================
def load_raw_data():
    """加载原始数据"""
    print("正在加载原始数据...")
    train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    return train, test


# ==================== 特征工程 ====================
def advanced_feature_engineering(train, test):
    """高级特征工程"""
    print("\n" + "="*60)
    print("进行高级特征工程...")
    print("="*60)
    
    train['is_train'] = 1
    test['is_train'] = 0
    combined = pd.concat([train, test], ignore_index=True)
    
    # 异常值处理
    combined['power'] = combined['power'].clip(0, 600)
    
    # ========== 深度特征工程 ==========
    v_cols = ['v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_11', 'v_14']
    
    # v特征统计
    combined['v_mean'] = combined[v_cols].mean(axis=1)
    combined['v_std'] = combined[v_cols].std(axis=1)
    combined['v_max'] = combined[v_cols].max(axis=1)
    combined['v_min'] = combined[v_cols].min(axis=1)
    combined['v_range'] = combined['v_max'] - combined['v_min']
    combined['v_median'] = combined[v_cols].median(axis=1)
    combined['v_skew'] = combined[v_cols].skew(axis=1)
    
    # v_0和v_3深度特征（最重要）
    combined['v_0_sq'] = combined['v_0'] ** 2
    combined['v_3_sq'] = combined['v_3'] ** 2
    combined['v_0_cube'] = combined['v_0'] ** 3
    combined['v_3_cube'] = combined['v_3'] ** 3
    combined['v_0_sqrt'] = np.sqrt(np.abs(combined['v_0']))
    combined['v_3_sqrt'] = np.sqrt(np.abs(combined['v_3']))
    combined['v_0_v_3'] = combined['v_0'] * combined['v_3']
    combined['v_0_v_3_ratio'] = combined['v_0'] / (combined['v_3'] + 1e-5)
    combined['v_0_v_3_diff'] = combined['v_0'] - combined['v_3']
    combined['v_0_v_3_diff_sq'] = combined['v_0_v_3_diff'] ** 2
    combined['v_0_sq_v_3'] = (combined['v_0'] ** 2) * combined['v_3']
    combined['v_0_v_3_sq'] = combined['v_0'] * (combined['v_3'] ** 2)
    
    # 三阶交互
    combined['v_0_v_2_v_3'] = combined['v_0'] * combined['v_2'] * combined['v_3']
    combined['v_1_v_2_v_3'] = combined['v_1'] * combined['v_2'] * combined['v_3']
    
    # 与power和kilometer交互
    combined['v_0_power'] = combined['v_0'] * combined['power']
    combined['v_3_power'] = combined['v_3'] * combined['power']
    combined['v_0_km'] = combined['v_0'] * combined['kilometer']
    combined['v_3_km'] = combined['v_3'] * combined['kilometer']
    combined['power_v_0_v_3'] = combined['power'] * combined['v_0_v_3']
    combined['km_power'] = combined['kilometer'] * combined['power']
    
    # 对数变换
    combined['log_power'] = np.log1p(combined['power'])
    combined['log_km'] = np.log1p(combined['kilometer'])
    
    # 分组统计
    combined['power_km_ratio'] = combined['power'] / (combined['kilometer'] + 1)
    
    # 分类变量编码
    cat_cols = ['brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    for col in cat_cols:
        combined[col] = combined[col].astype(str).fillna('missing')
        le = LabelEncoder()
        combined[col + '_enc'] = le.fit_transform(combined[col])
    
    # 品牌分组统计
    brand_stats = combined[combined['is_train'] == 1].groupby('brand_enc')['price'].agg(['mean', 'median', 'std', 'count'])
    brand_stats.columns = ['brand_price_mean', 'brand_price_median', 'brand_price_std', 'brand_count']
    combined = combined.merge(brand_stats, on='brand_enc', how='left')
    combined['brand_price_std'] = combined['brand_price_std'].fillna(0)
    combined['brand_price_mean'] = combined['brand_price_mean'].fillna(combined[combined['is_train']==1]['price'].mean())
    combined['brand_price_median'] = combined['brand_price_median'].fillna(combined[combined['is_train']==1]['price'].median())
    
    # 特征列表
    num_cols = [
        'power', 'kilometer',
        'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_11', 'v_14',
        'v_mean', 'v_std', 'v_max', 'v_min', 'v_range', 'v_median', 'v_skew',
        'v_0_sq', 'v_3_sq', 'v_0_cube', 'v_3_cube',
        'v_0_sqrt', 'v_3_sqrt', 'v_0_v_3', 'v_0_v_3_ratio',
        'v_0_v_3_diff', 'v_0_v_3_diff_sq',
        'v_0_sq_v_3', 'v_0_v_3_sq', 'v_0_v_2_v_3', 'v_1_v_2_v_3',
        'v_0_power', 'v_3_power', 'v_0_km', 'v_3_km', 'power_v_0_v_3', 'km_power',
        'log_power', 'log_km', 'power_km_ratio',
        'brand_price_mean', 'brand_price_median', 'brand_price_std', 'brand_count'
    ]
    
    cat_enc_cols = [c + '_enc' for c in cat_cols]
    
    # 分离数据
    train_data = combined[combined['is_train'] == 1]
    test_data = combined[combined['is_train'] == 0]
    
    X_num = train_data[num_cols].values.astype(np.float32)
    X_num_test = test_data[num_cols].values.astype(np.float32)
    X_cat = train_data[cat_enc_cols].values.astype(np.int64)
    X_cat_test = test_data[cat_enc_cols].values.astype(np.int64)
    
    y = train_data['price'].values.astype(np.float32)
    sale_ids = test_data['SaleID'].values
    
    # 标准化数值特征
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    X_num_test = scaler.transform(X_num_test)
    X_num = np.nan_to_num(X_num)
    X_num_test = np.nan_to_num(X_num_test)
    
    # 合并特征
    X = np.hstack([X_num, X_cat])
    X_test_final = np.hstack([X_num_test, X_cat_test])
    
    print(f"特征数: {X.shape[1]}")
    
    return X, y, X_test_final, sale_ids


# ==================== K折Stacking ====================
def kfold_stacking(X, y, X_test, n_splits=5):
    """K折Stacking训练四个模型"""
    print("\n" + "="*60)
    print("K折Stacking训练...")
    print("="*60)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 存储预测结果
    oof_cat = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_mlp = np.zeros(len(X))
    
    test_cat = np.zeros(len(X_test))
    test_lgb = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_mlp = np.zeros(len(X_test))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n{'='*20} Fold {fold+1}/{n_splits} {'='*20}")
        
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        # CatBoost
        print("训练CatBoost...")
        cat = CatBoostRegressor(
            iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=10,
            min_data_in_leaf=30, random_seed=42, verbose=0, loss_function='MAE'
        )
        cat.fit(X_tr, y_tr, verbose=0)
        oof_cat[val_idx] = cat.predict(X_val)
        test_cat += cat.predict(X_test) / n_splits
        cat_mae = mean_absolute_error(y_val, oof_cat[val_idx])
        print(f"  CatBoost MAE: {cat_mae:.2f}")
        
        # LightGBM
        print("训练LightGBM...")
        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'learning_rate': 0.02,
             'num_leaves': 63, 'max_depth': 8, 'verbose': -1, 'seed': 42},
            lgb_train, num_boost_round=2000
        )
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        test_lgb += lgb_model.predict(X_test) / n_splits
        lgb_mae = mean_absolute_error(y_val, oof_lgb[val_idx])
        print(f"  LightGBM MAE: {lgb_mae:.2f}")
        
        # XGBoost
        print("训练XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=2000, learning_rate=0.02, max_depth=8,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )
        xgb_model.fit(X_tr, y_tr, verbose=False)
        oof_xgb[val_idx] = xgb_model.predict(X_val)
        test_xgb += xgb_model.predict(X_test) / n_splits
        xgb_mae = mean_absolute_error(y_val, oof_xgb[val_idx])
        print(f"  XGBoost MAE: {xgb_mae:.2f}")
        
        # MLP深度网络
        print("训练MLP深度网络...")
        mlp = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=512,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False
        )
        mlp.fit(X_tr, y_tr)
        oof_mlp[val_idx] = mlp.predict(X_val)
        test_mlp += mlp.predict(X_test) / n_splits
        mlp_mae = mean_absolute_error(y_val, oof_mlp[val_idx])
        print(f"  MLP MAE: {mlp_mae:.2f}")
    
    return oof_cat, oof_lgb, oof_xgb, oof_mlp, test_cat, test_lgb, test_xgb, test_mlp


# ==================== 权重优化 ====================
def optimize_weights(oof_preds, y):
    """优化融合权重"""
    from scipy.optimize import minimize
    
    def objective(weights):
        weights = np.abs(weights) / np.sum(np.abs(weights))
        pred = np.zeros(len(y))
        for p, w in zip(oof_preds, weights):
            pred += p * w
        return mean_absolute_error(y, pred)
    
    n_models = len(oof_preds)
    initial_weights = np.ones(n_models) / n_models
    
    result = minimize(objective, initial_weights, method='Nelder-Mead', 
                     options={'maxiter': 200, 'xatol': 0.001})
    
    return np.abs(result.x) / np.sum(np.abs(result.x))


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("四模型融合训练 (CatBoost + LightGBM + XGBoost + MLP)")
    print("=" * 60)
    
    # 加载数据
    train, test = load_raw_data()
    
    # 特征工程
    X, y, X_test, sale_ids = advanced_feature_engineering(train, test)
    
    # K折Stacking
    oof_cat, oof_lgb, oof_xgb, oof_mlp, test_cat, test_lgb, test_xgb, test_mlp = kfold_stacking(
        X, y, X_test, n_splits=5
    )
    
    # 各模型评估
    print("\n" + "="*60)
    print("各模型MAE评估")
    print("="*60)
    print(f"CatBoost: {mean_absolute_error(y, oof_cat):.2f}")
    print(f"LightGBM: {mean_absolute_error(y, oof_lgb):.2f}")
    print(f"XGBoost: {mean_absolute_error(y, oof_xgb):.2f}")
    print(f"MLP: {mean_absolute_error(y, oof_mlp):.2f}")
    
    # 优化权重
    print("\n" + "="*60)
    print("优化融合权重...")
    print("="*60)
    
    oof_preds = [oof_cat, oof_lgb, oof_xgb, oof_mlp]
    model_names = ['CatBoost', 'LightGBM', 'XGBoost', 'MLP']
    
    optimal_weights = optimize_weights(oof_preds, y)
    
    print("\n最优权重:")
    for name, w in zip(model_names, optimal_weights):
        print(f"  {name}: {w:.4f}")
    
    # 融合预测
    final_oof = sum(pred * w for pred, w in zip(oof_preds, optimal_weights))
    final_test = sum(pred * w for pred, w in zip([test_cat, test_lgb, test_xgb, test_mlp], optimal_weights))
    
    # 最终评估
    final_mae = mean_absolute_error(y, final_oof)
    final_rmse = np.sqrt(mean_squared_error(y, final_oof))
    final_r2 = r2_score(y, final_oof)
    
    print("\n" + "="*60)
    print("最终融合结果")
    print("="*60)
    print(f"MAE: {final_mae:.2f}")
    print(f"RMSE: {final_rmse:.2f}")
    print(f"R²: {final_r2:.4f}")
    
    # 保存结果
    final_test = np.clip(final_test, y.min()*0.9, y.max()*1.1)
    submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_test})
    submit.to_csv('optimized_ensemble_submit.csv', index=False)
    
    print(f"\n预测结果已保存到 optimized_ensemble_submit.csv")
    
    if final_mae < 450:
        print(f"\n🎯 恭喜！MAE={final_mae:.0f} 已达到目标 (<450)")
    else:
        print(f"\n距离目标: {final_mae - 450:.0f}")
    
    return final_mae


if __name__ == "__main__":
    main()
