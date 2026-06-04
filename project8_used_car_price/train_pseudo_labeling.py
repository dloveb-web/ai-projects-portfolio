# -*- coding: utf-8 -*-
"""
伪标签训练脚本
策略：使用高置信度预测样本增强训练数据
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import joblib
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
    train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    return train, test

def preprocess_data(train, test):
    """数据预处理"""
    # 合并处理
    train['is_train'] = 1
    test['is_train'] = 0
    test['price'] = -1  # 占位符
    
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    
    # 处理异常值
    combined['power'] = combined['power'].clip(0, 600)
    combined['kilometer'] = combined['kilometer'].clip(0, 50)
    
    # 基础特征工程
    # 多项式特征
    combined['v_0_sq'] = combined['v_0'] ** 2
    combined['v_3_sq'] = combined['v_3'] ** 2
    combined['v_0_cube'] = combined['v_0'] ** 3
    combined['v_3_cube'] = combined['v_3'] ** 3
    
    # 交互特征
    combined['v_0_v_3'] = combined['v_0'] * combined['v_3']
    combined['v_0_sq_v_3'] = (combined['v_0'] ** 2) * combined['v_3']
    combined['v_0_v_3_sq'] = combined['v_0'] * (combined['v_3'] ** 2)
    combined['v_0_v_2_v_3'] = combined['v_0'] * combined['v_2'] * combined['v_3']
    
    # 与power/kilometer的交叉特征
    combined['v_0_power'] = combined['v_0'] * combined['power']
    combined['v_0_kilometer'] = combined['v_0'] * combined['kilometer']
    combined['v_3_power'] = combined['v_3'] * combined['power']
    combined['v_3_kilometer'] = combined['v_3'] * combined['kilometer']
    
    # 统计特征
    v_cols = [f'v_{i}' for i in range(15)]
    combined['v_mean'] = combined[v_cols].mean(axis=1)
    combined['v_std'] = combined[v_cols].std(axis=1)
    combined['v_max'] = combined[v_cols].max(axis=1)
    combined['v_min'] = combined[v_cols].min(axis=1)
    combined['v_range'] = combined['v_max'] - combined['v_min']
    
    # K-Fold Target Encoding for brand
    combined = combined.sort_values('SaleID').reset_index(drop=True)
    train_mask = combined['is_train'] == 1
    train_idx = combined[train_mask].index
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    brand_te = np.zeros(len(combined))
    
    for train_fold_idx, val_fold_idx in kf.split(train_idx):
        train_fold = train_idx[train_fold_idx]
        val_fold = train_idx[val_fold_idx]
        
        brand_mean = combined.loc[train_fold, :].groupby('brand')['price'].mean()
        brand_te[val_fold] = combined.loc[val_fold, 'brand'].map(brand_mean).values
    
    # 测试集使用全局均值
    test_idx = combined[combined['is_train'] == 0].index
    global_brand_mean = combined.loc[train_idx, :].groupby('brand')['price'].mean()
    brand_te[test_idx] = combined.loc[test_idx, 'brand'].map(global_brand_mean).values
    
    combined['brand_te'] = brand_te
    combined['brand_te'] = combined['brand_te'].fillna(combined.loc[train_idx, 'price'].mean())
    
    # 分离数据
    train_data = combined[combined['is_train'] == 1].copy()
    test_data = combined[combined['is_train'] == 0].copy()
    
    # 特征列 - 只选择数值类型
    drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'model', 'price', 'is_train']
    feature_cols = [col for col in train_data.columns if col not in drop_cols]
    
    # 只保留数值类型的特征
    numeric_cols = train_data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    print(f"使用 {len(numeric_cols)} 个数值特征")
    
    X_train = train_data[numeric_cols].values.astype(np.float64)
    y_train = train_data['price'].values.astype(np.float64)
    X_test = test_data[numeric_cols].values.astype(np.float64)
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, test_data['SaleID'].values, numeric_cols

def train_base_models(X_train, y_train, X_test, feature_cols):
    """训练基础模型并生成预测"""
    print("=" * 60)
    print("训练基础模型...")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 存储预测结果
    catboost_oof = np.zeros(len(X_train))
    catboost_test = np.zeros(len(X_test))
    lightgbm_oof = np.zeros(len(X_train))
    lightgbm_test = np.zeros(len(X_test))
    xgboost_oof = np.zeros(len(X_train))
    xgboost_test = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n--- Fold {fold + 1}/5 ---")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # CatBoost
        cat_model = cb.CatBoostRegressor(
            iterations=3000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=200,
            od_type='IncToDec',
            od_pval=1e-5
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        catboost_oof[val_idx] = cat_model.predict(X_val)
        catboost_test += cat_model.predict(X_test) / 5
        
        # LightGBM
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        lgb_model = lgb.train(
            {'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1,
             'learning_rate': 0.05, 'num_leaves': 63, 'feature_fraction': 0.8,
             'bagging_fraction': 0.8, 'bagging_freq': 5, 'seed': 42},
            lgb_train, num_boost_round=3000, valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
        )
        lightgbm_oof[val_idx] = lgb_model.predict(X_val)
        lightgbm_test += lgb_model.predict(X_test) / 5
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            objective='reg:absoluteerror', eval_metric='mae',
            early_stopping_rounds=200, verbosity=0
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        xgboost_oof[val_idx] = xgb_model.predict(X_val)
        xgboost_test += xgb_model.predict(X_test) / 5
    
    # 计算MAE
    cat_mae = mean_absolute_error(y_train, catboost_oof)
    lgb_mae = mean_absolute_error(y_train, lightgbm_oof)
    xgb_mae = mean_absolute_error(y_train, xgboost_oof)
    
    print(f"\n基础模型MAE:")
    print(f"  CatBoost: {cat_mae:.2f}")
    print(f"  LightGBM: {lgb_mae:.2f}")
    print(f"  XGBoost: {xgb_mae:.2f}")
    
    # 加权平均
    weights = [0.4, 0.35, 0.25]  # CatBoost权重最高
    ensemble_oof = weights[0] * catboost_oof + weights[1] * lightgbm_oof + weights[2] * xgboost_oof
    ensemble_test = weights[0] * catboost_test + weights[1] * lightgbm_test + weights[2] * xgboost_test
    
    ensemble_mae = mean_absolute_error(y_train, ensemble_oof)
    print(f"  加权平均融合: {ensemble_mae:.2f}")
    
    return ensemble_oof, ensemble_test, {'catboost': catboost_oof, 'lightgbm': lightgbm_oof, 'xgboost': xgboost_oof}

def pseudo_labeling_iteration(X_train, y_train, X_test, pseudo_labels, confidence_threshold, feature_cols, iteration):
    """伪标签迭代"""
    print(f"\n{'='*60}")
    print(f"伪标签迭代 {iteration} - 置信度阈值: {confidence_threshold}")
    
    # 选择高置信度样本
    # 置信度定义：预测值的稳定性（使用多个模型的预测方差）
    # 这里简化处理：选择预测值在合理范围内的样本
    
    # 使用分位数确定高置信度区间
    lower_bound = np.percentile(pseudo_labels, confidence_threshold * 100)
    upper_bound = np.percentile(pseudo_labels, (1 - confidence_threshold) * 100)
    
    # 选择在中间区间的样本（更稳定）
    selected_mask = (pseudo_labels >= lower_bound) & (pseudo_labels <= upper_bound)
    selected_ratio = selected_mask.sum() / len(pseudo_labels)
    
    print(f"选择样本数: {selected_mask.sum()} ({selected_ratio*100:.1f}%)")
    
    # 创建增强训练集
    X_pseudo = X_test[selected_mask]
    y_pseudo = pseudo_labels[selected_mask]
    
    X_augmented = np.vstack([X_train, X_pseudo])
    y_augmented = np.concatenate([y_train, y_pseudo])
    
    print(f"增强后训练集大小: {len(X_augmented)} (原始: {len(X_train)}, 新增: {len(X_pseudo)})")
    
    # 训练模型
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    catboost_oof = np.zeros(len(X_train))
    catboost_test = np.zeros(len(X_test))
    lightgbm_oof = np.zeros(len(X_train))
    lightgbm_test = np.zeros(len(X_test))
    xgboost_oof = np.zeros(len(X_train))
    xgboost_test = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold + 1}/5...", end=' ')
        X_tr, X_val = X_augmented[train_idx], X_train[val_idx]
        y_tr, y_val = y_augmented[train_idx], y_train[val_idx]
        
        # CatBoost
        cat_model = cb.CatBoostRegressor(
            iterations=3000, learning_rate=0.05, depth=8,
            l2_leaf_reg=3, random_seed=42, verbose=0,
            early_stopping_rounds=200
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        catboost_oof[val_idx] = cat_model.predict(X_val)
        catboost_test += cat_model.predict(X_test) / 5
        
        # LightGBM
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        lgb_model = lgb.train(
            {'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1,
             'learning_rate': 0.05, 'num_leaves': 63, 'feature_fraction': 0.8,
             'bagging_fraction': 0.8, 'bagging_freq': 5, 'seed': 42},
            lgb_train, num_boost_round=3000, valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
        )
        lightgbm_oof[val_idx] = lgb_model.predict(X_val)
        lightgbm_test += lgb_model.predict(X_test) / 5
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            objective='reg:absoluteerror', early_stopping_rounds=200, verbosity=0
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        xgboost_oof[val_idx] = xgb_model.predict(X_val)
        xgboost_test += xgb_model.predict(X_test) / 5
        
        # 计算fold MAE
        fold_mae = mean_absolute_error(y_val, (catboost_oof[val_idx] + lightgbm_oof[val_idx] + xgboost_oof[val_idx]) / 3)
        print(f"Fold MAE: {fold_mae:.2f}")
    
    # 加权平均
    weights = [0.4, 0.35, 0.25]
    ensemble_oof = weights[0] * catboost_oof + weights[1] * lightgbm_oof + weights[2] * xgboost_oof
    ensemble_test = weights[0] * catboost_test + weights[1] * lightgbm_test + weights[2] * xgboost_test
    
    mae = mean_absolute_error(y_train, ensemble_oof)
    print(f"\n迭代 {iteration} MAE: {mae:.2f}")
    
    return ensemble_oof, ensemble_test, mae

def main():
    print("=" * 60)
    print("伪标签训练")
    print("=" * 60)
    
    # 加载数据
    print("\n加载和预处理数据...")
    train, test = load_data()
    X_train, y_train, X_test, sale_ids, feature_cols = preprocess_data(train, test)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 第一步：训练基础模型
    base_oof, base_test, model_oofs = train_base_models(X_train, y_train, X_test, feature_cols)
    base_mae = mean_absolute_error(y_train, base_oof)
    
    # 记录结果
    results = [('基础模型', base_mae, base_test)]
    
    # 伪标签迭代
    pseudo_labels = base_test.copy()
    
    # 多轮迭代，逐步放宽置信度阈值
    confidence_thresholds = [0.15, 0.20, 0.25, 0.30]  # 逐步增加选择比例
    
    for i, threshold in enumerate(confidence_thresholds):
        iter_oof, iter_test, iter_mae = pseudo_labeling_iteration(
            X_train, y_train, X_test, pseudo_labels, threshold, feature_cols, i + 1
        )
        results.append((f'伪标签迭代{i+1}', iter_mae, iter_test))
        
        # 更新伪标签
        pseudo_labels = iter_test
        
        # 如果MAE不再下降，提前停止
        if iter_mae >= results[-2][1]:
            print(f"\nMAE未改善，停止迭代")
            break
    
    # 选择最佳结果
    print("\n" + "=" * 60)
    print("最终结果汇总:")
    print("=" * 60)
    for name, mae, _ in results:
        print(f"  {name}: MAE = {mae:.2f}")
    
    best_idx = np.argmin([r[1] for r in results])
    best_name, best_mae, best_pred = results[best_idx]
    print(f"\n最佳结果: {best_name}, MAE = {best_mae:.2f}")
    
    # 保存预测结果
    # 裁剪预测值到合理范围
    best_pred = np.clip(best_pred, y_train.min() * 0.9, y_train.max() * 1.1)
    
    submission = pd.DataFrame({
        'SaleID': sale_ids,
        'price': best_pred
    })
    submission.to_csv('pseudo_labeling_submit.csv', index=False)
    print(f"\n预测结果已保存: pseudo_labeling_submit.csv")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE对比
    names = [r[0] for r in results]
    maes = [r[1] for r in results]
    axes[0].bar(range(len(names)), maes, color=['steelblue', 'coral', 'seagreen', 'purple', 'orange'][:len(names)])
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('各迭代MAE对比')
    axes[0].axhline(y=400, color='r', linestyle='--', label='目标MAE=400')
    axes[0].legend()
    
    for i, v in enumerate(maes):
        axes[0].text(i, v + 5, f'{v:.1f}', ha='center', va='bottom')
    
    # 价格分布
    axes[1].hist(y_train, bins=50, alpha=0.5, label='训练集', density=True)
    axes[1].hist(best_pred, bins=50, alpha=0.5, label='预测集', density=True)
    axes[1].set_xlabel('价格')
    axes[1].set_ylabel('密度')
    axes[1].set_title('价格分布对比')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('pseudo_labeling_results.png', dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存: pseudo_labeling_results.png")
    
    return best_mae

if __name__ == '__main__':
    best_mae = main()
    print(f"\n最终MAE: {best_mae:.2f}")
    if best_mae < 400:
        print("目标达成！MAE < 400")
    else:
        print(f"距离目标还差: {best_mae - 400:.2f}")
