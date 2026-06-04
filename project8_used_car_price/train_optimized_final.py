# -*- coding: utf-8 -*-
"""
二手车价格预测 - 优化版PyTorch残差网络
目标：MAE <= 450
优化点：更多特征、更深网络、更长训练
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import KFold
import joblib
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cpu')


# ==================== 数据加载 ====================
def load_data():
    print("加载数据...")
    train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    return train, test


# ==================== 高级特征工程 ====================
def advanced_feature_engineering(train, test):
    print("高级特征工程...")
    
    test['price'] = -1
    data = pd.concat([train, test], axis=0, ignore_index=True)
    
    # 日期特征
    data['reg_year'] = data['regDate'] // 10000
    data['reg_month'] = (data['regDate'] // 100) % 100
    data['creat_year'] = data['creatDate'] // 10000
    data['car_age'] = data['creat_year'] - data['reg_year']
    data['car_age'] = data['car_age'].clip(lower=0)
    data['car_age_squared'] = data['car_age'] ** 2
    data['log_age'] = np.log1p(data['car_age'])
    
    # 处理notRepairedDamage
    data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', np.nan)
    data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)
    
    # v特征 - 核心特征挖掘
    v_cols = [f'v_{i}' for i in range(15)]
    data['v_mean'] = data[v_cols].mean(axis=1)
    data['v_std'] = data[v_cols].std(axis=1)
    data['v_max'] = data[v_cols].max(axis=1)
    data['v_min'] = data[v_cols].min(axis=1)
    data['v_range'] = data['v_max'] - data['v_min']
    data['v_skew'] = data[v_cols].skew(axis=1)
    
    # 关键v特征交互（参考文档中的高增益特征）
    data['v_0_v_3'] = data['v_0'] * data['v_3']
    data['v_0_v_2'] = data['v_0'] * data['v_2']
    data['v_0_v_12'] = data['v_0'] * data['v_12']
    data['v_3_v_12'] = data['v_3'] * data['v_12']
    data['v_0_v_3_v_12'] = data['v_0'] * data['v_3'] * data['v_12']
    
    # 文档中的高增益特征
    data['v_3_minus_v_10'] = data['v_3'] - data['v_10']
    data['v_3_minus_v_14'] = data['v_3'] - data['v_14']
    data['v_8_v_13'] = data['v_8'] * data['v_13']
    
    # v特征多项式
    data['v_0_sq'] = data['v_0'] ** 2
    data['v_3_sq'] = data['v_3'] ** 2
    data['v_0_sqrt'] = np.sqrt(np.abs(data['v_0']))
    data['v_3_sqrt'] = np.sqrt(np.abs(data['v_3']))
    
    # 三角函数特征
    data['sin_v_0'] = np.sin(data['v_0'])
    data['cos_v_3'] = np.cos(data['v_3'])
    data['sin_diff_v_0_v_3'] = np.sin(data['v_0'] - data['v_3'])
    
    # 功率特征
    data['power_km'] = data['power'] * data['kilometer']
    data['age_km'] = data['car_age'] * data['kilometer']
    data['power_squared'] = data['power'] ** 2
    data['log_power'] = np.log1p(data['power'])
    data['km_per_age'] = data['kilometer'] / (data['car_age'] + 1)
    
    # 目标编码
    for col in ['brand', 'model', 'regionCode']:
        data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')
    
    # 品牌价格统计（使用训练数据）
    train_mask = data['price'] != -1
    for col in ['brand', 'model']:
        stats = data[train_mask].groupby(col)['price'].agg(['mean', 'median', 'std']).reset_index()
        stats.columns = [col, f'{col}_price_mean', f'{col}_price_median', f'{col}_price_std']
        data = data.merge(stats, on=col, how='left')
        data[f'{col}_price_std'] = data[f'{col}_price_std'].fillna(0)
    
    # 保值率代理特征
    data['age_km_retention'] = data['kilometer'] / (data['car_age'] + 1)
    if 'brand_price_mean' in data.columns:
        data['relative_brand_value'] = data['brand_price_mean'] / (data['brand_price_mean'].mean() + 1)
    
    # 删除无用特征
    drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
    data = data.drop(columns=drop_cols, errors='ignore')
    
    # 分离
    train_data = data[data['price'] != -1].reset_index(drop=True)
    test_data = data[data['price'] == -1].reset_index(drop=True)
    test_data = test_data.drop(columns=['price'])
    
    print(f"特征数: {train_data.shape[1] - 1}")
    return train_data, test_data


# ==================== PyTorch模型 ====================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.act(x + self.net(x)))


class AdvancedNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 512, 256, 256, 128, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # 输入层
        layers.append(nn.Linear(prev_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Dropout(0.15))
        prev_dim = hidden_dims[0]
        
        # 隐藏层
        for i, dim in enumerate(hidden_dims[1:]):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(0.1))
            prev_dim = dim
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[-1], dropout=0.1) for _ in range(3)
        ])
        
        self.feature_layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, x):
        x = self.feature_layers(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.output(x).squeeze(-1)


def train_nn_model(X_train, y_train, X_val, y_val, epochs=200, batch_size=256):
    """训练神经网络"""
    # 使用QuantileTransformer进行更好的特征变换
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled), 
        torch.FloatTensor(y_train.values)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 模型
    model = AdvancedNN(X_train.shape[1]).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    best_mae = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy()
        val_mae = mean_absolute_error(y_val, val_pred)
        
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 30:
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} Val MAE: {val_mae:.2f} Best: {best_mae:.2f}")
    
    model.load_state_dict(best_state)
    return model, scaler, best_mae


# ==================== 树模型 ====================
def train_catboost(X_train, y_train, X_val, y_val):
    model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=5,
        min_data_in_leaf=20,
        loss_function='MAE',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    return model, mae


def train_lightgbm(X_train, y_train, X_val, y_val):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'dart',
        'drop_rate': 0.3,
        'learning_rate': 0.015,
        'num_leaves': 127,
        'max_depth': 9,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'seed': 42
    }
    
    model = lgb.train(params, train_data, num_boost_round=5000, 
                       valid_sets=[val_data], 
                       callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    return model, mae


# ==================== 主函数 ====================
def main():
    # 加载数据
    train, test = load_data()
    sale_ids = test['SaleID'].values
    
    # 特征工程
    train_data, test_data = advanced_feature_engineering(train, test)
    
    # 准备数据
    X = train_data.drop(columns=['price'])
    y = train_data['price']
    
    # 填充缺失值
    X = X.fillna(X.median())
    test_data = test_data.fillna(X.median())
    
    # 确保列一致
    common_cols = [c for c in X.columns if c in test_data.columns]
    X = X[common_cols]
    test_data = test_data[common_cols]
    
    print(f"最终特征数: {X.shape[1]}")
    
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cat_maes = []
    lgb_maes = []
    nn_maes = []
    
    test_preds_cat = np.zeros(len(test_data))
    test_preds_lgb = np.zeros(len(test_data))
    test_preds_nn = np.zeros(len(test_data))
    scalers = []
    models_nn = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/5")
        print(f"{'='*50}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # CatBoost
        cat_model, cat_mae = train_catboost(X_train, y_train, X_val, y_val)
        test_preds_cat += cat_model.predict(test_data) / 5
        cat_maes.append(cat_mae)
        print(f"CatBoost MAE: {cat_mae:.2f}")
        
        # LightGBM
        lgb_model, lgb_mae = train_lightgbm(X_train, y_train, X_val, y_val)
        test_preds_lgb += lgb_model.predict(test_data) / 5
        lgb_maes.append(lgb_mae)
        print(f"LightGBM MAE: {lgb_mae:.2f}")
        
        # 神经网络
        nn_model, scaler, nn_mae = train_nn_model(X_train, y_train, X_val, y_val, epochs=200)
        test_scaled = scaler.transform(test_data)
        nn_model.eval()
        with torch.no_grad():
            test_preds_nn += nn_model(torch.FloatTensor(test_scaled).to(device)).cpu().numpy() / 5
        nn_maes.append(nn_mae)
        print(f"NeuralNet MAE: {nn_mae:.2f}")
    
    # 最终结果
    print(f"\n{'='*50}")
    print("最终结果")
    print(f"{'='*50}")
    print(f"CatBoost 平均 MAE: {np.mean(cat_maes):.2f}")
    print(f"LightGBM 平均 MAE: {np.mean(lgb_maes):.2f}")
    print(f"NeuralNet 平均 MAE: {np.mean(nn_maes):.2f}")
    
    # 加权融合 (基于MAE倒数)
    cat_weight = 1 / np.mean(cat_maes)
    lgb_weight = 1 / np.mean(lgb_maes)
    nn_weight = 1 / np.mean(nn_maes)
    total = cat_weight + lgb_weight + nn_weight
    
    final_pred = (cat_weight/total * test_preds_cat + 
                  lgb_weight/total * test_preds_lgb + 
                  nn_weight/total * test_preds_nn)
    final_pred = np.maximum(final_pred, 50)
    
    # 保存结果
    submit = pd.DataFrame({
        'SaleID': sale_ids,
        'price': final_pred
    })
    submit.to_csv('optimized_final_submit.csv', index=False)
    
    print(f"\n结果已保存到 optimized_final_submit.csv")
    
    if np.mean(nn_maes) <= 450:
        print(f"\n成功！NeuralNet MAE达到目标: {np.mean(nn_maes):.2f} <= 450")
    else:
        print(f"\nNeuralNet距离目标还差: {np.mean(nn_maes) - 450:.2f}")


if __name__ == "__main__":
    main()
