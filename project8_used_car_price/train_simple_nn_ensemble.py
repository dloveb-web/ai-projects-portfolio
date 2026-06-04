# -*- coding: utf-8 -*-
"""
二手车价格预测 - 简化版PyTorch神经网络 + 树模型融合
目标：MAE <= 450
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device('cpu')


# ==================== 数据加载 ====================
def load_data():
    print("加载数据...")
    train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    return train, test


# ==================== 特征工程 ====================
def feature_engineering(train, test):
    print("特征工程...")
    
    # 合并处理
    test['price'] = -1
    data = pd.concat([train, test], axis=0, ignore_index=True)
    
    # 处理日期特征
    data['reg_year'] = data['regDate'] // 10000
    data['creat_year'] = data['creatDate'] // 10000
    data['car_age'] = data['creat_year'] - data['reg_year']
    data['car_age'] = data['car_age'].clip(lower=0)
    
    # 处理notRepairedDamage
    data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', np.nan)
    data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce')
    data['notRepairedDamage'] = data['notRepairedDamage'].fillna(0)
    
    # v特征统计
    v_cols = [f'v_{i}' for i in range(15)]
    data['v_mean'] = data[v_cols].mean(axis=1)
    data['v_std'] = data[v_cols].std(axis=1)
    data['v_max'] = data[v_cols].max(axis=1)
    data['v_min'] = data[v_cols].min(axis=1)
    
    # 关键v特征交互
    data['v_0_v_3'] = data['v_0'] * data['v_3']
    data['v_0_v_2'] = data['v_0'] * data['v_2']
    data['v_0_v_12'] = data['v_0'] * data['v_12']
    data['v_3_v_12'] = data['v_3'] * data['v_12']
    
    # 重要特征
    data['power_km'] = data['power'] * data['kilometer']
    data['age_km'] = data['car_age'] * data['kilometer']
    
    # 目标编码
    for col in ['brand', 'model', 'regionCode']:
        data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')
    
    # 删除无用特征
    drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
    data = data.drop(columns=drop_cols, errors='ignore')
    
    # 分离
    train_data = data[data['price'] != -1].reset_index(drop=True)
    test_data = data[data['price'] == -1].reset_index(drop=True)
    test_data = test_data.drop(columns=['price'])
    
    return train_data, test_data


# ==================== PyTorch模型 ====================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.LeakyReLU(0.01)
    
    def forward(self, x):
        return self.act(x + self.net(x))


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 256, 128, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(0.1))
            prev_dim = dim
        
        # 残差块
        layers.append(ResidualBlock(hidden_dims[-1]))
        
        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_nn_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=256):
    """训练神经网络"""
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled), 
        torch.FloatTensor(y_train.values)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 模型
    model = SimpleNN(X_train.shape[1]).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
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
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy()
        val_mae = mean_absolute_error(y_val, val_pred)
        scheduler.step(val_mae)
        
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} Val MAE: {val_mae:.2f}")
    
    # 加载最佳模型
    model.load_state_dict(best_state)
    return model, scaler, best_mae


# ==================== 树模型 ====================
def train_catboost(X_train, y_train, X_val, y_val):
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=10,
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
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 63,
        'max_depth': 8,
        'verbose': -1,
        'seed': 42
    }
    
    model = lgb.train(params, train_data, num_boost_round=3000, 
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
    train_data, test_data = feature_engineering(train, test)
    
    # 准备数据
    X = train_data.drop(columns=['price'])
    y = train_data['price']
    
    # 填充缺失值
    X = X.fillna(X.median())
    test_data = test_data.fillna(X.median())
    
    # 确保列一致
    common_cols = list(set(X.columns) & set(test_data.columns))
    X = X[common_cols]
    test_data = test_data[common_cols]
    
    print(f"特征数: {X.shape[1]}")
    
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cat_maes = []
    lgb_maes = []
    nn_maes = []
    ensemble_maes = []
    
    test_preds_cat = np.zeros(len(test_data))
    test_preds_lgb = np.zeros(len(test_data))
    test_preds_nn = np.zeros(len(test_data))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/5")
        print(f"{'='*50}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # CatBoost
        cat_model, cat_mae = train_catboost(X_train, y_train, X_val, y_val)
        cat_pred = cat_model.predict(X_val)
        cat_maes.append(cat_mae)
        test_preds_cat += cat_model.predict(test_data) / 5
        print(f"CatBoost MAE: {cat_mae:.2f}")
        
        # LightGBM
        lgb_model, lgb_mae = train_lightgbm(X_train, y_train, X_val, y_val)
        lgb_pred = lgb_model.predict(X_val)
        lgb_maes.append(lgb_mae)
        test_preds_lgb += lgb_model.predict(test_data) / 5
        print(f"LightGBM MAE: {lgb_mae:.2f}")
        
        # 神经网络
        nn_model, scaler, nn_mae = train_nn_model(X_train, y_train, X_val, y_val, epochs=100)
        test_scaled = scaler.transform(test_data)
        nn_model.eval()
        with torch.no_grad():
            nn_pred_val = nn_model(torch.FloatTensor(test_scaled).to(device)).cpu().numpy()
        test_preds_nn += nn_pred_val / 5
        nn_maes.append(nn_mae)
        print(f"NeuralNet MAE: {nn_mae:.2f}")
        
        # 简单融合
        ensemble_pred = 0.4 * cat_pred + 0.35 * lgb_pred + 0.25 * nn_pred_val[:len(y_val)] if len(nn_pred_val) > len(y_val) else 0.4 * cat_pred + 0.35 * lgb_pred + 0.25 * nn_model(torch.FloatTensor(scaler.transform(X_val)).to(device)).cpu().numpy()
        ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
        ensemble_maes.append(ensemble_mae)
        print(f"Ensemble MAE: {ensemble_mae:.2f}")
    
    # 最终融合
    print(f"\n{'='*50}")
    print("最终结果")
    print(f"{'='*50}")
    print(f"CatBoost 平均 MAE: {np.mean(cat_maes):.2f}")
    print(f"LightGBM 平均 MAE: {np.mean(lgb_maes):.2f}")
    print(f"NeuralNet 平均 MAE: {np.mean(nn_maes):.2f}")
    print(f"Ensemble 平均 MAE: {np.mean(ensemble_maes):.2f}")
    
    # 最终预测
    final_pred = 0.4 * test_preds_cat + 0.35 * test_preds_lgb + 0.25 * test_preds_nn
    final_pred = np.maximum(final_pred, 50)  # 最小价格限制
    
    # 保存结果
    submit = pd.DataFrame({
        'SaleID': sale_ids,
        'price': final_pred
    })
    submit.to_csv('nn_ensemble_submit.csv', index=False)
    print(f"\n结果已保存到 nn_ensemble_submit.csv")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.bar(['CatBoost', 'LightGBM', 'NeuralNet', 'Ensemble'], 
            [np.mean(cat_maes), np.mean(lgb_maes), np.mean(nn_maes), np.mean(ensemble_maes)])
    plt.ylabel('MAE')
    plt.title('模型性能对比')
    plt.savefig('model_comparison.png')
    plt.close()
    
    if np.mean(ensemble_maes) <= 450:
        print(f"\n成功！MAE达到目标: {np.mean(ensemble_maes):.2f} <= 450")
    else:
        print(f"\n距离目标还差: {np.mean(ensemble_maes) - 450:.2f}")


if __name__ == "__main__":
    main()
