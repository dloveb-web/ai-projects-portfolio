# -*- coding: utf-8 -*-
"""
二手车价格预测 - 直接优化版
基于477 MAE成功策略 + 最小可靠改进
目标：MAE <= 450
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cpu')

print("="*60)
print("直接优化版 - 基于477 MAE成功策略")
print("目标: MAE <= 450")
print("="*60)

# ==================== 数据处理 ====================
print("\n数据处理...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

# 与477 MAE成功代码完全一致的特征工程
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = data['creat_year'] - data['reg_year']
data['car_age'] = data['car_age'].clip(lower=0)

data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', np.nan)
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)

v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)
data['v_max'] = data[v_cols].max(axis=1)
data['v_min'] = data[v_cols].min(axis=1)

data['v_0_v_3'] = data['v_0'] * data['v_3']
data['v_0_v_2'] = data['v_0'] * data['v_2']
data['v_0_v_12'] = data['v_0'] * data['v_12']
data['v_3_v_12'] = data['v_3'] * data['v_12']

data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']

for col in ['brand', 'model', 'regionCode']:
    data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')

drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
test_data = test_data.drop(columns=['price'])

X = train_data.drop(columns=['price'])
y = train_data['price']

X = X.fillna(X.median())
test_data = test_data.fillna(X.median())

common_cols = list(set(X.columns) & set(test_data.columns))
X = X[common_cols]
test_data = test_data[common_cols]

print(f"特征数: {X.shape[1]}")

# ==================== 神经网络（与477 MAE相同） ====================
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

class DirectNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        layers = []
        hidden_dims = [256, 256, 128, 128]  # 与477 MAE完全相同
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(0.1))
            prev_dim = dim

        layers.append(ResidualBlock(hidden_dims[-1]))
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_direct_nn(X_train, y_train, X_val, y_val, epochs=100):
    """训练神经网络（与477 MAE相同参数）"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train.values)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = DirectNN(X_train.shape[1]).to(device)
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

        if (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch+1}/{epochs} Val MAE: {val_mae:.2f}")

    model.load_state_dict(best_state)
    return model, scaler, best_mae

# ==================== 树模型（与477 MAE相同） ====================
def train_direct_catboost(X_train, y_train, X_val, y_val):
    model = CatBoostRegressor(
        iterations=3000,  # 与477 MAE相同
        learning_rate=0.03,  # 与477 MAE相同
        depth=6,  # 与477 MAE相同
        l2_leaf_reg=10,  # 与477 MAE相同
        loss_function='MAE',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    return model, mae

def train_direct_lightgbm(X_train, y_train, X_val, y_val):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,  # 与477 MAE相同
        'num_leaves': 63,  # 与477 MAE相同
        'max_depth': 8,  # 与477 MAE相同
        'verbose': -1,
        'seed': 42
    }

    model = lgb.train(params, train_data, num_boost_round=3000,
                       valid_sets=[val_data],
                       callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    return model, mae

# ==================== 主训练（与477 MAE相同） ====================
print("\n开始训练 (5折交叉验证)...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cat_maes = []
lgb_maes = []
nn_maes = []
ensemble_maes = []

test_preds_cat = np.zeros(len(test_data))
test_preds_lgb = np.zeros(len(test_data))
test_preds_nn = np.zeros(len(test_data))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # CatBoost
    cat_model, cat_mae = train_direct_catboost(X_train, y_train, X_val, y_val)
    cat_pred = cat_model.predict(X_val)
    cat_maes.append(cat_mae)
    test_preds_cat += cat_model.predict(test_data) / 5
    print(f"   CatBoost MAE: {cat_mae:.2f}")

    # LightGBM
    lgb_model, lgb_mae = train_direct_lightgbm(X_train, y_train, X_val, y_val)
    lgb_pred = lgb_model.predict(X_val)
    lgb_maes.append(lgb_mae)
    test_preds_lgb += lgb_model.predict(test_data) / 5
    print(f"   LightGBM MAE: {lgb_mae:.2f}")

    # 神经网络
    nn_model, scaler, nn_mae = train_direct_nn(X_train, y_train, X_val, y_val, epochs=100)
    test_scaled = scaler.transform(test_data)
    nn_model.eval()
    with torch.no_grad():
        nn_pred_val = nn_model(torch.FloatTensor(test_scaled).to(device)).cpu().numpy()
    test_preds_nn += nn_pred_val / 5
    nn_maes.append(nn_mae)
    print(f"   NeuralNet MAE: {nn_mae:.2f}")

    # 融合（与477 MAE相同：0.4, 0.35, 0.25）
    ensemble_pred = 0.4 * cat_pred + 0.35 * lgb_pred + 0.25 * nn_pred_val
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    ensemble_maes.append(ensemble_mae)
    print(f"   Ensemble MAE: {ensemble_mae:.2f}")

# 最终结果
print(f"\n{'='*60}")
print("最终结果 (5折平均)")
print(f"{'='*60}")
print(f"CatBoost 平均 MAE: {np.mean(cat_maes):.2f}")
print(f"LightGBM 平均 MAE: {np.mean(lgb_maes):.2f}")
print(f"NeuralNet 平均 MAE: {np.mean(nn_maes):.2f}")
print(f"Ensemble 平均 MAE: {np.mean(ensemble_maes):.2f}")

# 最终预测（与477 MAE相同权重）
final_pred = 0.4 * test_preds_cat + 0.35 * test_preds_lgb + 0.25 * test_preds_nn
final_pred = np.maximum(final_pred, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': sale_ids,
    'price': final_pred
})
submit.to_csv('direct_optimization_submit.csv', index=False)
print(f"\n结果已保存到 direct_optimization_submit.csv")

# 最终总结
final_mae = np.mean(ensemble_maes)
print(f"\n{'='*60}")
print(f"最终MAE: {final_mae:.2f}")
print(f"目标MAE: 450")
print(f"差距: {final_mae - 450:.2f}")

if final_mae <= 450:
    print(f"🎉 成功达到目标MAE: {final_mae:.2f} <= 450")
    print(f"比之前477改进: {477 - final_mae:.2f}")
else:
    print(f"⚠️ 距离目标: {final_mae - 450:.2f}")
    print(f"当前MAE: {final_mae:.2f}")
    if final_mae < 477:
        print(f"比之前477改进: {477 - final_mae:.2f}")
    else:
        print(f"比之前477差距: {final_mae - 477:.2f}")

print(f"{'='*60}")