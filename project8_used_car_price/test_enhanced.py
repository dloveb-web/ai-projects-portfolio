# -*- coding: utf-8 -*-
"""
快速测试版本 - 只运行1个fold，验证代码是否正常工作
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cpu')

print("="*60)
print("快速测试 - 1 Fold")
print("="*60)

# 加载数据
print("\n加载数据...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

# 简化特征工程
print("\n特征工程...")
test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

# 日期特征
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = data['creat_year'] - data['reg_year']
data['car_age'] = data['car_age'].clip(lower=0)

# 处理 notRepairedDamage
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', np.nan)
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce')
data['notRepairedDamage'] = data['notRepairedDamage'].fillna(0)

# v特征基础统计
v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)
data['v_max'] = data[v_cols].max(axis=1)
data['v_min'] = data[v_cols].min(axis=1)

# v特征二阶交互
data['v_0_v_3'] = data['v_0'] * data['v_3']
data['v_0_v_2'] = data['v_0'] * data['v_2']
data['v_0_v_12'] = data['v_0'] * data['v_12']

# 对数变换
data['log_power'] = np.log1p(data['power'])
data['log_km'] = np.log1p(data['kilometer'])

# 组合特征
data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']

# 分离
train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
test_data = test_data.drop(columns=['price'])

# 准备数据
X = train_data.drop(columns=['price'])
y = train_data['price']

# 删除无用列
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
X = X.drop(columns=[c for c in drop_cols if c in X.columns])
test_data = test_data.drop(columns=[c for c in drop_cols if c in test_data.columns])

# 填充缺失值
X = X.fillna(X.median())
test_data = test_data.fillna(X.median())

# 确保列一致
common_cols = list(set(X.columns) & set(test_data.columns))
X = X[common_cols]
test_data = test_data[common_cols]

print(f"特征数: {X.shape[1]}")

# 1折验证
kf = KFold(n_splits=2, shuffle=True, random_state=42)  # 用2折加快速度
train_idx, val_idx = next(kf.split(X))

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

print("\n" + "="*60)
print("开始训练...")
print("="*60)

# CatBoost
print("\n【1/3】训练 CatBoost...")
cat_model = CatBoostRegressor(
    iterations=1000,  # 减少迭代次数
    learning_rate=0.05,
    depth=6,
    loss_function='MAE',
    random_seed=42,
    verbose=100
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
cat_pred = cat_model.predict(X_val)
cat_mae = mean_absolute_error(y_val, cat_pred)
print(f"CatBoost MAE: {cat_mae:.2f}")

# XGBoost
print("\n【2/3】训练 XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,  # 减少迭代次数
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=1,  # 使用单核减少资源占用
    eval_metric='mae',
    verbosity=1
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred = xgb_model.predict(X_val)
xgb_mae = mean_absolute_error(y_val, xgb_pred)
print(f"XGBoost MAE: {xgb_mae:.2f}")

# Neural Net
print("\n【3/3】训练神经网络...")

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 简单神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

train_dataset = TensorDataset(
    torch.FloatTensor(X_train_scaled),
    torch.FloatTensor(y_train.values)
)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

model = SimpleNN(X_train.shape[1]).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_mae = float('inf')
for epoch in range(50):  # 减少epoch
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        nn_pred = model(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy()
    nn_mae = mean_absolute_error(y_val, nn_pred)

    if nn_mae < best_mae:
        best_mae = nn_mae

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/50 Val MAE: {nn_mae:.2f} Best: {best_mae:.2f}")

print(f"NeuralNet Best MAE: {best_mae:.2f}")

# 自适应融合
print("\n" + "="*60)
print("自适应融合")
print("="*60)

maes = [cat_mae, xgb_mae, best_mae]
inv_maes = [1 / max(m, 1) for m in maes]
weights = [w / sum(inv_maes) for w in inv_maes]

print(f"权重: CatBoost={weights[0]:.3f}, XGBoost={weights[1]:.3f}, NN={weights[2]:.3f}")

ensemble_pred = (weights[0] * cat_pred +
                 weights[1] * xgb_pred +
                 weights[2] * nn_pred)
ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
print(f"Ensemble MAE: {ensemble_mae:.2f}")

print("\n" + "="*60)
if ensemble_mae <= 450:
    print(f"🎉 测试通过！MAE: {ensemble_mae:.2f} ≤ 450")
else:
    print(f"📈 继续优化，MAE: {ensemble_mae:.2f}")
print("="*60)
