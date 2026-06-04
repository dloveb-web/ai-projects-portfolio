# -*- coding: utf-8 -*-
"""
简化版深度残差网络 - 最小配置
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 数据集
class SimpleDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else self.X[i]

# 残差网络
class SimpleResNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

print("加载数据...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

# 特征工程
train['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([train, test], ignore_index=True)
combined['power'] = combined['power'].clip(0, 600)
combined['v_0_sq'] = combined['v_0'] ** 2
combined['v_3_sq'] = combined['v_3'] ** 2
combined['v_0_v_3'] = combined['v_0'] * combined['v_3']
combined['v_0_power'] = combined['v_0'] * combined['power']

# 编码分类变量
for col in ['brand', 'bodyType', 'fuelType', 'gearbox']:
    combined[col] = LabelEncoder().fit_transform(combined[col].astype(str).fillna('missing'))

# 特征列
feat_cols = ['brand', 'bodyType', 'fuelType', 'gearbox', 'power', 'kilometer',
             'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_11', 'v_14',
             'v_0_sq', 'v_3_sq', 'v_0_v_3', 'v_0_power']

train_data = combined[combined['is_train']==1]
test_data = combined[combined['is_train']==0]

X = train_data[feat_cols].values.astype(np.float32)
y = train_data['price'].values.astype(np.float32)
X_test = test_data[feat_cols].values.astype(np.float32)
sale_ids = test_data['SaleID'].values

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
X = np.nan_to_num(X)
X_test = np.nan_to_num(X_test)

print(f"特征维度: {X.shape}")

# 划分验证集
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_ds = SimpleDataset(X_tr, y_tr)
val_ds = SimpleDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024)

model = SimpleResNet(X.shape[1]).to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

print("\n开始训练...")
best_mae = float('inf')
best_state = None

for epoch in range(30):
    model.train()
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # 验证
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            preds.extend(model(bx).cpu().numpy())
            targets.extend(by.cpu().numpy())
    
    mae = mean_absolute_error(targets, preds)
    if mae < best_mae:
        best_mae = mae
        best_state = model.state_dict().copy()
    
    print(f"Epoch {epoch+1}/30: MAE={mae:.2f} (Best={best_mae:.2f})")

# 加载最佳模型
model.load_state_dict(best_state)
print(f"\n验证集最佳MAE: {best_mae:.2f}")

# 测试集预测
test_ds = SimpleDataset(X_test)
test_loader = DataLoader(test_ds, batch_size=1024)
model.eval()
preds = []
with torch.no_grad():
    for bx in test_loader:
        bx = bx.to(device)
        preds.extend(model(bx).cpu().numpy())

preds = np.clip(preds, y.min()*0.9, y.max()*1.1)
pd.DataFrame({'SaleID': sale_ids, 'price': preds}).to_csv('deep_simple.csv', index=False)
print(f"\n结果保存: deep_simple.csv")

if best_mae < 400:
    print("🎉 目标达成！")
else:
    print(f"距离目标: {best_mae - 400:.2f}")
