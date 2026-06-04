# -*- coding: utf-8 -*-
"""极简神经网络 - 快速测试"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

device = torch.device('cpu')
print("设备: CPU")

print("加载数据...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

# 简单特征
feat = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_11', 'v_14']
train['power'] = train['power'].clip(0, 600)
test['power'] = test['power'].clip(0, 600)

X = train[feat].values.astype(np.float32)
y = train['price'].values.astype(np.float32)
X_test = test[feat].values.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
X = np.nan_to_num(X)
X_test = np.nan_to_num(X_test)

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 转为Tensor
X_tr_t = torch.FloatTensor(X_tr)
y_tr_t = torch.FloatTensor(y_tr)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val)
X_test_t = torch.FloatTensor(X_test)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=1024, shuffle=True)

# 简单模型
model = nn.Sequential(
    nn.Linear(len(feat), 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 1)
).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("训练...")
for epoch in range(20):
    model.train()
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(bx).squeeze(), by)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        pred = model(X_val_t).squeeze().numpy()
        mae = mean_absolute_error(y_val, pred)
    print(f"Epoch {epoch+1}: MAE={mae:.0f}")

print(f"\n最终MAE: {mae:.0f}")

# 预测
model.eval()
with torch.no_grad():
    pred_test = model(X_test_t).squeeze().numpy()

pred_test = np.clip(pred_test, y.min()*0.9, y.max()*1.1)
pd.DataFrame({'SaleID': test['SaleID'], 'price': pred_test}).to_csv('nn_simple.csv', index=False)
print("保存: nn_simple.csv")
