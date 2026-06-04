# -*- coding: utf-8 -*-
"""快速残差网络 - 3折 15epochs"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error

device = torch.device('cpu')
print("残差网络训练")

# 数据
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
train['is_train'], test['is_train'] = 1, 0
combined = pd.concat([train, test], ignore_index=True)
combined['power'] = combined['power'].clip(0, 600)
combined['v_0_v_3'] = combined['v_0'] * combined['v_3']
combined['v_0_power'] = combined['v_0'] * combined['power']

# 编码
for c in ['brand', 'bodyType', 'fuelType', 'gearbox']:
    combined[c + '_enc'] = LabelEncoder().fit_transform(combined[c].astype(str).fillna('x'))

cat_dims = [combined[c + '_enc'].nunique() for c in ['brand', 'bodyType', 'fuelType', 'gearbox']]

num_cols = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_11', 'v_14', 'v_0_v_3', 'v_0_power']
cat_cols = [c + '_enc' for c in ['brand', 'bodyType', 'fuelType', 'gearbox']]

tr = combined[combined['is_train']==1]
te = combined[combined['is_train']==0]

X_num = tr[num_cols].values.astype(np.float32)
X_cat = tr[cat_cols].values.astype(np.int64)
y = tr['price'].values.astype(np.float32)
X_num_te = te[num_cols].values.astype(np.float32)
X_cat_te = te[cat_cols].values.astype(np.int64)

scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)
X_num_te = scaler.transform(X_num_te)
X_num, X_num_te = np.nan_to_num(X_num), np.nan_to_num(X_num_te)

# 模型
class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(0.2), nn.Linear(d,d), nn.BatchNorm1d(d))
    def forward(self, x): return torch.relu(x + self.net(x))

class Model(nn.Module):
    def __init__(self, num_d, cat_ds):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(d, min(8,(d+1)//2)) for d in cat_ds])
        emb_d = sum(min(8,(d+1)//2) for d in cat_ds)
        self.num = nn.Sequential(nn.Linear(num_d, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1))
        self.cat = nn.Sequential(nn.Linear(emb_d, 32), nn.BatchNorm1d(32), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(96, 128), nn.BatchNorm1d(128), nn.ReLU(), ResBlock(128), nn.Linear(128, 1))
    def forward(self, n, c):
        e = torch.cat([self.emb[i](c[:,i]) for i in range(len(self.emb))], 1)
        return self.out(torch.cat([self.num(n), self.cat(e)], 1)).squeeze(-1)

# 训练
kf = KFold(n_splits=3, shuffle=True, random_state=42)
oof, pred_te = np.zeros(len(y)), np.zeros(len(te))

for f, (ti, vi) in enumerate(kf.split(y)):
    print(f"\nFold {f+1}/3")
    tr_ld = DataLoader(TensorDataset(torch.FloatTensor(X_num[ti]), torch.LongTensor(X_cat[ti]), torch.FloatTensor(y[ti])), batch_size=512, shuffle=True)
    
    model = Model(len(num_cols), cat_dims)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    crit = nn.L1Loss()
    best = float('inf')
    
    for ep in range(15):
        model.train()
        for n, c, t in tr_ld:
            opt.zero_grad()
            crit(model(n, c), t).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        
        model.eval()
        with torch.no_grad():
            mae = mean_absolute_error(y[vi], model(torch.FloatTensor(X_num[vi]), torch.LongTensor(X_cat[vi])).numpy())
        if mae < best: best = mae
        if (ep+1) % 5 == 0: print(f"Ep {ep+1}: MAE={mae:.0f}")
    
    model.eval()
    with torch.no_grad():
        oof[vi] = model(torch.FloatTensor(X_num[vi]), torch.LongTensor(X_cat[vi])).numpy()
        pred_te += model(torch.FloatTensor(X_num_te), torch.LongTensor(X_cat_te)).numpy() / 3
    print(f"Best: {best:.0f}")

mae = mean_absolute_error(y, oof)
print(f"\n总体MAE: {mae:.0f}")
if mae < 400: print("🎉 达成目标!")
else: print(f"差: {mae-400:.0f}")

pred_te = np.clip(pred_te, y.min()*0.9, y.max()*1.1)
pd.DataFrame({'SaleID': te['SaleID'].values, 'price': pred_te}).to_csv('res_fast.csv', index=False)
print("保存: res_fast.csv")
