# -*- coding: utf-8 -*-
"""
快速版深度残差网络 - 简化配置加速训练
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# ==================== 数据集 ====================
class CarDataset(Dataset):
    def __init__(self, num_x, cat_x, const_x, y=None):
        self.num_x = torch.FloatTensor(num_x)
        self.cat_x = torch.LongTensor(cat_x)
        self.const_x = torch.FloatTensor(const_x)
        self.y = torch.FloatTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.num_x)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.num_x[idx], self.cat_x[idx], self.const_x[idx], self.y[idx]
        return self.num_x[idx], self.cat_x[idx], self.const_x[idx]

# ==================== 残差块 ====================
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.act(x + self.net(x))

# ==================== 主模型 ====================
class QuickModel(nn.Module):
    def __init__(self, num_dim, cat_cards, const_dim):
        super().__init__()
        
        # 数值特征
        self.num_net = nn.Sequential(
            nn.Linear(num_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )
        
        # 分类特征嵌入
        self.embeds = nn.ModuleList([
            nn.Embedding(c, min(8, (c+1)//2)) for c in cat_cards
        ])
        embed_dim = sum(min(8, (c+1)//2) for c in cat_cards)
        self.cat_net = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
        )
        
        # 构造特征
        self.const_net = nn.Sequential(
            nn.Linear(const_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )
        
        # 融合残差网络
        fusion_dim = 64 + 32 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            ResBlock(128, dropout=0.2),
            ResBlock(128, dropout=0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, num_x, cat_x, const_x):
        # 数值
        num_out = self.num_net(num_x)
        
        # 分类嵌入
        cat_embeds = [self.embeds[i](cat_x[:, i]) for i in range(len(self.embeds))]
        cat_out = self.cat_net(torch.cat(cat_embeds, dim=1))
        
        # 构造
        const_out = self.const_net(const_x)
        
        # 融合
        fusion = torch.cat([num_out, cat_out, const_out], dim=1)
        return self.fusion(fusion).squeeze(-1)

# ==================== 数据加载 ====================
print("加载数据...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

train['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([train, test], ignore_index=True)

# 处理异常值
combined['power'] = combined['power'].clip(0, 600)

# 构造特征
v_cols = ['v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_11', 'v_14']
combined['v_mean'] = combined[v_cols].mean(axis=1)
combined['v_std'] = combined[v_cols].std(axis=1)

const_feats = ['v_0_sq', 'v_3_sq', 'v_0_v_3', 'v_0_power', 'v_3_power']
combined['v_0_sq'] = combined['v_0'] ** 2
combined['v_3_sq'] = combined['v_3'] ** 2
combined['v_0_v_3'] = combined['v_0'] * combined['v_3']
combined['v_0_power'] = combined['v_0'] * combined['power']
combined['v_3_power'] = combined['v_3'] * combined['power']

# 分类变量
cat_cols = ['brand', 'bodyType', 'fuelType', 'gearbox', 'regionCode']
cat_cards = []
for col in cat_cols:
    combined[col] = combined[col].fillna(-1).astype(str)
    le = LabelEncoder()
    combined[col + '_enc'] = le.fit_transform(combined[col])
    cat_cards.append(len(le.classes_))

# 数值特征
num_cols = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_11', 'v_14', 'v_mean', 'v_std']

# 分离数据
train_data = combined[combined['is_train'] == 1]
test_data = combined[combined['is_train'] == 0]

num_train = train_data[num_cols].values.astype(np.float32)
cat_train = train_data[[c + '_enc' for c in cat_cols]].values.astype(np.int64)
const_train = train_data[const_feats].values.astype(np.float32)
y_train = train_data['price'].values.astype(np.float32)

num_test = test_data[num_cols].values.astype(np.float32)
cat_test = test_data[[c + '_enc' for c in cat_cols]].values.astype(np.int64)
const_test = test_data[const_feats].values.astype(np.float32)
sale_ids = test_data['SaleID'].values

# 标准化
scaler1 = StandardScaler()
num_train = scaler1.fit_transform(num_train)
num_test = scaler1.transform(num_test)
scaler2 = StandardScaler()
const_train = scaler2.fit_transform(const_train)
const_test = scaler2.transform(const_test)

# 处理NaN
num_train = np.nan_to_num(num_train)
num_test = np.nan_to_num(num_test)
const_train = np.nan_to_num(const_train)
const_test = np.nan_to_num(const_test)

print(f"数值: {num_train.shape}, 分类: {cat_train.shape}, 构造: {const_train.shape}")
print(f"分类基数: {cat_cards}")

# ==================== 训练 ====================
print("\n开始训练...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(num_train))
test_preds = np.zeros(len(num_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(num_train)):
    print(f"\n=== Fold {fold+1}/5 ===")
    
    train_ds = CarDataset(num_train[tr_idx], cat_train[tr_idx], const_train[tr_idx], y_train[tr_idx])
    val_ds = CarDataset(num_train[val_idx], cat_train[val_idx], const_train[val_idx], y_train[val_idx])
    
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
    
    model = QuickModel(len(num_cols), cat_cards, len(const_feats)).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=50, steps_per_epoch=len(train_loader))
    
    best_mae = float('inf')
    best_state = None
    
    for epoch in range(50):
        model.train()
        for num_x, cat_x, const_x, y in train_loader:
            num_x, cat_x, const_x, y = num_x.to(device), cat_x.to(device), const_x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(num_x, cat_x, const_x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # 验证
        model.eval()
        val_preds = []
        val_y = []
        with torch.no_grad():
            for num_x, cat_x, const_x, y in val_loader:
                num_x, cat_x, const_x, y = num_x.to(device), cat_x.to(device), const_x.to(device), y.to(device)
                val_preds.extend(model(num_x, cat_x, const_x).cpu().numpy())
                val_y.extend(y.cpu().numpy())
        
        mae = mean_absolute_error(val_y, val_preds)
        
        if mae < best_mae:
            best_mae = mae
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: MAE={mae:.2f} (Best={best_mae:.2f})")
    
    # 加载最佳模型
    model.load_state_dict(best_state)
    
    # 预测
    model.eval()
    test_ds = CarDataset(num_test, cat_test, const_test)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)
    
    with torch.no_grad():
        for num_x, cat_x, const_x in test_loader:
            num_x, cat_x, const_x = num_x.to(device), cat_x.to(device), const_x.to(device)
            test_preds += model(num_x, cat_x, const_x).cpu().numpy() / 5
    
    # OOF预测
    oof_preds[val_idx] = val_preds
    
    print(f"Fold {fold+1} Best MAE: {best_mae:.2f}")

# ==================== 结果 ====================
overall_mae = mean_absolute_error(y_train, oof_preds)
print("\n" + "="*50)
print(f"总体MAE: {overall_mae:.2f}")

if overall_mae < 400:
    print("🎉 目标达成！MAE < 400")
else:
    print(f"距离目标: {overall_mae - 400:.2f}")

# 保存
test_preds = np.clip(test_preds, y_train.min() * 0.9, y_train.max() * 1.1)
pd.DataFrame({'SaleID': sale_ids, 'price': test_preds}).to_csv('deep_residual_quick.csv', index=False)
print(f"\n保存: deep_residual_quick.csv")
