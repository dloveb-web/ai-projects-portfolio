# -*- coding: utf-8 -*-
"""
残差网络 - 嵌入层 + 批归一化 + 梯度裁剪
优化版本，平衡速度和效果
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cpu')
print("=" * 50)
print("深度残差网络训练")
print("=" * 50)

# ==================== 数据加载 ====================
print("\n加载数据...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

# 预处理
train['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([train, test], ignore_index=True)

# 异常值处理
combined['power'] = combined['power'].clip(0, 600)

# 构造特征
combined['v_0_sq'] = combined['v_0'] ** 2
combined['v_3_sq'] = combined['v_3'] ** 2
combined['v_0_v_3'] = combined['v_0'] * combined['v_3']
combined['v_0_power'] = combined['v_0'] * combined['power']
combined['v_3_power'] = combined['v_3'] * combined['power']

# 分类变量编码
cat_cols = ['brand', 'bodyType', 'fuelType', 'gearbox']
cat_maps = {}
for col in cat_cols:
    combined[col] = combined[col].astype(str).fillna('missing')
    le = LabelEncoder()
    combined[col + '_enc'] = le.fit_transform(combined[col])
    cat_maps[col] = len(le.classes_)

print(f"分类变量基数: {cat_maps}")

# 数值特征
num_cols = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_11', 'v_14',
            'v_0_sq', 'v_3_sq', 'v_0_v_3', 'v_0_power', 'v_3_power']

# 分离数据
train_data = combined[combined['is_train'] == 1]
test_data = combined[combined['is_train'] == 0]

# 数值特征
X_num = train_data[num_cols].values.astype(np.float32)
X_num_test = test_data[num_cols].values.astype(np.float32)

# 分类特征
X_cat = train_data[[c + '_enc' for c in cat_cols]].values.astype(np.int64)
X_cat_test = test_data[[c + '_enc' for c in cat_cols]].values.astype(np.int64)

y = train_data['price'].values.astype(np.float32)
sale_ids = test_data['SaleID'].values

# 标准化数值特征
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)
X_num_test = scaler.transform(X_num_test)
X_num = np.nan_to_num(X_num)
X_num_test = np.nan_to_num(X_num_test)

print(f"数值特征: {X_num.shape}, 分类特征: {X_cat.shape}")

# ==================== 模型定义 ====================
class ResBlock(nn.Module):
    """残差块"""
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
    
    def forward(self, x):
        return nn.LeakyReLU(0.1)(x + self.net(x))

class DeepModel(nn.Module):
    def __init__(self, num_dim, cat_dims, embed_dim=8):
        super().__init__()
        
        # 嵌入层
        self.embeds = nn.ModuleList([
            nn.Embedding(dim, min(embed_dim, (dim+1)//2))
            for dim in cat_dims
        ])
        embed_out = sum(min(embed_dim, (dim+1)//2) for dim in cat_dims)
        
        # 数值特征处理
        self.num_net = nn.Sequential(
            nn.Linear(num_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )
        
        # 嵌入特征处理
        self.cat_net = nn.Sequential(
            nn.Linear(embed_out, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )
        
        # 融合残差网络
        fusion_dim = 64 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            ResBlock(128, dropout=0.2),
            ResBlock(128, dropout=0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, num_x, cat_x):
        # 数值
        num_out = self.num_net(num_x)
        
        # 嵌入
        embeds = [self.embeds[i](cat_x[:, i]) for i in range(len(self.embeds))]
        cat_out = self.cat_net(torch.cat(embeds, dim=1))
        
        # 融合
        fusion = torch.cat([num_out, cat_out], dim=1)
        return self.fusion(fusion).squeeze(-1)

# ==================== K折训练 ====================
print("\n开始K折训练...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cat_dims = [cat_maps[c] for c in cat_cols]

oof_preds = np.zeros(len(X_num))
test_preds = np.zeros(len(X_num_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_num)):
    print(f"\n=== Fold {fold+1}/5 ===")
    
    X_num_tr, X_num_val = X_num[tr_idx], X_num[val_idx]
    X_cat_tr, X_cat_val = X_cat[tr_idx], X_cat[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    train_ds = TensorDataset(
        torch.FloatTensor(X_num_tr),
        torch.LongTensor(X_cat_tr),
        torch.FloatTensor(y_tr)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_num_val),
        torch.LongTensor(X_cat_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024)
    
    model = DeepModel(X_num.shape[1], cat_dims).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_mae = float('inf')
    best_state = None
    
    for epoch in range(30):
        model.train()
        for num_x, cat_x, target in train_loader:
            optimizer.zero_grad()
            pred = model(num_x, cat_x)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
        scheduler.step()
        
        # 验证
        model.eval()
        val_preds = []
        with torch.no_grad():
            for num_x, cat_x, target in val_loader:
                val_preds.extend(model(num_x, cat_x).numpy())
        
        mae = mean_absolute_error(y_val, val_preds)
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: MAE={mae:.0f} (Best={best_mae:.0f})")
    
    # 加载最佳模型预测
    model.load_state_dict(best_state)
    model.eval()
    
    # OOF预测
    with torch.no_grad():
        oof_preds[val_idx] = model(torch.FloatTensor(X_num_val), torch.LongTensor(X_cat_val)).numpy()
    
    # 测试集预测
    with torch.no_grad():
        test_preds += model(torch.FloatTensor(X_num_test), torch.LongTensor(X_cat_test)).numpy() / 5
    
    print(f"Fold {fold+1} Best MAE: {best_mae:.0f}")

# ==================== 结果 ====================
overall_mae = mean_absolute_error(y, oof_preds)
print("\n" + "=" * 50)
print(f"总体MAE: {overall_mae:.0f}")

if overall_mae < 400:
    print("🎉 目标达成！MAE < 400")
else:
    print(f"距离目标: {overall_mae - 400:.0f}")

# 保存
test_preds = np.clip(test_preds, y.min()*0.9, y.max()*1.1)
pd.DataFrame({'SaleID': sale_ids, 'price': test_preds}).to_csv('residual_net.csv', index=False)
print(f"\n保存: residual_net.csv")
