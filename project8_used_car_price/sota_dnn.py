# -*- coding: utf-8 -*-
"""
二手车价格预测 - PyTorch DNN + LightGBM 融合版
架构：Embedding层 + Residual MLP + 5-fold CV
参考：天池竞赛第4名方案（MAE 396.8）
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ========== 配置 ==========
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

BATCH_SIZE = 1024
EPOCHS = 200
PATIENCE = 20
LR = 3e-4
WEIGHT_DECAY = 1e-5
N_FOLDS = 5
SEEDS = [42, 123, 2024]
CAT_EMBED_DIM = 24  # 增大embedding维度

# ========== 加载数据 ==========
BASE_PATH = ''
train = pd.read_csv(BASE_PATH + 'used_car_train_20200313.csv', sep=' ')
test = pd.read_csv(BASE_PATH + 'used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

# Concat做特征工程（基线已验证有效）
test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)
n_train = len(train)
print(f"训练: {n_train}, 测试: {len(test)}")

# ========== 特征工程 ==========
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0.0')
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
data['power_age'] = data['power'] * data['car_age']
data['usage_intensity'] = data['kilometer'] / (data['car_age'] + 1)
for col in ['brand', 'model', 'regionCode']:
    data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')

# 分类特征需保留原始值供 embedding
cat_features_raw = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'regionCode']
for col in cat_features_raw:
    data[col] = data[col].fillna(data[col].mode()[0] if col in data.select_dtypes(include=['number']).columns else 'unknown')

drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

# 数值特征缺失值填充
numeric_cols = data.select_dtypes(include=[np.number]).columns
for col in data.columns:
    if col in data.select_dtypes(include=[np.number]).columns and data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].median())

# ========== 分离 train/test ==========
train_data = data.iloc[:n_train].reset_index(drop=True)
test_data = data.iloc[n_train:].reset_index(drop=True)
y_orig = train_data['price'].values.copy()

# ========== 分类特征编码（embedding用）==========
cat_embed_info = {}  # {col: (num_unique, embedding_dim)}
for col in cat_features_raw:
    if col in train_data.columns:
        le = LabelEncoder()
        # 确保train+test标签一致
        combined = pd.concat([train_data[col].astype(str), test_data[col].astype(str)])
        le.fit(combined)
        train_data[f'{col}_idx'] = le.transform(train_data[col].astype(str))
        test_data[f'{col}_idx'] = le.transform(test_data[col].astype(str))
        n_cats = len(le.classes_)
        cat_embed_info[f'{col}_idx'] = (n_cats, min(CAT_EMBED_DIM, n_cats // 2 + 1))
        print(f"  {col}: {n_cats} categories -> embed_dim={min(CAT_EMBED_DIM, n_cats // 2 + 1)}")

# 数值特征（分隔后处理，不包含price）
exclude_cols = ['price'] + [f'{col}_idx' for col in cat_features_raw] + cat_features_raw
numeric_feat_cols = [c for c in train_data.columns if c not in exclude_cols and c in train_data.select_dtypes(include=[np.number]).columns]
print(f"数值特征数: {len(numeric_feat_cols)}")

# 标准化数值特征
scaler = StandardScaler()
train_data[numeric_feat_cols] = scaler.fit_transform(train_data[numeric_feat_cols])
test_data[numeric_feat_cols] = scaler.transform(test_data[numeric_feat_cols])

# 整理特征矩阵
X_numeric = train_data[numeric_feat_cols].values.astype(np.float32)
X_test_numeric = test_data[numeric_feat_cols].values.astype(np.float32)

cat_idx_cols = [f'{col}_idx' for col in cat_features_raw]
X_cat = train_data[cat_idx_cols].values.astype(np.int64)
X_test_cat = test_data[cat_idx_cols].values.astype(np.int64)

y = train_data['price'].values.astype(np.float32)
print(f"价格范围: [{y.min():.0f}, {y.max():.0f}], 均值: {y.mean():.0f}")

print(f"\n数值特征: {X_numeric.shape[1]}")
print(f"分类特征: {X_cat.shape[1]}")

print(f"\n数值特征: {X_numeric.shape[1]}")
print(f"分类特征: {X_cat.shape[1]}")
for col in cat_idx_cols:
    print(f"  {col}: {cat_embed_info[col]}")

# ========== DNN 模型 ==========
class PriceDNN(nn.Module):
    def __init__(self, num_numeric, cat_embed_info, hidden_dims=[512, 256, 128, 64]):
        super().__init__()
        # Embedding层
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0
        for col, (n_cats, e_dim) in cat_embed_info.items():
            self.embeddings[col] = nn.Embedding(n_cats, e_dim)
            total_embed_dim += e_dim

        input_dim = num_numeric + total_embed_dim
        print(f"  DNN输入维度: {input_dim} (数值{num_numeric} + embedding{total_embed_dim})")

        # Residual MLP blocks
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            block = []
            block.append(nn.Linear(prev_dim, h_dim))
            block.append(nn.BatchNorm1d(h_dim))
            block.append(nn.ReLU())
            block.append(nn.Dropout(0.2))
            layers.append(nn.Sequential(*block))
            prev_dim = h_dim

        self.hidden_layers = nn.ModuleList(layers)

        # 残差投影层（输入输出维度不匹配时）
        self.shortcuts = nn.ModuleList()
        prev = input_dim
        for h_dim in hidden_dims:
            if prev != h_dim:
                self.shortcuts.append(nn.Linear(prev, h_dim))
            else:
                self.shortcuts.append(nn.Identity())
            prev = h_dim

        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, numeric, categorical):
        # Embedding: iterate in order, mapping cat_idx_cols order to embedding order
        emb_list = []
        for i, (col, emb_layer) in enumerate(self.embeddings.items()):
            emb_list.append(emb_layer(categorical[:, i]))
        embed_concat = torch.cat(emb_list, dim=1) if emb_list else torch.empty(numeric.size(0), 0, device=numeric.device)

        x = torch.cat([numeric, embed_concat], dim=1)

        # Residual blocks
        for layer, shortcut in zip(self.hidden_layers, self.shortcuts):
            identity = shortcut(x)
            x = layer(x)
            x = x + identity

        return self.output(x).squeeze()


# ========== 训练函数 ==========
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for num_batch, cat_batch, y_batch in loader:
        num_batch, cat_batch, y_batch = num_batch.to(device), cat_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(num_batch, cat_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def eval_mae(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for num_batch, cat_batch, y_batch in loader:
            num_batch, cat_batch = num_batch.to(device), cat_batch.to(device)
            pred = model(num_batch, cat_batch).cpu().numpy()
            preds.append(pred)
            trues.append(y_batch.numpy())
    return mean_absolute_error(np.concatenate(trues), np.concatenate(preds))


# ========== 多 Seed 5-fold CV 训练 ==========
all_val_maes = []
all_test_preds = []
all_seed_maes = []

for seed_idx, SEED in enumerate(SEEDS):
    print(f"\n{'='*60}")
    print(f"Seed {seed_idx+1}/{len(SEEDS)}: {SEED}")
    print(f"{'='*60}")
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    seed_val_maes = []
    seed_test_preds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_numeric)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}")
        
        # Split
        num_tr, num_va = X_numeric[train_idx], X_numeric[val_idx]
        cat_tr, cat_va = X_cat[train_idx], X_cat[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

    # Dataset/DataLoader
    train_ds = TensorDataset(
        torch.from_numpy(num_tr),
        torch.from_numpy(cat_tr),
        torch.from_numpy(y_tr)
    )
    val_ds = TensorDataset(
        torch.from_numpy(num_va),
        torch.from_numpy(cat_va),
        torch.from_numpy(y_va)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = PriceDNN(X_numeric.shape[1], cat_embed_info).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # Train
    best_mae = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_mae = eval_mae(model, val_loader, DEVICE)
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}: loss={train_loss:.2f}, val_mae={val_mae:.2f}")

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best
    model.load_state_dict(best_state)
    val_mae = eval_mae(model, val_loader, DEVICE)
    seed_val_maes.append(val_mae)
    print(f"    Fold {fold+1} Best MAE: {val_mae:.2f}")

    # Predict test
    test_ds = TensorDataset(
        torch.from_numpy(X_test_numeric),
        torch.from_numpy(X_test_cat),
        torch.zeros(len(X_test_numeric))
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    fold_test_preds = []
    with torch.no_grad():
        for num_b, cat_b, _ in test_loader:
            num_b, cat_b = num_b.to(DEVICE), cat_b.to(DEVICE)
            p = model(num_b, cat_b).cpu().numpy()
            fold_test_preds.append(p)
    fold_test_pred = np.concatenate(fold_test_preds)
    seed_test_preds.append(fold_test_pred)

    # After seed's fold loop - compute seed average
    seed_avg = np.mean(seed_val_maes)
    all_seed_maes.append(seed_avg)
    all_val_maes.extend(seed_val_maes)
    all_test_preds.append(np.mean(seed_test_preds, axis=0))
    print(f"\n  Seed {SEED} Avg MAE: {seed_avg:.2f}")

# ========== DNN 结果 ==========
dnn_val_mae = np.mean(all_val_maes)
dnn_test_pred = np.mean(all_test_preds, axis=0)
print(f"\n{'='*60}")
print(f"DNN 5-fold CV MAE: {dnn_val_mae:.2f} ± {np.std(all_val_maes):.2f}")
print(f"{'='*60}")

# 先保存DNN预测结果（LightGBM/XGBoost融合由独立脚本完成）
dnn_submit = pd.DataFrame({'SaleID': sale_ids, 'price': np.maximum(dnn_test_pred, 50)})
dnn_submit.to_csv(BASE_PATH + 'sota_dnn_pred.csv', index=False)
print("DNN预测已保存: sota_dnn_pred.csv（用于后续融合）")

# DNN单独提交
dnn_submit.to_csv(BASE_PATH + 'sota_dnn_only_submit.csv', index=False)
print(f"\nDNN单独提交: sota_dnn_only_submit.csv")
print(f"预测范围: [{dnn_test_pred.min():.2f}, {dnn_test_pred.max():.2f}]")
print(f"预测均值: {dnn_test_pred.mean():.2f}")
print(f"\n{'='*60}")
print(f"✅ DNN 完成! CV MAE: {dnn_val_mae:.2f}")
print(f"{'='*60}")
