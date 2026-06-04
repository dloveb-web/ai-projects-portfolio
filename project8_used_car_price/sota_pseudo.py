# -*- coding: utf-8 -*-
"""
Pseudo-Labeling: 用最优提交（444.77）的预测作为伪标签扩充训练集
流程：train(150K) + test_pseudo(50K) = 200K → 重新训练DNN+LGB → 预测原始test
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = ''
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
BATCH_SIZE, EPOCHS, PATIENCE, LR, WEIGHT_DECAY = 1024, 200, 20, 3e-4, 1e-5
N_FOLDS = 5

print("="*60)
print("Pseudo-Labeling 训练")
print("="*60)

# ===== 1. 加载最优预测作为伪标签 =====
train_orig = pd.read_csv(BASE_PATH + 'used_car_train_20200313.csv', sep=' ')
test_orig = pd.read_csv(BASE_PATH + 'used_car_testB_20200421.csv', sep=' ')
sale_ids = test_orig['SaleID'].values

best_submit = pd.read_csv(BASE_PATH + 'sota_dnn_fusion_submit.csv')
pseudo_prices = best_submit['price'].values
print(f"加载伪标签: {len(pseudo_prices)} 条 (Test MAE 444.77)")

# ===== 2. 构造伪标签测试集 =====
test_pseudo = test_orig.copy()
test_pseudo['price'] = pseudo_prices

# 合并原始训练集 + 伪标签测试集
data = pd.concat([train_orig, test_pseudo], axis=0, ignore_index=True)
print(f"合并数据: {len(train_orig)} 原始 + {len(test_pseudo)} 伪标签 = {len(data)}")

# ===== 3. 特征工程（concat模式）=====
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

cat_features = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'regionCode']
for col in cat_features:
    data[col] = data[col].fillna(data[col].mode()[0] if col in data.select_dtypes(include=['number']).columns else 'unknown')

drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

for col in data.select_dtypes(include=[np.number]).columns:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].median())

# ===== 4. 分类特征编码（embedding用）=====
cat_embed_info = {}
for col in cat_features:
    if col in data.columns:
        le = LabelEncoder()
        le.fit(data[col].astype(str))
        data[f'{col}_idx'] = le.transform(data[col].astype(str))
        n_cats = len(le.classes_)
        cat_embed_info[f'{col}_idx'] = (n_cats, min(24, n_cats // 2 + 1))

exclude = ['price'] + [f'{col}_idx' for col in cat_features] + cat_features
numeric_feat = [c for c in data.columns if c not in exclude and c in data.select_dtypes(include=[np.number]).columns]
print(f"数值特征数: {len(numeric_feat)}")

# 标准化（全部数据）
scaler = StandardScaler()
data[numeric_feat] = scaler.fit_transform(data[numeric_feat])

# ===== 5. 准备训练数据 =====
n_orig = len(train_orig)
n_total = len(data)

# 数值和分类特征矩阵
X_numeric = data[numeric_feat].values.astype(np.float32)
cat_idx = [f'{col}_idx' for col in cat_features]
X_cat = data[cat_idx].values.astype(np.int64)
y_all = data['price'].values.astype(np.float32)

# 原始test集的索引（排在data最后50K）
test_start = 0  # 原始train和伪标签test已经合并
# 实际上，我们的test_pseudo已经包含了测试集（带伪标签）
# 但最终我们需要预测的是原始testB的数据
# 由于test_pseudo = test_orig + pseudo_prices，它们的位置是对应的
# 我们需要在data中找到哪些行是原始test数据

# 实际上，现在data = train_orig + test_pseudo(n=200K)
# 原始test是后50K，对应data的后50K行
# 我们需要预测的是这后50K行（带伪标签的那部分）
# 但由于伪标签质量很高（444.77），我们可以直接用它们

# ===== 6. PriceDNN 模型 =====
class PriceDNN(nn.Module):
    def __init__(self, num_numeric, cat_embed_info):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        total_embed = 0
        for col, (n_cats, e_dim) in cat_embed_info.items():
            self.embeddings[col] = nn.Embedding(n_cats, e_dim)
            total_embed += e_dim
        input_dim = num_numeric + total_embed
        
        layers, shortcuts, prev = [], [], input_dim
        for h in [512, 256, 128, 64]:
            layers.append(nn.Sequential(nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)))
            shortcuts.append(nn.Linear(prev, h) if prev != h else nn.Identity())
            prev = h
        self.hidden_layers, self.shortcuts = nn.ModuleList(layers), nn.ModuleList(shortcuts)
        self.output = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, numeric, categorical):
        embs = [emb(categorical[:, i]) for i, (_, emb) in enumerate(self.embeddings.items())]
        x = torch.cat([numeric] + embs, dim=1)
        for layer, sc in zip(self.hidden_layers, self.shortcuts):
            x = layer(x) + sc(x)
        return self.output(x).squeeze()


# ===== 7. 训练辅助函数 =====
def train_epoch_fn(model, loader, criterion, optimizer):
    model.train()
    total = 0
    for num_b, cat_b, y_b in loader:
        num_b, cat_b, y_b = num_b.to(DEVICE), cat_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(num_b, cat_b), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(y_b)
    return total / len(loader.dataset)

def eval_mae_fn(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for num_b, cat_b, y_b in loader:
            preds.append(model(num_b.to(DEVICE), cat_b.to(DEVICE)).cpu().numpy())
            trues.append(y_b.numpy())
    return mean_absolute_error(np.concatenate(trues), np.concatenate(preds))

# ===== 8. DNN 训练（5-fold CV on augmented data）=====
# 注：这里的fold会混合原始train和伪标签test
# 预测目标是最后50K行（原始test的对应位置）

print("\n训练 DNN (Pseudo-Labeling)...")
dnn_test_preds = np.zeros(n_total - n_orig)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_idx = list(kf.split(X_numeric))

# 获取最后n_test行的fold索引：它在data中位置是[n_orig:]
# 每个fold中，找出test行是否在val集 → 收集所有fold的OOF预测
test_vanilla_mask = np.ones(n_total, dtype=bool)
test_vanilla_mask[:n_orig] = False  # 只有后50K

for fold, (tr, va) in enumerate(fold_idx):
    print(f"\n  Fold {fold+1}/{N_FOLDS}")
    
    X_tr = torch.from_numpy(X_numeric[tr])
    X_va = torch.from_numpy(X_numeric[va])
    C_tr = torch.from_numpy(X_cat[tr])
    C_va = torch.from_numpy(X_cat[va])
    y_tr_t = torch.from_numpy(y_all[tr])
    y_va_t = torch.from_numpy(y_all[va])
    
    train_loader = DataLoader(TensorDataset(X_tr, C_tr, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, C_va, y_va_t), batch_size=BATCH_SIZE)
    
    model = PriceDNN(X_numeric.shape[1], cat_embed_info).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.L1Loss()
    
    best_mae, best_state, patience = float('inf'), None, 0
    for epoch in range(EPOCHS):
        tl = train_epoch_fn(model, train_loader, criterion, optimizer)
        vm = eval_mae_fn(model, val_loader)
        scheduler.step()
        
        if (epoch + 1) % 40 == 0:
            print(f"    Epoch {epoch+1}: loss={tl:.2f}, val_mae={vm:.2f}")
        
        if vm < best_mae:
            best_mae, best_state, patience = vm, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"    Early stop at epoch {epoch+1}, best val_mae={best_mae:.2f}")
                break
    
    # Predict original test set (last 50K rows)
    model.load_state_dict(best_state)
    model.eval()
    test_indices = np.arange(n_orig, n_total)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_numeric[test_indices]),
                      torch.from_numpy(X_cat[test_indices]),
                      torch.zeros(len(test_indices))),
        batch_size=BATCH_SIZE)
    
    fold_preds = []
    with torch.no_grad():
        for nb, cb, _ in test_loader:
            fold_preds.append(model(nb.to(DEVICE), cb.to(DEVICE)).cpu().numpy())
    dnn_test_preds += np.concatenate(fold_preds) / N_FOLDS

dnn_pred = np.maximum(dnn_test_preds, 50)
print(f"\nDNN Pseudo-Label OOF: 范围 [{dnn_pred.min():.0f}, {dnn_pred.max():.0f}], 均值 {dnn_pred.mean():.0f}")

# 保存DNN预测供后续使用
dnn_only_submit = pd.DataFrame({'SaleID': sale_ids, 'price': dnn_pred})
dnn_only_submit.to_csv(BASE_PATH + 'sota_pseudo_dnn.csv', index=False)
print("DNN预测已保存: sota_pseudo_dnn.csv")

# ===== 9. LightGBM 训练 (Pseudo-Labeling) =====
try:
    print("\n训练 LightGBM (Pseudo-Labeling)...")
    lgb_keep = [c for c in data.columns 
                if c not in ['price'] + cat_features + [f'{c}_idx' for c in cat_features]
                and c in data.select_dtypes(include=[np.number]).columns]

    lgb_data = data[lgb_keep].values.astype(np.float32)
    lgb_preds = np.zeros(n_total - n_orig)

    lgb_params = {
        'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
        'learning_rate': 0.020, 'num_leaves': 120, 'max_depth': 9,
        'min_data_in_leaf': 17, 'feature_fraction': 0.88, 'bagging_fraction': 0.88,
        'bagging_freq': 5, 'reg_alpha': 0.19, 'reg_lambda': 0.19,
        'verbose': -1, 'seed': 42
    }

    for fold, (tr, va) in enumerate(fold_idx):
        print(f"  LGB Fold {fold+1}/{N_FOLDS}")
        ds = lgb.Dataset(lgb_data[tr], label=y_all[tr])
        vs = lgb.Dataset(lgb_data[va], label=y_all[va], reference=ds)
        m = lgb.train(lgb_params, ds, valid_sets=[vs], num_boost_round=4500,
                      callbacks=[lgb.early_stopping(140), lgb.log_evaluation(0)])
        lgb_preds += m.predict(lgb_data[n_orig:]) / N_FOLDS

    lgb_pred = np.maximum(lgb_preds, 50)
    print(f"LGB Pseudo-Label: 范围 [{lgb_pred.min():.0f}, {lgb_pred.max():.0f}], 均值 {lgb_pred.mean():.0f}")
    
    # 融合
    for w in np.arange(0.3, 0.8, 0.1):
        pred = w * dnn_pred + (1-w) * lgb_pred
        print(f"  权重 DNN={w:.1f}: [{pred.min():.0f},{pred.max():.0f}] mean={pred.mean():.0f}")

    final_w = 0.5
    final_pred = final_w * dnn_pred + (1-final_w) * lgb_pred
    
except Exception as e:
    print(f"LightGBM融合失败: {e}")
    print("仅使用DNN Pseudo-Label预测")
    final_pred = dnn_pred

final_pred = np.maximum(final_pred, 50)

submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit.to_csv(BASE_PATH + 'sota_pseudo_submit.csv', index=False)
print(f"\n✅ Pseudo-Labeling 提交: sota_pseudo_submit.csv")
print(f"   范围: [{final_pred.min():.0f}, {final_pred.max():.0f}], 均值: {final_pred.mean():.0f}")
