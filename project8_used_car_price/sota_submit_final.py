# -*- coding: utf-8 -*-
"""
二手车价格预测 - 完整提交方案
架构：PyTorch DNN + LightGBM + Rank 集成融合
参考: 天池竞赛（最佳成绩 Test MAE 444.59）
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
import warnings, os, sys, json, time
warnings.filterwarnings('ignore')

# ========== 配置 ==========
SEED = 42
N_FOLDS = 5
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
BATCH_SIZE = 1024
EPOCHS = 200
PATIENCE = 20
LR = 3e-4
WEIGHT_DECAY = 1e-5
CAT_EMBED_DIM = 24

np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"设备: {DEVICE}")
print("=" * 60)
print("二手车价格预测 - DNN + LightGBM + Rank 融合")
print("=" * 60)

# ========== 1. 加载数据 ==========
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values
print(f"训练集: {len(train)}, 测试集: {len(test)}")

# ========== 2. 特征工程 ==========
test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)
n_train = len(train)

data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0.0')
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)
data['bodyType'] = data['bodyType'].fillna(-1)
data['model'] = data['model'].fillna(-1)

v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)
data['v_max'] = data[v_cols].max(axis=1)
data['v_min'] = data[v_cols].min(axis=1)
data['v_0_v_3'] = data['v_0'] * data['v_3']
data['v_0_v_2'] = data['v_0'] * data['v_2']
data['v_0_v_12'] = data['v_0'] * data['v_12']
data['v_3_v_12'] = data['v_3'] * data['v_12']
data['v_0_v_8'] = data['v_0'] * data['v_8']
data['v_3_v_8'] = data['v_3'] * data['v_8']
data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']
data['power_age'] = data['power'] * data['car_age']
data['usage_intensity'] = data['kilometer'] / (data['car_age'] + 1)
for col in ['brand', 'model', 'regionCode']:
    data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')

cat_features = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'regionCode']
for c in cat_features:
    data[c] = data[c].fillna(data[c].mode()[0] if not data[c].mode().empty else 'unknown')

drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')
for c in data.select_dtypes(include=[np.number]).columns:
    if data[c].isnull().sum() > 0:
        data[c] = data[c].fillna(data[c].median())
        if data[c].isnull().sum() > 0:
            data[c] = data[c].fillna(0)

# ========== 3. DNN 用 Embedding 编码 ==========
cat_embed_info = {}
for c in cat_features:
    le = LabelEncoder()
    le.fit(data[c].astype(str))
    data[f'{c}_idx'] = le.transform(data[c].astype(str))
    n_cats = len(le.classes_)
    cat_embed_info[f'{c}_idx'] = (n_cats, min(CAT_EMBED_DIM, max(2, n_cats // 2)))

exc = ['price'] + [f'{c}_idx' for c in cat_features] + cat_features
num_feats = [c for c in data.columns if c not in exc and c in data.select_dtypes(include=[np.number]).columns]

scaler = StandardScaler()
data[num_feats] = scaler.fit_transform(data[num_feats])

train_data = data.iloc[:n_train].reset_index(drop=True)
test_data = data.iloc[n_train:].reset_index(drop=True)
y = train_data['price'].values.astype(np.float32)

X_num = train_data[num_feats].values.astype(np.float32)
X_test_num = test_data[num_feats].values.astype(np.float32)
cat_idx_cols = [f'{c}_idx' for c in cat_features]
X_cat = train_data[cat_idx_cols].values.astype(np.int64)
X_test_cat = test_data[cat_idx_cols].values.astype(np.int64)

print(f"数值特征: {X_num.shape[1]}, 分类特征: {X_cat.shape[1]}, 总特征: {X_num.shape[1] + sum(d for _, d in cat_embed_info.values())}")

# ========== 4. Target 编码（供 LightGBM 使用）==========
y_prices = y.copy()

def target_encode(train_df, test_df, col, y_vals):
    kf_tmp = KFold(n_splits=5, shuffle=True, random_state=42)
    tr_enc = np.zeros(len(train_df))
    for tr, va in kf_tmp.split(train_df):
        m = pd.Series(y_vals[tr], index=train_df.iloc[tr].index).groupby(train_df.iloc[tr][col]).mean()
        tr_enc[va] = train_df.iloc[va][col].map(m).fillna(y_vals.mean())
    g = pd.Series(y_vals, index=train_df.index).groupby(train_df[col]).mean()
    te_enc = test_df[col].map(g).fillna(y_vals.mean()).values
    return tr_enc, te_enc

for col in ['brand', 'model', 'regionCode']:
    tr, te = target_encode(train_data, test_data, col, y_prices)
    train_data[f'{col}_te'] = tr
    test_data[f'{col}_te'] = te

# LightGBM 特征矩阵
lgb_feat_cols = [c for c in train_data.columns if c not in ['price'] + cat_features + [f'{c}_idx' for c in cat_features]
                 and c in train_data.select_dtypes(include=[np.number]).columns]
X_lgb_train = train_data[lgb_feat_cols].values.astype(np.float32)
X_lgb_test = test_data[lgb_feat_cols].values.astype(np.float32)
print(f"LightGBM 特征数: {X_lgb_train.shape[1]}")

# ========== 5. DNN 模型 ==========
class PriceDNN(nn.Module):
    def __init__(self, num_numeric, cat_embed_info):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        total_embed = 0
        for c, (n_cat, d) in cat_embed_info.items():
            self.embeddings[c] = nn.Embedding(n_cat, d)
            total_embed += d
        input_dim = num_numeric + total_embed
        layers, shortcuts, prev = [], [], input_dim
        for h in [512, 256, 128, 64]:
            block = []
            block.append(nn.Linear(prev, h))
            block.append(nn.BatchNorm1d(h))
            block.append(nn.ReLU())
            block.append(nn.Dropout(0.2))
            layers.append(nn.Sequential(*block))
            shortcuts.append(nn.Linear(prev, h) if prev != h else nn.Identity())
            prev = h
        self.hidden_layers = nn.ModuleList(layers)
        self.shortcuts = nn.ModuleList(shortcuts)
        self.output = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, numeric, categorical):
        emb_list = [emb(categorical[:, i]) for i, (_, emb) in enumerate(self.embeddings.items())]
        x = torch.cat([numeric] + emb_list, dim=1)
        for layer, short in zip(self.hidden_layers, self.shortcuts):
            x = layer(x) + short(x)
        return self.output(x).squeeze()


# ========== 6. 训练辅助函数 ==========
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0
    for nb, cb, yb in loader:
        nb, cb, yb = nb.to(device), cb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(nb, cb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)

def eval_mae(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for nb, cb, yb in loader:
            nb, cb = nb.to(device), cb.to(device)
            preds.append(model(nb, cb).cpu().numpy())
            trues.append(yb.numpy())
    return mean_absolute_error(np.concatenate(trues), np.concatenate(preds))


# ========== 7. DNN 5-fold CV 训练 ==========
print("\n" + "=" * 60)
print("训练 DNN")
print("=" * 60)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
dnn_val_maes = []
dnn_test_preds = []

for fold, (tr, va) in enumerate(kf.split(X_num)):
    print(f"\n  Fold {fold + 1}/{N_FOLDS}")
    dl_tr = DataLoader(TensorDataset(
        torch.from_numpy(X_num[tr]), torch.from_numpy(X_cat[tr]), torch.from_numpy(y[tr])),
        batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(TensorDataset(
        torch.from_numpy(X_num[va]), torch.from_numpy(X_cat[va]), torch.from_numpy(y[va])),
        batch_size=BATCH_SIZE)

    model = PriceDNN(X_num.shape[1], cat_embed_info).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_mae, best_state, patience_cnt = float('inf'), None, 0
    for ep in range(EPOCHS):
        tl = train_epoch(model, dl_tr, criterion, optimizer, DEVICE)
        vm = eval_mae(model, dl_va, DEVICE)
        scheduler.step()
        if (ep + 1) % 40 == 0:
            print(f"    Epoch {ep + 1}: loss={tl:.2f}, val_mae={vm:.2f}")
        if vm < best_mae:
            best_mae, best_state, patience_cnt = vm, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    Early stop at epoch {ep + 1}, best={best_mae:.2f}")
                break

    model.load_state_dict(best_state)
    vm = eval_mae(model, dl_va, DEVICE)
    dnn_val_maes.append(vm)
    print(f"  Fold {fold + 1} Best MAE: {vm:.2f}")

    dl_test = DataLoader(TensorDataset(
        torch.from_numpy(X_test_num), torch.from_numpy(X_test_cat), torch.zeros(len(X_test_num))),
        batch_size=BATCH_SIZE)
    model.eval()
    fp = []
    with torch.no_grad():
        for nb, cb, _ in dl_test:
            fp.append(model(nb.to(DEVICE), cb.to(DEVICE)).cpu().numpy())
    dnn_test_preds.append(np.concatenate(fp))

dnn_pred = np.mean(dnn_test_preds, axis=0)
dnn_pred = np.maximum(dnn_pred, 50)
print(f"\nDNN CV MAE: {np.mean(dnn_val_maes):.2f} ± {np.std(dnn_val_maes):.2f}")

# ========== 8. LightGBM 训练 ==========
print("\n" + "=" * 60)
print("训练 LightGBM")
print("=" * 60)

lgb_params = {
    'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
    'learning_rate': 0.020, 'num_leaves': 120, 'max_depth': 9,
    'min_data_in_leaf': 17, 'feature_fraction': 0.88, 'bagging_fraction': 0.88,
    'bagging_freq': 5, 'reg_alpha': 0.19, 'reg_lambda': 0.19,
    'verbose': -1, 'seed': SEED
}

lgb_preds = np.zeros(len(X_lgb_test))
for fold, (tr, va) in enumerate(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X_lgb_train)):
    print(f"  LGB Fold {fold + 1}/{N_FOLDS}")
    ds = lgb.Dataset(X_lgb_train[tr], label=y[tr])
    vs = lgb.Dataset(X_lgb_train[va], label=y[va], reference=ds)
    m = lgb.train(lgb_params, ds, valid_sets=[vs], num_boost_round=4500,
                  callbacks=[lgb.early_stopping(140), lgb.log_evaluation(0)])
    lgb_preds += m.predict(X_lgb_test) / N_FOLDS

lgb_pred = np.maximum(lgb_preds, 50)

# ========== 9. 简单平均融合 ==========
fusion_pred = 0.5 * dnn_pred + 0.5 * lgb_pred

# ========== 10. Rank 集成 ==========
rank_dnn = pd.Series(dnn_pred).rank().values
rank_lgb = pd.Series(lgb_pred).rank().values
rank_avg = (rank_dnn + rank_lgb) / 2
sorted_fusion = np.sort(fusion_pred)
rank_final = np.interp(rank_avg, np.arange(len(fusion_pred)), sorted_fusion)

# ========== 11. 最终融合 ==========
final_pred = 0.5 * rank_final + 0.5 * fusion_pred
final_pred = np.maximum(final_pred, 50)

# ========== 12. 输出 ==========
print("\n" + "=" * 60)
print("结果输出")
print("=" * 60)

submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit_name = 'sota_rank_final_submit.csv'
submit.to_csv(submit_name, index=False)

print(f"提交文件: {submit_name}")
print(f"预测范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"预测均值: {final_pred.mean():.2f}")
print("\nDNN CV MAE: {:.2f}".format(np.mean(dnn_val_maes)))
print("=" * 60)
print("完成!")
print("=" * 60)
