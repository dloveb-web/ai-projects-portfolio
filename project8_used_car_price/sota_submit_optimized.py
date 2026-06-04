# -*- coding: utf-8 -*-
"""
二手车价格预测 - 完整提交方案（优化版）
改进：
  1. log1p 变换用于树模型（LightGBM）
  2. v 特征全量交叉（所有 v_i × v_j 组合 + v_i² + v_i/v_j）
  3. 加权融合（基于 OOF 搜索最优权重）
架构：PyTorch DNN + LightGBM + Rank 集成 + 加权融合
参考: 天池竞赛方案（目标 Test MAE < 444）
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
print("二手车价格预测 - DNN + LightGBM + Rank + 加权融合")
print("=" * 60)

# ========== 1. 加载数据 ==========
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values
prices = train['price'].values.copy()
print(f"训练集: {len(train)}, 测试集: {len(test)}")

# ========== 2. 特征工程 ==========
test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)
n_train = len(train)

# 基础特征
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0.0')
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)
data['bodyType'] = data['bodyType'].fillna(-1)
data['model'] = data['model'].fillna(-1)

# v 特征统计
v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)
data['v_max'] = data[v_cols].max(axis=1)
data['v_min'] = data[v_cols].min(axis=1)

# 业务交互
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

# 分组计数
for col in ['brand', 'model', 'regionCode']:
    data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')

# ===== 改进点 1: v 特征全量交叉 =====
# 乘法交互：所有 v_i × v_j (i < j)
for i in range(15):
    for j in range(i + 1, 15):
        data[f'v_{i}xv_{j}'] = data[f'v_{i}'] * data[f'v_{j}']

# v 特征平方项
for i in range(15):
    data[f'v_{i}_sq'] = data[f'v_{i}'] ** 2

# v 特征除法交互（仅对绝对值大于 0.01 的列，避免除零）
v_eps = 0.01
for i in range(15):
    col_i = f'v_{i}'
    for j in range(15):
        if i == j:
            continue
        col_j = f'v_{j}'
        mean_j = data[col_j].abs().mean()
        if mean_j > v_eps:
            data[f'v_{i}_div_v_{j}'] = data[col_i] / (data[col_j].abs().clip(lower=v_eps) * np.sign(data[col_j].clip(lower=-1e10)))

print(f"v 特征交叉完成: 新增特征数 = {105 + 15 + 210}")

# 分类特征填充
cat_features = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'regionCode']
for c in cat_features:
    data[c] = data[c].fillna(data[c].mode()[0] if not data[c].mode().empty else 'unknown')

# 删除无用列
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

# 缺失值填充
for c in data.select_dtypes(include=[np.number]).columns:
    if data[c].isnull().sum() > 0:
        data[c] = data[c].fillna(data[c].median())
        if data[c].isnull().sum() > 0:
            data[c] = data[c].fillna(0)

# ========== 3. DNN Embedding 编码 ==========
cat_embed_info = {}
for c in cat_features:
    le = LabelEncoder()
    le.fit(data[c].astype(str))
    data[f'{c}_idx'] = le.transform(data[c].astype(str))
    n_cats = len(le.classes_)
    cat_embed_info[f'{c}_idx'] = (n_cats, min(CAT_EMBED_DIM, max(2, n_cats // 2)))

# DNN 特征矩阵（不含_idx列）
exc = ['price'] + [f'{c}_idx' for c in cat_features] + cat_features
num_feats = [c for c in data.columns if c not in exc and c in data.select_dtypes(include=[np.number]).columns]
scaler = StandardScaler()
data[num_feats] = scaler.fit_transform(data[num_feats])

train_data = data.iloc[:n_train].reset_index(drop=True)
test_data = data.iloc[n_train:].reset_index(drop=True)
y_raw = train_data['price'].values.astype(np.float32)

# ===== 改进点 2: log1p 变换用于 LightGBM（树模型）=====
y_log = np.log1p(y_raw).astype(np.float32)
print(f"价格范围: [{y_raw.min():.0f}, {y_raw.max():.0f}]")
print(f"log1p 价格范围: [{y_log.min():.3f}, {y_log.max():.3f}]")

X_num = train_data[num_feats].values.astype(np.float32)
X_test_num = test_data[num_feats].values.astype(np.float32)
cat_idx_cols = [f'{c}_idx' for c in cat_features]
X_cat = train_data[cat_idx_cols].values.astype(np.int64)
X_test_cat = test_data[cat_idx_cols].values.astype(np.int64)

print(f"DNN 输入: {X_num.shape}, LightGBM 特征: {len(num_feats)}")

# ========== 4. Target 编码（LightGBM 用）==========
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
    tr, te = target_encode(train_data, test_data, col, y_raw)
    train_data[f'{col}_te'] = tr
    test_data[f'{col}_te'] = te

# LightGBM 特征矩阵（不含 price / _idx / 原始分类）
lgb_feat_cols = [c for c in train_data.columns
                 if c not in ['price'] + cat_features + [f'{c}_idx' for c in cat_features]
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
            block = [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            layers.append(nn.Sequential(*block))
            shortcuts.append(nn.Linear(prev, h) if prev != h else nn.Identity())
            prev = h
        self.hidden_layers = nn.ModuleList(layers)
        self.shortcuts = nn.ModuleList(shortcuts)
        self.output = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, numeric, categorical):
        embs = [emb(categorical[:, i]) for i, (_, emb) in enumerate(self.embeddings.items())]
        x = torch.cat([numeric] + embs, dim=1)
        for layer, short in zip(self.hidden_layers, self.shortcuts):
            x = layer(x) + short(x)
        return self.output(x).squeeze()


def train_epoch_fn(model, loader, criterion, optimizer, device):
    model.train()
    total = 0
    for nb, cb, yb in loader:
        nb, cb, yb = nb.to(device), cb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(nb, cb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


def eval_mae_fn(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for nb, cb, yb in loader:
            nb, cb = nb.to(device), cb.to(device)
            preds.append(model(nb, cb).cpu().numpy())
            trues.append(yb.numpy())
    return mean_absolute_error(np.concatenate(trues), np.concatenate(preds))


# ========== 6. DNN 训练（原始 price）==========
print("\n" + "=" * 60)
print("训练 DNN")
print("=" * 60)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
dnn_val_maes = []
dnn_test_preds = []
dnn_oof_preds = np.zeros(len(y_raw))

for fold, (tr, va) in enumerate(kf.split(X_num)):
    print(f"\n  Fold {fold + 1}/{N_FOLDS}")
    dl_tr = DataLoader(TensorDataset(
        torch.from_numpy(X_num[tr]), torch.from_numpy(X_cat[tr]), torch.from_numpy(y_raw[tr])),
        batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(TensorDataset(
        torch.from_numpy(X_num[va]), torch.from_numpy(X_cat[va]), torch.from_numpy(y_raw[va])),
        batch_size=BATCH_SIZE)

    model = PriceDNN(X_num.shape[1], cat_embed_info).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.L1Loss()

    best_mae, best_state, patience_cnt = float('inf'), None, 0
    for ep in range(EPOCHS):
        tl = train_epoch_fn(model, dl_tr, criterion, optimizer, DEVICE)
        vm = eval_mae_fn(model, dl_va, DEVICE)
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
    vm = eval_mae_fn(model, dl_va, DEVICE)
    dnn_val_maes.append(vm)
    dnn_oof_preds[va] = model(torch.from_numpy(X_num[va]).to(DEVICE),
                               torch.from_numpy(X_cat[va]).to(DEVICE)).detach().cpu().numpy()
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

dnn_test_pred = np.mean(dnn_test_preds, axis=0)
dnn_test_pred = np.maximum(dnn_test_pred, 50)
dnn_oof_preds = np.maximum(dnn_oof_preds, 50)
print(f"\nDNN CV MAE: {np.mean(dnn_val_maes):.2f} ± {np.std(dnn_val_maes):.2f}")

# ========== 7. LightGBM 训练（log1p 变换）==========
print("\n" + "=" * 60)
print("训练 LightGBM（log1p 变换）")
print("=" * 60)

lgb_params = {
    'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
    'learning_rate': 0.020, 'num_leaves': 120, 'max_depth': 9,
    'min_data_in_leaf': 17, 'feature_fraction': 0.88, 'bagging_fraction': 0.88,
    'bagging_freq': 5, 'reg_alpha': 0.19, 'reg_lambda': 0.19,
    'verbose': -1, 'seed': SEED
}

lgb_test_preds = np.zeros(len(X_lgb_test))
lgb_oof_preds = np.zeros(len(y_raw))

for fold, (tr, va) in enumerate(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X_lgb_train)):
    print(f"  LGB Fold {fold + 1}/{N_FOLDS}")
    # LightGBM 使用 log1p 变换后的价格
    ds = lgb.Dataset(X_lgb_train[tr], label=y_log[tr])
    vs = lgb.Dataset(X_lgb_train[va], label=y_log[va], reference=ds)
    m = lgb.train(lgb_params, ds, valid_sets=[vs], num_boost_round=4500,
                  callbacks=[lgb.early_stopping(140), lgb.log_evaluation(0)])
    lgb_test_preds += np.expm1(m.predict(X_lgb_test)) / N_FOLDS
    lgb_oof_preds[va] = np.expm1(m.predict(X_lgb_train[va]))

lgb_test_pred = np.maximum(lgb_test_preds, 50)
lgb_oof_preds = np.maximum(lgb_oof_preds, 50)
lgb_oof_mae = mean_absolute_error(y_raw, lgb_oof_preds)
print(f"LGB OOF MAE: {lgb_oof_mae:.2f}")

# ========== 8. 简单平均融合（基线）===========
fusion_simple = 0.5 * dnn_test_pred + 0.5 * lgb_test_pred

# ========== 9. Rank 集成 ==========
rank_dnn = pd.Series(dnn_test_pred).rank().values
rank_lgb = pd.Series(lgb_test_pred).rank().values
rank_avg = (rank_dnn + rank_lgb) / 2
sorted_fusion = np.sort(fusion_simple)
rank_final = np.interp(rank_avg, np.arange(len(fusion_simple)), sorted_fusion)

# ===== 改进点 3: 加权融合（基于 OOF 搜索最优权重）=====
print("\n" + "=" * 60)
print("搜索最优融合权重")
print("=" * 60)

best_mae, best_w_dnn, best_w_lgb = float('inf'), 0.5, 0.5
for w_dnn in np.arange(0.2, 0.9, 0.1):
    w_lgb = 1 - w_dnn
    oof_mae = mean_absolute_error(y_raw, w_dnn * dnn_oof_preds + w_lgb * lgb_oof_preds)
    print(f"  w_dnn={w_dnn:.1f} w_lgb={w_lgb:.1f}: OOF MAE={oof_mae:.2f}")
    if oof_mae < best_mae:
        best_mae, best_w_dnn, best_w_lgb = oof_mae, w_dnn, w_lgb

print(f"\n最佳权重: DNN={best_w_dnn:.1f}, LGB={best_w_lgb:.1f}, OOF MAE={best_mae:.2f}")

# 加权融合 + Rank 集成
weighted_fusion = best_w_dnn * dnn_test_pred + best_w_lgb * lgb_test_pred
rank_weighted = pd.Series(weighted_fusion).rank().values
sorted_weighted = np.sort(weighted_fusion)
rank_weighted_final = np.interp(rank_weighted, np.arange(len(weighted_fusion)), sorted_weighted)

# 最终：Rank 融合 + 加权融合 50/50
final_pred = 0.5 * rank_weighted_final + 0.5 * weighted_fusion
final_pred = np.maximum(final_pred, 50)

# 验证 OOF
final_oof = 0.5 * (best_w_dnn * dnn_oof_preds + best_w_lgb * lgb_oof_preds) + \
            0.5 * pd.Series(best_w_dnn * dnn_oof_preds + best_w_lgb * lgb_oof_preds).rank().values * \
            np.std(best_w_dnn * dnn_oof_preds + best_w_lgb * lgb_oof_preds) / len(y_raw)
final_oof_mae = mean_absolute_error(y_raw, np.maximum(final_oof, 50))
print(f"最终 OOF MAE: {final_oof_mae:.2f}")

# ========== 10. 输出 ==========
print("\n" + "=" * 60)
print("结果输出")
print("=" * 60)

submit = pd.DataFrame({'SaleID': sale_ids, 'price': final_pred})
submit_name = 'sota_optimized_final_submit.csv'
submit.to_csv(submit_name, index=False)

print(f"提交文件: {submit_name}")
print(f"预测范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"预测均值: {final_pred.mean():.2f}")
print(f"\nDNN CV MAE: {np.mean(dnn_val_maes):.2f}")
print(f"LGB OOF MAE: {lgb_oof_mae:.2f}")
print(f"最佳权重: DNN={best_w_dnn:.1f}, LGB={best_w_lgb:.1f}")
print(f"最终 OOF MAE: {final_oof_mae:.2f}")
print("=" * 60)
print("完成!")
print("=" * 60)
