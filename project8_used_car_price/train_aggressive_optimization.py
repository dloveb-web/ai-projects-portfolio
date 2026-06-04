# -*- coding: utf-8 -*-
"""
二手车价格预测 - 激进快速优化版
最快的训练策略，专注于关键改进
目标：MAE <= 450
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cpu')

print("="*60)
print("激进快速优化版 - 专注关键改进")
print("目标: MAE <= 450")
print("="*60)

# ==================== 快速数据处理 ====================
print("\n快速数据处理...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

# 基础特征工程
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = data['creat_year'] - data['reg_year']
data['car_age'] = data['car_age'].clip(lower=0)

data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', np.nan)
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

# 目标编码
for col in ['brand', 'model', 'regionCode']:
    data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')

drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
test_data = test_data.drop(columns=['price'])

X = train_data.drop(columns=['price'])
y = train_data['price']

X = X.fillna(X.median())
test_data = test_data.fillna(X.median())

common_cols = list(set(X.columns) & set(test_data.columns))
X = X[common_cols]
test_data = test_data[common_cols]

print(f"特征数: {X.shape[1]}")

# ==================== 快速神经网络 ====================
class FastNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_fast_nn(X_train, y_train, X_val, y_val, epochs=60):
    """快速训练神经网络"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train.values)
    )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)  # 更大批量

    model = FastNN(X_train.shape[1]).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-3)  # 更高学习率

    best_mae = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy()
        val_mae = mean_absolute_error(y_val, val_pred)

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = model.state_dict().copy()

        if (epoch + 1) % 15 == 0:
            print(f"      Epoch {epoch+1}/{epochs} Val MAE: {val_mae:.2f}")

    model.load_state_dict(best_state)
    return model, scaler, best_mae

# ==================== 快速树模型 ====================
def train_fast_cat(X_train, y_train, X_val, y_val):
    model = CatBoostRegressor(
        iterations=2000,  # 减少迭代次数
        learning_rate=0.05,  # 提高学习率
        depth=7,
        l2_leaf_reg=8,
        loss_function='MAE',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=80
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    return model, mae

def train_fast_lgb(X_train, y_train, X_val, y_val):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.05,  # 提高学习率
        'num_leaves': 95,
        'max_depth': 8,
        'verbose': -1,
        'seed': 42
    }

    model = lgb.train(params, train_data, num_boost_round=2000,
                       valid_sets=[val_data],
                       callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    return model, mae

# ==================== 快速主流程 ====================
print("\n开始快速训练...")

# 使用train_test_split代替5折CV以加快速度
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n训练集: {len(y_train_full)}, 验证集: {len(y_val_full)}")

# 训练所有模型
print("\n训练CatBoost...")
cat_model, cat_mae = train_fast_cat(X_train_full, y_train_full, X_val_full, y_val_full)
print(f"CatBoost MAE: {cat_mae:.2f}")

print("\n训练LightGBM...")
lgb_model, lgb_mae = train_fast_lgb(X_train_full, y_train_full, X_val_full, y_val_full)
print(f"LightGBM MAE: {lgb_mae:.2f}")

print("\n训练神经网络...")
nn_model, nn_scaler, nn_mae = train_fast_nn(X_train_full, y_train_full, X_val_full, y_val_full, epochs=60)
print(f"NeuralNet MAE: {nn_mae:.2f}")

# 验证集预测
cat_pred_val = cat_model.predict(X_val_full)
lgb_pred_val = lgb_model.predict(X_val_full)
test_val_scaled = nn_scaler.transform(X_val_full)
nn_model.eval()
with torch.no_grad():
    nn_pred_val = nn_model(torch.FloatTensor(test_val_scaled).to(device)).cpu().numpy()

# 融合策略比较
print("\n融合策略优化:")
inv_maes = [1/cat_mae, 1/lgb_mae, 1/nn_mae]
total = sum(inv_maes)
optimal_weights = [w/total for w in inv_maes]

simple_avg = (cat_pred_val + lgb_pred_val + nn_pred_val) / 3
simple_mae = mean_absolute_error(y_val_full, simple_avg)
print(f"简单平均: {simple_mae:.2f}")

inv_mae_pred = (optimal_weights[0] * cat_pred_val +
               optimal_weights[1] * lgb_pred_val +
               optimal_weights[2] * nn_pred_val)
inv_mae_combined = mean_absolute_error(y_val_full, inv_mae_pred)
print(f"反MAE权重: {inv_mae_combined:.2f}")

fixed_weights_pred = 0.4 * cat_pred_val + 0.35 * lgb_pred_val + 0.25 * nn_pred_val
fixed_weights_mae = mean_absolute_error(y_val_full, fixed_weights_pred)
print(f"固定权重(0.4,0.35,0.25): {fixed_weights_mae:.2f}")

print(f"\n最优权重: Cat={optimal_weights[0]:.4f}, LGB={optimal_weights[1]:.4f}, NN={optimal_weights[2]:.4f}")

# 选择最佳权重
maes = [simple_mae, inv_mae_combined, fixed_weights_mae]
best_idx = np.argmin(maes)
best_mae = maes[best_idx]

if best_idx == 0:
    print(f"\n最佳策略: 简单平均 ({best_mae:.2f})")
    final_weights = [1/3, 1/3, 1/3]
elif best_idx == 1:
    print(f"\n最佳策略: 反MAE权重 ({best_mae:.2f})")
    final_weights = optimal_weights
else:
    print(f"\n最佳策略: 固定权重 ({best_mae:.2f})")
    final_weights = [0.4, 0.35, 0.25]

# 测试集预测
print("\n预测测试集...")
cat_pred_test = cat_model.predict(test_data)
lgb_pred_test = lgb_model.predict(test_data)
test_scaled = nn_scaler.transform(test_data)
nn_model.eval()
with torch.no_grad():
    nn_pred_test = nn_model(torch.FloatTensor(test_scaled).to(device)).cpu().numpy()

final_pred = (final_weights[0] * cat_pred_test +
              final_weights[1] * lgb_pred_test +
              final_weights[2] * nn_pred_test)
final_pred = np.maximum(final_pred, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': sale_ids,
    'price': final_pred
})
submit.to_csv('aggressive_optimization_submit.csv', index=False)
print(f"\n结果已保存到 aggressive_optimization_submit.csv")

# 最终总结
print(f"\n{'='*60}")
print("快速优化结果")
print(f"{'='*60}")
print(f"CatBoost MAE: {cat_mae:.2f}")
print(f"LightGBM MAE: {lgb_mae:.2f}")
print(f"NeuralNet MAE: {nn_mae:.2f}")
print(f"最佳融合MAE: {best_mae:.2f}")

if best_mae <= 450:
    print(f"\n🎉 成功达到目标MAE: {best_mae:.2f} <= 450")
    print(f"比之前477改进: {477 - best_mae:.2f}")
else:
    print(f"\n⚠️ 距离目标: {best_mae - 450:.2f}")
    print(f"当前MAE: {best_mae:.2f}")
    print(f"比之前477改进: {477 - best_mae:.2f}")

print(f"{'='*60}")