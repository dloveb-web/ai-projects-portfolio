# -*- coding: utf-8 -*-
"""
二手车价格预测 - 元学习自动优化版
使用元学习自动寻找最优模型组合和权重
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
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cpu')

print("="*60)
print("元学习自动优化版 - 寻找最优组合和权重")
print("目标: MAE <= 450")
print("="*60)

# ==================== 数据处理 ====================
print("\n数据处理...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

# 特征工程（保持与最佳477代码一致）
test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

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

# ==================== 神经网络 ====================
class SimpleResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        return self.act(x + self.net(x))

class MetaNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        layers = []
        hidden_dims = [256, 256, 128, 128]
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(0.1))
            prev_dim = dim

        layers.append(SimpleResidualBlock(hidden_dims[-1]))
        layers.append(SimpleResidualBlock(hidden_dims[-1]))

        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_meta_nn(X_train, y_train, X_val, y_val, epochs=80):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train.values)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = MetaNN(X_train.shape[1]).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=12, factor=0.5)

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
        scheduler.step(val_mae)

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = model.state_dict().copy()

        if val_mae < best_mae * 0.95:  # 早期停止如果提升显著
            break

    model.load_state_dict(best_state)
    return model, scaler, best_mae

# ==================== 树模型训练 ====================
def train_cat_meta(X_train, y_train, X_val, y_val):
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=10,
        loss_function='MAE',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    return model, mae

def train_lgb_meta(X_train, y_train, X_val, y_val):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.02,
        'num_leaves': 63,
        'max_depth': 8,
        'verbose': -1,
        'seed': 42
    }

    model = lgb.train(params, train_data, num_boost_round=3000,
                       valid_sets=[val_data],
                       callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    return model, mae

# ==================== 元学习优化 ====================
def find_optimal_weights(preds_dict, y_true):
    """使用优化算法寻找最优权重"""
    def objective(weights):
        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化

        ensemble_pred = sum(weights[i] * preds_dict[i] for i in range(len(preds_dict)))
        mae = mean_absolute_error(y_true, ensemble_pred)
        return mae

    # 初始猜测（均匀权重）
    n_models = len(preds_dict)
    initial_weights = np.ones(n_models) / n_models

    # 约束条件：权重和为1，非负
    bounds = [(0, 1)] * n_models
    constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}

    # 优化
    result = minimize(objective, initial_weights, bounds=bounds, method='SLSQP',
                  constraints=constraints, options={'ftol': 1e-6})

    optimal_weights = result.x / result.x.sum()
    optimal_mae = result.fun

    return optimal_weights, optimal_mae

# ==================== 主训练流程 ====================
print("\n开始训练...")

# 分割训练集和验证集（用于元学习）
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练所有模型
print("\n训练CatBoost...")
cat_model, cat_mae = train_cat_meta(X_train_full, y_train_full, X_val_full, y_val_full)
print(f"CatBoost MAE: {cat_mae:.2f}")

print("\n训练LightGBM...")
lgb_model, lgb_mae = train_lgb_meta(X_train_full, y_train_full, X_val_full, y_val_full)
print(f"LightGBM MAE: {lgb_mae:.2f}")

print("\n训练神经网络...")
nn_model, nn_scaler, nn_mae = train_meta_nn(X_train_full, y_train_full, X_val_full, y_val_full)
print(f"NeuralNet MAE: {nn_mae:.2f}")

# 获取验证集预测
cat_pred_val = cat_model.predict(X_val_full)
lgb_pred_val = lgb_model.predict(X_val_full)
test_val_scaled = nn_scaler.transform(X_val_full)
nn_model.eval()
with torch.no_grad():
    nn_pred_val = nn_model(torch.FloatTensor(test_val_scaled).to(device)).cpu().numpy()

# 元学习：寻找最优权重
print("\n元学习：寻找最优权重...")
preds_for_optimization = np.array([cat_pred_val, lgb_pred_val, nn_pred_val])
optimal_weights, optimal_mae = find_optimal_weights(preds_for_optimization, y_val_full)

print(f"最优权重: CatBoost={optimal_weights[0]:.4f}, LightGBM={optimal_weights[1]:.4f}, NeuralNet={optimal_weights[2]:.4f}")
print(f"优化后MAE: {optimal_mae:.2f}")

# 对比不同融合策略
print("\n融合策略对比:")
simple_avg_pred = (cat_pred_val + lgb_pred_val + nn_pred_val) / 3
simple_avg_mae = mean_absolute_error(y_val_full, simple_avg_pred)
print(f"简单平均: MAE={simple_avg_mae:.2f}")

fixed_weights_pred = 0.4 * cat_pred_val + 0.35 * lgb_pred_val + 0.25 * nn_pred_val
fixed_weights_mae = mean_absolute_error(y_val_full, fixed_weights_pred)
print(f"固定权重(0.4,0.35,0.25): MAE={fixed_weights_mae:.2f}")

inv_mae_weights = np.array([1/cat_mae, 1/lgb_mae, 1/nn_mae])
inv_mae_weights = inv_mae_weights / inv_mae_weights.sum()
inv_mae_pred = (inv_mae_weights[0] * cat_pred_val +
                inv_mae_weights[1] * lgb_pred_val +
                inv_mae_weights[2] * nn_pred_val)
inv_mae_combined = mean_absolute_error(y_val_full, inv_mae_pred)
print(f"反MAE权重: MAE={inv_mae_combined:.2f}")

optimal_pred = (optimal_weights[0] * cat_pred_val +
               optimal_weights[1] * lgb_pred_val +
               optimal_weights[2] * nn_pred_val)
optimal_combined = mean_absolute_error(y_val_full, optimal_pred)
print(f"元学习优化: MAE={optimal_combined:.2f} ⭐")

# 使用最优权重预测测试集
print("\n预测测试集...")
cat_pred_test = cat_model.predict(test_data)
lgb_pred_test = lgb_model.predict(test_data)
test_scaled = nn_scaler.transform(test_data)
nn_model.eval()
with torch.no_grad():
    nn_pred_test = nn_model(torch.FloatTensor(test_scaled).to(device)).cpu().numpy()

final_pred = (optimal_weights[0] * cat_pred_test +
              optimal_weights[1] * lgb_pred_test +
              optimal_weights[2] * nn_pred_test)
final_pred = np.maximum(final_pred, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': sale_ids,
    'price': final_pred
})
submit.to_csv('meta_learning_submit.csv', index=False)
print(f"\n结果已保存到 meta_learning_submit.csv")

# 最终总结
print(f"\n{'='*60}")
print("最终结果")
print(f"{'='*60}")
print(f"CatBoost MAE: {cat_mae:.2f}")
print(f"LightGBM MAE: {lgb_mae:.2f}")
print(f"NeuralNet MAE: {nn_mae:.2f}")
print(f"元学习优化MAE: {optimal_combined:.2f}")

if optimal_combined <= 450:
    print(f"\n🎉 成功达到目标MAE: {optimal_combined:.2f} <= 450")
    print(f"比之前477改进: {477 - optimal_combined:.2f}")
else:
    print(f"\n⚠️ 距离目标: {optimal_combined - 450:.2f}")
    print(f"当前MAE: {optimal_combined:.2f}")
    print(f"比之前477改进: {477 - optimal_combined:.2f}")

print(f"{'='*60}")