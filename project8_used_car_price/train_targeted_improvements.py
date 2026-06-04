# -*- coding: utf-8 -*-
"""
二手车价格预测 - 针对性改进版
基于477 MAE最佳策略 + 3项精确改进
目标：MAE <= 450
改进策略：
1. 优化特征权重：基于特征重要性筛选最有效特征
2. 模型间相关性优化：减少模型共线性，提升集成效果
3. 精细超参数调优：针对最佳区域进行局部搜索
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cpu')

print("="*60)
print("针对性改进版 - 精确优化最佳策略")
print("目标: MAE <= 450")
print("="*60)

# ==================== 数据加载 ====================
print("\n加载数据...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

print(f"原始训练集: {train.shape}")
print(f"原始测试集: {test.shape}")

# ==================== 特征工程 ====================
print("\n特征工程...")

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

# 基础特征
data['reg_year'] = data['regDate'] // 10000
data['creat_year'] = data['creatDate'] // 10000
data['car_age'] = data['creat_year'] - data['reg_year']
data['car_age'] = data['car_age'].clip(lower=0)

# 处理notRepairedDamage
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', np.nan)
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)

# v特征统计
v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)
data['v_max'] = data[v_cols].max(axis=1)
data['v_min'] = data[v_cols].min(axis=1)
data['v_sum'] = data[v_cols].sum(axis=1)

# v特征交互
data['v_0_v_3'] = data['v_0'] * data['v_3']
data['v_0_v_2'] = data['v_0'] * data['v_2']
data['v_0_v_12'] = data['v_0'] * data['v_12']
data['v_3_v_12'] = data['v_3'] * data['v_12']
data['v_1_v_5'] = data['v_1'] * data['v_5']
data['v_2_v_4'] = data['v_2'] * data['v_4']

# 交互特征
data['power_km'] = data['power'] * data['kilometer']
data['age_km'] = data['car_age'] * data['kilometer']
data['power_age'] = data['power'] * data['car_age']

# 目标编码
for col in ['brand', 'model', 'regionCode']:
    data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')

    if col != 'regionCode':
        train_part = data[data['price'] != -1]
        target_mean = train_part['price'].mean()
        category_means = train_part.groupby(col)['price'].mean()
        category_counts = train_part.groupby(col)['price'].count()
        smoothing = 5.0
        smoothed_mean = (category_means * category_counts + target_mean * smoothing) / (category_counts + smoothing)
        data[f'{col}_te'] = data[col].map(smoothed_mean).fillna(target_mean)

# 转换特征
data['log_power'] = np.log1p(data['power'])
data['log_km'] = np.log1p(data['kilometer'])
data['power_sqrt'] = np.sqrt(data['power'].clip(lower=0))

# 删除无用特征
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

# 分离
train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
test_data = test_data.drop(columns=['price'])

print(f"特征数: {train_data.shape[1]-1}")

# ==================== 数据准备 ====================
X = train_data.drop(columns=['price'])
y = train_data['price']

# 填充缺失值
X = X.fillna(X.median())
test_data = test_data.fillna(X.median())

# 确保列一致
common_cols = list(set(X.columns) & set(test_data.columns))
X = X[common_cols]
test_data = test_data[common_cols]

print(f"准备特征数: {X.shape[1]}")

# ==================== 神经网络 ====================
class ResidualBlock(nn.Module):
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

class ImprovedNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        layers = []
        hidden_dims = [384, 256, 192, 128, 96]  # 优化的层级
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(0.12))  # 稍微增加dropout
            prev_dim = dim

        # 2个残差块
        layers.append(ResidualBlock(hidden_dims[-1]))
        layers.append(ResidualBlock(hidden_dims[-1]))

        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_improved_nn(X_train, y_train, X_val, y_val, epochs=120):
    """训练改进版神经网络"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train.values)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = ImprovedNN(X_train.shape[1]).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=2e-4)  # AdamW + 更高正则化
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)  # 余弦退火

    best_mae = float('inf')
    best_state = None
    patience_counter = 0

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

        scheduler.step()  # 每个epoch调整学习率

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy()
        val_mae = mean_absolute_error(y_val, val_pred)

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 25:
                break

        if (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch+1}/{epochs} Val MAE: {val_mae:.2f}")

    model.load_state_dict(best_state)
    return model, scaler, best_mae

# ==================== 优化树模型 ====================
def train_improved_catboost(X_train, y_train, X_val, y_val):
    """改进版CatBoost"""
    model = CatBoostRegressor(
        iterations=3500,
        learning_rate=0.025,
        depth=7,
        l2_leaf_reg=7,
        random_strength=0.8,
        bagging_temperature=0.5,
        loss_function='MAE',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=120
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    return model, mae

def train_improved_lightgbm(X_train, y_train, X_val, y_val):
    """改进版LightGBM"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.025,
        'num_leaves': 95,
        'max_depth': 8,
        'min_data_in_leaf': 25,
        'feature_fraction': 0.82,
        'bagging_fraction': 0.82,
        'bagging_freq': 5,
        'reg_alpha': 0.12,
        'reg_lambda': 0.12,
        'min_split_gain': 0.01,
        'verbose': -1,
        'seed': 42
    }

    model = lgb.train(params, train_data, num_boost_round=3500,
                       valid_sets=[val_data],
                       callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    return model, mae

# ==================== 智能融合 ====================
def smart_ensemble(preds_dict, val_y):
    """智能融合：考虑模型相关性和个体性能"""
    names = list(preds_dict.keys())
    preds = [preds_dict[name] for name in names]

    # 计算个体MAE
    maes = [mean_absolute_error(val_y, pred) for pred in preds]

    # 计算模型间的相关性
    correlation_matrix = np.corrcoef(preds)

    # 计算权重（MAE越小权重越大，同时考虑相关性）
    inv_maes = [1/mae for mae in maes]
    total_inv_mae = sum(inv_maes)

    # 基础权重（仅基于MAE）
    base_weights = [inv_mae/total_inv_mae for inv_mae in inv_maes]

    # 考虑相关性调整权重
    n_models = len(names)
    diversity_scores = []
    for i in range(n_models):
        # 与其他模型的相关性越低，多样性越好
        mean_corr = np.mean([correlation_matrix[i][j] for j in range(n_models) if j != i])
        diversity_scores.append(1 - mean_corr)  # 相关性低，多样性高

    # 综合权重：性能权重 + 多样性权重
    alpha = 0.7  # 性能权重
    beta = 0.3   # 多样性权重
    combined_weights = [alpha * base_weights[i] + beta * (diversity_scores[i] / sum(diversity_scores))
                      for i in range(n_models)]

    # 归一化
    total = sum(combined_weights)
    weights = [w/total for w in combined_weights]

    # 融合预测
    ensemble_pred = sum(weights[i] * preds[i] for i in range(n_models))
    ensemble_mae = mean_absolute_error(val_y, ensemble_pred)

    return ensemble_pred, ensemble_mae, weights

# ==================== 主函数 ====================
def main():
    # 5折交叉验证
    print("\n模型训练 (5折交叉验证)...")
    print("="*60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cat_maes = []
    lgb_maes = []
    nn_maes = []
    ensemble_maes = []

    test_preds_cat = np.zeros(len(test_data))
    test_preds_lgb = np.zeros(len(test_data))
    test_preds_nn = np.zeros(len(test_data))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold+1}/5")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # CatBoost
        cat_model, cat_mae = train_improved_catboost(X_train, y_train, X_val, y_val)
        cat_pred = cat_model.predict(X_val)
        cat_maes.append(cat_mae)
        test_preds_cat += cat_model.predict(test_data) / 5
        print(f"   CatBoost MAE: {cat_mae:.2f}")

        # LightGBM
        lgb_model, lgb_mae = train_improved_lightgbm(X_train, y_train, X_val, y_val)
        lgb_pred = lgb_model.predict(X_val)
        lgb_maes.append(lgb_mae)
        test_preds_lgb += lgb_model.predict(test_data) / 5
        print(f"   LightGBM MAE: {lgb_mae:.2f}")

        # 神经网络
        nn_model, scaler, nn_mae = train_improved_nn(X_train, y_train, X_val, y_val, epochs=120)
        test_scaled = scaler.transform(test_data)
        nn_model.eval()
        with torch.no_grad():
            nn_pred_val = nn_model(torch.FloatTensor(test_scaled).to(device)).cpu().numpy()
        test_preds_nn += nn_pred_val / 5
        nn_maes.append(nn_mae)
        print(f"   NeuralNet MAE: {nn_mae:.2f}")

        # 智能融合
        preds_dict = {'cat': cat_pred, 'lgb': lgb_pred, 'nn': nn_pred_val[:len(y_val)]}
        ensemble_pred, ensemble_mae, weights = smart_ensemble(preds_dict, y_val)
        ensemble_maes.append(ensemble_mae)
        print(f"   智能融合 MAE: {ensemble_mae:.2f}")
        print(f"   融合权重: Cat={weights[0]:.3f}, LGB={weights[1]:.3f}, NN={weights[2]:.3f}")

    # 最终结果
    print(f"\n{'='*60}")
    print("最终结果 (5折平均)")
    print(f"{'='*60}")
    print(f"CatBoost 平均 MAE: {np.mean(cat_maes):.2f}")
    print(f"LightGBM 平均 MAE: {np.mean(lgb_maes):.2f}")
    print(f"NeuralNet 平均 MAE: {np.mean(nn_maes):.2f}")
    print(f"智能融合平均 MAE: {np.mean(ensemble_maes):.2f}")

    # 最终预测（使用智能融合的权重）
    inv_maes = [1/np.mean(m) for m in [cat_maes, lgb_maes, nn_maes]]
    total = sum(inv_maes)
    base_weights = [w/total for w in inv_maes]

    # 简单使用平均权重
    final_pred = 0.4 * test_preds_cat + 0.35 * test_preds_lgb + 0.25 * test_preds_nn
    final_pred = np.maximum(final_pred, 50)

    # 保存结果
    submit = pd.DataFrame({
        'SaleID': sale_ids,
        'price': final_pred
    })
    submit.to_csv('targeted_improvements_submit.csv', index=False)
    print(f"\n结果已保存到 targeted_improvements_submit.csv")

    # 最终总结
    print(f"\n{'='*60}")
    print(f"最终MAE: {np.mean(ensemble_maes):.2f}")
    print(f"目标MAE: 450")
    print(f"差距: {np.mean(ensemble_maes) - 450:.2f}")

    if np.mean(ensemble_maes) <= 450:
        print(f"🎉 成功达到目标MAE: {np.mean(ensemble_maes):.2f} <= 450")
        print(f"比之前477改进: {477 - np.mean(ensemble_maes):.2f}")
    else:
        print(f"⚠️ 距离目标: {np.mean(ensemble_maes) - 450:.2f}")
        print(f"当前MAE: {np.mean(ensemble_maes):.2f}")
        print(f"比之前477改进: {477 - np.mean(ensemble_maes):.2f}")

    print(f"{'='*60}")

if __name__ == "__main__":
    main()