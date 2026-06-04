# -*- coding: utf-8 -*-
"""
二手车价格预测 - PyTorch残差网络 + 三模型融合优化版
目标：MAE ≤ 450
特性：
- PyTorch残差网络处理特征
- 数值特征和分类特征分别处理
- 嵌入层处理分类变量
- 批归一化 + 梯度裁剪
- 与树模型融合
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ==================== 数据加载 ====================
def load_processed_data():
    """加载预处理后的数据"""
    print("正在加载预处理后的数据...")
    X_train = joblib.load('processed_data/X_train.joblib')
    X_val = joblib.load('processed_data/X_val.joblib')
    y_train = joblib.load('processed_data/y_train.joblib')
    y_val = joblib.load('processed_data/y_val.joblib')
    test_data = joblib.load('processed_data/test_data.joblib')
    sale_ids = joblib.load('processed_data/sale_ids.joblib')
    return X_train, X_val, y_train, y_val, test_data, sale_ids


# ==================== 噪声特征删除 ====================
NOISE_FEATURES = ['seller', 'offerType', 'SaleID', 'name', 
                   'creat_day', 'creat_year', 'creat_month', 'reg_day', 'reg_month']


# ==================== K-Fold目标编码 ====================
def kfold_target_encode(X_train, y_train, X_val, col, n_splits=5, smoothing=10.0):
    """K-Fold目标编码，避免数据泄露"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, index=X_train.index)
    
    train_encoded = np.zeros(len(X_train))
    
    for train_idx, val_idx in kf.split(X_train):
        fold_train_col = X_train.iloc[train_idx][col]
        fold_y = y_train.iloc[train_idx]
        target_mean = fold_y.mean()
        category_means = fold_y.groupby(fold_train_col).mean()
        category_counts = fold_train_col.groupby(fold_train_col).count()
        
        smoothed_mean = (category_means * category_counts + target_mean * smoothing) / (category_counts + smoothing)
        val_data = X_train.iloc[val_idx][col]
        train_encoded[val_idx] = val_data.map(smoothed_mean).fillna(target_mean)
    
    target_mean = y_train.mean()
    category_means = y_train.groupby(X_train[col]).mean()
    category_counts = X_train[col].groupby(X_train[col]).count()
    smoothed_mean = (category_means * category_counts + target_mean * smoothing) / (category_counts + smoothing)
    val_encoded = X_val[col].map(smoothed_mean).fillna(target_mean)
    
    return train_encoded, val_encoded


def kfold_group_stats(X_train, y_train, X_val, group_col, n_splits=5):
    """K-Fold分组统计，避免数据泄露"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, index=X_train.index)
    
    train_mean = np.zeros(len(X_train))
    train_median = np.zeros(len(X_train))
    train_std = np.zeros(len(X_train))
    
    for train_idx, val_idx in kf.split(X_train):
        fold_group = X_train.iloc[train_idx][group_col]
        fold_y = y_train.iloc[train_idx]
        
        group_stats = fold_y.groupby(fold_group).agg(['mean', 'median', 'std'])
        
        val_groups = X_train.iloc[val_idx][group_col]
        train_mean[val_idx] = val_groups.map(group_stats['mean']).fillna(fold_y.mean())
        train_median[val_idx] = val_groups.map(group_stats['median']).fillna(fold_y.median())
        train_std[val_idx] = val_groups.map(group_stats['std']).fillna(0)
    
    group_stats = y_train.groupby(X_train[group_col]).agg(['mean', 'median', 'std'])
    val_mean = X_val[group_col].map(group_stats['mean']).fillna(y_train.mean())
    val_median = X_val[group_col].map(group_stats['median']).fillna(y_train.median())
    val_std = X_val[group_col].map(group_stats['std']).fillna(0)
    
    return (train_mean, train_median, train_std), (val_mean, val_median, val_std)


# ==================== 深度特征工程 ====================
def deep_feature_engineering(X_train, y_train, X_val, test_data):
    """深度特征工程"""
    print("\n" + "="*50)
    print("开始深度特征工程...")
    print("="*50)
    
    # 1. 删除噪声特征
    noise_cols = [col for col in NOISE_FEATURES if col in X_train.columns]
    if noise_cols:
        print(f"删除噪声特征: {noise_cols}")
        X_train = X_train.drop(columns=noise_cols)
        X_val = X_val.drop(columns=noise_cols)
        test_data = test_data.drop(columns=noise_cols)
    
    # 2. 删除高相关特征
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = X_train[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
        if high_corr_features:
            print(f"删除高相关特征: {high_corr_features}")
            X_train = X_train.drop(columns=high_corr_features)
            X_val = X_val.drop(columns=high_corr_features)
            test_data = test_data.drop(columns=high_corr_features)
    
    # ==================== v特征深度挖掘 ====================
    v_cols = [col for col in X_train.columns if col.startswith('v_') and col not in ['v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_12', 'v_13']]
    print(f"\n发现的v特征: {v_cols}")
    
    # v特征统计
    if len(v_cols) >= 3:
        X_train['v_mean'] = X_train[v_cols].mean(axis=1)
        X_val['v_mean'] = X_val[v_cols].mean(axis=1)
        test_data['v_mean'] = test_data[v_cols].mean(axis=1)
        
        X_train['v_std'] = X_train[v_cols].std(axis=1)
        X_val['v_std'] = X_val[v_cols].std(axis=1)
        test_data['v_std'] = test_data[v_cols].std(axis=1)
        
        X_train['v_max'] = X_train[v_cols].max(axis=1)
        X_val['v_max'] = X_val[v_cols].max(axis=1)
        test_data['v_max'] = test_data[v_cols].max(axis=1)
        
        X_train['v_min'] = X_train[v_cols].min(axis=1)
        X_val['v_min'] = X_val[v_cols].min(axis=1)
        test_data['v_min'] = test_data[v_cols].min(axis=1)
        
        X_train['v_range'] = X_train['v_max'] - X_train['v_min']
        X_val['v_range'] = X_val['v_max'] - X_val['v_min']
        test_data['v_range'] = test_data['v_max'] - test_data['v_min']
    
    # v_0和v_3特殊处理
    if 'v_0' in X_train.columns and 'v_3' in X_train.columns:
        X_train['v_0_v_3_mul'] = X_train['v_0'] * X_train['v_3']
        X_val['v_0_v_3_mul'] = X_val['v_0'] * X_val['v_3']
        test_data['v_0_v_3_mul'] = test_data['v_0'] * test_data['v_3']
        
        X_train['v_0_v_3_ratio'] = X_train['v_0'] / (X_train['v_3'] + 1e-5)
        X_val['v_0_v_3_ratio'] = X_val['v_0'] / (X_val['v_3'] + 1e-5)
        test_data['v_0_v_3_ratio'] = test_data['v_0'] / (test_data['v_3'] + 1e-5)
        
        X_train['v_0_sq'] = X_train['v_0'] ** 2
        X_val['v_0_sq'] = X_val['v_0'] ** 2
        test_data['v_0_sq'] = test_data['v_0'] ** 2
        
        X_train['v_3_sq'] = X_train['v_3'] ** 2
        X_val['v_3_sq'] = X_val['v_3'] ** 2
        test_data['v_3_sq'] = test_data['v_3'] ** 2
        
        X_train['v_0_sqrt'] = np.sqrt(np.abs(X_train['v_0']))
        X_val['v_0_sqrt'] = np.sqrt(np.abs(X_val['v_0']))
        test_data['v_0_sqrt'] = np.sqrt(np.abs(test_data['v_0']))
        
        X_train['v_0_v_3_diff'] = X_train['v_0'] - X_train['v_3']
        X_val['v_0_v_3_diff'] = X_val['v_0'] - X_val['v_3']
        test_data['v_0_v_3_diff'] = test_data['v_0'] - test_data['v_3']
    
    # 三阶交互
    if 'v_0' in X_train.columns and 'v_3' in X_train.columns and 'v_2' in X_train.columns:
        X_train['v_0_v_2_v_3'] = X_train['v_0'] * X_train['v_2'] * X_train['v_3']
        X_val['v_0_v_2_v_3'] = X_val['v_0'] * X_val['v_2'] * X_val['v_3']
        test_data['v_0_v_2_v_3'] = test_data['v_0'] * test_data['v_2'] * test_data['v_3']
    
    # 对数变换
    if 'kilometer' in X_train.columns:
        X_train['log_km'] = np.log1p(X_train['kilometer'])
        X_val['log_km'] = np.log1p(X_val['kilometer'])
        test_data['log_km'] = np.log1p(test_data['kilometer'])
    
    if 'power' in X_train.columns:
        X_train['log_power'] = np.log1p(X_train['power'])
        X_val['log_power'] = np.log1p(X_val['power'])
        test_data['log_power'] = np.log1p(test_data['power'])
    
    # 与v特征交叉
    if 'power' in X_train.columns and 'v_0' in X_train.columns:
        X_train['power_v_0'] = X_train['power'] * X_train['v_0']
        X_val['power_v_0'] = X_val['power'] * X_val['v_0']
        test_data['power_v_0'] = test_data['power'] * test_data['v_0']
        
        X_train['power_v_3'] = X_train['power'] * X_train['v_3']
        X_val['power_v_3'] = X_val['power'] * X_val['v_3']
        test_data['power_v_3'] = test_data['power'] * test_data['v_3']
    
    # K-Fold目标编码
    if 'brand' in X_train.columns:
        print("\n添加brand目标编码...")
        train_encoded, val_encoded = kfold_target_encode(X_train, y_train, X_val, 'brand', n_splits=5, smoothing=15.0)
        X_train['brand_te'] = train_encoded
        X_val['brand_te'] = val_encoded
        target_mean = y_train.mean()
        category_means = y_train.groupby(X_train['brand']).mean()
        category_counts = X_train['brand'].groupby(X_train['brand']).count()
        smoothed_mean = (category_means * category_counts + target_mean * 15.0) / (category_counts + 15.0)
        test_data['brand_te'] = test_data['brand'].map(smoothed_mean).fillna(target_mean)
    
    # 分组统计
    if 'brand' in X_train.columns:
        print("添加brand分组统计...")
        train_stats, val_stats = kfold_group_stats(X_train, y_train, X_val, 'brand', n_splits=5)
        X_train['brand_mean'] = train_stats[0]
        X_val['brand_mean'] = val_stats[0]
        
        group_stats = y_train.groupby(X_train['brand']).agg(['mean'])
        test_data['brand_mean'] = test_data['brand'].map(group_stats['mean']).fillna(y_train.mean())
    
    # 处理无穷值和缺失值
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    test_data = test_data.replace([np.inf, -np.inf], np.nan)
    
    for col in X_train.columns:
        if X_train[col].dtype in [np.float64, np.int64]:
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_val[col].fillna(median_val, inplace=True)
            test_data[col].fillna(median_val, inplace=True)
    
    print(f"\n特征工程完成，最终特征数: {X_train.shape[1]}")
    
    return X_train, X_val, test_data


# ==================== PyTorch残差网络 ====================
class ResidualBlock(nn.Module):
    """残差块：包含批归一化"""
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # 如果输入输出维度不同，使用投影
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.relu(out)
        return out


class EmbeddingLayer(nn.Module):
    """嵌入层处理分类特征"""
    def __init__(self, num_categories, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
    
    def forward(self, x):
        x = self.embedding(x.long())
        x = x.squeeze(1) if x.dim() == 3 else x
        return self.bn(x)


class ResNetRegressor(nn.Module):
    """
    残差网络回归器
    - 数值特征分支：通过残差块处理
    - 分类特征分支：通过嵌入层处理
    - 最终融合输出
    """
    def __init__(self, num_numeric_features, cat_cardinalities, 
                 embedding_dim=16, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        
        # 分类特征嵌入层
        self.cat_embeddings = nn.ModuleList([
            EmbeddingLayer(card, min(embedding_dim, (card + 1) // 2))
            for card in cat_cardinalities
        ])
        
        total_embed_dim = sum(min(embedding_dim, (card + 1) // 2) for card in cat_cardinalities)
        
        # 数值特征处理
        self.numeric_bn = nn.BatchNorm1d(num_numeric_features)
        
        # 总输入维度
        total_input = num_numeric_features + total_embed_dim
        
        # 残差块堆叠
        self.input_layer = nn.Sequential(
            nn.Linear(total_input, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout)
            )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1)
        )
    
    def forward(self, numeric_x, cat_x):
        # 数值特征处理
        numeric_out = self.numeric_bn(numeric_x)
        
        # 分类特征嵌入
        cat_outs = []
        for i, embed_layer in enumerate(self.cat_embeddings):
            cat_outs.append(embed_layer(cat_x[:, i]))
        cat_out = torch.cat(cat_outs, dim=1) if cat_outs else torch.empty(numeric_x.size(0), 0).to(numeric_x.device)
        
        # 合并特征
        x = torch.cat([numeric_out, cat_out], dim=1)
        
        # 残差网络处理
        x = self.input_layer(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 输出
        out = self.output_layer(x)
        return out.squeeze(-1)


class MAELoss(nn.Module):
    """MAE损失函数"""
    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))


class HuberLoss(nn.Module):
    """Huber损失：对异常值更鲁棒"""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = torch.abs(pred - target)
        quadratic = torch.min(error, torch.tensor(self.delta).to(error.device))
        linear = error - quadratic
        return torch.mean(0.5 * quadratic ** 2 + self.delta * linear)


def prepare_pytorch_data(X_train, X_val, test_data, cat_cols=['brand', 'model', 'gearbox', 'fuelType', 'notRepairedDamage']):
    """准备PyTorch训练数据"""
    print("\n准备PyTorch数据...")
    
    # 识别分类列
    available_cat_cols = [col for col in cat_cols if col in X_train.columns]
    print(f"分类特征: {available_cat_cols}")
    
    # 数值列
    numeric_cols = [col for col in X_train.columns if col not in available_cat_cols]
    print(f"数值特征数: {len(numeric_cols)}")
    
    # 标签编码分类特征
    label_encoders = {}
    cat_cardinalities = []
    
    X_train_cat = np.zeros((len(X_train), len(available_cat_cols)), dtype=np.int64)
    X_val_cat = np.zeros((len(X_val), len(available_cat_cols)), dtype=np.int64)
    test_cat = np.zeros((len(test_data), len(available_cat_cols)), dtype=np.int64)
    
    for i, col in enumerate(available_cat_cols):
        le = LabelEncoder()
        # 合并所有数据进行编码
        all_values = pd.concat([X_train[col], X_val[col], test_data[col]]).astype(str).fillna('missing')
        le.fit(all_values)
        
        X_train_cat[:, i] = le.transform(X_train[col].astype(str).fillna('missing'))
        X_val_cat[:, i] = le.transform(X_val[col].astype(str).fillna('missing'))
        test_cat[:, i] = le.transform(test_data[col].astype(str).fillna('missing'))
        
        label_encoders[col] = le
        cat_cardinalities.append(len(le.classes_))
    
    print(f"分类特征基数: {cat_cardinalities}")
    
    # 标准化数值特征
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numeric_cols])
    X_val_num = scaler.transform(X_val[numeric_cols])
    test_num = scaler.transform(test_data[numeric_cols])
    
    return {
        'train_num': X_train_num,
        'train_cat': X_train_cat,
        'val_num': X_val_num,
        'val_cat': X_val_cat,
        'test_num': test_num,
        'test_cat': test_cat,
        'cat_cardinalities': cat_cardinalities,
        'num_numeric': len(numeric_cols),
        'label_encoders': label_encoders,
        'scaler': scaler,
        'numeric_cols': numeric_cols,
        'cat_cols': available_cat_cols
    }


def train_pytorch_model(data_dict, y_train, y_val, epochs=100, batch_size=512, lr=0.001, 
                        weight_decay=1e-5, max_grad_norm=1.0):
    """训练PyTorch残差网络模型"""
    print("\n" + "="*50)
    print("训练PyTorch残差网络模型...")
    print("="*50)
    
    # 准备数据
    train_dataset = TensorDataset(
        torch.FloatTensor(data_dict['train_num']),
        torch.LongTensor(data_dict['train_cat']),
        torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data_dict['val_num']),
        torch.LongTensor(data_dict['val_cat']),
        torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)
    
    # 初始化模型
    model = ResNetRegressor(
        num_numeric_features=data_dict['num_numeric'],
        cat_cardinalities=data_dict['cat_cardinalities'],
        embedding_dim=16,
        hidden_dims=[256, 128, 64],
        dropout=0.3
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = HuberLoss(delta=100.0)  # 使用Huber损失，对异常值鲁棒
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # 训练
    best_val_mae = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 15
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for numeric_x, cat_x, y_batch in train_loader:
            numeric_x, cat_x, y_batch = numeric_x.to(device), cat_x.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(numeric_x, cat_x)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for numeric_x, cat_x, y_batch in val_loader:
                numeric_x, cat_x, y_batch = numeric_x.to(device), cat_x.to(device), y_batch.to(device)
                outputs = model(numeric_x, cat_x)
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())
        
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_mae = mean_absolute_error(val_targets, val_preds)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Val MAE: {val_mae:.2f}")
        
        # 早停
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停于 Epoch {epoch+1}")
                break
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    print(f"\n最佳验证 MAE: {best_val_mae:.2f}")
    
    # 测试集预测
    model.eval()
    with torch.no_grad():
        test_numeric = torch.FloatTensor(data_dict['test_num']).to(device)
        test_cat = torch.LongTensor(data_dict['test_cat']).to(device)
        test_pred = model(test_numeric, test_cat).cpu().numpy()
    
    # 验证集预测
    model.eval()
    with torch.no_grad():
        val_numeric = torch.FloatTensor(data_dict['val_num']).to(device)
        val_cat = torch.LongTensor(data_dict['val_cat']).to(device)
        val_pred = model(val_numeric, val_cat).cpu().numpy()
    
    return model, val_pred, test_pred, best_val_mae


# ==================== 树模型训练 ====================
def train_catboost(X_train, X_val, y_train, y_val):
    """训练CatBoost模型"""
    print("\n" + "="*50)
    print("训练CatBoost模型...")
    print("="*50)
    
    params = {
        'iterations': 5000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 10,
        'min_data_in_leaf': 30,
        'rsm': 0.8,
        'random_seed': 42,
        'od_type': 'Iter',
        'od_wait': 50,
        'verbose': 500,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'task_type': 'CPU',
        'thread_count': -1,
        'use_best_model': True,
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, verbose=500)
    
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"CatBoost MAE: {mae:.2f}")
    
    return model, y_pred


def train_lightgbm(X_train, X_val, y_train, y_val):
    """训练LightGBM模型"""
    print("\n" + "="*50)
    print("训练LightGBM模型...")
    print("="*50)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 63,
        'max_depth': 8,
        'min_data_in_leaf': 30,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'verbose': -1,
        'seed': 42,
    }
    
    callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=500)]
    
    model = lgb.train(params, train_data, num_boost_round=5000, valid_sets=[train_data, val_data],
                      valid_names=['train', 'valid'], callbacks=callbacks)
    
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"LightGBM MAE: {mae:.2f}")
    
    return model, y_pred


def train_xgboost(X_train, X_val, y_train, y_val):
    """训练XGBoost模型"""
    print("\n" + "="*50)
    print("训练XGBoost模型...")
    print("="*50)
    
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.02,
        'max_depth': 8,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 5000,
        'random_state': 42,
        'eval_metric': 'mae',
        'early_stopping_rounds': 50,
        'n_jobs': -1,
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=500)
    
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"XGBoost MAE: {mae:.2f}")
    
    return model, y_pred


# ==================== 模型融合 ====================
def ensemble_predictions(predictions_list, weights=None, y_val=None):
    """模型融合：加权平均"""
    print("\n" + "="*50)
    print("模型融合（加权平均）...")
    print("="*50)
    
    if weights is None:
        # 基于验证集MAE自动计算权重
        maes = [mean_absolute_error(y_val, pred) for pred in predictions_list]
        inv_maes = [1/max(m, 1) for m in maes]  # 防止除零
        total = sum(inv_maes)
        weights = [w/total for w in inv_maes]
        print(f"自动计算权重: {[f'{w:.3f}' for w in weights]}")
    
    # 加权平均
    ensemble_pred = np.zeros(len(predictions_list[0]))
    for pred, weight in zip(predictions_list, weights):
        ensemble_pred += pred * weight
    
    if y_val is not None:
        mae = mean_absolute_error(y_val, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        r2 = r2_score(y_val, ensemble_pred)
        print(f"\n融合模型 MAE: {mae:.2f}")
        print(f"融合模型 RMSE: {rmse:.2f}")
        print(f"融合模型 R²: {r2:.4f}")
    
    return ensemble_pred, weights


def stacking_ensemble(X_train, y_train, X_val, y_val, test_data, data_dict, n_folds=5):
    """
    Stacking融合：包含PyTorch残差网络
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler as SS
    
    print("\n" + "="*50)
    print("Stacking融合训练（包含PyTorch残差网络）...")
    print("="*50)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 4个基模型
    train_meta = np.zeros((len(X_train), 4))
    val_meta = np.zeros((len(X_val), 4))
    test_meta = np.zeros((len(test_data), 4))
    
    # K-Fold训练
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # CatBoost
        cat = CatBoostRegressor(
            iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=10,
            min_data_in_leaf=30, random_seed=42, verbose=0, loss_function='MAE'
        )
        cat.fit(X_tr, y_tr, verbose=0)
        train_meta[val_idx, 0] = cat.predict(X_va)
        val_meta[:, 0] += cat.predict(X_val) / n_folds
        test_meta[:, 0] += cat.predict(test_data) / n_folds
        
        # LightGBM
        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'learning_rate': 0.02,
             'num_leaves': 63, 'max_depth': 8, 'verbose': -1, 'seed': 42},
            lgb_train, num_boost_round=3000
        )
        train_meta[val_idx, 1] = lgb_model.predict(X_va)
        val_meta[:, 1] += lgb_model.predict(X_val) / n_folds
        test_meta[:, 1] += lgb_model.predict(test_data) / n_folds
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.02, max_depth=8,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )
        xgb_model.fit(X_tr, y_tr, verbose=False)
        train_meta[val_idx, 2] = xgb_model.predict(X_va)
        val_meta[:, 2] += xgb_model.predict(X_val) / n_folds
        test_meta[:, 2] += xgb_model.predict(test_data) / n_folds
        
        fold_mae = mean_absolute_error(y_va, train_meta[val_idx, 0])
        print(f"Fold {fold+1} CatBoost MAE: {fold_mae:.2f}")
    
    # PyTorch模型单独训练（使用完整训练集）
    print("\n训练PyTorch残差网络...")
    pytorch_model, pytorch_val_pred, pytorch_test_pred, pytorch_mae = train_pytorch_model(
        data_dict, y_train, y_val, epochs=150, batch_size=512, lr=0.001
    )
    
    train_meta[:, 3] = pytorch_val_pred  # 验证集预测
    val_meta[:, 3] = pytorch_val_pred
    test_meta[:, 3] = pytorch_test_pred
    
    # 训练元学习器
    print("\n训练元学习器（Ridge）...")
    scaler = SS()
    train_meta_scaled = scaler.fit_transform(train_meta)
    val_meta_scaled = scaler.transform(val_meta)
    test_meta_scaled = scaler.transform(test_meta)
    
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(train_meta_scaled, y_train)
    
    val_pred = meta_model.predict(val_meta_scaled)
    test_pred = meta_model.predict(test_meta_scaled)
    
    # 评估
    mae = mean_absolute_error(y_val, val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    r2 = r2_score(y_val, val_pred)
    
    print(f"\nStacking模型 MAE: {mae:.2f}")
    print(f"Stacking模型 RMSE: {rmse:.2f}")
    print(f"Stacking模型 R²: {r2:.4f}")
    
    weights = meta_model.coef_
    weights = np.abs(weights) / np.abs(weights).sum()
    print(f"\n学习到的权重: CatBoost={weights[0]:.3f}, LightGBM={weights[1]:.3f}, XGBoost={weights[2]:.3f}, PyTorch={weights[3]:.3f}")
    
    return val_pred, test_pred, mae


# ==================== 主函数 ====================
def main():
    # 加载数据
    X_train, X_val, y_train, y_val, test_data, sale_ids = load_processed_data()
    
    # 深度特征工程
    X_train_eng, X_val_eng, test_data_eng = deep_feature_engineering(X_train, y_train, X_val, test_data)
    
    # 确保列一致
    common_cols = list(set(X_train_eng.columns) & set(X_val_eng.columns) & set(test_data_eng.columns))
    X_train_eng = X_train_eng[common_cols].reset_index(drop=True)
    X_val_eng = X_val_eng[common_cols].reset_index(drop=True)
    test_data_eng = test_data_eng[common_cols].reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    
    # 准备PyTorch数据
    data_dict = prepare_pytorch_data(X_train_eng, X_val_eng, test_data_eng)
    
    # Stacking融合
    val_pred, test_predictions, stacking_mae = stacking_ensemble(
        X_train_eng, y_train, X_val_eng, y_val, test_data_eng, data_dict, n_folds=5
    )
    
    # 保存结果
    submit_data = pd.DataFrame({
        'SaleID': sale_ids,
        'price': test_predictions
    })
    submit_data.to_csv('pytorch_ensemble_submit.csv', index=False)
    
    print(f"\n预测结果已保存到 pytorch_ensemble_submit.csv")
    
    # 绘制预测对比图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, val_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('PyTorch残差网络融合模型 - 预测价格 vs 实际价格')
    plt.tight_layout()
    plt.savefig('pytorch_ensemble_prediction_vs_actual.png')
    plt.close()
    
    # 最终评估
    final_mae = mean_absolute_error(y_val, val_pred)
    final_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    final_r2 = r2_score(y_val, val_pred)
    
    print("\n" + "="*50)
    print("最终结果")
    print("="*50)
    print(f"MAE: {final_mae:.2f}")
    print(f"RMSE: {final_rmse:.2f}")
    print(f"R²: {final_r2:.4f}")
    
    if final_mae <= 450:
        print(f"\n✓ 成功！MAE已达到目标 (≤450)")
    else:
        print(f"\n还需继续优化，距离目标还差 {final_mae - 450:.2f}")
    
    return final_mae


if __name__ == "__main__":
    main()
