# -*- coding: utf-8 -*-
"""
二手车价格预测 - 神经网络模型 (MLP)
目标：MAE < 400
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ==================== MLP模型定义 ====================
class MLP(nn.Module):
    """多层感知机模型"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout_rate=0.2):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class ResidualMLP(nn.Module):
    """带残差连接的MLP"""
    def __init__(self, input_dim, hidden_dim=256, num_blocks=4, dropout_rate=0.2):
        super(ResidualMLP, self).__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ))
        
        self.output = nn.Linear(hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.blocks:
            residual = x
            x = block(x)
            x = self.leaky_relu(x + residual)
            x = self.dropout(x)
        
        return self.output(x).squeeze(-1)


# ==================== 数据准备 ====================
def load_and_prepare_data():
    """加载并准备数据"""
    print("正在加载数据...")
    X_train = joblib.load('processed_data/X_train.joblib')
    X_val = joblib.load('processed_data/X_val.joblib')
    y_train = joblib.load('processed_data/y_train.joblib')
    y_val = joblib.load('processed_data/y_val.joblib')
    test_data = joblib.load('processed_data/test_data.joblib')
    sale_ids = joblib.load('processed_data/sale_ids.joblib')
    
    # 删除非数值列
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train = X_train[numeric_cols]
    X_val = X_val[numeric_cols]
    test_data = test_data[numeric_cols]
    
    # 填充缺失值
    X_train = X_train.fillna(X_train.median())
    X_val = X_val.fillna(X_train.median())
    test_data = test_data.fillna(X_train.median())
    
    # 处理无穷值
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_val = X_val.replace([np.inf, -np.inf], 0)
    test_data = test_data.replace([np.inf, -np.inf], 0)
    
    print(f"特征数: {X_train.shape[1]}")
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    
    return X_train, X_val, y_train, y_val, test_data, sale_ids


def create_features(X_train, X_val, test_data, y_train):
    """创建特征"""
    # v特征交互
    v_cols = [col for col in X_train.columns if col.startswith('v_')]
    
    if len(v_cols) >= 2:
        # v_0 * v_3
        if 'v_0' in X_train.columns and 'v_3' in X_train.columns:
            X_train['v_0_v_3_mul'] = X_train['v_0'] * X_train['v_3']
            X_val['v_0_v_3_mul'] = X_val['v_0'] * X_val['v_3']
            test_data['v_0_v_3_mul'] = test_data['v_0'] * test_data['v_3']
        
        # v特征统计
        X_train['v_mean'] = X_train[v_cols].mean(axis=1)
        X_val['v_mean'] = X_val[v_cols].mean(axis=1)
        test_data['v_mean'] = test_data[v_cols].mean(axis=1)
        
        X_train['v_std'] = X_train[v_cols].std(axis=1)
        X_val['v_std'] = X_val[v_cols].std(axis=1)
        test_data['v_std'] = test_data[v_cols].std(axis=1)
    
    # 对数变换
    if 'kilometer' in X_train.columns:
        X_train['log_km'] = np.log1p(X_train['kilometer'])
        X_val['log_km'] = np.log1p(X_val['kilometer'])
        test_data['log_km'] = np.log1p(test_data['kilometer'])
    
    if 'power' in X_train.columns:
        X_train['log_power'] = np.log1p(X_train['power'])
        X_val['log_power'] = np.log1p(X_val['power'])
        test_data['log_power'] = np.log1p(test_data['power'])
    
    return X_train, X_val, test_data


# ==================== 训练函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * len(y_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    return total_loss / len(dataloader.dataset), np.array(all_preds), np.array(all_targets)


def train_mlp(X_train, y_train, X_val, y_val, hidden_dims=[512, 256, 128, 64], 
              epochs=500, batch_size=256, lr=0.001, patience=30, model_type='mlp'):
    """训练MLP模型"""
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 转换为Tensor
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val)
    
    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    input_dim = X_train.shape[1]
    
    # 创建模型
    if model_type == 'residual':
        model = ResidualMLP(input_dim, hidden_dim=256, num_blocks=4).to(device)
    else:
        model = MLP(input_dim, hidden_dims=hidden_dims).to(device)
    
    # 损失函数和优化器
    criterion = nn.L1Loss()  # MAE
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    # 训练
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"\n开始训练 {model_type.upper()} 模型...")
    print(f"输入维度: {input_dim}")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 50 == 0:
            mae = mean_absolute_error(val_targets, val_preds)
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.2f}, Val MAE={mae:.2f}")
        
        if patience_counter >= patience:
            print(f"早停于 Epoch {epoch+1}")
            break
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    # 最终评估
    _, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
    mae = mean_absolute_error(val_targets, val_preds)
    rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    r2 = r2_score(val_targets, val_preds)
    
    print(f"\n{model_type.upper()} 最终结果:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    return model, scaler, val_preds, mae


# ==================== K-Fold训练 ====================
def kfold_train(X_train, y_train, X_val, y_val, test_data, n_splits=5):
    """K-Fold训练多个MLP模型"""
    print("\n" + "="*50)
    print("K-Fold MLP训练")
    print("="*50)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X_train))
    val_preds = np.zeros(len(X_val))
    test_preds = np.zeros(len(test_data))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        
        X_tr = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
        y_tr = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        X_va = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
        y_va = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
        
        # 训练模型
        model, scaler, fold_preds, _ = train_mlp(
            X_tr, y_tr, X_va, y_va,
            hidden_dims=[512, 256, 128],
            epochs=300, batch_size=256, lr=0.001, patience=25,
            model_type='mlp'
        )
        
        oof_preds[val_idx] = fold_preds
        
        # 预测验证集和测试集
        X_val_scaled = scaler.transform(X_val)
        test_scaled = scaler.transform(test_data)
        
        model.eval()
        with torch.no_grad():
            val_preds += model(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy() / n_splits
            test_preds += model(torch.FloatTensor(test_scaled).to(device)).cpu().numpy() / n_splits
    
    # 评估
    mae = mean_absolute_error(y_train, oof_preds)
    print(f"\nK-Fold OOF MAE: {mae:.2f}")
    
    val_mae = mean_absolute_error(y_val, val_preds)
    print(f"K-Fold Val MAE: {val_mae:.2f}")
    
    return val_preds, test_preds


# ==================== 主函数 ====================
def main():
    # 加载数据
    X_train, X_val, y_train, y_val, test_data, sale_ids = load_and_prepare_data()
    
    # 创建特征
    X_train, X_val, test_data = create_features(X_train, X_val, test_data, y_train)
    print(f"创建特征后维度: {X_train.shape[1]}")
    
    # 方案1: 单个MLP
    print("\n" + "="*50)
    print("方案1: 单个MLP模型")
    print("="*50)
    mlp_model, mlp_scaler, mlp_preds, mlp_mae = train_mlp(
        X_train, y_train, X_val, y_val,
        hidden_dims=[512, 256, 128, 64],
        epochs=500, batch_size=256, lr=0.001, patience=40,
        model_type='mlp'
    )
    
    # 方案2: 残差MLP
    print("\n" + "="*50)
    print("方案2: 残差MLP模型")
    print("="*50)
    res_model, res_scaler, res_preds, res_mae = train_mlp(
        X_train, y_train, X_val, y_val,
        hidden_dims=[256, 256, 256],
        epochs=500, batch_size=256, lr=0.001, patience=40,
        model_type='residual'
    )
    
    # 方案3: K-Fold MLP
    print("\n" + "="*50)
    print("方案3: K-Fold MLP")
    print("="*50)
    kfold_val_preds, kfold_test_preds = kfold_train(X_train, y_train, X_val, y_val, test_data, n_splits=5)
    
    # 模型融合
    print("\n" + "="*50)
    print("神经网络模型融合")
    print("="*50)
    
    # 基于MAE反比加权
    maes = [mlp_mae, res_mae, mean_absolute_error(y_val, kfold_val_preds)]
    weights = [1/m for m in maes]
    weights = [w/sum(weights) for w in weights]
    print(f"融合权重: MLP={weights[0]:.3f}, ResMLP={weights[1]:.3f}, KFold={weights[2]:.3f}")
    
    # 融合预测
    ensemble_val = weights[0] * mlp_preds + weights[1] * res_preds + weights[2] * kfold_val_preds
    ensemble_mae = mean_absolute_error(y_val, ensemble_val)
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val))
    ensemble_r2 = r2_score(y_val, ensemble_val)
    
    print(f"\n融合结果:")
    print(f"MAE: {ensemble_mae:.2f}")
    print(f"RMSE: {ensemble_rmse:.2f}")
    print(f"R²: {ensemble_r2:.4f}")
    
    # 预测测试集
    print("\n预测测试集...")
    mlp_test = mlp_model(torch.FloatTensor(mlp_scaler.transform(test_data)).to(device)).detach().cpu().numpy()
    res_test = res_model(torch.FloatTensor(res_scaler.transform(test_data)).to(device)).detach().cpu().numpy()
    
    ensemble_test = weights[0] * mlp_test + weights[1] * res_test + weights[2] * kfold_test_preds
    
    # 保存结果
    submit = pd.DataFrame({'SaleID': sale_ids, 'price': ensemble_test})
    submit.to_csv('neural_network_submit.csv', index=False)
    print("结果已保存到 neural_network_submit.csv")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, ensemble_val, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('神经网络模型 - 预测价格 vs 实际价格')
    plt.tight_layout()
    plt.savefig('neural_network_prediction.png')
    plt.close()
    
    # 最终输出
    print("\n" + "="*50)
    print("最终结果")
    print("="*50)
    print(f"MAE: {ensemble_mae:.2f}")
    print(f"RMSE: {ensemble_rmse:.2f}")
    print(f"R²: {ensemble_r2:.4f}")
    
    if ensemble_mae < 400:
        print("\n🎉 恭喜！MAE已达到目标 (<400)")
    else:
        print(f"\n距离目标还差: {ensemble_mae - 400:.2f}")
    
    return ensemble_mae


if __name__ == "__main__":
    main()
