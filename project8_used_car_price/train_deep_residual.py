# -*- coding: utf-8 -*-
"""
高级深度学习模型 - 残差网络 + 嵌入层 + 批归一化
特征分离处理：数值特征 + 分类特征嵌入
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==================== 数据集类 ====================
class CarPriceDataset(Dataset):
    def __init__(self, num_features, cat_features, constructed_features, targets=None):
        self.num_features = torch.FloatTensor(num_features)
        self.cat_features = torch.LongTensor(cat_features)
        self.constructed_features = torch.FloatTensor(constructed_features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        
    def __len__(self):
        return len(self.num_features)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.num_features[idx], self.cat_features[idx], self.constructed_features[idx], self.targets[idx]
        return self.num_features[idx], self.cat_features[idx], self.constructed_features[idx]

# ==================== 残差块 ====================
class ResidualBlock(nn.Module):
    """残差块：包含批归一化、Dropout和跳跃连接"""
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out = out + residual  # 残差连接
        out = self.activation(out)
        return out

# ==================== 特征处理网络 ====================
class NumericalFeatureNet(nn.Module):
    """数值特征处理网络"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.net(x)

class CategoricalFeatureNet(nn.Module):
    """分类特征嵌入网络"""
    def __init__(self, cat_cardinalities, embed_dim=16):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, min(embed_dim, (card + 1) // 2))
            for card in cat_cardinalities
        ])
        total_embed_dim = sum(min(embed_dim, (card + 1) // 2) for card in cat_cardinalities)
        self.fc = nn.Sequential(
            nn.Linear(total_embed_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        embeddings = [embed(x[:, i]) for i, embed in enumerate(self.embeddings)]
        embed_concat = torch.cat(embeddings, dim=1)
        return self.fc(embed_concat)

# ==================== 构造特征残差网络 ====================
class ConstructedFeatureNet(nn.Module):
    """构造特征专用残差网络"""
    def __init__(self, input_dim, hidden_dims=[64, 64, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hidden_dim, dropout=0.2))
            prev_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# ==================== 主模型 ====================
class DeepResidualModel(nn.Module):
    """
    深度残差模型：
    - 数值特征单独处理
    - 分类特征嵌入处理
    - 构造特征残差网络
    - 融合后多层残差块
    """
    def __init__(self, num_features_dim, cat_cardinalities, constructed_dim, 
                 hidden_dims=[256, 256, 128, 64]):
        super().__init__()
        
        # 数值特征处理
        self.num_net = NumericalFeatureNet(num_features_dim, hidden_dim=64)
        
        # 分类特征嵌入
        self.cat_net = CategoricalFeatureNet(cat_cardinalities, embed_dim=16)
        
        # 构造特征残差网络
        self.constructed_net = ConstructedFeatureNet(constructed_dim, hidden_dims=[64, 64, 64])
        
        # 计算融合维度
        fusion_dim = 64 + 32 + 64  # num_net输出 + cat_net输出 + constructed_net输出
        
        # 融合后的残差网络
        fusion_layers = []
        prev_dim = fusion_dim
        for hidden_dim in hidden_dims:
            fusion_layers.append(ResidualBlock(prev_dim, hidden_dim, dropout=0.3))
            prev_dim = hidden_dim
        
        self.fusion_net = nn.Sequential(*fusion_layers)
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, num_x, cat_x, constructed_x):
        # 分别处理
        num_out = self.num_net(num_x)
        cat_out = self.cat_net(cat_x)
        const_out = self.constructed_net(constructed_x)
        
        # 融合
        fusion = torch.cat([num_out, cat_out, const_out], dim=1)
        
        # 残差网络
        out = self.fusion_net(fusion)
        
        # 输出
        return self.output(out).squeeze(-1)

# ==================== 数据处理 ====================
def load_and_process_data():
    print("加载数据...")
    train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    
    # 合并处理
    train['is_train'] = 1
    test['is_train'] = 0
    test['price'] = -1
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    
    # 处理异常值
    combined['power'] = combined['power'].clip(0, 600)
    combined['kilometer'] = combined['kilometer'].clip(0, 50)
    
    # ==================== 构造特征 ====================
    print("构造特征...")
    
    # v特征统计
    v_cols = ['v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_11', 'v_14']
    combined['v_mean'] = combined[v_cols].mean(axis=1)
    combined['v_std'] = combined[v_cols].std(axis=1)
    combined['v_max'] = combined[v_cols].max(axis=1)
    combined['v_min'] = combined[v_cols].min(axis=1)
    
    # 构造特征列表
    constructed_features = []
    
    # 高阶多项式
    combined['v_0_sq'] = combined['v_0'] ** 2
    combined['v_3_sq'] = combined['v_3'] ** 2
    combined['v_0_cube'] = combined['v_0'] ** 3
    combined['v_3_cube'] = combined['v_3'] ** 3
    constructed_features.extend(['v_0_sq', 'v_3_sq', 'v_0_cube', 'v_3_cube'])
    
    # 交互特征
    combined['v_0_v_3'] = combined['v_0'] * combined['v_3']
    combined['v_0_v_2'] = combined['v_0'] * combined['v_2']
    combined['v_0_v_2_v_3'] = combined['v_0'] * combined['v_2'] * combined['v_3']
    combined['v_0_sq_v_3'] = (combined['v_0'] ** 2) * combined['v_3']
    combined['v_0_v_3_sq'] = combined['v_0'] * (combined['v_3'] ** 2)
    constructed_features.extend(['v_0_v_3', 'v_0_v_2', 'v_0_v_2_v_3', 'v_0_sq_v_3', 'v_0_v_3_sq'])
    
    # 与power/kilometer交叉
    combined['v_0_power'] = combined['v_0'] * combined['power']
    combined['v_3_power'] = combined['v_3'] * combined['power']
    combined['v_0_kilometer'] = combined['v_0'] * combined['kilometer']
    constructed_features.extend(['v_0_power', 'v_3_power', 'v_0_kilometer'])
    
    # ==================== 分类变量 ====================
    print("处理分类变量...")
    cat_cols = ['brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode']
    
    # 处理缺失值
    for col in cat_cols:
        combined[col] = combined[col].fillna(-1).astype(str)
    
    # Label Encoding
    label_encoders = {}
    cat_cardinalities = []
    for col in cat_cols:
        le = LabelEncoder()
        combined[col + '_encoded'] = le.fit_transform(combined[col])
        label_encoders[col] = le
        cat_cardinalities.append(len(le.classes_))
    
    print(f"分类变量基数: {cat_cardinalities}")
    
    # ==================== 数值特征 ====================
    num_cols = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 
                'v_11', 'v_14', 'v_mean', 'v_std', 'v_max', 'v_min']
    
    # 分离数据
    train_data = combined[combined['is_train'] == 1].copy()
    test_data = combined[combined['is_train'] == 0].copy()
    
    # 提取特征
    num_features_train = train_data[num_cols].values.astype(np.float32)
    cat_features_train = train_data[[col + '_encoded' for col in cat_cols]].values.astype(np.int64)
    constructed_train = train_data[constructed_features].values.astype(np.float32)
    y_train = train_data['price'].values.astype(np.float32)
    
    num_features_test = test_data[num_cols].values.astype(np.float32)
    cat_features_test = test_data[[col + '_encoded' for col in cat_cols]].values.astype(np.int64)
    constructed_test = test_data[constructed_features].values.astype(np.float32)
    sale_ids = test_data['SaleID'].values
    
    # 标准化数值特征
    scaler_num = StandardScaler()
    num_features_train = scaler_num.fit_transform(num_features_train)
    num_features_test = scaler_num.transform(num_features_test)
    
    # 标准化构造特征
    scaler_const = StandardScaler()
    constructed_train = scaler_const.fit_transform(constructed_train)
    constructed_test = scaler_const.transform(constructed_test)
    
    # 处理NaN
    num_features_train = np.nan_to_num(num_features_train)
    num_features_test = np.nan_to_num(num_features_test)
    constructed_train = np.nan_to_num(constructed_train)
    constructed_test = np.nan_to_num(constructed_test)
    
    print(f"数值特征: {num_features_train.shape}")
    print(f"分类特征: {cat_features_train.shape}")
    print(f"构造特征: {constructed_train.shape}")
    
    return (num_features_train, cat_features_train, constructed_train, y_train,
            num_features_test, cat_features_test, constructed_test, sale_ids,
            cat_cardinalities, len(num_cols), len(constructed_features))

# ==================== 训练函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    
    for num_x, cat_x, const_x, y in dataloader:
        num_x = num_x.to(device)
        cat_x = cat_x.to(device)
        const_x = const_x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        pred = model(num_x, cat_x, const_x)
        loss = criterion(pred, y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        total_loss += loss.item() * len(y)
    
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    preds = []
    targets = []
    
    with torch.no_grad():
        for num_x, cat_x, const_x, y in dataloader:
            num_x = num_x.to(device)
            cat_x = cat_x.to(device)
            const_x = const_x.to(device)
            y = y.to(device)
            
            pred = model(num_x, cat_x, const_x)
            loss = criterion(pred, y)
            
            total_loss += loss.item() * len(y)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())
    
    mae = mean_absolute_error(targets, preds)
    return total_loss / len(dataloader.dataset), mae

def predict(model, dataloader, device):
    model.eval()
    preds = []
    
    with torch.no_grad():
        for num_x, cat_x, const_x in dataloader:
            num_x = num_x.to(device)
            cat_x = cat_x.to(device)
            const_x = const_x.to(device)
            
            pred = model(num_x, cat_x, const_x)
            preds.extend(pred.cpu().numpy())
    
    return np.array(preds)

# ==================== 主训练流程 ====================
def main():
    print("=" * 60)
    print("深度残差网络训练")
    print("=" * 60)
    
    # 加载数据
    (num_features_train, cat_features_train, constructed_train, y_train,
     num_features_test, cat_features_test, constructed_test, sale_ids,
     cat_cardinalities, num_dim, const_dim) = load_and_process_data()
    
    # K折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(num_features_train))
    test_preds = np.zeros(len(num_features_test))
    
    batch_size = 1024
    epochs = 100
    patience = 15
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(num_features_train)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/5")
        print("="*60)
        
        # 准备数据
        train_dataset = CarPriceDataset(
            num_features_train[train_idx],
            cat_features_train[train_idx],
            constructed_train[train_idx],
            y_train[train_idx]
        )
        val_dataset = CarPriceDataset(
            num_features_train[val_idx],
            cat_features_train[val_idx],
            constructed_train[val_idx],
            y_train[val_idx]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 初始化模型
        model = DeepResidualModel(
            num_features_dim=num_dim,
            cat_cardinalities=cat_cardinalities,
            constructed_dim=const_dim,
            hidden_dims=[256, 256, 128, 64]
        ).to(device)
        
        # 损失函数和优化器
        criterion = nn.L1Loss()  # MAE Loss
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # 训练
        best_val_mae = float('inf')
        best_model_state = None
        no_improve = 0
        
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm=1.0)
            val_loss, val_mae = validate(model, val_loader, criterion, device)
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.2f}, Val MAE={val_mae:.2f}")
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        # 预测
        val_dataset_pred = CarPriceDataset(
            num_features_train[val_idx],
            cat_features_train[val_idx],
            constructed_train[val_idx]
        )
        val_loader_pred = DataLoader(val_dataset_pred, batch_size=batch_size, shuffle=False)
        oof_preds[val_idx] = predict(model, val_loader_pred, device)
        
        test_dataset = CarPriceDataset(
            num_features_test, cat_features_test, constructed_test
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_preds += predict(model, test_loader, device) / 5
        
        print(f"Fold {fold+1} Best MAE: {best_val_mae:.2f}")
    
    # 计算总体MAE
    overall_mae = mean_absolute_error(y_train, oof_preds)
    print("\n" + "="*60)
    print(f"总体MAE: {overall_mae:.2f}")
    
    if overall_mae < 400:
        print("目标达成！MAE < 400")
    else:
        print(f"距离目标: {overall_mae - 400:.2f}")
    
    # 保存预测
    test_preds = np.clip(test_preds, y_train.min() * 0.9, y_train.max() * 1.1)
    submission = pd.DataFrame({'SaleID': sale_ids, 'price': test_preds})
    submission.to_csv('deep_residual_submit.csv', index=False)
    print(f"\n结果已保存: deep_residual_submit.csv")
    
    return overall_mae

if __name__ == '__main__':
    main()
