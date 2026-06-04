# -*- coding: utf-8 -*-
"""
二手车价格预测 - PyTorch深度残差网络 + 三模型融合
特性：
1. 构造特征用残差连接
2. 数值特征和分类特征分别处理
3. 嵌入层处理分类变量
4. 批归一化
5. 梯度裁剪
目标：MAE < 450
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 60)
print("PyTorch深度残差网络 + 三模型融合训练")
print("=" * 60)
print(f"使用设备: {device}")


# ==================== 数据加载与特征工程 ====================
def load_and_process_data():
    """加载数据并进行特征工程"""
    print("\n加载数据...")
    train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    
    train['is_train'] = 1
    test['is_train'] = 0
    combined = pd.concat([train, test], ignore_index=True)
    
    # ========== 异常值处理 ==========
    combined['power'] = combined['power'].clip(0, 600)
    
    # ========== 深度特征工程 ==========
    print("进行深度特征工程...")
    
    # v特征统计
    v_cols = ['v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_11', 'v_14']
    combined['v_mean'] = combined[v_cols].mean(axis=1)
    combined['v_std'] = combined[v_cols].std(axis=1)
    combined['v_max'] = combined[v_cols].max(axis=1)
    combined['v_min'] = combined[v_cols].min(axis=1)
    combined['v_range'] = combined['v_max'] - combined['v_min']
    
    # v_0和v_3深度特征（最重要）
    combined['v_0_sq'] = combined['v_0'] ** 2
    combined['v_3_sq'] = combined['v_3'] ** 2
    combined['v_0_cube'] = combined['v_0'] ** 3
    combined['v_3_cube'] = combined['v_3'] ** 3
    combined['v_0_sqrt'] = np.sqrt(np.abs(combined['v_0']))
    combined['v_3_sqrt'] = np.sqrt(np.abs(combined['v_3']))
    combined['v_0_v_3'] = combined['v_0'] * combined['v_3']
    combined['v_0_v_3_ratio'] = combined['v_0'] / (combined['v_3'] + 1e-5)
    combined['v_0_v_3_diff'] = combined['v_0'] - combined['v_3']
    combined['v_0_v_3_diff_sq'] = combined['v_0_v_3_diff'] ** 2
    combined['v_0_sq_v_3'] = (combined['v_0'] ** 2) * combined['v_3']
    combined['v_0_v_3_sq'] = combined['v_0'] * (combined['v_3'] ** 2)
    
    # 三阶交互
    combined['v_0_v_2_v_3'] = combined['v_0'] * combined['v_2'] * combined['v_3']
    
    # 与power和kilometer交互
    combined['v_0_power'] = combined['v_0'] * combined['power']
    combined['v_3_power'] = combined['v_3'] * combined['power']
    combined['v_0_km'] = combined['v_0'] * combined['kilometer']
    combined['v_3_km'] = combined['v_3'] * combined['kilometer']
    combined['power_v_0_v_3'] = combined['power'] * combined['v_0_v_3']
    
    # 对数变换
    combined['log_power'] = np.log1p(combined['power'])
    combined['log_km'] = np.log1p(combined['kilometer'])
    
    # ========== 分类变量编码 ==========
    cat_cols = ['brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    cat_dims = {}
    
    for col in cat_cols:
        combined[col] = combined[col].astype(str).fillna('missing')
        le = LabelEncoder()
        combined[col + '_enc'] = le.fit_transform(combined[col])
        cat_dims[col] = len(le.classes_)
    
    print(f"分类变量基数: {cat_dims}")
    
    # ========== 数值特征 ==========
    num_cols = [
        'power', 'kilometer',
        'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_11', 'v_14',
        'v_mean', 'v_std', 'v_max', 'v_min', 'v_range',
        'v_0_sq', 'v_3_sq', 'v_0_cube', 'v_3_cube',
        'v_0_sqrt', 'v_3_sqrt', 'v_0_v_3', 'v_0_v_3_ratio',
        'v_0_v_3_diff', 'v_0_v_3_diff_sq',
        'v_0_sq_v_3', 'v_0_v_3_sq', 'v_0_v_2_v_3',
        'v_0_power', 'v_3_power', 'v_0_km', 'v_3_km', 'power_v_0_v_3',
        'log_power', 'log_km'
    ]
    
    # 分离数据
    train_data = combined[combined['is_train'] == 1]
    test_data = combined[combined['is_train'] == 0]
    
    # 数值特征
    X_num = train_data[num_cols].values.astype(np.float32)
    X_num_test = test_data[num_cols].values.astype(np.float32)
    
    # 分类特征
    cat_enc_cols = [c + '_enc' for c in cat_cols]
    X_cat = train_data[cat_enc_cols].values.astype(np.int64)
    X_cat_test = test_data[cat_enc_cols].values.astype(np.int64)
    
    y = train_data['price'].values.astype(np.float32)
    sale_ids = test_data['SaleID'].values
    
    # 标准化数值特征
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    X_num_test = scaler.transform(X_num_test)
    X_num = np.nan_to_num(X_num)
    X_num_test = np.nan_to_num(X_num_test)
    
    print(f"数值特征: {X_num.shape}, 分类特征: {X_cat.shape}")
    
    return X_num, X_cat, y, X_num_test, X_cat_test, sale_ids, cat_dims, num_cols


# ==================== 模型定义 ====================
class ResidualBlock(nn.Module):
    """残差块 - 带批归一化和Dropout"""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.activation(x + self.block(x))


class NumericalFeatureNetwork(nn.Module):
    """数值特征处理网络 - 带残差连接"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # 初始投影层
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )
        
        # 残差块堆叠
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout=0.15),
            ResidualBlock(hidden_dim, dropout=0.15),
            ResidualBlock(hidden_dim, dropout=0.1),
        )
    
    def forward(self, x):
        x = self.input_bn(x)
        x = self.project(x)
        x = self.res_blocks(x)
        return x


class CategoricalFeatureNetwork(nn.Module):
    """分类特征处理网络 - 嵌入层 + 残差连接"""
    def __init__(self, cat_dims, embed_dim=16, hidden_dim=64):
        super().__init__()
        
        # 嵌入层
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, min(embed_dim, (dim + 1) // 2))
            for dim in cat_dims
        ])
        
        total_embed_dim = sum(min(embed_dim, (dim + 1) // 2) for dim in cat_dims)
        
        # 嵌入特征处理网络
        self.embed_net = nn.Sequential(
            nn.Linear(total_embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )
        
        # 残差块
        self.res_block = ResidualBlock(hidden_dim, dropout=0.1)
    
    def forward(self, x):
        # 嵌入各分类特征
        embeds = [self.embeddings[i](x[:, i]) for i in range(len(self.embeddings))]
        embed_concat = torch.cat(embeds, dim=1)
        
        # 残差网络处理
        out = self.embed_net(embed_concat)
        out = self.res_block(out)
        return out


class DeepResidualModel(nn.Module):
    """深度残差融合模型"""
    def __init__(self, num_dim, cat_dims, hidden_dim=128):
        super().__init__()
        
        # 数值特征网络
        self.num_net = NumericalFeatureNetwork(num_dim, hidden_dim)
        
        # 分类特征网络
        self.cat_net = CategoricalFeatureNetwork(cat_dims, embed_dim=16, hidden_dim=64)
        
        # 融合网络
        fusion_dim = hidden_dim + 64
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            ResidualBlock(256, dropout=0.15),
            ResidualBlock(256, dropout=0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            ResidualBlock(128, dropout=0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, num_x, cat_x):
        num_out = self.num_net(num_x)
        cat_out = self.cat_net(cat_x)
        fusion = torch.cat([num_out, cat_out], dim=1)
        return self.fusion(fusion).squeeze(-1)


# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, grad_clip=1.0):
    """训练模型，带梯度裁剪"""
    criterion = nn.L1Loss()  # MAE损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    best_mae = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for num_x, cat_x, target in train_loader:
            num_x, cat_x, target = num_x.to(device), cat_x.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred = model(num_x, cat_x)
            loss = criterion(pred, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # 验证
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for num_x, cat_x, target in val_loader:
                num_x, cat_x = num_x.to(device), cat_x.to(device)
                val_preds.extend(model(num_x, cat_x).cpu().numpy())
                val_targets.extend(target.numpy())
        
        mae = mean_absolute_error(val_targets, val_preds)
        
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: MAE={mae:.0f} (Best={best_mae:.0f})")
        
        # 早停
        if patience_counter >= 15:
            print(f"早停于 epoch {epoch+1}")
            break
    
    return best_mae, best_state


# ==================== K折交叉验证 ====================
def kfold_train(X_num, X_cat, y, X_num_test, X_cat_test, cat_dims, n_splits=5):
    """K折训练"""
    print("\n" + "=" * 60)
    print("开始K折交叉验证训练...")
    print("=" * 60)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cat_dims_list = [cat_dims[c] for c in ['brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']]
    
    oof_preds = np.zeros(len(X_num))
    test_preds = np.zeros(len(X_num_test))
    fold_maes = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_num)):
        print(f"\n{'='*20} Fold {fold+1}/{n_splits} {'='*20}")
        
        X_num_tr, X_num_val = X_num[tr_idx], X_num[val_idx]
        X_cat_tr, X_cat_val = X_cat[tr_idx], X_cat[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        # 数据加载器
        train_ds = TensorDataset(
            torch.FloatTensor(X_num_tr),
            torch.LongTensor(X_cat_tr),
            torch.FloatTensor(y_tr)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_num_val),
            torch.LongTensor(X_cat_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1024)
        
        # 模型
        model = DeepResidualModel(X_num.shape[1], cat_dims_list).to(device)
        
        # 训练
        best_mae, best_state = train_model(
            model, train_loader, val_loader,
            epochs=60, lr=0.002, grad_clip=1.0
        )
        
        # 加载最佳模型
        model.load_state_dict(best_state)
        model.eval()
        
        # OOF预测
        with torch.no_grad():
            oof_preds[val_idx] = model(
                torch.FloatTensor(X_num_val).to(device),
                torch.LongTensor(X_cat_val).to(device)
            ).cpu().numpy()
        
        # 测试集预测
        with torch.no_grad():
            test_preds += model(
                torch.FloatTensor(X_num_test).to(device),
                torch.LongTensor(X_cat_test).to(device)
            ).cpu().numpy() / n_splits
        
        fold_maes.append(best_mae)
        print(f"Fold {fold+1} Best MAE: {best_mae:.0f}")
    
    return oof_preds, test_preds, fold_maes


# ==================== 主函数 ====================
def main():
    # 加载数据
    X_num, X_cat, y, X_num_test, X_cat_test, sale_ids, cat_dims, num_cols = load_and_process_data()
    
    # K折训练
    oof_preds, test_preds, fold_maes = kfold_train(
        X_num, X_cat, y, X_num_test, X_cat_test, cat_dims, n_splits=5
    )
    
    # 整体评估
    overall_mae = mean_absolute_error(y, oof_preds)
    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    overall_r2 = r2_score(y, oof_preds)
    
    print("\n" + "=" * 60)
    print("深度残差网络结果")
    print("=" * 60)
    print(f"各折MAE: {[f'{m:.0f}' for m in fold_maes]}")
    print(f"平均MAE: {np.mean(fold_maes):.0f}")
    print(f"整体MAE: {overall_mae:.0f}")
    print(f"整体RMSE: {overall_rmse:.0f}")
    print(f"整体R²: {overall_r2:.4f}")
    
    if overall_mae < 450:
        print(f"\n🎯 深度模型目标达成！MAE={overall_mae:.0f} < 450")
    else:
        print(f"\n深度模型距离目标: {overall_mae - 450:.0f}")
    
    # 保存深度模型预测结果
    test_preds = np.clip(test_preds, y.min()*0.9, y.max()*1.1)
    deep_submit = pd.DataFrame({'SaleID': sale_ids, 'price': test_preds})
    deep_submit.to_csv('deep_residual_submit.csv', index=False)
    
    # 保存OOF预测（用于后续融合）
    np.save('deep_oof_preds.npy', oof_preds)
    np.save('deep_test_preds.npy', test_preds)
    
    print(f"\n预测结果已保存:")
    print(f"  - deep_residual_submit.csv")
    print(f"  - deep_oof_preds.npy")
    print(f"  - deep_test_preds.npy")
    
    return overall_mae, oof_preds, test_preds


if __name__ == "__main__":
    main()
