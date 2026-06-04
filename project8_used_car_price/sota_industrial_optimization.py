# -*- coding: utf-8 -*-
"""
二手车价格预测 - SOTA工业级优化方案
目标: 测试集MAE < 450
策略: LightGBM + FT-Transformer + Stacking融合
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("="*70)
print("SOTA工业级优化 - 二手车价格预测")
print("目标: MAE < 450")
print("="*70)

# ==================== 1. 数据预处理 ====================
print("\n【步骤1】数据预处理...")

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
sale_ids = test['SaleID'].values

test['price'] = -1
data = pd.concat([train, test], axis=0, ignore_index=True)

print(f"训练集大小: {len(train)}, 测试集大小: {len(test)}")

# ==================== 2. 特征工程 ====================
print("\n【步骤2】特征工程...")

# 2.1 基础时间特征
data['reg_year'] = data['regDate'] // 10000
data['reg_month'] = (data['regDate'] % 10000) // 100
data['creat_year'] = data['creatDate'] // 10000
data['creat_month'] = (data['creatDate'] % 10000) // 100
data['car_age'] = (data['creat_year'] - data['reg_year']).clip(lower=0)

# 2.2 异常值处理
def winsorize_series(series, lower=1, upper=99):
    """缩尾处理：将异常值限制在指定分位数范围内"""
    lower_bound = np.percentile(series, lower)
    upper_bound = np.percentile(series, upper)
    return series.clip(lower=lower_bound, upper=upper_bound)

for col in ['power', 'kilometer', 'car_age']:
    data[col] = winsorize_series(data[col])

# 2.3 业务交叉特征（核心）
data['power_km'] = data['power'] * data['kilometer']  # 功率×里程
data['age_km'] = data['car_age'] * data['kilometer']  # 车龄×里程
data['power_age'] = data['power'] * data['car_age']  # 功率×车龄
data['usage_intensity'] = data['kilometer'] / (data['car_age'] + 1)  # 使用强度（年均里程）
data['power_per_km'] = data['power'] / (data['kilometer'] + 1)  # 每公里功率
data['km_per_year'] = data['kilometer'] / (data['car_age'] + 1)  # 年均里程

# 2.4 v特征统计和交互
v_cols = [f'v_{i}' for i in range(15)]
data['v_mean'] = data[v_cols].mean(axis=1)
data['v_std'] = data[v_cols].std(axis=1)
data['v_max'] = data[v_cols].max(axis=1)
data['v_min'] = data[v_cols].min(axis=1)
data['v_range'] = data['v_max'] - data['v_min']
data['v_skew'] = data[v_cols].skew(axis=1)
data['v_kurt'] = data[v_cols].kurt(axis=1)

# v特征交叉（高阶特征）
data['v_0_v_3'] = data['v_0'] * data['v_3']
data['v_0_v_2'] = data['v_0'] * data['v_2']
data['v_0_v_12'] = data['v_0'] * data['v_12']
data['v_3_v_12'] = data['v_3'] * data['v_12']
data['v_1_v_5'] = data['v_1'] * data['v_5']
data['v_2_v_4'] = data['v_2'] * data['v_4']

# 2.5 分类特征处理
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0.0')
data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce').fillna(0)

# 2.6 分组统计特征
for col in ['brand', 'model', 'regionCode', 'bodyType', 'fuelType']:
    data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')

# 删除无用列
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
data = data.drop(columns=drop_cols, errors='ignore')

print(f"特征工程完成，总特征数: {data.shape[1]}")

# ==================== 3. 缺失值处理 ====================
print("\n【步骤3】缺失值处理...")

# 数值特征用中位数填充，分类特征用众数填充
numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=['object']).columns

for col in numeric_cols:
    median_val = data[col].median()
    data[col] = data[col].fillna(median_val)

for col in categorical_cols:
    mode_val = data[col].mode()[0] if not data[col].mode().empty else 'unknown'
    data[col] = data[col].fillna(mode_val)

print(f"数值特征: {len(numeric_cols)}, 分类特征: {len(categorical_cols)}")

# ==================== 4. 分类特征编码 ====================
print("\n【步骤4】分类特征编码...")

# Label Encoding for tree models
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Target Encoding for high cardinality features (on train data only)
def target_encode(train_df, test_df, col, target='price', n_folds=5, noise_level=0.01):
    """Target编码：使用交叉验证避免过拟合"""
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[f'{col}_te'] = 0.0
    test_df[f'{col}_te'] = 0.0

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    for train_idx, val_idx in kf.split(train_df):
        # 计算训练部分的target均值
        target_mean = train_df.iloc[train_idx].groupby(col)[target].mean()
        # 应用到验证集
        train_df.loc[val_idx, f'{col}_te'] = train_df.loc[val_idx, col].map(target_mean)

    # 对测试集使用全量训练数据的统计
    target_mean = train_df.groupby(col)[target].mean()
    test_df[f'{col}_te'] = test_df[col].map(target_mean).fillna(train_df[target].mean())

    # 添加噪声避免过拟合
    noise = np.random.normal(0, noise_level, len(train_df))
    train_df[f'{col}_te'] = train_df[f'{col}_te'] + noise

    return train_df[f'{col}_te'], test_df[f'{col}_te']

# 分离训练和测试
train_data = data[data['price'] != -1].reset_index(drop=True)
test_data = data[data['price'] == -1].reset_index(drop=True)
test_data = test_data.drop(columns=['price'])

# 对高基数分类特征进行Target编码
high_cardinality_cols = ['brand', 'model', 'regionCode']
for col in high_cardinality_cols:
    if col in train_data.columns:
        train_te, test_te = target_encode(train_data, test_data, col)
        train_data[f'{col}_te'] = train_te
        test_data[f'{col}_te'] = test_te
        print(f"  Target编码完成: {col}")

# ==================== 5. 特征标准化 ====================
print("\n【步骤5】特征标准化...")

X = train_data.drop(columns=['price'])
y = train_data['price'].values

# 数值特征标准化
scaler = StandardScaler()
numeric_cols_for_std = X.select_dtypes(include=[np.number]).columns
X[numeric_cols_for_std] = scaler.fit_transform(X[numeric_cols_for_std])
test_data[numeric_cols_for_std] = scaler.transform(test_data[numeric_cols_for_std])

print(f"标准化完成，特征数: {X.shape[1]}")

# ==================== 6. Optuna自动调参 ====================
print("\n【步骤6】LightGBM Optuna调参...")

def objective_lgb(trial):
    """Optuna目标函数：优化LightGBM参数"""
    param = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'random_state': SEED,
        'verbose': -1,

        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 0.95),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 3, 8),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.01, 0.5),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.01, 0.5),
        'min_split_gain': trial.suggest_loguniform('min_split_gain', 0.001, 0.01),
    }

    # 使用3折CV快速评估
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    mae_scores = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        train_data_lgb = lgb.Dataset(X_tr, label=y_tr)
        val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

        model = lgb.train(
            param, train_data_lgb,
            num_boost_round=1000,
            valid_sets=[val_data_lgb],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        mae_scores.append(mae)

    return np.mean(mae_scores)

# 运行Optuna优化（减少试验次数以节省时间）
print("  Optuna调参中（30次试验）...")
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective_lgb, n_trials=30, show_progress_bar=True)

best_lgb_params = study.best_params
print(f"  最佳参数: {best_lgb_params}")
print(f"  最佳MAE: {study.best_value:.2f}")

# 更新完整参数
lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'random_state': SEED,
    'verbose': -1,
    **best_lgb_params
}

# ==================== 7. FT-Transformer模型 ====================
print("\n【步骤7】FT-Transformer深度学习模型...")

class TabularDataset(Dataset):
    """表格数据Dataset"""
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class FTTransformer(nn.Module):
    """简化的FT-Transformer模型（表格数据专用）"""
    def __init__(self, num_features, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super(FTTransformer, self).__init__()

        self.num_features = num_features
        self.d_model = d_model

        # 特征嵌入层
        self.feature_embedding = nn.Linear(num_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.BatchNorm1d(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 特征嵌入
        x = self.feature_embedding(x)  # [batch, d_model]
        x = x.unsqueeze(1)  # [batch, 1, d_model]

        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, 1+num_features, d_model]

        # Transformer编码
        x = self.transformer(x)

        # 取CLS token
        cls_output = x[:, 0, :]  # [batch, d_model]

        # 输出
        output = self.output_layer(cls_output).squeeze(-1)
        return output

def train_ft_transformer(X_train, y_train, X_val, y_val, num_epochs=50, batch_size=2048):
    """训练FT-Transformer模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")

    # 创建数据集
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 初始化模型
    model = FTTransformer(num_features=X_train.shape[1], d_model=64, n_heads=4, n_layers=2, dropout=0.2)
    model = model.to(device)

    # 优化器（带权重衰减）
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    # 损失函数（MAE）
    criterion = nn.L1Loss()

    best_val_mae = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.numpy())

        val_mae = mean_absolute_error(val_targets, val_predictions)
        val_loss = np.mean([abs(p - t) for p, t in zip(val_predictions, val_targets)])

        # 学习率调度
        scheduler.step(val_mae)

        print(f"  Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val MAE: {val_mae:.2f}")

        # 早停
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停于 epoch {epoch+1}")
                break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_mae

# ==================== 8. Stacking训练 ====================
print("\n【步骤8】Stacking融合训练...")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 存储基模型预测
lgb_train_preds = np.zeros(len(X))
ft_train_preds = np.zeros(len(X))
lgb_test_preds = np.zeros(len(test_data))
ft_test_preds = np.zeros(len(test_data))

fold_maes_lgb = []
fold_maes_ft = []
fold_maes_stack = []

print("  开始5折交叉验证...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n  Fold {fold+1}/5")

    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    # --- LightGBM ---
    print("    训练LightGBM...")
    train_data_lgb = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data_lgb = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data_lgb)

    lgb_model = lgb.train(
        lgb_params, train_data_lgb,
        num_boost_round=5000,
        valid_sets=[val_data_lgb],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
    )

    lgb_pred_val = lgb_model.predict(X_val_fold)
    lgb_pred_test = lgb_model.predict(test_data)

    lgb_mae = mean_absolute_error(y_val_fold, lgb_pred_val)
    fold_maes_lgb.append(lgb_mae)
    lgb_train_preds[val_idx] = lgb_pred_val
    lgb_test_preds += lgb_pred_test / 5

    print(f"    LightGBM MAE: {lgb_mae:.2f}")

    # --- FT-Transformer ---
    print("    训练FT-Transformer...")
    ft_model, ft_mae = train_ft_transformer(
        X_train_fold, y_train_fold,
        X_val_fold, y_val_fold,
        num_epochs=40, batch_size=4096
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ft_model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val_fold.values).to(device)
        X_test_tensor = torch.FloatTensor(test_data.values).to(device)

        ft_pred_val = ft_model(X_val_tensor).cpu().numpy()
        ft_pred_test = ft_model(X_test_tensor).cpu().numpy()

    fold_maes_ft.append(ft_mae)
    ft_train_preds[val_idx] = ft_pred_val
    ft_test_preds += ft_pred_test / 5

    print(f"    FT-Transformer MAE: {ft_mae:.2f}")

    # --- Stacking（第二层：线性回归）---
    print("    Stacking融合...")
    stack_X_train = np.column_stack([lgb_train_preds[train_idx], ft_train_preds[train_idx]])
    stack_X_val = np.column_stack([lgb_train_preds[val_idx], ft_train_preds[val_idx]])

    # 简单平均作为基线
    stack_pred_avg = (lgb_pred_val + ft_pred_val) / 2
    stack_mae_avg = mean_absolute_error(y_val_fold, stack_pred_avg)

    # 线性回归作为元学习器
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(stack_X_train, y_train_fold)
    stack_pred_lr = meta_model.predict(stack_X_val)
    stack_mae_lr = mean_absolute_error(y_val_fold, stack_pred_lr)
    fold_maes_stack.append(stack_mae_lr)

    print(f"    Stacking (平均) MAE: {stack_mae_avg:.2f}")
    print(f"    Stacking (LR) MAE: {stack_mae_lr:.2f}")
    print(f"    最终权重: LightGBM={meta_model.coef_[0]:.4f}, FT-Transformer={meta_model.coef_[1]:.4f}")

# ==================== 9. 最终结果 ====================
print(f"\n{'='*70}")
print("【最终结果】5折交叉验证平均")
print(f"{'='*70}")
print(f"LightGBM 平均 MAE: {np.mean(fold_maes_lgb):.2f} ± {np.std(fold_maes_lgb):.2f}")
print(f"FT-Transformer 平均 MAE: {np.mean(fold_maes_ft):.2f} ± {np.std(fold_maes_ft):.2f}")
print(f"Stacking (线性回归) 平均 MAE: {np.mean(fold_maes_stack):.2f} ± {np.std(fold_maes_stack):.2f}")

# 计算RMSE和R²
final_stack_preds = (lgb_train_preds + ft_train_preds) / 2
rmse = np.sqrt(mean_squared_error(y, final_stack_preds))
r2 = r2_score(y, final_stack_preds)

print(f"\n附加指标:")
print(f"  RMSE: {rmse:.2f}")
print(f"  R² Score: {r2:.4f}")

# ==================== 10. 测试集预测 ====================
print("\n【步骤9】生成测试集预测...")

# 训练最终元学习器
final_stack_X = np.column_stack([lgb_train_preds, ft_train_preds])
final_meta_model = Ridge(alpha=1.0)
final_meta_model.fit(final_stack_X, y)

# 测试集预测
test_stack_X = np.column_stack([lgb_test_preds, ft_test_preds])
final_pred = final_meta_model.predict(test_stack_X)

# 价格裁剪（确保合理范围）
final_pred = np.maximum(final_pred, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': sale_ids,
    'price': final_pred
})
submit.to_csv('sota_industrial_submit.csv', index=False)
print(f"\n结果已保存到: sota_industrial_submit.csv")
print(f"预测价格范围: [{final_pred.min():.2f}, {final_pred.max():.2f}]")
print(f"预测价格均值: {final_pred.mean():.2f}")

# ==================== 总结 ====================
print(f"\n{'='*70}")
print("【项目总结】")
print(f"{'='*70}")
final_mae = np.mean(fold_maes_stack)
print(f"最终MAE: {final_mae:.2f}")
print(f"目标MAE: 450")
print(f"差距: {final_mae - 450:.2f}")

if final_mae < 450:
    print(f"\n🎉 成功达到目标！MAE: {final_mae:.2f} < 450")
else:
    print(f"\n⚠️ 距离目标: {final_mae - 450:.2f}")

print(f"{'='*70}")
print("优化完成！")
