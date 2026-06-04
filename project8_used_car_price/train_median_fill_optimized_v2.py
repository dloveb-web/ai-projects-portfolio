# -*- coding: utf-8 -*-
"""
二手车价格预测 - 中位数填充异常值 + 优化特征工程版 V2
修复数据类型问题
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device('cpu')


# ==================== 数据加载 ====================
def load_processed_data():
    """加载中位数填充后的数据"""
    print("="*60)
    print("加载中位数填充后的训练数据...")
    print("="*60)

    train = pd.read_csv('train_median_fill.csv')
    test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

    # 立即处理 notRepairedDamage 列中的字符串值
    for df in [train, test]:
        if 'notRepairedDamage' in df.columns and df['notRepairedDamage'].dtype == object:
            df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
            df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce').fillna(0)

    print(f"训练集大小: {train.shape}")
    print(f"测试集大小: {test.shape}")

    return train, test


# ==================== 增强特征工程 ====================
def enhanced_feature_engineering(train, test):
    """增强的特征工程"""
    print("\n" + "="*60)
    print("增强特征工程...")
    print("="*60)

    # 处理日期特征
    for df in [train, test]:
        df['reg_year'] = df['regDate'] // 10000
        df['reg_month'] = (df['regDate'] // 100) % 100
        df['creat_year'] = df['creatDate'] // 10000
        df['creat_month'] = (df['creatDate'] // 100) % 100

        # 计算车龄
        if 'car_age' not in df.columns:
            df['car_age'] = df['creat_year'] - df['reg_year']
            df['car_age'] = df['car_age'].clip(lower=0)
            df['car_age_squared'] = df['car_age'] ** 2
            df['log_age'] = np.log1p(df['car_age'])

    # v特征深度挖掘
    v_cols = [f'v_{i}' for i in range(15)]
    available_v_cols = [col for col in v_cols if col in train.columns]

    print(f"\n【1/8】处理日期特征...")
    print(f"【2/8】处理 notRepairedDamage...")

    if len(available_v_cols) >= 3:
        print(f"【3/8】v 特征深度挖掘...发现 {len(available_v_cols)} 个v特征")

        # 基础统计特征
        for df in [train, test]:
            df['v_mean'] = df[available_v_cols].mean(axis=1)
            df['v_std'] = df[available_v_cols].std(axis=1)
            df['v_max'] = df[available_v_cols].max(axis=1)
            df['v_min'] = df[available_v_cols].min(axis=1)
            df['v_range'] = df['v_max'] - df['v_min']
            df['v_skew'] = df[available_v_cols].skew(axis=1)
            df['v_kurt'] = df[available_v_cols].kurt(axis=1)

        # v特征二阶交互
        print(f"【4/8】v 特征二阶交互...")
        key_pairs = [('v_0', 'v_3'), ('v_0', 'v_2'), ('v_0', 'v_12'),
                   ('v_3', 'v_12'), ('v_0', 'v_5'), ('v_3', 'v_5')]

        for col1, col2 in key_pairs:
            if col1 in available_v_cols and col2 in available_v_cols:
                for df in [train, test]:
                    df[f'{col1}_{col2}'] = df[col1] * df[col2]

        # v特征差值
        diff_pairs = [('v_3', 'v_10'), ('v_3', 'v_14'), ('v_0', 'v_3')]
        for col1, col2 in diff_pairs:
            if col1 in available_v_cols and col2 in available_v_cols:
                for df in [train, test]:
                    df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]

        # v特征三阶交互
        print(f"【5/8】v 特征三阶交互...")
        triad_pairs = [('v_0', 'v_2', 'v_3'), ('v_0', 'v_3', 'v_12')]
        for col1, col2, col3 in triad_pairs:
            if all(c in available_v_cols for c in [col1, col2, col3]):
                for df in [train, test]:
                    df[f'{col1}_{col2}_{col3}'] = df[col1] * df[col2] * df[col3]

        # v特征多项式变换
        print(f"【6/8】v 特征多项式变换...")
        poly_cols = ['v_0', 'v_3']
        for col in poly_cols:
            if col in available_v_cols:
                for df in [train, test]:
                    df[f'{col}_sq'] = df[col] ** 2
                    df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
                    df[f'{col}_cb'] = df[col] ** 3

        # 三角函数特征
        print(f"【7/8】三角函数特征...")
        trig_cols = [('v_0', 'sin'), ('v_3', 'cos'), ('v_12', 'tan')]
        for col, func in trig_cols:
            if col in available_v_cols:
                for df in [train, test]:
                    if func == 'sin':
                        df[f'{func}_{col}'] = np.sin(df[col])
                    elif func == 'cos':
                        df[f'{func}_{col}'] = np.cos(df[col])
                    else:  # tan
                        df[f'{func}_{col}'] = np.tan(df[col] / 10)

        # sin_diff_v_0_v_3
        if 'v_0' in available_v_cols and 'v_3' in available_v_cols:
            for df in [train, test]:
                df['sin_diff_v_0_v_3'] = np.sin(df['v_0'] - df['v_3'])

    # 功率和里程特征
    print(f"【8/8】功率和里程特征...")

    for df in [train, test]:
        # 组合特征
        df['power_km'] = df['power'] * df['kilometer']
        df['age_km'] = df['car_age'] * df['kilometer']
        df['power_age'] = df['power'] * df['car_age']

        # 对数变换
        df['log_power'] = np.log1p(df['power'])
        df['log_km'] = np.log1p(df['kilometer'])

        # 平方变换
        df['power_sq'] = df['power'] ** 2
        df['km_sq'] = df['kilometer'] ** 2

        # 比率特征
        df['km_per_age'] = df['kilometer'] / (df['car_age'] + 1)
        df['power_per_age'] = df['power'] / (df['car_age'] + 1)

        # v特征与其他特征的交互
        if 'v_0' in available_v_cols:
            df['power_v_0'] = df['power'] * df['v_0']
            df['km_v_0'] = df['kilometer'] * df['v_0']
        if 'v_3' in available_v_cols:
            df['power_v_3'] = df['power'] * df['v_3']

    # 分组统计特征
    print("\n" + "-"*60)
    print("添加分组统计特征...")
    print("-"*60)
    for df in [train, test]:
        for col in ['brand', 'model', 'regionCode']:
            if col in df.columns:
                df[f'{col}_count'] = df.groupby(col).cumcount()

    # 删除无用特征
    drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
    train = train.drop(columns=[c for c in drop_cols if c in train.columns])
    test = test.drop(columns=[c for c in drop_cols if c in test.columns])

    print(f"\n✓ 特征工程完成！")
    print(f"  训练集特征数: {train.shape[1]}")
    print(f"  测试集特征数: {test.shape[1]}")

    return train, test


# ==================== K-Fold 目标编码 ====================
def kfold_target_encode(train, cat_col, n_splits=5, smoothing=15.0):
    """K-Fold 目标编码，避免数据泄露"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_encoded = np.zeros(len(train))

    target_mean = train['price'].mean()

    for train_idx, val_idx in kf.split(train):
        fold_train = train.iloc[train_idx]
        fold_val = train.iloc[val_idx]

        # 计算每个类别的均值
        category_means = fold_train.groupby(cat_col)['price'].mean()
        category_counts = fold_train.groupby(cat_col).size()

        # 平滑
        smoothed = (category_means * category_counts + target_mean * smoothing) / (category_counts + smoothing)

        # 映射到验证集
        train_encoded[val_idx] = fold_val[cat_col].map(smoothed).fillna(target_mean)

    return train_encoded


def apply_target_encoding(train, test, cat_cols, n_splits=5):
    """对多个分类列应用目标编码"""
    print("\n" + "-"*60)
    print(f"应用 K-Fold 目标编码，列: {cat_cols}")
    print("-"*60)

    train_result = train.copy()
    test_result = test.copy()

    for col in cat_cols:
        if col in train.columns and col in test.columns:
            print(f"编码 {col}...")

            # 训练集 K-Fold 编码
            train_encoded = kfold_target_encode(train_result, col, n_splits=n_splits)

            # 测试集使用所有训练数据编码
            target_mean = train_result['price'].mean()
            category_means = train_result.groupby(col)['price'].mean()
            category_counts = train_result.groupby(col).size()

            smoothed = (category_means * category_counts + target_mean * 15.0) / (category_counts + 15.0)

            test_encoded = test_result[col].map(smoothed).fillna(target_mean)

            # 添加新列
            train_result[f'{col}_te'] = train_encoded
            test_result[f'{col}_te'] = test_encoded

    return train_result, test_result


# ==================== 数据清理 ====================
def clean_data_types(train, test):
    """清理数据类型，确保模型可以处理"""
    print("\n" + "-"*60)
    print("清理数据类型...")
    print("-"*60)

    train_clean = train.copy()
    test_clean = test.copy()

    # 识别数值列和分类列
    numeric_cols_train = train_clean.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_test = test_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols_train = train_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    categorical_cols_test = test_clean.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"训练集: {len(numeric_cols_train)} 个数值列, {len(categorical_cols_train)} 个分类列")
    print(f"测试集: {len(numeric_cols_test)} 个数值列, {len(categorical_cols_test)} 个分类列")

    # 填充数值列的缺失值
    common_numeric = [c for c in numeric_cols_train if c in numeric_cols_test]
    for col in common_numeric:
        median_val = train_clean[col].median()
        train_clean[col].fillna(median_val, inplace=True)
        if col in test_clean.columns:
            test_clean[col].fillna(median_val, inplace=True)

    # CatBoost 可以处理分类列，但需要转换为 category 类型
    for col in categorical_cols_train:
        if col in train_clean.columns:
            train_clean[col] = train_clean[col].astype('category')
        if col in test_clean.columns:
            test_clean[col] = test_clean[col].astype('category')

    print(f"✓ 数据类型清理完成")

    return train_clean, test_clean


# ==================== PyTorch 模型 ====================
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


class EnhancedNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 256, 128, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(0.1))
            prev_dim = dim

        # 残差块
        layers.append(ResidualBlock(hidden_dims[-1]))

        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_nn_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=256):
    """训练神经网络"""
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train.values)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 模型
    model = EnhancedNN(X_train.shape[1]).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

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

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy()
        val_mae = mean_absolute_error(y_val, val_pred)
        scheduler.step(val_mae)

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} Val MAE: {val_mae:.2f}")

    # 加载最佳模型
    model.load_state_dict(best_state)
    return model, scaler, best_mae


# ==================== 树模型 ====================
def train_catboost(X_train, y_train, X_val, y_val):
    # 识别分类特征
    cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=10,
        loss_function='MAE',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100,
        cat_features=cat_features if cat_features else None  # 显式指定分类特征
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    return model, mae


def train_xgboost(X_train, y_train, X_val, y_val):
    """XGBoost 模型"""
    model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=100,
        eval_metric='mae',
        verbosity=0,
        enable_categorical=True  # 启用分类特征支持
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    return model, mae


# ==================== 自适应权重融合 ====================
def adaptive_ensemble(X_train, y_train, X_val, y_val, test_data, fold_idx):
    """
    基于验证集 MAE 的自适应权重融合 (3模型: CatBoost + XGBoost + NeuralNet)

    返回:
        ensemble_pred: 融合后的预测
        weights: 各模型权重
        model_maes: 各模型MAE
    """
    # 训练各模型
    cat_model, cat_mae = train_catboost(X_train, y_train, X_val, y_val)
    xgb_model, xgb_mae = train_xgboost(X_train, y_train, X_val, y_val)
    nn_model, scaler, nn_mae = train_nn_model(X_train, y_train, X_val, y_val, epochs=100)

    # 获取验证集预测
    cat_pred = cat_model.predict(X_val)
    xgb_pred = xgb_model.predict(X_val)
    X_val_scaled = scaler.transform(X_val)
    nn_model.eval()
    with torch.no_grad():
        nn_pred = nn_model(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy()

    # 基于 MAE 的倒数计算权重 (MAE越小，权重越大)
    maes = [cat_mae, xgb_mae, nn_mae]
    inv_maes = [1 / max(m, 1) for m in maes]
    weights = [w / sum(inv_maes) for w in inv_maes]

    print(f"\n  【Fold {fold_idx+1}】模型性能:")
    print(f"    CatBoost MAE: {cat_mae:.2f} (权重: {weights[0]:.3f})")
    print(f"    XGBoost   MAE: {xgb_mae:.2f} (权重: {weights[1]:.3f})")
    print(f"    NeuralNet MAE: {nn_mae:.2f} (权重: {weights[2]:.3f})")

    # 加权融合
    ensemble_pred = (weights[0] * cat_pred +
                     weights[1] * xgb_pred +
                     weights[2] * nn_pred)

    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    print(f"    Ensemble  MAE: {ensemble_mae:.2f}")

    # 获取测试集预测
    test_cat = cat_model.predict(test_data)
    test_xgb = xgb_model.predict(test_data)
    test_scaled = scaler.transform(test_data)
    with torch.no_grad():
        test_nn = nn_model(torch.FloatTensor(test_scaled).to(device)).cpu().numpy()

    test_ensemble = (weights[0] * test_cat +
                     weights[1] * test_xgb +
                     weights[2] * test_nn)

    return ensemble_pred, test_ensemble, weights, maes


# ==================== 主函数 ====================
def main():
    print("\n" + "="*60)
    print("中位数填充异常值 + 优化特征工程 + XGBoost + 自适应融合 V2")
    print("="*60)

    # 加载处理后的数据
    train, test = load_processed_data()
    sale_ids = test['SaleID'].values

    # 增强特征工程
    train, test = enhanced_feature_engineering(train, test)

    # 清理数据类型
    train, test = clean_data_types(train, test)

    # K-Fold 目标编码
    train, test = apply_target_encoding(
        train, test,
        cat_cols=['brand', 'model', 'regionCode'],
        n_splits=5
    )

    # 准备数据
    X = train.drop(columns=['price'])
    y = train['price']

    # 确保列一致
    common_cols = list(set(X.columns) & set(test.columns))
    X = X[common_cols]
    test = test[common_cols]

    print(f"\n{'='*60}")
    print(f"最终特征数: {X.shape[1]}")
    print(f"{'='*60}")

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_weights = []
    all_maes = {'cat': [], 'xgb': [], 'nn': [], 'ensemble': []}

    test_ensemble_pred = np.zeros(len(test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/5")
        print(f"{'='*60}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 自适应融合
        ensemble_pred, test_pred, weights, maes = adaptive_ensemble(
            X_train, y_train, X_val, y_val, test, fold
        )

        # 累积测试集预测
        test_ensemble_pred += test_pred / 5

        # 保存权重和MAE
        all_weights.append(weights)
        all_maes['cat'].append(maes[0])
        all_maes['xgb'].append(maes[1])
        all_maes['nn'].append(maes[2])
        all_maes['ensemble'].append(mean_absolute_error(y_val, ensemble_pred))

    # 最终结果
    print(f"\n{'='*60}")
    print("最终结果 (5折平均)")
    print(f"{'='*60}")
    print(f"CatBoost MAE: {np.mean(all_maes['cat']):.2f} ± {np.std(all_maes['cat']):.2f}")
    print(f"XGBoost   MAE: {np.mean(all_maes['xgb']):.2f} ± {np.std(all_maes['xgb']):.2f}")
    print(f"NeuralNet MAE: {np.mean(all_maes['nn']):.2f} ± {np.std(all_maes['nn']):.2f}")
    print(f"Ensemble  MAE: {np.mean(all_maes['ensemble']):.2f} ± {np.std(all_maes['ensemble']):.2f}")

    # 平均权重
    avg_weights = np.mean(all_weights, axis=0)
    print(f"\n平均融合权重:")
    print(f"  CatBoost:  {avg_weights[0]:.3f}")
    print(f"  XGBoost:   {avg_weights[1]:.3f}")
    print(f"  NeuralNet: {avg_weights[2]:.3f}")

    # 价格限制
    final_pred = np.maximum(test_ensemble_pred, 50)

    # 保存结果
    submit = pd.DataFrame({
        'SaleID': sale_ids,
        'price': final_pred
    })
    output_file = 'median_fill_optimized_v2.csv'
    submit.to_csv(output_file, index=False)
    print(f"\n✓ 结果已保存到: {output_file}")

    # 绘制性能对比图
    plt.figure(figsize=(12, 6))
    models = ['CatBoost', 'XGBoost', 'NeuralNet', 'Ensemble']
    mean_maes = [np.mean(all_maes[k]) for k in ['cat', 'xgb', 'nn', 'ensemble']]
    std_maes = [np.std(all_maes[k]) for k in ['cat', 'xgb', 'nn', 'ensemble']]

    bars = plt.bar(models, mean_maes, yerr=std_maes, capsize=5, alpha=0.7,
                  color=['#1f77b4', '#2ca02c', '#d62728', '#9467bd'])
    plt.ylabel('MAE', fontsize=12)
    plt.title('中位数填充异常值 + 优化特征工程 V2', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('median_fill_comparison_v2.png', dpi=150)
    plt.close()
    print(f"✓ 性能对比图已保存: median_fill_comparison_v2.png")

    # 结果判断
    final_mae = np.mean(all_maes['ensemble'])
    print(f"\n{'='*60}")
    if final_mae <= 450:
        print(f"🎉 成功！MAE 达到目标: {final_mae:.2f} ≤ 450")
    else:
        print(f"📈 距离目标还差: {final_mae - 450:.2f}")
        print(f"   建议继续优化...")
    print(f"{'='*60}")

    return final_mae


if __name__ == "__main__":
    main()
