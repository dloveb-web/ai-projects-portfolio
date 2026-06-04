# -*- coding: utf-8 -*-
"""
二手车价格预测 - 增强特征工程版 (方案1)
目标：MAE <= 450
优化内容：
1. K-Fold 目标编码 (brand, model, regionCode)
2. 增强 v 特征交互（三阶交互、多项式、三角函数）
3. 对数/平方根变换
4. 添加 XGBoost 模型
5. 基于 MAE 的自适应权重融合
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device('cpu')


# ==================== 数据加载 ====================
def load_data():
    print("="*60)
    print("加载数据...")
    print("="*60)
    train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    print(f"训练集大小: {train.shape}")
    print(f"测试集大小: {test.shape}")
    return train, test


# ==================== K-Fold 目标编码 ====================
def kfold_target_encode(train, target_col, cat_col, n_splits=5, smoothing=15.0):
    """
    K-Fold 目标编码，避免数据泄露

    Args:
        train: 训练数据
        target_col: 目标列名 (price)
        cat_col: 分类列名
        n_splits: 折数
        smoothing: 平滑系数
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_encoded = np.zeros(len(train))

    target_mean = train[target_col].mean()

    for train_idx, val_idx in kf.split(train):
        fold_train = train.iloc[train_idx]
        fold_val = train.iloc[val_idx]

        # 计算每个类别的均值
        category_means = fold_train.groupby(cat_col)[target_col].mean()
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
            train_encoded = kfold_target_encode(
                train_result, 'price', col, n_splits=n_splits
            )

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


# ==================== 增强特征工程 ====================
def enhanced_feature_engineering(train, test):
    """增强的特征工程"""
    print("\n" + "="*60)
    print("增强特征工程...")
    print("="*60)

    # 合并处理
    test['price'] = -1
    data = pd.concat([train, test], axis=0, ignore_index=True)

    # ========== 1. 日期特征 ==========
    print("\n【1/8】处理日期特征...")
    data['reg_year'] = data['regDate'] // 10000
    data['reg_month'] = (data['regDate'] // 100) % 100
    data['creat_year'] = data['creatDate'] // 10000
    data['creat_month'] = (data['creatDate'] // 100) % 100
    data['car_age'] = data['creat_year'] - data['reg_year']
    data['car_age'] = data['car_age'].clip(lower=0)

    # 车龄多项式
    data['car_age_squared'] = data['car_age'] ** 2
    data['log_age'] = np.log1p(data['car_age'])

    # ========== 2. 处理 notRepairedDamage ==========
    print("【2/8】处理 notRepairedDamage...")
    data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', np.nan)
    data['notRepairedDamage'] = pd.to_numeric(data['notRepairedDamage'], errors='coerce')
    data['notRepairedDamage'] = data['notRepairedDamage'].fillna(0)

    # ========== 3. v 特征深度挖掘 ==========
    print("【3/8】v 特征深度挖掘...")
    v_cols = [f'v_{i}' for i in range(15)]

    # 基础统计特征
    data['v_mean'] = data[v_cols].mean(axis=1)
    data['v_std'] = data[v_cols].std(axis=1)
    data['v_max'] = data[v_cols].max(axis=1)
    data['v_min'] = data[v_cols].min(axis=1)
    data['v_range'] = data['v_max'] - data['v_min']
    data['v_skew'] = data[v_cols].skew(axis=1)
    data['v_kurt'] = data[v_cols].kurt(axis=1)

    # ========== 4. v 特征二阶交互 ==========
    print("【4/8】v 特征二阶交互...")
    # 高重要性组合
    data['v_0_v_3'] = data['v_0'] * data['v_3']
    data['v_0_v_2'] = data['v_0'] * data['v_2']
    data['v_0_v_12'] = data['v_0'] * data['v_12']
    data['v_3_v_12'] = data['v_3'] * data['v_12']
    data['v_0_v_5'] = data['v_0'] * data['v_5']
    data['v_3_v_5'] = data['v_3'] * data['v_5']

    # v 特征差值
    data['v_3_minus_v_10'] = data['v_3'] - data['v_10']
    data['v_3_minus_v_14'] = data['v_3'] - data['v_14']
    data['v_0_minus_v_3'] = data['v_0'] - data['v_3']

    # ========== 5. v 特征三阶交互 ==========
    print("【5/8】v 特征三阶交互...")
    data['v_0_v_2_v_3'] = data['v_0'] * data['v_2'] * data['v_3']
    data['v_0_v_3_v_12'] = data['v_0'] * data['v_3'] * data['v_12']
    data['v_0_v_2_v_12'] = data['v_0'] * data['v_2'] * data['v_12']

    # ========== 6. v 特征多项式变换 ==========
    print("【6/8】v 特征多项式变换...")
    # 平方
    data['v_0_sq'] = data['v_0'] ** 2
    data['v_3_sq'] = data['v_3'] ** 2

    # 平方根
    data['v_0_sqrt'] = np.sqrt(np.abs(data['v_0']))
    data['v_3_sqrt'] = np.sqrt(np.abs(data['v_3']))

    # 立方
    data['v_0_cb'] = data['v_0'] ** 3
    data['v_3_cb'] = data['v_3'] ** 3

    # ========== 7. 三角函数特征 ==========
    print("【7/8】三角函数特征...")
    data['sin_v_0'] = np.sin(data['v_0'])
    data['cos_v_3'] = np.cos(data['v_3'])
    data['tan_v_12'] = np.tan(data['v_12'] / 10)
    data['sin_diff_v_0_v_3'] = np.sin(data['v_0'] - data['v_3'])

    # ========== 8. 功率和里程特征 ==========
    print("【8/8】功率和里程特征...")

    # 组合特征
    data['power_km'] = data['power'] * data['kilometer']
    data['age_km'] = data['car_age'] * data['kilometer']
    data['power_age'] = data['power'] * data['car_age']

    # 对数变换
    data['log_power'] = np.log1p(data['power'])
    data['log_km'] = np.log1p(data['kilometer'])

    # 平方变换
    data['power_sq'] = data['power'] ** 2
    data['km_sq'] = data['kilometer'] ** 2

    # 比率特征
    data['km_per_age'] = data['kilometer'] / (data['car_age'] + 1)
    data['power_per_age'] = data['power'] / (data['car_age'] + 1)

    # v 特征与其他特征的交互
    data['power_v_0'] = data['power'] * data['v_0']
    data['power_v_3'] = data['power'] * data['v_3']
    data['km_v_0'] = data['kilometer'] * data['v_0']

    # ========== 分组统计特征 ==========
    print("\n" + "-"*60)
    print("添加分组统计特征...")
    print("-"*60)
    for col in ['brand', 'model', 'regionCode']:
        if col in data.columns:
            data[f'{col}_count'] = data.groupby(col)['SaleID'].transform('count')

    # 删除无用特征
    drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
    data = data.drop(columns=drop_cols, errors='ignore')

    # 分离
    train_data = data[data['price'] != -1].reset_index(drop=True)
    test_data = data[data['price'] == -1].reset_index(drop=True)
    test_data = test_data.drop(columns=['price'])

    print(f"\n✓ 特征工程完成！")
    print(f"  训练集特征数: {train_data.shape[1] - 1}")
    print(f"  测试集特征数: {test_data.shape[1]}")

    return train_data, test_data


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


def train_lightgbm(X_train, y_train, X_val, y_val):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
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


def train_xgboost(X_train, y_train, X_val, y_val):
    """新增：XGBoost 模型"""
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
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    return model, mae


# ==================== 自适应权重融合 ====================
def adaptive_ensemble(X_train, y_train, X_val, y_val, test_data, fold_idx):
    """
    基于验证集 MAE 的自适应权重融合

    返回:
        ensemble_pred: 融合后的预测
        weights: 各模型权重
        model_maes: 各模型MAE
    """
    # 训练各模型
    cat_model, cat_mae = train_catboost(X_train, y_train, X_val, y_val)
    lgb_model, lgb_mae = train_lightgbm(X_train, y_train, X_val, y_val)
    xgb_model, xgb_mae = train_xgboost(X_train, y_train, X_val, y_val)
    nn_model, scaler, nn_mae = train_nn_model(X_train, y_train, X_val, y_val, epochs=100)

    # 获取验证集预测
    cat_pred = cat_model.predict(X_val)
    lgb_pred = lgb_model.predict(X_val)
    xgb_pred = xgb_model.predict(X_val)
    X_val_scaled = scaler.transform(X_val)
    nn_model.eval()
    with torch.no_grad():
        nn_pred = nn_model(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy()

    # 基于 MAE 的倒数计算权重 (MAE越小，权重越大)
    maes = [cat_mae, lgb_mae, xgb_mae, nn_mae]
    inv_maes = [1 / max(m, 1) for m in maes]
    weights = [w / sum(inv_maes) for w in inv_maes]

    print(f"\n  【Fold {fold_idx+1}】模型性能:")
    print(f"    CatBoost MAE: {cat_mae:.2f} (权重: {weights[0]:.3f})")
    print(f"    LightGBM MAE: {lgb_mae:.2f} (权重: {weights[1]:.3f})")
    print(f"    XGBoost   MAE: {xgb_mae:.2f} (权重: {weights[2]:.3f})")
    print(f"    NeuralNet MAE: {nn_mae:.2f} (权重: {weights[3]:.3f})")

    # 加权融合
    ensemble_pred = (weights[0] * cat_pred +
                     weights[1] * lgb_pred +
                     weights[2] * xgb_pred +
                     weights[3] * nn_pred)

    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    print(f"    Ensemble  MAE: {ensemble_mae:.2f}")

    # 获取测试集预测
    test_cat = cat_model.predict(test_data)
    test_lgb = lgb_model.predict(test_data)
    test_xgb = xgb_model.predict(test_data)
    test_scaled = scaler.transform(test_data)
    with torch.no_grad():
        test_nn = nn_model(torch.FloatTensor(test_scaled).to(device)).cpu().numpy()

    test_ensemble = (weights[0] * test_cat +
                     weights[1] * test_lgb +
                     weights[2] * test_xgb +
                     weights[3] * test_nn)

    return ensemble_pred, test_ensemble, weights, maes


# ==================== 主函数 ====================
def main():
    print("\n" + "="*60)
    print("方案1：增强特征工程 + XGBoost + 自适应融合")
    print("="*60)

    # 加载数据
    train, test = load_data()
    sale_ids = test['SaleID'].values

    # 增强特征工程
    train_data, test_data = enhanced_feature_engineering(train, test)

    # K-Fold 目标编码
    train_data, test_data = apply_target_encoding(
        train_data, test_data,
        cat_cols=['brand', 'model', 'regionCode'],
        n_splits=5
    )

    # 准备数据
    X = train_data.drop(columns=['price'])
    y = train_data['price']

    # 填充缺失值
    X = X.fillna(X.median())
    test_data = test_data.fillna(X.median())

    # 确保列一致
    common_cols = list(set(X.columns) & set(test_data.columns))
    X = X[common_cols]
    test_data = test_data[common_cols]

    print(f"\n{'='*60}")
    print(f"最终特征数: {X.shape[1]}")
    print(f"{'='*60}")

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_weights = []
    all_maes = {'cat': [], 'lgb': [], 'xgb': [], 'nn': [], 'ensemble': []}

    test_ensemble_pred = np.zeros(len(test_data))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/5")
        print(f"{'='*60}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 自适应融合
        ensemble_pred, test_pred, weights, maes = adaptive_ensemble(
            X_train, y_train, X_val, y_val, test_data, fold
        )

        # 累积测试集预测
        test_ensemble_pred += test_pred / 5

        # 保存权重和MAE
        all_weights.append(weights)
        all_maes['cat'].append(maes[0])
        all_maes['lgb'].append(maes[1])
        all_maes['xgb'].append(maes[2])
        all_maes['nn'].append(maes[3])
        all_maes['ensemble'].append(mean_absolute_error(y_val, ensemble_pred))

    # 最终结果
    print(f"\n{'='*60}")
    print("最终结果 (5折平均)")
    print(f"{'='*60}")
    print(f"CatBoost MAE: {np.mean(all_maes['cat']):.2f} ± {np.std(all_maes['cat']):.2f}")
    print(f"LightGBM MAE: {np.mean(all_maes['lgb']):.2f} ± {np.std(all_maes['lgb']):.2f}")
    print(f"XGBoost   MAE: {np.mean(all_maes['xgb']):.2f} ± {np.std(all_maes['xgb']):.2f}")
    print(f"NeuralNet MAE: {np.mean(all_maes['nn']):.2f} ± {np.std(all_maes['nn']):.2f}")
    print(f"Ensemble  MAE: {np.mean(all_maes['ensemble']):.2f} ± {np.std(all_maes['ensemble']):.2f}")

    # 平均权重
    avg_weights = np.mean(all_weights, axis=0)
    print(f"\n平均融合权重:")
    print(f"  CatBoost:  {avg_weights[0]:.3f}")
    print(f"  LightGBM:  {avg_weights[1]:.3f}")
    print(f"  XGBoost:   {avg_weights[2]:.3f}")
    print(f"  NeuralNet: {avg_weights[3]:.3f}")

    # 价格限制
    final_pred = np.maximum(test_ensemble_pred, 50)

    # 保存结果
    submit = pd.DataFrame({
        'SaleID': sale_ids,
        'price': final_pred
    })
    output_file = 'enhanced_v1_submit.csv'
    submit.to_csv(output_file, index=False)
    print(f"\n✓ 结果已保存到: {output_file}")

    # 绘制性能对比图
    plt.figure(figsize=(12, 6))
    models = ['CatBoost', 'LightGBM', 'XGBoost', 'NeuralNet', 'Ensemble']
    mean_maes = [np.mean(all_maes[k]) for k in ['cat', 'lgb', 'xgb', 'nn', 'ensemble']]
    std_maes = [np.std(all_maes[k]) for k in ['cat', 'lgb', 'xgb', 'nn', 'ensemble']]

    bars = plt.bar(models, mean_maes, yerr=std_maes, capsize=5, alpha=0.7,
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    plt.ylabel('MAE', fontsize=12)
    plt.title('方案1：增强特征工程 + XGBoost + 自适应融合', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('enhanced_v1_comparison.png', dpi=150)
    plt.close()
    print(f"✓ 性能对比图已保存: enhanced_v1_comparison.png")

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
