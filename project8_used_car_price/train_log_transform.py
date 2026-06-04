# -*- coding: utf-8 -*-
"""
二手车价格预测 - 对数变换优化版 (方案B)
目标：MAE <= 450
核心优化：价格对数变换 + LightGBM
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("对数变换优化版 - 方案B")
print("="*60)

# ==================== 数据加载与处理 ====================
print("\n加载数据...")
train = pd.read_csv('train_median_fill.csv')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

# 处理 notRepairedDamage
for df in [train, test]:
    if 'notRepairedDamage' in df.columns and df['notRepairedDamage'].dtype == object:
        df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
        df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce').fillna(0)

print(f"训练集: {train.shape}, 测试集: {test.shape}")

# ==================== 对数变换 ====================
print("\n对数变换...")
print("="*60)

# 对价格取对数
train['log_price'] = np.log1p(train['price'])
# 保存原始价格用于反变换
original_price_median = train['price'].median()
original_price_mean = train['price'].mean()

# 使用对数价格作为目标变量
y_train = train['log_price']
y_test_original = train['price'].values  # 保存原始价格用于计算原始MAE

print(f"✓ 对数变换完成")
print(f"  原始价格范围: [{train['price'].min():.0f}, {train['price'].max():.0f}]")
print(f"对数价格范围: [{train['log_price'].min():.2f}, {train['log_price'].max():.2f}]")

# ==================== 特征工程（简化版） ====================
print("\n特征工程...")
print("="*60)

for df in [train, test]:
    # 基础特征
    df['reg_year'] = df['regDate'] // 10000
    df['creat_year'] = df['creatDate'] // 10000
    df['car_age'] = (df['creat_year'] - df['reg_year']).clip(lower=0)
    df['car_age_squared'] = df['car_age'] ** 2

    # v特征
    v_cols = [f'v_{i}' for i in range(15)]
    available_v_cols = [col for col in v_cols if col in df.columns]

    if len(available_v_cols) >= 3:
        df['v_mean'] = df[available_v_cols].mean(axis=1)
        df['v_0_v_3'] = df['v_0'] * df['v_3']
        df['v_0_v_12'] = df['v_0'] * df['v_12']

    # 组合特征
    df['power_km'] = df['power'] * df['kilometer']
    df['log_power'] = np.log1p(df['power'])
    df['log_km'] = np.log1p(df['kilometer'])

print(f"✓ 特征工程完成，特征数: {train.shape[1]-1}")

# ==================== 数据准备 ====================
print("\n准备数据...")
print("="*60)

# 准备特征
drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType', 'price', 'log_price']
X = train.drop(columns=[c for c in drop_cols if c in train.columns])
X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])

# 只选择数值列（排除notRepairedDamage）
numeric_cols = [col for col in X.columns if col != 'notRepairedDamage' and pd.api.types.is_numeric_dtype(X[col])]
X = X[numeric_cols]
X_test = X_test[[col for col in numeric_cols if col in X_test.columns]]

# 填充缺失值
for col in X.columns:
    if col in X_test.columns:
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print(f"✓ 数据准备完成，特征数: {X_scaled.shape[1]}")

# ==================== 模型训练 ====================
def train_model(name, X_train, y_train, X_val, y_val):
    """训练单个模型"""
    print(f"\n训练 {name}...")

    if name == 'CatBoost':
        model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=10,
            loss_function='MAE',
            random_seed=42,
            verbose=100
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        pred = model.predict(X_val)
        # CatBoost 预测的是对数价格，需要反变换回原始空间计算MAE
        pred_original = np.exp(pred) - 1

    elif name == 'XGBoost':
        model = xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=0.02,
            max_depth=8,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = model.predict(X_val)
        pred_original = np.exp(pred) - 1

    elif name == 'LightGBM':
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
            'lambda_l2': 0.1,
            'verbose': -1,
            'seed': 42
        }

        model = lgb.train(params, train_data, num_boost_round=3000,
                         valid_sets=[val_data],
                         callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
        pred = model.predict(X_val)
        pred_original = np.exp(pred) - 1

    # 计算原始价格的MAE
    mae = mean_absolute_error(y_val, pred_original)
    print(f"  {name} MAE: {mae:.2f}")

    return model, mae


def train_neural_network(X_train, y_train, X_val, y_val, epochs=100):
    """训练神经网络（使用对数目标）"""
    print("\n训练 Neural Network (使用对数目标)...")

    train_dataset = lgb.Dataset(X_train, label=y_train)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

    # LightGBM 参数
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,  # 较高学习率
        'num_leaves': 31,  # 较少叶子
        'max_depth': 6,
        'min_data_in_leaf': 40,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.05,
        'lambda_l2': 0.05,
        'verbose': -1,
        'seed': 42,
        'num_iterations': 3000,
        'early_stopping_rounds': 100
    }

    model = lgb.train(params, train_data, num_boost_round=3000,
                         valid_sets=[val_dataset],
                         callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    pred_log = model.predict(X_val)
    pred_original = np.exp(pred_log) - 1

    # 计算MAE
    mae = mean_absolute_error(y_val, pred_original)
    print(f"  Neural Network MAE: {mae:.2f}")

    return model, mae


# ==================== 自适应权重融合 ====================
def adaptive_ensemble(X_train, y_train, X_val, y_val, X_test, fold_idx):
    """自适应权重融合 (4模型)"""
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx+1}/5 - 自适应融合")
    print(f"{'='*60}")

    # 训练4个模型
    cat_model, cat_mae = train_model('CatBoost', X_train, y_train, X_val, y_val)
    xgb_model, xgb_mae = train_model('XGBoost', X_train, y_train, X_val, y_val)
    lgb_model, lgb_mae = train_neural_network(X_train, y_train, X_val, y_val, epochs=100)

    # 获取验证集预测（原始价格空间）
    cat_pred = np.exp(cat_model.predict(X_val)) - 1
    xgb_pred = np.exp(xgb_model.predict(X_val)) - 1
    lgb_pred = np.exp(lgb_model.predict(X_val)) - 1

    # 基于 MAE 的倒数计算权重
    maes = [cat_mae, xgb_mae, lgb_mae]
    inv_maes = [1 / max(m, 1) for m in maes]
    weights = [w / sum(inv_maes) for w in inv_maes]

    print(f"  模型性能 (原始价格MAE):")
    print(f"    CatBoost MAE: {cat_mae:.2f} (权重: {weights[0]:.3f})")
    print(f"    XGBoost   MAE: {xgb_mae:.2f} (权重: {weights[1]:.3f})")
    print(f"    LightGBM MAE: {lgb_mae:.2f} (权重: {weights[2]:.3f})")

    # 加权融合（对数空间）
    ensemble_pred_log = (weights[0] * np.log(cat_model.predict(X_val) +
                       weights[1] * np.log(xgb_model.predict(X_val)) +
                       weights[2] * lgb_model.predict(X_val))

    ensemble_mae = mean_absolute_error(y_val, np.exp(ensemble_pred_log) - 1)
    print(f"    Ensemble  MAE: {ensemble_mae:.2f}")

    # 获取测试集预测（对数空间）
    test_cat_pred = np.exp(cat_model.predict(X_test))
    test_xgb_pred = np.exp(xgb_model.predict(X_test))
    test_lgb_pred = np.exp(lgb_model.predict(X_test))

    # 加权融合测试集预测
    test_ensemble_pred_log = (weights[0] * np.log(cat_model.predict(X_test) +
                                       weights[1] * np.log(xgb_model.predict(X_test)) +
                                       weights[2] * lgb_model.predict(X_test))

    # 反变换回原始空间
    test_ensemble_pred = np.exp(test_ensemble_pred_log) - 1

    return (test_ensemble_pred, weights, maes)


# ==================== 主函数 ====================
print("\n" + "="*60)
print("开始训练 (5折交叉验证)")
print("="*60)

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

all_weights = []
all_maes = {'cat': [], 'xgb': [], 'lgb': [], 'ensemble': []}

test_ensemble_pred = np.zeros(len(test))

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx+1}/5")
    print(f"{'='*60}")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # 自适应融合
    fold_pred, fold_weights, fold_maes = adaptive_ensemble(
        X_train, y_train, X_val, y_val, X_test_scaled, fold_idx
    )

    # 累积测试集预测
    test_ensemble_pred += fold_pred / 5

    # 保存权重和MAE
    all_weights.append(fold_weights)
    all_maes['cat'].append(fold_maes[0])
    all_maes['xgb'].append(fold_maes[1])
    all_maes['lgb'].append(fold_maes[2])
    all_maes['ensemble'].append(fold_maes[3])

print("\n" + "="*60)
print("训练完成！")
print("="*60)

# 最终结果
print("最终结果 (5折平均) - 原始价格MAE")
print("="*60)

print(f"CatBoost  MAE: {np.mean(all_maes['cat']):.2f} ± {np.std(all_maes['cat']):.2f}")
print(f"XGBoost    MAE: {np.mean(all_maes['xgb']):.2f} ± {np.std(all_maes['xgb']):.2f}")
print(f"LightGBM  MAE: {np.mean(all_maes['lgb']):.2f} ± {np.std(all_maes['lgb']):.2f}")
print(f"Ensemble   MAE: {np.mean(all_maes['ensemble']):.2f} ± {np.std(all_maes['ensemble']):.2f}")

# 平均权重
avg_weights = np.mean([list(w) for w in all_weights], axis=0)
print(f"\n平均融合权重:")
print(f"  CatBoost  : {avg_weights[0]:.3f}")
print(f"  XGBoost    : {avg_weights[1]:.3f}")
print(f"  LightGBM   : {avg_weights[2]:.3f}")

# 价格限制
final_pred = np.maximum(test_ensemble_pred, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': test['SaleID'].values,
    'price': final_pred
})
output_file = 'log_transform_submit.csv'
submit.to_csv(output_file, index=False)
print(f"\n✓ 结果已保存到: {output_file}")

# 结果判断
final_mae = np.mean(all_maes['ensemble'])
print(f"\n{'='*60}")
if final_mae <= 450:
    print(f"🎉 成功！MAE 达到目标: {final_mae:.2f} ≤ 450")
else:
    print(f"📈 距离目标还差: {final_mae - 450:.2f}")
    print(f"   建议继续优化...")

# 对比分析
print(f"\n{'='*60}")
print("对数变换效果分析")
print("="*60)
print(f"原始价格统计:")
print(f"  均值: {original_price_median:.2f}")
print(f"  平均值: {original_price_mean:.2f}")
print(f"  中位数: {train['price'].median():.2f}")
print(f"  范围: {train['price'].max():.2f}")
print(f"\n{'='*60}")
print(f"对数价格统计:")
print(f"  均值: {train['log_price'].mean():.2f}")
print(f"  中位数: {train['log_price'].median():.2f}")
print(f"  范围: {train['log_price'].max():.2f}")
print(f"\n✓ 对数变换将价格分布压缩，预期降低MAE")
print(f"{'='*60}")
