# -*- coding: utf-8 -*-
"""二手车价格预测 - 神经网络优化版
策略：Stacking + 深度网络 + 更优训练策略
目标：MAE <= 450
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("神经网络优化版 - Stacking + 深度网络")
print("="*60)

# ==================== 数据加载 ====================
print("\n加载数据...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"原始训练集: {train.shape}")
print(f"原始测试集: {test.shape}")

# ==================== 保守数据清洗 ====================
print("\n保守数据清洗...")

# 1. notRepairedDamage处理
for df in [train, test]:
    if 'notRepairedDamage' in df.columns and df['notRepairedDamage'].dtype == object:
        df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', '0.0')
        df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce').fillna(0)

# 2. 删除price<=0的异常
price_before = len(train)
train = train[train['price'] > 0].copy()
price_after = len(train)
print(f"   删除price<=0的样本: {price_before - price_after} 个")

# 3. Power截断
for df in [train, test]:
    df['power'] = df['power'].clip(lower=0, upper=600)

# 4. 填充缺失值
def simple_fillna(df):
    """简单填充缺失值"""
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
    return df

simple_fillna(train)
simple_fillna(test)

# ==================== 特征工程 ====================
print("\n特征工程...")

for df in [train, test]:
    # 基础时间特征
    df['reg_year'] = df['regDate'] // 10000
    df['creat_year'] = df['creatDate'] // 10000
    df['car_age'] = (df['creat_year'] - df['reg_year']).clip(lower=0)

    # 车龄特征
    df['car_age_squared'] = df['car_age'] ** 2
    df['car_age_log'] = np.log1p(df['car_age'])

    # v特征交互
    v_cols = [f'v_{i}' for i in range(15)]
    available_v_cols = [col for col in v_cols if col in df.columns]
    if len(available_v_cols) >= 3:
        df['v_mean'] = df[available_v_cols].mean(axis=1)
        df['v_std'] = df[available_v_cols].std(axis=1)
        df['v_0_v_3'] = df['v_0'] * df['v_3']
        df['v_1_v_2'] = df['v_1'] * df['v_2']
        df['v_4_v_5'] = df['v_4'] * df['v_5']

    # Power相关
    df['log_power'] = np.log1p(df['power'])
    df['power_squared'] = df['power'] ** 2
    df['power_km'] = df['power'] * df['kilometer']

    # Kilometer相关
    df['log_km'] = np.log1p(df['kilometer'])
    df['km_squared'] = df['kilometer'] ** 2

    # 组合特征
    df['power_per_year'] = df['power'] / (df['car_age'] + 1)
    df['km_per_year'] = df['kilometer'] / (df['car_age'] + 1)

# 分类特征编码
categorical_cols = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox']
for col in categorical_cols:
    if col in train.columns:
        le = LabelEncoder()
        train[col + '_encoded'] = le.fit_transform(train[col].astype(str))
        test[col + '_encoded'] = le.transform(test[col].astype(str))

print(f"   特征数: {train.shape[1]-1}")

# ==================== 数据准备 ====================
print("\n数据准备...")

drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType', 'notRepairedDamage', 'price']
drop_cols.extend(['brand', 'model', 'bodyType', 'fuelType', 'gearbox'])

X = train.drop(columns=[c for c in drop_cols if c in train.columns])
X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])
y = train['price']  # 直接预测原始价格

numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
X = X[numeric_cols]
X_test = X_test[[col for col in numeric_cols if col in X_test.columns]]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print(f"   准备完成，特征数: {X_scaled.shape[1]}")

# ==================== 基模型训练（5折） ====================
print("\n基模型训练 (5折交叉验证)...")
print("="*60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储基模型预测
train_cat_preds = np.zeros(len(train))
train_lgb_preds = np.zeros(len(train))
train_xgb_preds = np.zeros(len(train))

test_cat_preds = np.zeros(len(test))
test_lgb_preds = np.zeros(len(test))
test_xgb_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"\nFold {fold+1}/5")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=10,
        loss_function='MAE',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # 存储训练集和测试集预测
    train_cat_preds[train_idx] = cat_model.predict(X_train)
    train_cat_preds[val_idx] = cat_model.predict(X_val)
    test_cat_preds += cat_model.predict(X_test_scaled) / 5

    cat_val_pred = train_cat_preds[val_idx]
    cat_mae = mean_absolute_error(y_val, cat_val_pred)
    print(f"   CatBoost MAE: {cat_mae:.2f}")

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        num_leaves=63,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=2000,
        min_data_in_leaf=30,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # 存储训练集和测试集预测
    train_lgb_preds[train_idx] = lgb_model.predict(X_train)
    train_lgb_preds[val_idx] = lgb_model.predict(X_val)
    test_lgb_preds += lgb_model.predict(X_test_scaled) / 5

    lgb_val_pred = train_lgb_preds[val_idx]
    lgb_mae = mean_absolute_error(y_val, lgb_val_pred)
    print(f"   LightGBM MAE: {lgb_mae:.2f}")

    # XGBoost
    xgb_model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mae',
        early_stopping_rounds=100,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # 存储训练集和测试集预测
    train_xgb_preds[train_idx] = xgb_model.predict(X_train)
    train_xgb_preds[val_idx] = xgb_model.predict(X_val)
    test_xgb_preds += xgb_model.predict(X_test_scaled) / 5

    xgb_val_pred = train_xgb_preds[val_idx]
    xgb_mae = mean_absolute_error(y_val, xgb_val_pred)
    print(f"   XGBoost MAE: {xgb_mae:.2f}")

    # 简单融合
    ensemble_val_pred = (cat_val_pred + lgb_val_pred + xgb_val_pred) / 3
    ensemble_mae = mean_absolute_error(y_val, ensemble_val_pred)
    print(f"   Ensemble MAE: {ensemble_mae:.2f}")

# ==================== 神经网络作为元学习器 ====================
print("\n" + "="*60)
print("神经网络元学习器训练...")
print("="*60)

# 构造Stacking特征
stacking_features_train = np.column_stack((train_cat_preds, train_lgb_preds, train_xgb_preds))

stacking_features_test = np.column_stack((test_cat_preds, test_lgb_preds, test_xgb_preds))

print(f"Stacking特征形状: {stacking_features_train.shape}")

# 深度神经网络
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

def create_deep_stacking_model(input_dim):
    """创建深度Stacking神经网络"""
    inputs = Input(shape=(input_dim,))

    # 第一层 - 更深的网络
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # 第二层
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # 第三层
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # 第四层
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # 输出层
    outputs = Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001, decay=1e-6),
                  loss='mae',
                  metrics=['mae'])
    return model

print("   神经网络架构:")
print("   Input(3) -> Dense(512) -> BN -> Dropout(0.4)")
print("   -> Dense(256) -> BN -> Dropout(0.3)")
print("   -> Dense(128) -> BN -> Dropout(0.2)")
print("   -> Dense(64) -> BN -> Dropout(0.1)")
print("   -> Dense(1)")

# 5折训练神经网络
kf = KFold(n_splits=5, shuffle=True, random_state=42)

all_maes = []
nn_test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(stacking_features_train)):
    print(f"\n神经神经网络 Fold {fold+1}/5")

    X_train_nn, X_val_nn = stacking_features_train[train_idx], stacking_features_train[val_idx]
    y_train_nn, y_val_nn = y.iloc[train_idx], y.iloc[val_idx]

    # 创建模型
    nn_model = create_deep_stacking_model(X_train_nn.shape[1])

    # 回调函数
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=0)

    # 训练神经网络
    history = nn_model.fit(
        X_train_nn, y_train_nn,
        validation_data=(X_val_nn, y_val_nn),
        epochs=100,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # 预测
    nn_val_pred = nn_model.predict(X_val_nn).flatten()
    nn_mae = mean_absolute_error(y_val_nn, nn_val_pred)
    all_maes.append(nn_mae)
    print(f"   神经网络 MAE: {nn_mae:.2f}")

    # 测试集预测
    nn_test_preds += nn_model.predict(stacking_features_test).flatten() / 5

print(f"\n神经网络平均MAE: {np.mean(all_maes):.2f}")

# ==================== 最终融合 ====================
print("\n" + "="*60)
print("最终融合...")
print("="*60)

# 测试集简单平均
test_preds_simple_avg = (test_cat_preds + test_lgb_preds + test_xgb_preds) / 3
simple_avg_mae_test = np.mean([cat_mae, lgb_mae, xgb_mae])
print(f"简单平均融合MAE: {simple_avg_mae_test:.2f}")

# 神经网络加权融合
# 基于验证MAE计算权重
val_maes = [cat_mae, lgb_mae, xgb_mae]
inv_maes = [1/max(m, 0.01) for m in val_maes]
weights = [w/sum(inv_maes) for w in inv_maes]

print(f"基模型权重: Cat={weights[0]:.3f}, LGB={weights[1]:.3f}, XGB={weights[2]:.3f}")

# 加权基模型预测
test_preds_weighted_base = (weights[0] * test_cat_preds +
                        weights[1] * test_lgb_preds +
                        weights[2] * test_xgb_preds)

# 最终融合：基模型 + 神经网络
# 假设神经网络MAE约为基模型平均MAE的80%
nn_mae_avg = np.mean(all_maes)
test_preds_final = 0.5 * test_preds_weighted_base + 0.5 * nn_test_preds

# 确保预测值合理
final_pred = np.maximum(test_preds_final, 50)

# 保存结果
submit = pd.DataFrame({
    'SaleID': test['SaleID'].values,
    'price': final_pred
})
submit.to_csv('nn_stacking_optimized_submit.csv', index=False)
print(f"\n结果已保存到: nn_stacking_optimized_submit.csv")

print(f"\n" + "="*60)
print("最终结果")
print("="*60)
print(f"基模型简单平均MAE: {simple_avg_mae_test:.2f}")
print(f"神经网络MAE: {nn_mae_avg:.2f}")
print(f"距离目标450: {simple_avg_mae_test - 450:.2f}")
print("="*60)
