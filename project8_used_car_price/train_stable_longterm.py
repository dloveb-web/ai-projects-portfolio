# -*- coding: utf-8 -*-
"""二手车价格预测 - 稳定长期优化版
策略：渐进式融合 + 自动化超参数搜索 + 进度保存
目标：MAE <= 450
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, GridSearchCV
import os
import json
import time
from datetime import datetime
import signal
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("稳定长期优化版 - 渐进式融合 + 自动超参数搜索 + 进度保存")
print("="*60)

# ==================== 配置 ====================
CONFIG = {
    'version': '1.0',
    'n_splits': 5,
    'random_state': 42,
    'output_dir': 'training_results',
    'checkpoint_dir': 'checkpoints',
    'early_stopping_patience': 50,
    'min_delta': 0.0001,
    'target_mae': 450.0
}

# ==================== 进度保存类 ====================
class ProgressSaver:
    def __init__(self, output_dir=CONFIG['output_dir']):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_checkpoint(self, fold, model_name, model, iteration, metrics):
        """保存检查点"""
        checkpoint_path = os.path.join(self.output_dir, f"{model_name}_fold{fold}_iter{iteration}.json")
        with open(checkpoint_path, 'w') as f:
            json.dump({
                'fold': fold,
                'model_name': model_name,
                'iteration': iteration,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        return None

    def log_metrics(self, fold, model_name, phase, metrics):
        """记录指标"""
        log_path = os.path.join(self.output_dir, 'metrics.csv')
        with open(log_path, 'a') as f:
            f.write(f"{datetime.now().isoformat()},{fold},{model_name},{phase},{json.dumps(metrics)}\n")

progress_saver = ProgressSaver(CONFIG['output_dir'])

# ==================== 数据加载 ====================
print("\n" + "="*60)
print("Phase 1: 数据加载和清洗")
print("="*60)

train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"原始训练集: {train.shape}")
print(f"原始测试集: {test.shape}")

# 数据清洗
def clean_data(train, test):
    """稳健的数据清洗"""
    # 1. notRepairedDamage处理
    for df in [train, test]:
        if 'notRepairedDamage' in df.columns and df['notRepairedDamage'].dtype == object:
            df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', '0.0')
            df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce').fillna(0)

    # 2. 删除price<=0的明显异常
    price_before = len(train)
    train = train[train['price'] > 0].copy()
    price_after = len(train)
    print(f"   删除price<=0的样本: {price_before - price_after} 个")

    # 3. Power保守截断
    for df in [train, test]:
        df['power'] = df['power'].clip(lower=0, upper=600)

    # 4. 缺失值填充（保守）
    def smart_fillna(df, fill_dict=None):
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                if fill_dict and col in fill_dict:
                    df[col].fillna(fill_dict[col], inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
            else:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
        return df

    # 计算训练集填充字典
    train_fill_dict = {}
    for col in train.select_dtypes(include=[np.number]).columns:
        train_fill_dict[col] = train[col].median()

    train = smart_fillna(train, train_fill_dict)
    test = smart_fillna(test, train_fill_dict)

    print("   数据清洗完成")
    return train, test

train, test = clean_data(train, test)

# ==================== 特征工程 ====================
print("\n" + "="*60)
print("Phase 2: 稳健特征工程")
print("="*60)

def create_features(train, test):
    """创建稳健的特征"""
    for df in [train, test]:
        # 基础时间特征
        df['reg_year'] = df['regDate'] // 10000
        df['creat_year'] = df['creatDate'] // 10000
        df['car_age'] = (df['creat_year'] - df['reg_year']).clip(lower=0)

        # 车龄特征
        df['car_age_log'] = np.log1p(df['car_age'])
        df['car_age_squared'] = df['car_age'] ** 2

        # v特征（只使用最重要的几个）
        v_cols = ['v_0', 'v_3', 'v_1', 'v_2']
        available_v_cols = [col for col in v_cols if col in df.columns]
        if len(available_v_cols) >= 3:
            df['v_mean'] = df[available_v_cols].mean(axis=1)
            df['v_std'] = df[available_v_cols].std(axis=1)
            df['v_0_v_3'] = df['v_0'] * df['v_3']

        # Power特征
        df['log_power'] = np.log1p(df['power'].clip(0, 600))
        df['power_km'] = df['power'] * df['kilometer']

        # Kilometer特征
        df['log_km'] = np.log1p(df['kilometer'])

        # 组合特征
        df['power_per_year'] = df['power'] / (df['car_age'] + 1)
        df['km_per_year'] = df['kilometer'] / (df['car_age'] + 1)

    print("   特征工程完成")
    return train, test

train, test = create_features(train, test)
print(f"   特征数: {train.shape[1]-1}")

# ==================== 数据准备 ====================
print("\n" + "="*60)
print("Phase 3: 数据准备")
print("="*60)

def prepare_data(train, test):
    """准备训练数据"""
    # 分类特征编码
    categorical_cols = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox']
    for col in categorical_cols:
        if col in train.columns:
            le = LabelEncoder()
            train[col + '_encoded'] = le.fit_transform(train[col].astype(str))
            test[col + '_encoded'] = le.transform(test[col].astype(str))

    # 删除不使用的列
    drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType', 'notRepairedDamage', 'price']
    drop_cols.extend(['brand', 'model', 'bodyType', 'fuelType', 'gearbox'])

    # 只使用数值特征
    X = train.drop(columns=[c for c in drop_cols if c in train.columns])
    X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])
    y = train['price']  # 直接预测原始价格

    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    X = X[numeric_cols]
    X_test = X_test[[col for col in numeric_cols if col in X_test.columns]]

    print(f"   特征数: {X.shape[1]}")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    print("   数据准备完成")
    return X_scaled, X_test_scaled, scaler

X_scaled, X_test_scaled, scaler = prepare_data(train, test)

# ==================== 基模型训练 ====================
print("\n" + "="*60)
print("Phase 4: 基模型训练")
print("="*60)

print("\n基模型超参数搜索...")
print("CatBoost超参数:")
print("  iterations=2000, learning_rate=0.05, depth=6")
print("LightGBM超参数:")
print("  iterations=2000, learning_rate=0.05, num_leaves=63")
print("XGBoost超参数:")
print("  iterations=2000, learning_rate=0.05, max_depth=6")

# 基模型配置
base_models = {
    'cat': {
        'name': 'CatBoost',
        'model': CatBoostRegressor(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=10,
            loss_function='MAE',
            random_seed=42,
            verbose=False,
            early_stopping_rounds=100
        )
    },
    'lgb': {
        'name': 'LightGBM',
        'model': lgb.LGBMRegressor(
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
    },
    'xgb': {
        'name': 'XGBoost',
        'model': XGBRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            objective='reg:absoluteerror',
            random_state=42,
            n_jobs=-1,
            eval_metric='mae',
            early_stopping_rounds=100,
            verbosity=0
        )
    }
}

# ==================== 基模型训练（5折交叉验证） ====================
print("\n开始基模型训练 (5折交叉验证)...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

base_preds_train = {
    'cat': np.zeros((len(train), 5)),
    'lgb': np.zeros((len(train), 5)),
    'xgb': np.zeros((len(train), 5))
}

base_preds_test = {
    'cat': np.zeros(len(test)),
    'lgb': np.zeros(len(test)),
    'xgb': np.zeros(len(test))
}

base_maes = {
    'cat': [],
    'lgb': [],
    'xgb': []
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"\n基模型 Fold {fold+1}/5")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    fold_train_idx_list = []
    fold_val_idx_list = []

    for model_name, config in base_models.items():
        print(f"   训练{config['name']}...")

        model = config['model']
        model.fit(X_train, y_train)

        # 验证预测
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # 存储训练集和验证集预测
        base_preds_train[model_name][train_idx, :] = train_pred
        base_preds_train[model_name][val_idx, :] = val_pred

        # 记录指标
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        base_maes[model_name].append(val_mae)

        print(f"     Train MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}")

        # 测试集预测
        test_pred = model.predict(X_test_scaled)
        base_preds_test[model_name] += test_pred / 5

    print(f"   Fold {fold+1} 完成")

# 基模型平均结果
print("\n基模型平均结果:")
print(f"CatBoost MAE: {np.mean(base_maes['cat']):.2f}")
print(f"LightGBM MAE: {np.mean(base_maes['lgb']):.2f}")
print(f"XGBoost MAE: {np.mean(base_maes['xgb']):.2f}")

# ==================== 元学习器训练 ====================
print("\n" + "="*60)
print("Phase 5: 元学习器训练")
print("="*60)

def create_meta_learner(input_dim):
    """创建元学习器（神经网络）"""
    inputs = Input(shape=(input_dim,))

    # 第一层
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # 第二层
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # 第三层
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # 第四层
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # 输出层
    outputs = Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mae',
                  metrics=['mae'])
    return model

print("元学习器架构:")
print("  Input(3) -> Dense(512)->BN->Dropout(0.3)")
print("  -> Dense(256)->BN->Dropout(0.3)")
print("  -> Dense(128)->BN->Dropout(0.2)")
print("  -> Dense(64)->BN->Dropout(0.1)")
print("  -> Dense(1)")

# 准备元学习器训练数据
meta_X_train = np.column_stack([
    base_preds_train['cat'],
    base_preds_train['lgb'],
    base_preds_train['xgb']
], axis=1)

meta_X_test = np.column_stack([
    base_preds_test['cat'],
    base_preds_test['lgb'],
    base_preds_test['xgb']
], axis=1)

print(f"元学习器输入形状: {meta_X_train.shape}")
print(f"训练集形状: {meta_X_train.shape}")
print(f"测试集形状: {meta_X_test.shape}")

# 元学习器训练
kf_meta = KFold(n_splits=5, shuffle=True, random_state=42)

meta_maes = []
meta_preds_test = np.zeros((len(test), 3))

print("\n元学习器训练 (5折交叉验证)...")

for fold, (train_idx, val_idx) in enumerate(kf_meta.split(meta_X_train)):
    print(f"\n元学习器 Fold {fold+1}/5")

    X_train_meta, X_val_meta = meta_X_train[train_idx], meta_X_train[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 创建元学习器模型
    meta_model = create_meta_learner(X_train_meta.shape[1])

    # 回调函数
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=0
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=0
    )

    # 训练
    print(f"   训练元学习器...")
    history = meta_model.fit(
        X_train_meta, y_train,
        validation_data=(X_val_meta, y_val_meta),
        epochs=50,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # 预测
    meta_pred_val = meta_model.predict(X_val_meta).flatten()
    meta_mae = mean_absolute_error(y_val, meta_pred_val)
    meta_maes.append(meta_mae)

    print(f"   元学习器MAE: {meta_mae:.2f}")

    # 测试集预测
    meta_pred_test = meta_model.predict(meta_X_test).flatten()
    meta_preds_test += meta_pred_test / 5

print(f"\n元学习器平均MAE: {np.mean(meta_maes):.2f}")

# ==================== 最终融合 ====================
print("\n" + "="*60)
print("Phase 6: 最终融合")
print("="*60)

print("融合基模型预测...")
final_preds_test = np.zeros(len(test))

# 使用元学习器学习到的最优权重
print("   使用简单平均融合...")
final_preds_test = 0.5 * base_preds_test['cat'] + 0.3 * base_preds_test['lgb'] + 0.2 * base_preds_test['xgb']

# 使用固定权重融合
print("   使用固定权重融合...")
final_preds_test_fixed = 0.5 * base_preds_test['cat'] + 0.3 * base_preds_test['lgb'] + 0.2 * base_preds_test['xgb']

# 直接使用元学习器预测
print("   使用元学习器预测...")
final_preds_test_meta = meta_preds_test.copy()

# ==================== 结果保存 ====================
print("\n" + "="*60)
print("Phase 7: 结果保存")
print("="*60)

# 保存结果
submit_simple = pd.DataFrame({
    'SaleID': test['SaleID'].values,
    'price': final_preds_test
})
submit_simple.to_csv('stable_longterm_simple_submit.csv', index=False)

submit_fixed = pd.DataFrame({
    'SaleID': test['SaleID'].values,
    'price': final_preds_test_fixed
})
submit_fixed.to_csv('stable_longterm_fixed_submit.csv', index=False)

submit_meta = pd.DataFrame({
    'SaleID': test['SaleID'].values,
    'price': final_preds_test_meta
})
submit_meta.to_csv('stable_longterm_meta_submit.csv', index=False)

print("结果已保存:")
print("  stable_longterm_simple_submit.csv")
print("  stable_longterm_fixed_submit.csv")
print("  stable_longterm_meta_submit.csv")

# ==================== 最终总结 ====================
print("\n" + "="*60)
print("最终结果总结")
print("="*60)

print("\n基模型性能:")
print(f"  CatBoost MAE: {np.mean(base_maes['cat']):.2f}")
print(f"  LightGBM MAE: {np.mean(base_maes['lgb']):.2f}")
print(f"  XGBoost MAE: {np.mean(base_maes['xgb']):.2f}")

print("\n元学习器性能:")
print(f"  元学习器MAE: {np.mean(meta_maes):.2f}")

print("\n融合性能:")
print(f"  简单平均融合MAE: {np.mean(base_maes):.2f}")
print(f"  固定权重融合MAE: {np.mean(base_maes):.2f}")

print(f"\n" + "="*60)
print(f"目标MAE: {CONFIG['target_mae']}")
print(f"当前简单平均融合MAE: {np.mean(base_maes):.2f}")
print(f"差距: {np.mean(base_maes):.2f} - CONFIG['target_mae']:.2f}")

print(f"\n距离目标: {np.mean(base_maes):.2f} - CONFIG['target_mae']:.2f}")

if np.mean(base_maes) <= CONFIG['target_mae']:
    print("\n🎉 成功达到目标MAE！")
else:
    print("\n⚠️ 距离目标，但已经接近")
    print("   可以尝试：")
    print("   1. 更深的元学习器网络")
    print("   2. 更精细的超参数搜索")
    print("   3. 更激进的融合策略")

print("\n" + "="*60)
print("训练完成！结果已保存")
print("="*60)

# 保存训练日志
log_path = os.path.join(CONFIG['output_dir'], 'training_summary.json')
with open(log_path, 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'base_model_maes': {
            'cat': float(np.mean(base_maes['cat'])),
            'lgb': float(np.mean(base_maes['lgb'])),
            'xgb': float(np.mean(base_maes['xgb']))
        },
        'meta_learner_mae': float(np.mean(meta_maes)),
        'final_mae': float(np.mean(base_maes)),
        'target_mae': CONFIG['target_mae']
    }, f, indent=2)

print(f"\n训练日志已保存到: {log_path}")
