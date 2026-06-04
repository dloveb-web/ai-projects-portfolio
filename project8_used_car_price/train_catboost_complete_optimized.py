# -*- coding: utf-8 -*-
"""
二阶段 Optuna 微调脚本（在已有 best_params 周围做更细粒度搜索）
说明：
- 第一阶段（可选）：如果未提供 baseline，会做一次短的粗粒度搜索以获得 baseline。
- 第二阶段：基于 baseline 的 best_params，在其周围构造窄搜索空间做精细搜索（fine tuning）。
- 最后使用最优参数训练最终模型，可选择在 train+val 上重新训练以产出生产模型。
- 所有注释与输出均为中文，结果保存在 output_dir（默认：models_optuna_two_stage）。

用法示例：
  # 使用已有 best params 文件进行微调（推荐）
  python optuna_two_stage_finetune.py --baseline models_optuna/optuna_best_params.json --n-trials-fine 50 --use-log --fit-final

  # 如果没有 baseline，先做短的粗搜索再微调
  python optuna_two_stage_finetune.py --no-baseline --n-trials-coarse 20 --n-trials-fine 60 --use-log --fit-final

注意：
- 需要 processed_data/X_train.joblib, X_val.joblib, y_train.joblib, y_val.joblib 可用
- 依赖：optuna, catboost, scikit-learn, joblib, pandas, numpy
"""

import os
import time
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import optuna
from functools import partial

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Known categorical columns (integer-encoded in processed_data)
KNOWN_CAT_COLS = [
    "name", "model", "brand", "bodyType", "fuelType",
    "gearbox", "notRepairedDamage", "regionCode", "seller", "offerType"
]
DEFAULT_DROP_COLS = ["SaleID"]

# -------------------- 默认路径与目录 --------------------
PROCESSED_DIR = "processed_data"
DEFAULT_BASELINE_PATH = os.path.join("models_optuna", "optuna_best_params.json")
OUTPUT_DIR = "models_optuna_two_stage"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------- 工具：载入数据与基础预处理 --------------------
def load_data():
    """加载 processed_data 下的数据，返回 X_train, X_val, y_train, y_val（均为 pandas）"""
    X_train = joblib.load(os.path.join(PROCESSED_DIR, "X_train.joblib"))
    X_val = joblib.load(os.path.join(PROCESSED_DIR, "X_val.joblib"))
    y_train = joblib.load(os.path.join(PROCESSED_DIR, "y_train.joblib"))
    y_val = joblib.load(os.path.join(PROCESSED_DIR, "y_val.joblib"))
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, index=X_train.index)
    else:
        y_train = y_train.reindex(X_train.index)
    if not isinstance(y_val, pd.Series):
        y_val = pd.Series(y_val, index=X_val.index)
    else:
        y_val = y_val.reindex(X_val.index)
    return X_train.copy(), X_val.copy(), y_train.copy(), y_val.copy()


def get_cat_cols(X):
    """Return categorical column names that exist in X."""
    return [c for c in KNOWN_CAT_COLS if c in X.columns]


def drop_low_variance_cols(X_train, X_val, min_unique=2):
    """Drop columns with too few unique values in training set."""
    low_cols = [c for c in X_train.columns if X_train[c].nunique(dropna=False) < min_unique]
    if low_cols:
        X_train = X_train.drop(columns=low_cols, errors="ignore")
        X_val = X_val.drop(columns=low_cols, errors="ignore")
    return X_train, X_val, low_cols


def engineer_features(X_train, X_val, cat_cols):
    """
    Light feature engineering for numeric columns only.
    Adds a few robust interactions and log features for skewed columns.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Simple, robust interactions
    if "kilometer" in X_train.columns and "car_age" in X_train.columns:
        X_train["km_per_year"] = X_train["kilometer"] / (X_train["car_age"] + 0.1)
        X_val["km_per_year"] = X_val["kilometer"] / (X_val["car_age"] + 0.1)
        X_train["km_age"] = X_train["kilometer"] * X_train["car_age"]
        X_val["km_age"] = X_val["kilometer"] * X_val["car_age"]
    if "power" in X_train.columns and "kilometer" in X_train.columns:
        X_train["power_per_km"] = X_train["power"] / (X_train["kilometer"] + 1.0)
        X_val["power_per_km"] = X_val["power"] / (X_val["kilometer"] + 1.0)
    if "power" in X_train.columns and "car_age" in X_train.columns:
        X_train["power_age"] = X_train["power"] * X_train["car_age"]
        X_val["power_age"] = X_val["power"] * X_val["car_age"]

    # Log features for skewed, non-negative numeric columns
    skew = X_train[num_cols].skew(numeric_only=True)
    skew_cols = skew[skew.abs() > 1.0].index.tolist()
    for c in skew_cols:
        if X_train[c].min() >= 0:
            X_train[f"log_{c}"] = np.log1p(X_train[c])
            X_val[f"log_{c}"] = np.log1p(X_val[c])

    return X_train, X_val


def base_preprocess(X_train, X_val, y_train, y_val, winsor=(1.0, 99.0), save_scaler_path=None,
                    cat_cols=None, scale_numeric=False):
    """
    基础预处理（winzor + median 填充 + 标准化）
    - 返回处理后的 X_train, X_val, y_train, y_val, scaler（scaler 已保存到 save_scaler_path）
    """
    # 目标缩尾
    def winsorize(arr, lo, hi):
        lo_v = np.percentile(arr, lo)
        hi_v = np.percentile(arr, hi)
        return np.clip(arr, lo_v, hi_v)

    y_train = pd.Series(winsorize(y_train.values, winsor[0], winsor[1]), index=y_train.index)
    y_val = pd.Series(winsorize(y_val.values, winsor[0], winsor[1]), index=y_val.index)

    if cat_cols is None:
        cat_cols = []

    # Numeric winsor/fill (exclude categorical columns)
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    for c in num_cols:
        X_train[c] = winsorize(X_train[c].values, winsor[0], winsor[1])
        if c in X_val.columns:
            X_val[c] = winsorize(X_val[c].values, winsor[0], winsor[1])

    if num_cols:
        med = X_train[num_cols].median()
        X_train[num_cols] = X_train[num_cols].fillna(med)
        X_val[num_cols] = X_val[num_cols].fillna(med)

    # Fill categorical columns (integer-encoded in processed_data)
    for c in cat_cols:
        mv = X_train[c].mode().iloc[0] if not X_train[c].mode().empty else ""
        X_train[c] = X_train[c].fillna(mv)
        if c in X_val.columns:
            X_val[c] = X_val[c].fillna(mv)

    # Optional: scale numeric features only
    scaler = None
    if scale_numeric and num_cols:
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val[num_cols] = scaler.transform(X_val[num_cols])
        if save_scaler_path:
            joblib.dump(scaler, save_scaler_path)
    return X_train, X_val, y_train, y_val, scaler


# -------------------- 从 baseline 构造微调搜索空间 --------------------
def construct_fine_search_space(baseline_params):
    """
    根据 baseline best_params 构造更窄的搜索空间用于 fine tuning
    返回一个函数 make_param_space(trial) 用于在 Optuna objective 内调用 trial.suggest_*
    支持的 baseline_keys: learning_rate, depth, l2_leaf_reg, min_data_in_leaf, subsample, rsm, random_strength, bagging_temperature
    """
    # 解析 baseline（可能是包含 'best_params' 的文件）
    params = baseline_params.copy()

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # 取基线值或默认
    lr0 = float(params.get('learning_rate', 0.02))
    depth0 = int(params.get('depth', 5))
    l2_0 = int(params.get('l2_leaf_reg', 30))
    min_leaf0 = int(params.get('min_data_in_leaf', 20))
    subsample0 = float(params.get('subsample', 0.8))
    rsm0 = float(params.get('rsm', 0.8))
    randstr0 = float(params.get('random_strength', 1.5))
    bagtemp0 = float(params.get('bagging_temperature', 0.0))
    max_leaves0 = int(params.get('max_leaves', 64))

    # 构造窄范围（注意边界）
    lr_lo = max(1e-4, lr0 * 0.5)
    lr_hi = min(0.3, lr0 * 1.8)
    depth_lo = max(3, depth0 - 1)
    depth_hi = min(10, depth0 + 2)
    l2_lo = max(1, int(l2_0 * 0.5))
    l2_hi = min(200, int(max(l2_0 * 2, l2_0 + 10)))
    min_leaf_lo = max(1, int(min_leaf0 * 0.5))
    min_leaf_hi = max(min_leaf_lo + 1, int(min_leaf0 * 2))
    subsample_lo = clamp(subsample0 - 0.2, 0.2, 1.0)
    subsample_hi = clamp(subsample0 + 0.2, 0.2, 1.0)
    rsm_lo = clamp(rsm0 - 0.2, 0.2, 1.0)
    rsm_hi = clamp(rsm0 + 0.2, 0.2, 1.0)
    randstr_lo = max(0.0, randstr0 - 1.5)
    randstr_hi = randstr0 + 1.5
    bagtemp_lo = max(0.0, bagtemp0 - 0.4)
    bagtemp_hi = min(1.0, bagtemp0 + 0.4)
    max_leaves_lo = max(31, int(max_leaves0 * 0.5))
    max_leaves_hi = min(255, int(max_leaves0 * 2))

    def suggest_params(trial):
        """在 trial 中建议参数（用于 fine Optuna objective）"""
        bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])
        grow_policy = trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
        params_out = {
            'learning_rate': trial.suggest_float('learning_rate', lr_lo, lr_hi, log=True),
            'depth': trial.suggest_int('depth', depth_lo, depth_hi),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', l2_lo, l2_hi),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', min_leaf_lo, min_leaf_hi),
            'rsm': trial.suggest_float('rsm', rsm_lo, rsm_hi),
            'random_strength': trial.suggest_float('random_strength', randstr_lo, randstr_hi),
            'bootstrap_type': bootstrap_type,
            'grow_policy': grow_policy
        }
        if bootstrap_type == 'Bayesian':
            params_out['bagging_temperature'] = trial.suggest_float('bagging_temperature', bagtemp_lo, bagtemp_hi)
        else:
            params_out['subsample'] = trial.suggest_float('subsample', subsample_lo, subsample_hi)
        if grow_policy == 'Lossguide':
            params_out['max_leaves'] = trial.suggest_int('max_leaves', max_leaves_lo, max_leaves_hi)
        return params_out

    # 输出日志说明搜索区间
    print("Fine search ranges constructed from baseline:")
    print(f" learning_rate: [{lr_lo}, {lr_hi}]")
    print(f" depth: [{depth_lo}, {depth_hi}]")
    print(f" l2_leaf_reg: [{l2_lo}, {l2_hi}]")
    print(f" min_data_in_leaf: [{min_leaf_lo}, {min_leaf_hi}]")
    print(f" subsample: [{subsample_lo}, {subsample_hi}]")
    print(f" rsm: [{rsm_lo}, {rsm_hi}]")
    print(f" random_strength: [{randstr_lo}, {randstr_hi}]")
    print(f" bagging_temperature: [{bagtemp_lo}, {bagtemp_hi}]")
    print(f" max_leaves: [{max_leaves_lo}, {max_leaves_hi}] (only for Lossguide)")

    return suggest_params


# -------------------- Optuna objective（可注入 suggest function） --------------------
def catboost_cv_with_suggest(trial, X, y, suggest_fn, n_folds=4, use_log=False, seed=42,
                             iterations=2000, cat_features=None):
    """
    CV objective，suggest_fn(trial) 返回 dict of suggested hyperparams (subset)
    其余固定超参在此处设置
    """
    suggested = suggest_fn(trial)
    bootstrap_type = suggested.get('bootstrap_type', 'Bayesian')
    grow_policy = suggested.get('grow_policy', 'SymmetricTree')
    params = {
        'iterations': iterations,
        'learning_rate': suggested.get('learning_rate', 0.02),
        'depth': suggested.get('depth', 5),
        'l2_leaf_reg': suggested.get('l2_leaf_reg', 30),
        'min_data_in_leaf': suggested.get('min_data_in_leaf', 20),
        'rsm': suggested.get('rsm', 0.8),
        'random_strength': suggested.get('random_strength', 1.5),
        'bootstrap_type': bootstrap_type,
        'grow_policy': grow_policy,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'random_seed': seed,
        'task_type': 'CPU',
        'verbose': 0,
        'allow_writing_files': False,
        'thread_count': -1
    }
    if bootstrap_type == 'Bayesian':
        params['bagging_temperature'] = suggested.get('bagging_temperature', 0.0)
    else:
        params['subsample'] = suggested.get('subsample', 0.8)
    if grow_policy == 'Lossguide':
        params['max_leaves'] = suggested.get('max_leaves', 64)

    # target transform if use_log
    if use_log:
        y_t = np.log1p(y)
    else:
        y_t = y.values

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    maes = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y_t[train_idx], y_t[val_idx]
        model = CatBoostRegressor(**params)
        try:
            model.fit(
                X_tr, y_tr,
                eval_set=(X_va, y_va),
                use_best_model=True,
                early_stopping_rounds=200,
                cat_features=cat_features
            )
        except Exception:
            model.fit(X_tr, y_tr, cat_features=cat_features)
        pred_t = model.predict(X_va)
        if use_log:
            pred = np.expm1(pred_t); true = np.expm1(y_va)
        else:
            pred = pred_t; true = y_va
        maes.append(mean_absolute_error(true, pred))
    return float(np.mean(maes))


# -------------------- 两阶段流程主逻辑 --------------------
def run_two_stage(baseline_params_path=None, no_baseline=False,
                  n_trials_coarse=20, n_trials_fine=50,
                  use_log=False, fit_on_train_val=False,
                  quick=False, output_dir=OUTPUT_DIR):
    """
    主流程：
    - 若提供 baseline_params_path（JSON），读取 baseline params，否则当 no_baseline=True 时进行短的粗搜索获得 baseline
    - 基于 baseline 构造 fine search 空间并执行 fine Optuna
    - 用 fine 最优参数训练最终模型并评估，按需在 train+val 上重训练生产模型
    - 所有结果、study 与模型保存在 output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    # Load data and preprocess (winsor + light feature engineering)
    X_train, X_val, y_train, y_val = load_data()

    drop_cols = [c for c in DEFAULT_DROP_COLS if c in X_train.columns]
    if drop_cols:
        X_train = X_train.drop(columns=drop_cols, errors="ignore")
        X_val = X_val.drop(columns=drop_cols, errors="ignore")
        print("Dropped ID columns:", drop_cols)

    X_train, X_val, low_cols = drop_low_variance_cols(X_train, X_val, min_unique=2)
    if low_cols:
        print("Dropped low-variance columns:", low_cols)

    cat_cols = get_cat_cols(X_train)
    X_tr, X_va, y_tr, y_va, scaler = base_preprocess(
        X_train.copy(), X_val.copy(), y_train.copy(), y_val.copy(),
        winsor=(0.5, 99.5),
        save_scaler_path=os.path.join(output_dir, "scaler.joblib"),
        cat_cols=cat_cols,
        scale_numeric=False
    )
    X_tr, X_va = engineer_features(X_tr, X_va, cat_cols)

    cat_features = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]
    if not cat_features:
        cat_features = None

    # 如果没有 baseline，则做短的粗搜索（coarse）
    baseline = None
    if baseline_params_path and os.path.exists(baseline_params_path) and not no_baseline:
        print("加载基线参数文件：", baseline_params_path)
        try:
            with open(baseline_params_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 兼容不同文件格式：如果文件为 {'best_params': {...}}，取 inner dict
            if isinstance(data, dict) and 'best_params' in data:
                baseline = data['best_params']
            elif isinstance(data, dict) and 'optuna_best_params' in data:
                baseline = data['optuna_best_params']
            else:
                # assume data is dict of params
                baseline = data
            print("已加载 baseline:", baseline)
        except Exception as e:
            print("加载 baseline 失败，将进行粗搜索来获取 baseline。错误：", e)
            baseline = None

    if baseline is None:
        print("未提供或解析 baseline，进行短的粗搜索以获得 baseline（trials=", n_trials_coarse, ")")
        sampler = optuna.samplers.TPESampler(seed=42)
        study_coarse = optuna.create_study(direction='minimize', sampler=sampler)
        def coarse_suggest(trial):
            # Wider ranges for coarse search
            bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])
            grow_policy = trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
            params_out = {
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
                'depth': trial.suggest_int('depth', 3, 9),
                'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 3, 150),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 200),
                'rsm': trial.suggest_float('rsm', 0.5, 1.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
                'bootstrap_type': bootstrap_type,
                'grow_policy': grow_policy
            }
            if bootstrap_type == 'Bayesian':
                params_out['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 1.0)
            else:
                params_out['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
            if grow_policy == 'Lossguide':
                params_out['max_leaves'] = trial.suggest_int('max_leaves', 31, 255)
            return params_out

        objective_coarse = partial(
            catboost_cv_with_suggest,
            suggest_fn=coarse_suggest,
            X=X_tr,
            y=y_tr,
            n_folds=3,
            use_log=use_log,
            seed=42,
            iterations=1000 if quick else 1500,
            cat_features=cat_features
        )
        study_coarse.optimize(objective_coarse, n_trials=n_trials_coarse, show_progress_bar=True)
        baseline = study_coarse.best_params
        joblib.dump(study_coarse, os.path.join(output_dir, "optuna_study_coarse.joblib"))
        with open(os.path.join(output_dir, "optuna_coarse_best.json"), "w", encoding='utf-8') as f:
            json.dump({'best_params': baseline, 'best_value': study_coarse.best_value}, f, ensure_ascii=False, indent=2)
        print("粗搜索完成，baseline:", baseline)

    # 构造 fine 搜索空间
    suggest_fn = construct_fine_search_space(baseline)

    # 运行 fine Optuna
    print("开始 fine Optuna（trials=", n_trials_fine, ")，在 baseline 周围做更细搜索...")
    sampler = optuna.samplers.TPESampler(seed=42)
    study_fine = optuna.create_study(direction='minimize', sampler=sampler)
    objective_fine = partial(
        catboost_cv_with_suggest,
        suggest_fn=suggest_fn,
        X=X_tr,
        y=y_tr,
        n_folds=4,
        use_log=use_log,
        seed=42,
        iterations=(1500 if quick else 3500),
        cat_features=cat_features
    )
    study_fine.optimize(objective_fine, n_trials=n_trials_fine, show_progress_bar=True)

    # 保存 fine study 与结果
    joblib.dump(study_fine, os.path.join(output_dir, "optuna_study_fine.joblib"))
    with open(os.path.join(output_dir, "optuna_fine_best.json"), "w", encoding='utf-8') as f:
        json.dump({'best_params': study_fine.best_params, 'best_value': study_fine.best_value}, f, ensure_ascii=False, indent=2)
    print("Fine search 完成，best_params:", study_fine.best_params, "best_CV_MAE:", study_fine.best_value)

    # 用 fine 最优参数训练最终模型并评估
    best_params = study_fine.best_params
    bootstrap_type = best_params.get('bootstrap_type', 'Bayesian')
    grow_policy = best_params.get('grow_policy', 'SymmetricTree')
    final_params = {
        'iterations': 3000 if quick else 6000,
        'learning_rate': best_params.get('learning_rate', 0.02),
        'depth': best_params.get('depth', 5),
        'l2_leaf_reg': best_params.get('l2_leaf_reg', 30),
        'min_data_in_leaf': best_params.get('min_data_in_leaf', 20),
        'rsm': best_params.get('rsm', 0.8),
        'random_strength': best_params.get('random_strength', 1.5),
        'bootstrap_type': bootstrap_type,
        'grow_policy': grow_policy,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'random_seed': 42,
        'od_type': 'Iter',
        'od_wait': 150 if quick else 250,
        'task_type': 'CPU',
        'verbose': 100 if not quick else 50,
        'allow_writing_files': False,
        'thread_count': -1
    }
    if bootstrap_type == 'Bayesian':
        final_params['bagging_temperature'] = best_params.get('bagging_temperature', 0.0)
    else:
        final_params['subsample'] = best_params.get('subsample', 0.8)
    if grow_policy == 'Lossguide':
        final_params['max_leaves'] = best_params.get('max_leaves', 64)

    print("训练最终模型（使用 fine 最优参数）...")
    model = CatBoostRegressor(**final_params)
    if use_log:
        y_tr_t = np.log1p(y_tr.values)
        y_va_t = np.log1p(y_va.values)
    else:
        y_tr_t = y_tr.values
        y_va_t = y_va.values
    model.fit(
        X_tr, y_tr_t,
        eval_set=(X_va, y_va_t),
        use_best_model=True,
        early_stopping_rounds=250,
        cat_features=cat_features
    )

    # 验证评估
    pred_t = model.predict(X_va)
    pred = np.expm1(pred_t) if use_log else pred_t
    mae = mean_absolute_error(y_va.values, pred)
    rmse = np.sqrt(mean_squared_error(y_va.values, pred))
    r2 = np.corrcoef(y_va.values, pred)[0, 1] ** 2 if len(y_va) > 1 else 0.0

    print(f"最终模型验证结果：MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

    # 保存模型与结果
    model_path = os.path.join(output_dir, "catboost_two_stage_final.cbm")
    model.save_model(model_path)
    joblib.dump(model, os.path.join(output_dir, "catboost_two_stage_final.joblib"))

    # 可选：在 train+val 上重训练生产模型并保存
    if fit_on_train_val:
        print("在 train+val 上重训练生产模型（为部署准备）...")
        X_comb = pd.concat([X_tr, X_va], axis=0)
        y_comb = pd.concat([y_tr, y_va], axis=0)
        if use_log:
            y_comb_t = np.log1p(y_comb.values)
        else:
            y_comb_t = y_comb.values
        prod_model = CatBoostRegressor(**final_params)
        prod_model.fit(X_comb, y_comb_t, use_best_model=False, cat_features=cat_features)
        prod_path = os.path.join(output_dir, "catboost_two_stage_prod.cbm")
        prod_model.save_model(prod_path)
        joblib.dump(prod_model, os.path.join(output_dir, "catboost_two_stage_prod.joblib"))
        print("已保存生产模型到：", prod_path)

    # 保存预测 CSV（验证集）与 summary
    val_df = pd.DataFrame({'y_true': y_va.values, 'y_pred': pred})
    val_df.to_csv(os.path.join(output_dir, "val_predictions_two_stage.csv"), index=False)
    summary = {
        'baseline': baseline,
        'fine_best_params': study_fine.best_params,
        'fine_cv_mae': float(study_fine.best_value),
        'val_mae': float(mae),
        'val_rmse': float(rmse),
        'val_r2': float(r2)
    }
    with open(os.path.join(output_dir, "two_stage_summary.json"), "w", encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Two-stage fine-tuning 完成，结果保存在：", output_dir)
    print("验证 MAE =", mae)
    return summary


# -------------------- CLI --------------------
def main():
    parser = argparse.ArgumentParser(description="两阶段 Optuna 微调（在 baseline 周围做更细搜索）")
    parser.add_argument("--baseline", type=str, default=DEFAULT_BASELINE_PATH, help="baseline 参数文件路径（JSON），若包含 {'best_params':...} 将自动读取内部")
    parser.add_argument("--no-baseline", action="store_true", help="忽略 baseline，先做粗搜索获得 baseline")
    parser.add_argument("--n-trials-coarse", type=int, default=20, help="若无 baseline，粗搜索试验数")
    parser.add_argument("--n-trials-fine", type=int, default=50, help="fine 阶段试验数（在 baseline 周围搜索）")
    parser.add_argument("--use-log", action="store_true", help="训练时对目标使用 log1p 变换")
    parser.add_argument("--fit-final", action="store_true", help="是否在 train+val 上重训练生产模型并保存")
    parser.add_argument("--quick", action="store_true", help="快速模式：fine 使用较少 iterations 以加速验证")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="输出目录")
    args = parser.parse_args()

    start = time.time()
    summary = run_two_stage(baseline_params_path=args.baseline,
                            no_baseline=args.no_baseline,
                            n_trials_coarse=args.n_trials_coarse,
                            n_trials_fine=args.n_trials_fine,
                            use_log=args.use_log,
                            fit_on_train_val=args.fit_final,
                            quick=args.quick,
                            output_dir=args.output_dir)
    elapsed = (time.time() - start) / 60.0
    print(f"总耗时 {elapsed:.1f} 分钟")
    print("Summary:", summary)


if __name__ == "__main__":
    main()
