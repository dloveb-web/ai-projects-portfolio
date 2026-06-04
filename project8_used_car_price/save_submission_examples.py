# -*- coding: utf-8 -*-
"""
快速实验脚本（含按 SaleID 输出提交文件）
说明（中文注释）：
- 在原有 run_quick_experiment 脚本基础上，增加并严格化提交文件保存逻辑：
  1) 若 processed_data/test_data.joblib 和 processed_data/sale_ids.joblib 存在，则为测试集生成 submission_price.csv（仅 SaleID 与 price）。
  2) 若验证集 X_val 包含 SaleID 列，则生成 val_submission_by_saleid.csv（仅 SaleID 与 price），便于人工复核或提交特殊格式。
  3) 所有文件保存到实验输出目录 outdir（按时间戳）。
- 保持 quick 模式以便快速试验（默认迭代与试验次数较小）。
用法示例：
    python run_quick_experiment_with_submission.py --quick --do-stacking --do-local --use-log
"""
import os
import time
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from functools import partial
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna

# 全局配置
PROCESSED_DIR = "processed_data"
OUT_DIR_ROOT = "experiments_quick"
os.makedirs(OUT_DIR_ROOT, exist_ok=True)
OUT_DIR = OUT_DIR_ROOT

# -------------------- 辅助函数（中文注释） --------------------
def load_processed_data():
    """加载 processed_data 下的训练/验证数据（返回 pandas 类型）"""
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
    return X_train, X_val, y_train, y_val

def winsorize_array(arr, lo=1.0, hi=99.0):
    """缩尾函数（防极端值影响）"""
    lo_v = np.percentile(arr, lo)
    hi_v = np.percentile(arr, hi)
    return np.clip(arr, lo_v, hi_v)

def preprocess(X_train, X_val, y_train, y_val, winsor=(1.0,99.0), scale=True, save_dir=None):
    """预处理：目标 & 数值缩尾、填充、标准化（保存 scaler）"""
    y_train = pd.Series(winsorize_array(y_train.values, winsor[0], winsor[1]), index=y_train.index)
    y_val = pd.Series(winsorize_array(y_val.values, winsor[0], winsor[1]), index=y_val.index)
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        X_train[c] = winsorize_array(X_train[c].values, winsor[0], winsor[1])
        if c in X_val.columns:
            X_val[c] = winsorize_array(X_val[c].values, winsor[0], winsor[1])
    if num_cols:
        med = X_train[num_cols].median()
        X_train[num_cols] = X_train[num_cols].fillna(med)
        X_val[num_cols] = X_val[num_cols].fillna(med)
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in cat_cols:
        mv = X_train[c].mode().iloc[0] if not X_train[c].mode().empty else ""
        X_train[c] = X_train[c].fillna(mv)
        if c in X_val.columns:
            X_val[c] = X_val[c].fillna(mv)
    scaler = None
    if scale and num_cols:
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val[num_cols] = scaler.transform(X_val[num_cols])
        if save_dir:
            joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
    return X_train, X_val, y_train, y_val, scaler

def feature_engineer(X_train, X_val, y_train=None, save_dir=None):
    """轻量特征工程：删除高相关、交互、log、brand target-encode"""
    X_tr = X_train.copy(); X_va = X_val.copy()
    num_cols = X_tr.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        corr = X_tr[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = [col for col in upper.columns if any(upper[col] > 0.94)]
        if high_corr:
            X_tr = X_tr.drop(columns=high_corr, errors='ignore')
            X_va = X_va.drop(columns=high_corr, errors='ignore')
    if 'age' in X_tr.columns and 'mileage' in X_tr.columns:
        X_tr['age_mileage'] = X_tr['age'] * X_tr['mileage']
        X_va['age_mileage'] = X_va['age'] * X_va['mileage']
    if 'mileage' in X_tr.columns:
        X_tr['log_mileage'] = np.log1p(X_tr['mileage'])
        X_va['log_mileage'] = np.log1p(X_va['mileage'])
    if 'engine_power' in X_tr.columns and 'weight' in X_tr.columns:
        X_tr['power_to_weight'] = X_tr['engine_power'] / (X_tr['weight'] + 1e-6)
        X_va['power_to_weight'] = X_va['engine_power'] / (X_va['weight'] + 1e-6)
    encoder = None
    if 'brand' in X_tr.columns and y_train is not None:
        encoder = TargetEncoder(min_samples_leaf=20, smoothing=10.0, handle_unknown='value', handle_missing='value')
        X_tr['brand_te'] = encoder.fit_transform(X_tr['brand'], y_train)
        X_va['brand_te'] = encoder.transform(X_va['brand'])
        if save_dir:
            joblib.dump(encoder, os.path.join(save_dir, "brand_te.joblib"))
    if save_dir:
        joblib.dump(X_tr.columns.tolist(), os.path.join(save_dir, "train_columns.joblib"))
    return X_tr, X_va

# Optuna 目标（quick 模式用较少迭代）
def catboost_cv_objective(trial, X, y, n_folds=3, use_log=False, random_seed=42, iterations=800):
    params = {
        'iterations': iterations,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.06, log=True),
        'depth': trial.suggest_int('depth', 3, 7),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 3, 80),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'rsm': trial.suggest_float('rsm', 0.5, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 3.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'random_seed': random_seed,
        'task_type': 'CPU',
        'verbose': 0
    }
    if use_log:
        y_trans = np.log1p(y)
    else:
        y_trans = y.values
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    maes = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y_trans[train_idx], y_trans[val_idx]
        model = CatBoostRegressor(**params)
        try:
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, early_stopping_rounds=50)
        except Exception:
            model.fit(X_tr, y_tr)
        y_va_pred_t = model.predict(X_va)
        if use_log:
            y_va_pred = np.expm1(y_va_pred_t); y_va_true = np.expm1(y_va)
        else:
            y_va_pred = y_va_pred_t; y_va_true = y_va
        maes.append(mean_absolute_error(y_va_true, y_va_pred))
    return float(np.mean(maes))

def run_optuna_quick(X_train, y_train, n_trials=6, n_folds=2, use_log=False, seed=42, quick=True, save_dir=None):
    iterations = 800 if quick else 2500
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    objective = partial(catboost_cv_objective, X=X_train, y=y_train, n_folds=n_folds, use_log=use_log, random_seed=seed, iterations=iterations)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    if save_dir:
        joblib.dump(study, os.path.join(save_dir, "optuna_study_quick.joblib"))
        with open(os.path.join(save_dir, "optuna_best_params_quick.json"), "w", encoding="utf-8") as f:
            json.dump({'best_params': study.best_params, 'best_value': study.best_value}, f, ensure_ascii=False, indent=2)
    return study.best_params

def train_final_catboost(best_params, X_train, y_train, X_val, y_val, use_log=False, quick=True, save_dir=None):
    params = {
        'iterations': 1000 if quick else 5000,
        'learning_rate': best_params.get('learning_rate', 0.02),
        'depth': best_params.get('depth', 5),
        'l2_leaf_reg': best_params.get('l2_leaf_reg', 30),
        'min_data_in_leaf': best_params.get('min_data_in_leaf', 20),
        'subsample': best_params.get('subsample', 0.8),
        'rsm': best_params.get('rsm', 0.8),
        'random_strength': best_params.get('random_strength', 1.5),
        'bagging_temperature': best_params.get('bagging_temperature', 0.0),
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'random_seed': 42,
        'od_type': 'Iter',
        'od_wait': 80 if quick else 200,
        'task_type': 'CPU',
        'verbose': 50 if quick else 100
    }
    if use_log:
        y_train_t = np.log1p(y_train.values); y_val_t = np.log1p(y_val.values)
    else:
        y_train_t = y_train.values; y_val_t = y_val.values
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train_t, eval_set=(X_val, y_val_t), use_best_model=True)
    y_val_pred_t = model.predict(X_val)
    y_val_pred = np.expm1(y_val_pred_t) if use_log else y_val_pred_t
    mae = mean_absolute_error(y_val.values, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val.values, y_val_pred))
    r2 = r2_score(y_val.values, y_val_pred)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model.save_model(os.path.join(save_dir, "catboost_final.cbm"))
        joblib.dump(model, os.path.join(save_dir, "catboost_final.joblib"))
    return model, mae, rmse, r2, y_val_pred

# 轻量 stacking 与 local functions 保留（略去重复注释以节省篇幅）
def run_stacking_quick(X_train, y_train, X_val, y_val, use_log=False, quick=True, save_dir=None):
    n_folds = 3 if quick else 5
    X_tr = X_train.copy(); X_va = X_val.copy()
    y_tr = y_train.copy(); y_va = y_val.copy()
    if use_log:
        y_tr_t = np.log1p(y_tr)
    else:
        y_tr_t = y_tr
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X_tr), 3))
    val_preds_folds = np.zeros((n_folds, len(X_va), 3))
    cat_models = []; xgb_models = []; lgb_models = []
    fold = 0
    for train_idx, hold_idx in kf.split(X_tr):
        X_t, X_h = X_tr.iloc[train_idx], X_tr.iloc[hold_idx]
        y_t, y_h = y_tr_t.iloc[train_idx], y_tr_t.iloc[hold_idx]
        cb = CatBoostRegressor(iterations=800 if quick else 2500, learning_rate=0.02, depth=5, l2_leaf_reg=30, verbose=0, random_seed=42)
        cb.fit(X_t, y_t, eval_set=(X_h, y_h), use_best_model=True)
        if use_log:
            oof_preds[hold_idx,0] = np.expm1(cb.predict(X_h))
            val_preds_folds[fold,:,0] = np.expm1(cb.predict(X_va))
        else:
            oof_preds[hold_idx,0] = cb.predict(X_h)
            val_preds_folds[fold,:,0] = cb.predict(X_va)
        cat_models.append(cb)
        xgb = XGBRegressor(n_estimators=400 if quick else 800, learning_rate=0.03, max_depth=6, subsample=0.8, random_state=42, verbosity=0)
        xgb.fit(X_t, (np.expm1(y_t) if use_log else y_t).values)
        oof_preds[hold_idx,1] = xgb.predict(X_h); val_preds_folds[fold,:,1] = xgb.predict(X_va)
        xgb_models.append(xgb)
        lgb = LGBMRegressor(n_estimators=400 if quick else 800, learning_rate=0.04, max_depth=6, subsample=0.8, random_state=42)
        lgb.fit(X_t, (np.expm1(y_t) if use_log else y_t).values)
        oof_preds[hold_idx,2] = lgb.predict(X_h); val_preds_folds[fold,:,2] = lgb.predict(X_va)
        lgb_models.append(lgb)
        fold += 1
    meta = Ridge(alpha=1.0); meta.fit(oof_preds, y_tr.values)
    val_preds_avg = val_preds_folds.mean(axis=0); val_meta_pred = meta.predict(val_preds_avg)
    mae = mean_absolute_error(y_val.values, val_meta_pred)
    rmse = np.sqrt(mean_squared_error(y_val.values, val_meta_pred))
    r2 = r2_score(y_val.values, val_meta_pred)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(cat_models, os.path.join(save_dir, "cat_models.joblib"))
        joblib.dump(xgb_models, os.path.join(save_dir, "xgb_models.joblib"))
        joblib.dump(lgb_models, os.path.join(save_dir, "lgb_models.joblib"))
        joblib.dump(meta, os.path.join(save_dir, "meta_ridge.joblib"))
        pd.DataFrame({'y_true': y_val.values, 'y_pred_stack': val_meta_pred}).to_csv(os.path.join(save_dir, "val_pred_stack.csv"), index=False)
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'pred': val_meta_pred, 'meta': meta}

def train_local_models_quick(X_train, y_train, X_val, y_val, split_feature_candidates=['mileage','km_per_year','age'], bins=3, quick=True, save_dir=None):
    split_feature = None
    for f in split_feature_candidates:
        if f in X_train.columns and f in X_val.columns and X_train[f].nunique() > 5:
            split_feature = f; break
    if split_feature is None:
        return None
    Xtr = X_train.copy(); Xva = X_val.copy()
    try:
        _, edges = pd.qcut(Xtr[split_feature], q=bins, retbins=True, duplicates='drop')
        Xtr['bin'] = pd.cut(Xtr[split_feature], bins=edges, labels=False, include_lowest=True)
        Xva['bin'] = pd.cut(Xva[split_feature], bins=edges, labels=False, include_lowest=True)
    except Exception:
        edges = np.linspace(Xtr[split_feature].min(), Xtr[split_feature].max(), bins+1)
        Xtr['bin'] = pd.cut(Xtr[split_feature], bins=edges, labels=False, include_lowest=True)
        Xva['bin'] = pd.cut(Xva[split_feature], bins=edges, labels=False, include_lowest=True)
    models_info = {}
    for b in sorted(Xtr['bin'].dropna().unique()):
        idx_tr = Xtr.index[Xtr['bin'] == b]; idx_va = Xva.index[Xva['bin'] == b]
        X_tr_bin = Xtr.loc[idx_tr].drop(columns=['bin']); y_tr_bin = y_train.loc[idx_tr]
        X_va_bin = Xva.loc[idx_va].drop(columns=['bin']); y_va_bin = y_val.loc[idx_va]
        if len(X_tr_bin) < max(50, int(0.01 * len(X_train))):
            continue
        model = CatBoostRegressor(iterations=800 if quick else 2500, learning_rate=0.02, depth=4, l2_leaf_reg=30, verbose=0, random_seed=42)
        if len(X_va_bin) > 0:
            model.fit(X_tr_bin, y_tr_bin, eval_set=(X_va_bin, y_va_bin), use_best_model=True)
            val_mae = mean_absolute_error(y_va_bin.values, model.predict(X_va_bin))
        else:
            model.fit(X_tr_bin, y_tr_bin); val_mae = None
        models_info[int(b)] = {'model': model, 'val_index': list(idx_va), 'val_mae': val_mae}
        if save_dir:
            os.makedirs(save_dir, exist_ok=True); model.save_model(os.path.join(save_dir, f"local_bin_{int(b)}.cbm"))
    return {'feature': split_feature, 'edges': edges.tolist(), 'models': models_info}

# -------------------- 新增：保存提交文件函数（SaleID, price） --------------------
def save_test_submission(preds, out_dir, filename="submission_price.csv"):
    """若存在 processed_data/sale_ids.joblib 与 test_data.joblib，则生成提交文件（SaleID, price）"""
    sale_ids_path = os.path.join(PROCESSED_DIR, "sale_ids.joblib")
    test_data_path = os.path.join(PROCESSED_DIR, "test_data.joblib")
    if not os.path.exists(sale_ids_path) or not os.path.exists(test_data_path):
        print("未找到测试数据或 SaleID，跳过生成测试提交文件")
        return None
    sale_ids = joblib.load(sale_ids_path)
    # 对齐长度（以最小长度为准）
    n = min(len(sale_ids), len(preds))
    df = pd.DataFrame({"SaleID": sale_ids[:n], "price": np.array(preds)[:n]})
    out_path = os.path.join(out_dir, filename)
    df.to_csv(out_path, index=False)
    print("测试提交文件已保存:", out_path)
    return out_path

def save_val_submission_with_saleid(X_val_raw, preds, out_dir, saleid_col="SaleID", filename="val_submission_by_saleid.csv"):
    """若验证集含 SaleID 列，则按 SaleID+price 保存验证预测文件"""
    if saleid_col not in X_val_raw.columns:
        print(f"验证集不包含列 {saleid_col}，跳过按 SaleID 输出验证预测")
        return None
    df = pd.DataFrame({"SaleID": X_val_raw[saleid_col].values, "price": np.array(preds)})
    out_path = os.path.join(out_dir, filename)
    df.to_csv(out_path, index=False)
    print("验证集按 SaleID 的预测文件已保存:", out_path)
    return out_path

# -------------------- 主流程（中文注释） --------------------
def main():
    parser = argparse.ArgumentParser(description="快速实验脚本（含 SaleID 提交保存）")
    parser.add_argument("--quick", action="store_true", help="启用 quick 模式（少量 trials / 少量迭代）")
    parser.add_argument("--do-stacking", action="store_true", help="开启 OOF stacking")
    parser.add_argument("--do-local", action="store_true", help="开启局部分箱模型")
    parser.add_argument("--use-log", action="store_true", help="对目标使用 log1p 变换")
    parser.add_argument("--n-trials", type=int, default=6, help="Optuna 试验次数（quick 模式默认6）")
    parser.add_argument("--n-folds", type=int, default=2, help="Optuna CV 折数（quick 模式默认2）")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(OUT_DIR_ROOT, f"exp_{ts}_quick{int(args.quick)}_log{int(args.use_log)}")
    os.makedirs(outdir, exist_ok=True)
    print("实验输出目录:", outdir)

    # 加载数据
    X_train, X_val_raw, y_train, y_val = load_processed_data()
    print("数据加载：", X_train.shape, X_val_raw.shape, len(y_train), len(y_val))

    # 预处理与特征工程（保存 train_columns/scaler/encoder 到 outdir）
    winsor = (2.0, 98.0) if args.quick else (1.0, 99.0)
    X_tr, X_va, y_tr, y_va, scaler = preprocess(X_train.copy(), X_val_raw.copy(), y_train.copy(), y_val.copy(), winsor=winsor, scale=True, save_dir=outdir)
    X_tr_fe, X_va_fe = feature_engineer(X_tr, X_va, y_tr, save_dir=outdir)
    print("预处理与特征工程完成，特征数量:", X_tr_fe.shape[1])

    # Optuna 快速调参
    n_trials = args.n_trials if not args.quick else 6
    n_folds = args.n_folds if not args.quick else 2
    print("开始 Optuna 快速调参...")
    best_params = run_optuna_quick(X_tr_fe, y_tr, n_trials=n_trials, n_folds=n_folds, use_log=args.use_log, seed=42, quick=args.quick, save_dir=outdir)
    print("Optuna 最佳参数:", best_params)
    with open(os.path.join(outdir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    # 用最佳参���训练最终模型并评估
    model_dir = os.path.join(outdir, "final_model")
    os.makedirs(model_dir, exist_ok=True)
    final_model, val_mae, val_rmse, val_r2, y_val_pred = train_final_catboost(best_params, X_tr_fe, y_tr, X_va_fe, y_va, use_log=args.use_log, quick=args.quick, save_dir=model_dir)
    print(f"最终模型验证 MAE={val_mae:.2f}, RMSE={val_rmse:.2f}, R2={val_r2:.4f}")

    # 保存验证集预测（包含真实值与预测值）
    pd.DataFrame({'y_true': y_va.values, 'y_pred': y_val_pred}).to_csv(os.path.join(outdir, "val_predictions_final.csv"), index=False)

    # 可选 stacking（快速）
    final_preds_for_submission = y_val_pred
    if args.do_stacking:
        print("运行快速 stacking...")
        stack_dir = os.path.join(outdir, "stacking")
        os.makedirs(stack_dir, exist_ok=True)
        stack_res = run_stacking_quick(X_tr_fe, y_tr, X_va_fe, y_va, use_log=args.use_log, quick=args.quick, save_dir=stack_dir)
        print("Stacking MAE=", stack_res['mae'])
        if stack_res['mae'] < val_mae:
            final_preds_for_submission = stack_res['pred']

    # 可选局部分箱并融合（优先分箱）
    if args.do_local:
        print("训练局部分箱模型...")
        local_dir = os.path.join(outdir, "local_models")
        os.makedirs(local_dir, exist_ok=True)
        local_info = train_local_models_quick(X_tr_fe, y_tr, X_va_fe, y_va, quick=args.quick, save_dir=local_dir)
        if local_info:
            try:
                fused = final_preds_for_submission.copy()
                for b, info in local_info['models'].items():
                    val_idx = info['val_index']
                    if not val_idx: continue
                    preds_local = info['model'].predict(X_va_fe.loc[val_idx])
                    locs = [X_va_fe.index.get_loc(i) for i in val_idx]
                    fused[locs] = preds_local
                # 更新最终预测（用于提交）
                final_preds_for_submission = fused
            except Exception as e:
                print("局部分箱融合失败：", e)

    # 现在保存验证集按 SaleID 输出（如果 X_val_raw 包含 SaleID 列）
    save_val_submission_with_saleid(X_val_raw, final_preds_for_submission, outdir, saleid_col="SaleID", filename="val_submission_by_saleid.csv")

    # 对测试集做预测并保存 submission（SaleID, price）
    test_data_path = os.path.join(PROCESSED_DIR, "test_data.joblib")
    sale_ids_path = os.path.join(PROCESSED_DIR, "sale_ids.joblib")
    if os.path.exists(test_data_path) and os.path.exists(sale_ids_path):
        print("加载测试数据，准备对测试集做预测并生成提交（SaleID, price）...")
        test_data = joblib.load(test_data_path)
        sale_ids = joblib.load(sale_ids_path)
        # 对测试集应用与训练相同的特征工程与标准化（尽量按训练时保存的列/encoder进行）
        # 添加交互特征
        if 'age' in test_data.columns and 'mileage' in test_data.columns:
            test_data['age_mileage'] = test_data['age'] * test_data['mileage']
        if 'mileage' in test_data.columns:
            test_data['log_mileage'] = np.log1p(test_data['mileage'])
        if 'engine_power' in test_data.columns and 'weight' in test_data.columns:
            test_data['power_to_weight'] = test_data['engine_power'] / (test_data['weight'] + 1e-6)
        # brand encoder
        encoder_path = os.path.join(outdir, "brand_te.joblib")
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
            if 'brand' in test_data.columns:
                test_data['brand_te'] = encoder.transform(test_data['brand'])
            else:
                test_data['brand_te'] = 0
        # 标准化（尽量使用保存的 scaler）
        scaler_path = os.path.join(outdir, "scaler.joblib")
        scaler = None
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            num_cols_test = test_data.select_dtypes(include=[np.number]).columns.tolist()
            try:
                test_data[num_cols_test] = scaler.transform(test_data[num_cols_test])
            except Exception:
                # 兼容性回退：只对交集列尝试 transform
                common = [c for c in num_cols_test if c in getattr(scaler, 'mean_', {})]
                try:
                    test_data[common] = scaler.transform(test_data[common])
                except Exception:
                    pass
        # 对齐训练列
        train_cols_path = os.path.join(outdir, "train_columns.joblib")
        if os.path.exists(train_cols_path):
            train_cols = joblib.load(train_cols_path)
            test_data = test_data.reindex(columns=train_cols, fill_value=0)
        else:
            test_data = test_data.reindex(columns=X_tr_fe.columns.tolist(), fill_value=0)
        # 预测优先级：stacking -> local fusion -> final_model
        final_test_preds = None
        if args.do_stacking and os.path.exists(os.path.join(outdir, "stacking", "meta_ridge.joblib")):
            try:
                cat_models = joblib.load(os.path.join(outdir, "stacking", "cat_models.joblib"))
                xgb_models = joblib.load(os.path.join(outdir, "stacking", "xgb_models.joblib"))
                lgb_models = joblib.load(os.path.join(outdir, "stacking", "lgb_models.joblib"))
                meta = joblib.load(os.path.join(outdir, "stacking", "meta_ridge.joblib"))
                preds_cat = np.mean([m.predict(test_data) for m in cat_models], axis=0)
                preds_xgb = np.mean([m.predict(test_data) for m in xgb_models], axis=0)
                preds_lgb = np.mean([m.predict(test_data) for m in lgb_models], axis=0)
                stack_feat = np.vstack([preds_cat, preds_xgb, preds_lgb]).T
                final_test_preds = meta.predict(stack_feat)
            except Exception as e:
                print("Stacking 对测试集预测失败：", e)
                final_test_preds = None
        # 若 stacking 失败或未启用，则用 final_model
        if final_test_preds is None:
            final_model_joblib = os.path.join(model_dir, "catboost_final.joblib")
            cbm_path = os.path.join(model_dir, "catboost_final.cbm")
            final_model_obj = None
            if os.path.exists(final_model_joblib):
                final_model_obj = joblib.load(final_model_joblib)
            elif os.path.exists(cbm_path):
                final_model_obj = CatBoostRegressor(); final_model_obj.load_model(cbm_path)
            else:
                final_model_obj = final_model
            try:
                final_test_preds = final_model_obj.predict(test_data)
            except Exception as e:
                print("最终模型对测试集预测失败：", e)
                final_test_preds = np.zeros(len(test_data))
        # 若存在局部分箱模型，可对分箱样本做替换（若 local models 存在）
        local_models_dir = os.path.join(outdir, "local_models")
        if os.path.exists(local_models_dir):
            # 若本次训练生成了局部模型文件，则尝试加载并替换
            # 简化处理：若 local models 保存在 local_models/，则跳过复杂融合（本脚本优先使用 final_test_preds）
            pass
        # 保存测试提交（SaleID, price）
        save_test_submission(final_test_preds, outdir, filename="submission_price.csv")
    else:
        print("未找到测试数据或 SaleID，跳过生成测试提交文件（processed_data/test_data.joblib / sale_ids.joblib）")

    # 保存汇总 summary
    summary = {'val_mae': float(val_mae), 'val_rmse': float(val_rmse), 'val_r2': float(val_r2), 'optuna_best_params': best_params, 'timestamp': ts}
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("实验完成，结果保存在:", outdir)
    print("验证 MAE =", val_mae)

if __name__ == "__main__":
    main()