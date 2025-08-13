# hyperparam_search.py
"""
Randomized hyperparameter search for XGBoost using AUPRC (average precision) as the
early-stopping metric. Uses xgb.train fallback to ensure early stopping is supported
across xgboost builds. Saves the best model (JSON), metadata, and top-k results.

Usage:
    python hyperparam_search.py --trials 40 --n_jobs 2

Notes:
 - This script runs model training many times; set trials to a reasonable number for your compute budget.
 - It will write artifacts to `models/hp_search/`.
"""
import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "PS_20174392719_1491204439457_log.csv"
OUT_DIR = ROOT / "models" / "hp_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Load & prep (same filtering/feature engineering as training script)
df = pd.read_csv(DATA_PATH)
df = df[(df["type"] == "TRANSFER") | (df["type"] == "CASH_OUT")].copy()
df["errorBalanceOrig"] = (df["newbalanceOrig"] + df["amount"]) - df["oldbalanceOrg"]
df["errorBalanceDest"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
df["isBlackHoleTransaction"] = ((df['oldbalanceDest'] == df['newbalanceDest']) & (df['amount'] > 0)).astype(int)
df["hourOfDay"] = df["step"] % 24
df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=True)
y = df["isFraud"]
X = df.drop(["isFraud", "nameOrig", "nameDest", "isFlaggedFraud"], axis=1)

# Train/val/test split (keep test holdout for final evaluation)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train_full)

pos = y_train.value_counts().get(1, 1)
neg = y_train.value_counts().get(0, 1)

def sample_hyperparams():
    # Randomized sampling space (you can expand ranges)
    return {
        "learning_rate": random.choice([0.01, 0.03, 0.05, 0.08, 0.1]),
        "max_depth": random.choice([4, 6, 7, 8]),
        "subsample": random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
        "reg_alpha": random.choice([0.0, 0.1, 0.5, 1.0]),
        "reg_lambda": random.choice([1.0, 1.5, 2.0]),
        "n_estimators": random.choice([200, 300, 400]),
        "scale_pos_weight": max(1.0, neg / max(1, pos)),
    }

def train_evaluate(params, X_tr, y_tr, X_val_, y_val_, early_rounds=50):
    # Use AUPRC as eval metric for early stopping
    xgb_params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "seed": RANDOM_STATE,
        "verbosity": 0,
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "reg_alpha": params["reg_alpha"],
        "reg_lambda": params["reg_lambda"],
        "scale_pos_weight": params["scale_pos_weight"],
        # monitor aucpr
        "eval_metric": "aucpr",
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val_, label=y_val_)
    evals = [(dval, "validation")]
    booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=params["n_estimators"],
        evals=evals,
        early_stopping_rounds=early_rounds,
        verbose_eval=False,
    )
    # predict on validation and test
    val_proba = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    val_auprc = average_precision_score(y_val_, val_proba)
    return booster, val_auprc

def save_booster(booster, params, score, idx):
    fname = OUT_DIR / f"booster_trial_{idx:03d}.json"
    booster.save_model(str(fname))
    meta = {
        "params": params,
        "score_aucpr": float(score),
        "timestamp": int(time()),
        "model_file": str(fname.name),
    }
    with open(OUT_DIR / f"meta_trial_{idx:03d}.json", "w") as f:
        json.dump(meta, f, indent=2)
    return fname, meta

if __name__ == "__main__":
    parser = argparse = argparse = None
    import argparse as _arg
    parser = _arg.ArgumentParser()
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--early_rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    results = []
    for i in range(args.trials):
        params = sample_hyperparams()
        try:
            booster, val_score = train_evaluate(params, X_train, y_train, X_val, y_val, early_rounds=args.early_rounds)
            fname, meta = save_booster(booster, params, val_score, i)
            meta["trial"] = i
            meta["val_aucpr"] = float(val_score)
            results.append(meta)
            print(f"Trial {i:03d} done - AUPRC (val)={val_score:.4f}")
        except Exception as e:
            print(f"Trial {i:03d} failed: {e}")

    results_sorted = sorted(results, key=lambda r: r["val_aucpr"], reverse=True)
    with open(OUT_DIR / "hp_search_summary.json", "w") as f:
        json.dump({"results": results_sorted}, f, indent=2)

    if results_sorted:
        best = results_sorted[0]
        print("Best trial:", best["trial"], "val_aucpr=", best["val_aucpr"])
        # Save best booster as `fraud_detection_model_tuned.json`
        best_model_file = OUT_DIR / best["model_file"]
        final_dst = ROOT / "models" / "fraud_detection_model_tuned.json"
        final_dst.write_bytes(best_model_file.read_bytes())
        # Also copy metadata
        (ROOT / "models" / "fraud_detection_model_tuned.json.meta").write_text(json.dumps(best, indent=2))
        print("Saved tuned model to", final_dst)
    else:
        print("No successful trials.")