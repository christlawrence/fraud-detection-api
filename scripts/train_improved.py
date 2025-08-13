# train_improved.py
"""
Improved training script:
- Monitors AUPRC (average precision) in early stopping using xgb.train fallback when needed.
- Saves model as JSON and stores hyperparams + metadata.
- Keeps shap generation and evaluation as before.
"""
import os
import json
import inspect
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_recall_curve, roc_curve, auc,
                             accuracy_score, precision_score, recall_score, f1_score,
                             average_precision_score)
import xgboost as xgb

# Paths
ROOT = Path(__file__).resolve().parent.parent if (Path(__file__).name == "train_improved.py") else Path.cwd()
DATA_PATH = ROOT / "data" / "PS_20174392719_1491204439457_log.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "figures"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

df = pd.read_csv(DATA_PATH)
df = df[(df["type"] == "TRANSFER") | (df["type"] == "CASH_OUT")].copy()
df["errorBalanceOrig"] = (df["newbalanceOrig"] + df["amount"]) - df["oldbalanceOrg"]
df["errorBalanceDest"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
df["isBlackHoleTransaction"] = ((df['oldbalanceDest'] == df['newbalanceDest']) & (df['amount'] > 0)).astype(int)
df["hourOfDay"] = df["step"] % 24
df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=True)

y = df["isFraud"]
X = df.drop(["isFraud", "nameOrig", "nameDest", "isFlaggedFraud"], axis=1)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train_full)

pos = y_train.value_counts().get(1, 1)
neg = y_train.value_counts().get(0, 1)
scale_pos_weight = max(1.0, neg / max(1, pos))

params = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": scale_pos_weight,
    "eval_metric": "aucpr",  # monitor AUPRC
    "seed": RANDOM_STATE,
    "verbosity": 0,
}

n_estimators = 300
early_stopping_rounds = 50

def train_with_xgb_train(params, X_tr, y_tr, X_val_, y_val_, num_boost_round=300, early_rounds=50):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val_, label=y_val_)
    evals = [(dval, "validation")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_rounds,
        verbose_eval=False,
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.close()
    booster.save_model(tmp.name)
    skl = xgb.XGBClassifier()
    skl.load_model(tmp.name)
    os.unlink(tmp.name)
    return skl, booster

print("Training using xgb.train with AUPRC early stopping...")
model, booster = train_with_xgb_train(params, X_train, y_train, X_val, y_val, num_boost_round=n_estimators, early_rounds=early_stopping_rounds)

best_iter = getattr(booster, "best_iteration", None)
if best_iter is None:
    best_iter = getattr(model, "best_iteration", None)
print("Best iteration:", best_iter)

# Save model as JSON
model_path = MODELS_DIR / "fraud_detection_model.json"
model.save_model(str(model_path))

# Persist feature list
features_path = model_path.with_suffix(model_path.suffix + ".features")
with open(features_path, "w") as f:
    f.write("\n".join(list(X_train.columns)))

# Evaluate & compute AUPRC explicitly on test
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
accuracy = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auroc = auc(fpr, tpr)
auprc = average_precision_score(y_test, y_pred_proba)

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "auroc": float(auroc),
    "auprc": float(auprc),
    "best_iteration": int(best_iter) if best_iter is not None else None,
    "scale_pos_weight": float(scale_pos_weight),
    "n_estimators": n_estimators,
    "early_stopping_rounds": early_stopping_rounds,
    "training_method": "xgb.train",
}

# Save metrics JSON
metrics_path = REPORTS_DIR / "metrics_summary.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved metrics and model. AUPRC:", metrics["auprc"])

# ---- NEW: Select and persist decision threshold from validation ----
# Ensure validation features align with training order
feature_names = [ln.strip() for ln in features_path.read_text().splitlines() if ln.strip()]
X_val_aligned = X_val.reindex(columns=feature_names, fill_value=0.0)

val_proba = model.predict_proba(X_val_aligned)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, val_proba)

# Align shapes: thresholds has length N; precision/recall have length N+1
prec_t = prec[:-1]
rec_t = rec[:-1]

target_precision = 0.95
decision_threshold = 0.5
strategy = f"precision>={target_precision}"

mask = prec_t >= target_precision
if mask.any():
    # Among those meeting precision target, choose the one with max recall
    best_idx = int(np.argmax(rec_t[mask]))
    candidate_thr = thr[mask]  # same length as rec_t[mask]
    # best_idx is within candidate arrays; pick that threshold
    decision_threshold = float(candidate_thr[best_idx])
else:
    # Fallback: maximize F1 on aligned arrays
    denom = prec_t + rec_t
    f1_curve = np.where(denom > 0, 2 * (prec_t * rec_t) / denom, 0.0)
    best_idx = int(np.argmax(f1_curve))
    if len(thr) > 0:
        decision_threshold = float(thr[best_idx])
    else:
        decision_threshold = 0.5
    strategy = "max_f1_fallback"

# Persist metadata next to the model for the API
metadata_path = model_path.with_suffix(model_path.suffix + ".metadata.json")
metadata = {
    "decision_threshold": decision_threshold,
    "threshold_selection": {
        "strategy": strategy,
        "precision_target": target_precision,
    },
    "best_iteration": int(best_iter) if best_iter is not None else None,
    "scale_pos_weight": float(scale_pos_weight),
    "eval_metric": "aucpr",
}
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved decision_threshold={decision_threshold:.4f} to {metadata_path}")