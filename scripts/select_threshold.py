# scripts/select_threshold.py

# Choose a decision threshold from validation to target a desired precision or recall, then persist it to the final model metadata for the API.

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import xgboost as xgb

ROOT = Path.cwd()
DATA = ROOT / "data" / "PS_20174392719_1491204439457_log.csv"
MODEL = ROOT / "models" / "fraud_detection_model_final.json"
FEATURES = MODEL.with_suffix(MODEL.suffix + ".features")
META = MODEL.with_suffix(MODEL.suffix + ".metadata.json")

# 1) Load data and re-create val split as in training
df = pd.read_csv(DATA)
df = df[(df["type"] == "TRANSFER") | (df["type"] == "CASH_OUT")].copy()
df["errorBalanceOrig"] = (df["newbalanceOrig"] + df["amount"]) - df["oldbalanceOrg"]
df["errorBalanceDest"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
df["isBlackHoleTransaction"] = ((df['oldbalanceDest'] == df['newbalanceDest']) & (df['amount'] > 0)).astype(int)
df["hourOfDay"] = df["step"] % 24
df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=True)

y = df["isFraud"]
X = df.drop(["isFraud", "nameOrig", "nameDest", "isFlaggedFraud"], axis=1)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.10, random_state=42, stratify=y_train_full
)

# 2) Load model and ensure correct feature order
model = xgb.XGBClassifier()
model.load_model(str(MODEL))
feature_names = [ln.strip() for ln in FEATURES.read_text().splitlines() if ln.strip()]
X_val = X_val.reindex(columns=feature_names, fill_value=0.0)

# 3) Select threshold (optimize for target precision)
proba = model.predict_proba(X_val)[:, 1]
# Guard against pathological all-NaN probs
if not np.any(np.isfinite(proba)):
    th = 0.5
    strategy = "fallback_nonfinite_probs"
else:
    prec, rec, thr = precision_recall_curve(y_val, proba)
    # Align shapes: thr has length N; prec/rec have length N+1
    if len(thr) == 0:
        th = 0.5
        strategy = "fallback_no_thresholds"
    else:
        prec_t = prec[:-1]
        rec_t = rec[:-1]

        target_precision = 0.95
        mask = prec_t >= target_precision
        if mask.any():
            # Among thresholds meeting precision target, choose the one with max recall
            best_idx = int(np.argmax(rec_t[mask]))
            candidate_thr = thr[mask]  # aligned length with rec_t[mask]
            th = float(candidate_thr[best_idx])
            strategy = f"precision>={target_precision}"
        else:
            # Fallback: maximize F1 on aligned arrays
            denom = prec_t + rec_t
            f1_curve = np.where(denom > 0, 2 * (prec_t * rec_t) / denom, 0.0)
            best_idx = int(np.argmax(f1_curve))
            th = float(thr[best_idx])
            strategy = "max_f1_fallback"

# 4) Persist into model metadata
meta = {}
if META.exists():
    try:
        meta = json.loads(META.read_text())
    except Exception:
        meta = {}

meta["decision_threshold"] = float(th)
meta["threshold_selection"] = {
    "strategy": strategy,
    "precision_target": 0.95
}
META.write_text(json.dumps(meta, indent=2))
print(f"Saved decision_threshold={th:.4f} to {META} using strategy={strategy}")