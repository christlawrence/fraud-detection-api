# scripts/predict.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import json
#import os
from pathlib import Path
#import numpy as np
import pandas as pd
import xgboost as xgb

app = FastAPI(title="Fraud Detection (Hybrid)")

# Configuration: canonical model paths (update if you use different filenames)
ROOT = Path(__file__).resolve().parent.parent if (Path(__file__).name == "predict.py") else Path.cwd()
MODEL_PATH = ROOT / "models" / "fraud_detection_model_final.json"
FEATURES_PATH = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".features")
METADATA_PATH = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".metadata.json")

# Load model and artifacts at startup
if not MODEL_PATH.exists():
    # fallback to tuned or default model
    if (ROOT / "models" / "fraud_detection_model_tuned.json").exists():
        MODEL_PATH = ROOT / "models" / "fraud_detection_model_tuned.json"
    elif (ROOT / "models" / "fraud_detection_model.json").exists():
        MODEL_PATH = ROOT / "models" / "fraud_detection_model.json"
    else:
        raise RuntimeError("No model file found in models/. Train a model or place final JSON model in models/.")

if not FEATURES_PATH.exists():
    # try alternative path name (model + .features)
    FEATURES_PATH = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".features")
    if not FEATURES_PATH.exists():
        raise RuntimeError("Feature list file not found alongside the model. Save the feature list during training.")

feature_names = [ln.strip() for ln in FEATURES_PATH.read_text().splitlines() if ln.strip()]

# Load model
model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))

# Load metadata if present
metadata = {}
if METADATA_PATH.exists():
    try:
        metadata = json.loads(METADATA_PATH.read_text())
    except Exception:
        metadata = {}

DECISION_THRESHOLD = float(metadata.get("decision_threshold", metadata.get("optimal_threshold", 0.5)))


# Pydantic input model: accept free-form dict while requiring certain keys later
class Transaction(BaseModel):
    payload: Dict[str, Any] = Field(..., description="Transaction fields (raw)")


def _compute_engineered_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    # safe reads with defaults
    amount = float(row.get("amount", 0.0))
    newbalanceOrig = float(row.get("newbalanceOrig", 0.0))
    oldbalanceOrg = float(row.get("oldbalanceOrg", 0.0))
    oldbalanceDest = float(row.get("oldbalanceDest", 0.0))
    newbalanceDest = float(row.get("newbalanceDest", 0.0))
    row["errorBalanceOrig"] = (newbalanceOrig + amount) - oldbalanceOrg
    row["errorBalanceDest"] = (oldbalanceDest + amount) - newbalanceDest
    row["hourOfDay"] = int(row.get("step", 0)) % 24
    # one-hot dummies for type - preserve names used in training; default 0
    # Training used get_dummies(..., drop_first=True) so expected dummies may be 'type_TRANSFER' or 'type_CASH_OUT'
    for t in ("type_TRANSFER", "type_CASH_OUT"):
        row.setdefault(t, 0)
    return row


def _is_black_hole(row: Dict[str, Any]) -> bool:
    # replicate the safety-net rule
    try:
        return (float(row.get("oldbalanceDest", 0.0)) == float(row.get("newbalanceDest", 0.0))) and (
                    float(row.get("amount", 0.0)) > 0)
    except Exception:
        return False


def _build_feature_vector(payload: Dict[str, Any]) -> pd.DataFrame:
    row = dict(payload)  # shallow copy
    row = _compute_engineered_fields(row)
    # ensure all feature columns present
    for col in feature_names:
        if col not in row:
            row[col] = 0.0
    # Order columns to match training
    vec = pd.DataFrame([{c: row[c] for c in feature_names}])
    return vec


@app.post("/predict")
def predict_tx(t: Transaction):
    payload = t.payload
    # 1) Rule-based safety net
    if _is_black_hole(payload):
        return {"fraud": True, "reason": "black_hole_rule", "model_used": False, "model_prob": None}

    # 2) Build feature vector (ensures correct ordering)
    try:
        X = _build_feature_vector(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload or missing required numeric fields: {e}")

    # 3) Model inference
    try:
        proba = float(model.predict_proba(X)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    is_fraud = proba >= DECISION_THRESHOLD
    return {
        "fraud": bool(is_fraud),
        "model_used": True,
        "model_prob": proba,
        "decision_threshold": DECISION_THRESHOLD
    }


# optional health endpoint
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_path": str(MODEL_PATH)}
