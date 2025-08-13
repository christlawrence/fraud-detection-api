"""
Inference smoke test

What it does:
- Loads the canonical model and its companion files from models/.
- Reads the saved decision_threshold from the metadata file.
- Builds a few synthetic edge cases (including ones that intentionally miss some
  one-hot/dummy columns).
- Aligns features to the model's .features list (order + missing -> 0.0).
- Applies rule-first logic, then the model threshold to produce final predictions.

Expected behavior at a high decision_threshold (~0.9931):
- Any case with rule_flag == 1 must end in final_pred == 1, regardless of model_prob.
- Any case with rule_flag == 0 and model_prob < decision_threshold must end in final_pred == 0.
- Any case with rule_flag == 0 and model_prob >= decision_threshold must end in final_pred == 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import xgboost as xgb


# ------------------------------------------------------------------------------
# Configs
# ------------------------------------------------------------------------------
MODELS_DIR = Path("models")
CANONICAL_MODEL = MODELS_DIR / "fraud_detection_model.json"
FEATURES_PATH = Path(str(CANONICAL_MODEL) + ".features")
METADATA_PATH = Path(str(CANONICAL_MODEL) + ".metadata.json")


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def load_feature_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing features file: {path}")
    feats = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not feats:
        raise ValueError(f"Empty features list in: {path}")
    return feats


def load_decision_threshold(path: Path, default: float = 0.5) -> float:
    try:
        meta = json.loads(path.read_text())
        thr = float(meta.get("decision_threshold", default))
        # numeric guardrails
        thr = min(max(thr, 1e-9), 1 - 1e-9)
        return thr
    except Exception:
        return default


def load_model(model_path: Path) -> xgb.XGBClassifier:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model


def align_to_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    # Reindex to exact columns + order, fill missing with 0.0
    aligned = df.reindex(columns=feature_list, fill_value=0.0).astype(np.float32)
    return aligned


def rule_black_hole(row: pd.Series) -> int:
    # Safety rule: if isBlackHoleTransaction==1 then flag regardless of model score
    # If column missing in the synthetic case, default to 0
    val = row.get("isBlackHoleTransaction", 0)
    try:
        return int(val == 1 or val == 1.0)
    except Exception:
        return 0


# ------------------------------------------------------------------------------
# Synthetic cases for smoke test
# Note: We only specify a subset of columns on purpose to test alignment.
# Any missing columns are filled with 0 and ordered by the .features file.
# ------------------------------------------------------------------------------
def build_cases() -> pd.DataFrame:
    # Minimal common base fields used in training/engineering
    base = dict(
        step=300.0,
        amount=9500.0,
        oldbalanceOrg=10000.0,
        newbalanceOrig=500.0,
        oldbalanceDest=4000.0,
        newbalanceDest=13500.0,
        errorBalanceOrig=(500.0 + 9500.0) - 10000.0,  # (newbalanceOrig + amount) - oldbalanceOrg
        errorBalanceDest=(4000.0 + 9500.0) - 13500.0,  # (oldbalanceDest + amount) - newbalanceDest
        hourOfDay=float(300 % 24),
        # one-hot example columns may include e.g. "type_CASH_OUT" depending on your training
        # They will be filled/kept via feature alignment below.
    )

    rows: List[Dict[str, Any]] = []

    # 1) legit_typical: no black-hole flag, reasonable balances
    rows.append({
        **base,
        "case_name": "legit_typical",
        "isBlackHoleTransaction": 0,
        # Optionally include a common one-hot if you know it exists; otherwise rely on alignment
        # "type_CASH_OUT": 1.0,
    })

    # 2) black_hole: explicit safety rule trigger
    rows.append({
        **base,
        "case_name": "black_hole",
        "isBlackHoleTransaction": 1,
        # Keep amount/balances moderate
        "amount": 2500.0,
        "newbalanceOrig": 7500.0,
        "errorBalanceOrig": (7500.0 + 2500.0) - 10000.0,
        "oldbalanceDest": 1000.0,
        "newbalanceDest": 3500.0,
        "errorBalanceDest": (1000.0 + 2500.0) - 3500.0,
    })

    # 3) small_amount_drain: realistic low-value drain without rule trigger
    rows.append({
        **base,
        "case_name": "small_amount_drain",
        "isBlackHoleTransaction": 0,
        "amount": 120.0,
        "newbalanceOrig": 9880.0,
        "errorBalanceOrig": (9880.0 + 120.0) - 10000.0,
        "oldbalanceDest": 5000.0,
        "newbalanceDest": 5120.0,
        "errorBalanceDest": (5000.0 + 120.0) - 5120.0,
    })

    # 4) edge_missing_dummies: intentionally omit any type_* one-hots
    #    to confirm feature alignment (missing -> 0) and threshold logic
    rows.append({
        **base,
        "case_name": "edge_missing_dummies",
        "isBlackHoleTransaction": 1,  # also triggers rule override
        # don't add any "type_*" columns here on purpose
    })

    df = pd.DataFrame(rows).set_index("case_name", drop=True)
    return df


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    # Load assets
    feature_list = load_feature_list(FEATURES_PATH)
    model = load_model(CANONICAL_MODEL)
    decision_threshold = load_decision_threshold(METADATA_PATH, default=0.5)

    print(f"Loaded feature list with {len(feature_list)} features.")
    print(f"Model loaded from: {CANONICAL_MODEL}")
    # Uncomment if you want to see the active threshold in console
    # print(f"Using decision_threshold: {decision_threshold:.6f}")

    # Build synthetic cases
    raw_cases = build_cases()

    # Compute rule flags on raw data (prior to numeric alignment)
    rule_flags = raw_cases.apply(rule_black_hole, axis=1)

    # Align to model features
    X = align_to_features(raw_cases, feature_list)

    # Score
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        # Some older XGBoost wrappers may require DMatrix under certain settings
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        booster = model.get_booster()
        proba = booster.predict(dmatrix)
        proba = np.asarray(proba, dtype=float)

    # Compose results
    outputs: List[Dict[str, Any]] = []
    for i, case_name in enumerate(X.index.tolist()):
        p = float(proba[i])
        rf = int(rule_flags.loc[case_name])

        # Reference model_pred at 0.5 (for context)
        model_pred_default = int(p >= 0.5)

        # Final decision: rule-first OR metadata threshold
        final_pred = int((rf == 1) or (p >= decision_threshold))

        outputs.append({
            "case": case_name,
            "rule_flag": rf,
            "model_prob": p,
            "model_pred": model_pred_default,
            "final_pred": final_pred,
        })

    # Pretty-print JSON list
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Smoke Test Error] {e}", file=sys.stderr)
        sys.exit(1)
