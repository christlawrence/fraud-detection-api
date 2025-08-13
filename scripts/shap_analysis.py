"""
Goal
- Produce quick, reproducible explainability artifacts for the finalized fraud model.
- Prefer SHAP TreeExplainer; gracefully fall back to built-in and permutation importances if SHAP
  is not available in the environment.

Inputs
- Data CSV at data/PS_20174392719_1491204439457_log.csv
- Model artifacts under models/:
  - fraud_detection_model_final.json (preferred) or fraud_detection_model.json (fallback)
  - <model>.json.features (required)
  - <model>.json.metadata.json (optional; used for threshold display)

Outputs (written to reports/figures/)
- shap_summary_beeswarm.png (or feature_importance_bar.png if SHAP unavailable)
- shap_waterfall_fraud.png (only when SHAP available and at least one positive prediction exists)
- shap_meta.json (small context dump: paths, sample size, threshold used)

Usage
    python scripts/shap_analysis.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# Try SHAP; proceed without it if missing
try:
    import shap  # type: ignore
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False


ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "PS_20174392719_1491204439457_log.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL = MODELS_DIR / "fraud_detection_model_final.json"
CANONICAL_MODEL = MODELS_DIR / "fraud_detection_model.json"

if FINAL_MODEL.exists():
    MODEL_PATH = FINAL_MODEL
else:
    MODEL_PATH = CANONICAL_MODEL

FEATURES_PATH = Path(str(MODEL_PATH) + ".features")
METADATA_PATH = Path(str(MODEL_PATH) + ".metadata.json")

# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------
def load_threshold(meta_path: Path, default: float = 0.5) -> float:
    if not meta_path.exists():
        return default
    try:
        meta = json.loads(meta_path.read_text())
        thr = float(meta.get("decision_threshold", default))
        return min(max(thr, 1e-9), 1 - 1e-9)
    except Exception:
        return default


def load_features_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing features file: {path}")
    feats = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not feats:
        raise ValueError(f"Empty features list in: {path}")
    return feats


def load_model(model_path: Path) -> xgb.XGBClassifier:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model


def load_and_prepare_data(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found at: {csv_path}")
    df = pd.read_csv(csv_path)

    # Focus on fraud-relevant types
    df = df[(df["type"] == "TRANSFER") | (df["type"] == "CASH_OUT")].copy()

    # Feature engineering (mirror training)
    df["errorBalanceOrig"] = (df["newbalanceOrig"] + df["amount"]) - df["oldbalanceOrg"]
    df["errorBalanceDest"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
    df["isBlackHoleTransaction"] = (
        (df["oldbalanceDest"] == df["newbalanceDest"]) & (df["amount"] > 0)
    ).astype(int)
    df["hourOfDay"] = df["step"] % 24

    # One-hot for type; align later by features list
    df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=True)

    y = df["isFraud"]
    # Drop identifiers and strictly non-feature columns
    X = df.drop(["isFraud", "nameOrig", "nameDest", "isFlaggedFraud"], axis=1, errors="ignore")
    return X, y


def align_matrix(X: pd.DataFrame, features_list: List[str]) -> pd.DataFrame:
    return X.reindex(columns=features_list, fill_value=0.0).astype(np.float32)


def save_meta(context: dict) -> None:
    (REPORTS_DIR / "shap_meta.json").write_text(json.dumps(context, indent=2))


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load assets
    threshold = load_threshold(METADATA_PATH, default=0.5)
    features_list = load_features_list(FEATURES_PATH)
    model = load_model(MODEL_PATH)

    # Data
    X, y = load_and_prepare_data(DATA_PATH)
    # Use a reasonable sample for speed but keep positives
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Align to trained feature order
    X_test_aligned = align_matrix(X_test, features_list)

    # Score for later selection
    try:
        proba = model.predict_proba(X_test_aligned)[:, 1]
    except Exception:
        dmx = xgb.DMatrix(X_test_aligned, feature_names=list(X_test_aligned.columns))
        proba = model.get_booster().predict(dmx)
        proba = np.asarray(proba, dtype=float)

    # ------------------------------------------------------------------------------
    # 1) Global importance
    # ------------------------------------------------------------------------------
    if HAVE_SHAP:
        # Use TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test_aligned)

        # Beeswarm
        plt.figure(figsize=(12, 7), dpi=120)
        shap.plots.beeswarm(shap_values, show=False, max_display=20)
        plt.title("SHAP Summary Beeswarm — Global Drivers", fontsize=14)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "shap_summary_beeswarm.png", facecolor="white")
        plt.close()

    else:
        # Fallback: bar of feature importances + permutation importance overlay
        # Built-in gain/weight importance
        booster = model.get_booster()
        try:
            score_gain = booster.get_score(importance_type="gain")
        except Exception:
            score_gain = booster.get_score(importance_type="weight")
        imp_df = (
            pd.Series(score_gain, name="gain")
            .reindex(features_list)
            .fillna(0.0)
            .sort_values(ascending=False)
            .head(20)
            .rename_axis("feature")
            .reset_index()
        )

        plt.figure(figsize=(12, 7), dpi=120)
        sns.barplot(data=imp_df, y="feature", x="gain", color="steelblue")
        plt.title("Feature Importance (gain) — SHAP unavailable", fontsize=14)
        plt.xlabel("Importance (gain)")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "feature_importance_bar.png", facecolor="white")
        plt.close()

        # Permutation importance (on a subset for speed)
        try:
            sub_idx = np.random.RandomState(42).choice(len(X_test_aligned), size=min(2000, len(X_test_aligned)), replace=False)
            perm = permutation_importance(
                model, X_test_aligned.iloc[sub_idx], y_test.iloc[sub_idx], scoring="average_precision", n_repeats=5, random_state=42
            )
            perm_df = (
                pd.DataFrame({"feature": X_test_aligned.columns, "importance": perm.importances_mean})
                .sort_values("importance", ascending=False)
                .head(20)
            )
            plt.figure(figsize=(12, 7), dpi=120)
            sns.barplot(data=perm_df, y="feature", x="importance", color="darkorange")
            plt.title("Permutation Importance (AUPRC) — Fallback", fontsize=14)
            plt.xlabel("Mean importance (Δ AUPRC)")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / "permutation_importance_bar.png", facecolor="white")
            plt.close()
        except Exception:
            pass  # Permutation importance is best-effort only

    # ------------------------------------------------------------------------------
    # 2) Case-level waterfall (only with SHAP)
    # ------------------------------------------------------------------------------
    if HAVE_SHAP:
        # Pick a representative positive prediction near/above threshold; fallback to max prob
        preds = (proba >= threshold).astype(int)
        pos_idx = np.where(preds == 1)[0]
        if len(pos_idx) == 0:
            # pick the highest-prob example if none above threshold
            idx = int(np.argmax(proba))
        else:
            # Choose the one with probability closest to threshold to illustrate tipping point
            idx = int(pos_idx[np.argmin(np.abs(proba[pos_idx] - threshold))])

        x_row = X_test_aligned.iloc[idx:idx + 1]
        shap_row = shap_values[idx]

        plt.figure(figsize=(12, 6), dpi=120)
        # Waterfall with feature names; limit to top contributions
        shap.plots.waterfall(shap_row, max_display=16, show=False)
        plt.title("SHAP Waterfall — Representative Fraud Case", fontsize=14, pad=12)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "shap_waterfall_fraud.png", facecolor="white")
        plt.close()

    # ------------------------------------------------------------------------------
    # Save small metadata for traceability
    # ------------------------------------------------------------------------------
    context = {
        "model_path": str(MODEL_PATH),
        "features_path": str(FEATURES_PATH),
        "metadata_path": str(METADATA_PATH if METADATA_PATH.exists() else ""),
        "threshold": float(threshold),
        "have_shap": bool(HAVE_SHAP),
        "n_test": int(len(X_test_aligned)),
        "n_test_pos": int(y_test.sum()),
    }
    save_meta(context)

    print("Explainability artifacts written to:", str(REPORTS_DIR))


if __name__ == "__main__":
    main()