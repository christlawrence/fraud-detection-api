# scripts/train.py

# This script trains the final fraud detection model, saves the model + feature list and exports evaluation artifacts (confusion matrices, ROC/PR curves, metrics JSON, SHAP summary).

import os
import json
import inspect
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import xgboost as xgb

print("Starting the model training script...")
print(f"xgboost version: {getattr(xgb, '__version__', 'unknown')}")

# --- Paths---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 1. Load Data ==================================================
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "PS_20174392719_1491204439457_log.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully.")

# 2. Final Feature Engineering & Preprocessing ==================================================
print("Performing final, comprehensive feature engineering...")

# Focus on types where fraud occurs
df_filtered = df[(df["type"] == "TRANSFER") | (df["type"] == "CASH_OUT")].copy()

# Engineered features
df_filtered["errorBalanceOrig"] = (df_filtered["newbalanceOrig"] + df_filtered["amount"]) - df_filtered["oldbalanceOrg"]
df_filtered["errorBalanceDest"] = (df_filtered["oldbalanceDest"] + df_filtered["amount"]) - df_filtered["newbalanceDest"]
df_filtered["isBlackHoleTransaction"] = (
    (df_filtered['oldbalanceDest'] == df_filtered['newbalanceDest']) &
    (df_filtered['amount'] > 0)
).astype(int)
# Temporal feature to align with analysis/visuals
df_filtered["hourOfDay"] = df_filtered["step"] % 24

# One-hot encode type
df_filtered = pd.get_dummies(df_filtered, columns=["type"], prefix="type", drop_first=True)

# Target and features
y = df_filtered["isFraud"]
# Exclude identifiers and flags; keep all numeric signals used in analysis
X = df_filtered.drop(["isFraud", "nameOrig", "nameDest", "isFlaggedFraud"], axis=1)

# 3. Train/Val/Test Split ==================================================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
# Small validation set for early stopping
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train_full
)

print(f"Data split into {len(X_train_full)} training+validation and {len(X_test)} testing samples.")
print(f" - Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

# 4. Model Training w/ Tuned Hyperparameters + Early Stopping ===================
print("Training the final XGBoost model with tuned hyperparameters...")
# Imbalance handling
pos = y_train.value_counts().get(1, 1)
neg = y_train.value_counts().get(0, 1)
scale_pos_weight = max(1.0, neg / max(1, pos))

model = xgb.XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    seed=RANDOM_STATE,
    tree_method="hist",
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
)

# Determine whether the sklearn-style fit supports 'callbacks' in this environment
_fit_sig = inspect.signature(xgb.XGBClassifier.fit)
supports_callbacks = "callbacks" in _fit_sig.parameters
# Also detect whether early_stopping_rounds param exists on fit (some versions do)
supports_early_stopping_rounds = "early_stopping_rounds" in _fit_sig.parameters

print(f"XGBClassifier.fit supports 'callbacks': {supports_callbacks}")
print(f"XGBClassifier.fit supports 'early_stopping_rounds': {supports_early_stopping_rounds}")

# Prepare an early stopping callback if available in this xgboost build
early_stop_cb = None
try:
    early_stop_cb = xgb.callback.EarlyStopping(rounds=30, save_best=True, min_delta=0.0)
except Exception:
    early_stop_cb = None

def _train_with_xgb_train(params, X_tr, y_tr, X_val_, y_val_, num_boost_round=300, early_rounds=30):
    """
    Fallback training using the low-level xgb.train API which reliably supports early_stopping_rounds
    across builds. After training, save the booster to a temporary file and load it into an XGBClassifier
    so the remainder of the code can interact with a sklearn-style object.
    """
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val_, label=y_val_)
    evals = [(dval, "validation")]
    # xgb.train expects parameters using booster style (no sklearn wrapper params like n_estimators)
    xgb_params = dict(params)
    # Remove sklearn-only args if present
    xgb_params.pop("n_estimators", None)
    xgb_params.pop("seed", None)
    xgb_params.pop("verbosity", None)
    # Use training API
    booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_rounds,
        verbose_eval=False,
    )
    # Save booster to temp file and load into sklearn wrapper
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xgb")
    tmp.close()
    booster.save_model(tmp.name)
    skl_model = xgb.XGBClassifier(**{
        "objective": params.get("objective", "binary:logistic"),
        "tree_method": params.get("tree_method", "auto"),
    })
    # load_model will populate the underlying Booster
    skl_model.load_model(tmp.name)
    os.unlink(tmp.name)
    return skl_model, booster.best_iteration if hasattr(booster, "best_iteration") else None

# Choose training path
trained_model = None
best_ntree = None

if supports_callbacks and early_stop_cb is not None:
    print("Using callback-based early stopping.")
    # Call sklearn-style fit with callbacks
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stop_cb],
            verbose=False,
        )
        trained_model = model
    except Exception as e:
        print(f"Callback-based fit failed with error: {e}. Falling back to xgb.train approach.")
        trained_model, best_ntree = _train_with_xgb_train(
            params=model.get_xgb_params(), X_tr=X_train, y_tr=y_train, X_val_=X_val, y_val_=y_val,
            num_boost_round=model.get_xgb_params().get("n_estimators", 300), early_rounds=30
        )
elif supports_early_stopping_rounds:
    print("Using early_stopping_rounds param on sklearn fit.")
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False,
        )
        trained_model = model
    except TypeError as e:
        # Some builds may claim to support early_stopping_rounds in signature but still reject; fall back
        print(f"sklearn.fit with early_stopping_rounds raised TypeError: {e}. Falling back to xgb.train.")
        trained_model, best_ntree = _train_with_xgb_train(
            params=model.get_xgb_params(), X_tr=X_train, y_tr=y_train, X_val_=X_val, y_val_=y_val,
            num_boost_round=model.get_xgb_params().get("n_estimators", 300), early_rounds=30
        )
else:
    # Last resort: use xgb.train to guarantee early stopping support
    print("Using xgb.train fallback for early stopping.")
    trained_model, best_ntree = _train_with_xgb_train(
        params=model.get_xgb_params(), X_tr=X_train, y_tr=y_train, X_val_=X_val, y_val_=y_val,
        num_boost_round=model.get_xgb_params().get("n_estimators", 300), early_rounds=30
    )

if trained_model is None:
    # If something went wrong above, fall back to no-early-stopping sklearn fit to avoid crash (warn the user).
    print("Warning: Could not run early stopping. Training without early stopping (last resort).")
    model.fit(X_train, y_train)
    trained_model = model

model = trained_model
print("Model training complete.")

# Try multiple attribute names for compatibility across xgboost versions
if best_ntree is None:
    best_ntree = getattr(model, "best_iteration", None)
if best_ntree is None:
    best_ntree = getattr(model, "best_iteration_", None)
if best_ntree is None:
    # Older versions expose best_ntree_limit; convert to 0-based iteration for logging if available
    btnl = getattr(model, "best_ntree_limit", None)
    if isinstance(btnl, (int, np.integer)) and btnl > 0:
        best_ntree = int(btnl - 1)
# As a last resort, try booster attribute if present
if best_ntree is None:
    try:
        booster = model.get_booster()
        if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
            best_ntree = int(booster.best_iteration)
    except Exception:
        pass

if best_ntree is not None:
    print(f"Best iteration (0-based): {best_ntree}")

# Helper predict functions to respect best_ntree_limit on older versions
def _predict_with_best(m, X_):
    # Newer sklearn API ignores ntree_limit so this will be a safe call
    btnl = getattr(m, "best_ntree_limit", None)
    if isinstance(btnl, (int, np.integer)) and btnl > 0:
        try:
            return m.predict(X_, ntree_limit=int(btnl))
        except TypeError:
            # Some builds may not accept ntree_limit param on predict; fall back
            return m.predict(X_)
    return m.predict(X_)

def _predict_proba_with_best(m, X_):
    btnl = getattr(m, "best_ntree_limit", None)
    if isinstance(btnl, (int, np.integer)) and btnl > 0:
        try:
            return m.predict_proba(X_, ntree_limit=int(btnl))
        except TypeError:
            return m.predict_proba(X_)
    return m.predict_proba(X_)

# 5. Save the Model + Feature List ============================================
model_path = os.path.join(MODELS_DIR, "fraud_detection_model.xgb")
# save_model will work because model is an XGBClassifier with a loaded booster
model.save_model(model_path)
# Persist feature list for downstream scripts
features_path = model_path + ".features"
with open(features_path, "w") as f:
    f.write("\n".join(list(X_train.columns)))
print(f"Model saved to {model_path}")
print(f"Feature list saved to {features_path}")

# 6. Model Evaluation ==================================================
print("Evaluating the model...")
y_pred = _predict_with_best(model, X_test)
y_pred_proba = _predict_proba_with_best(model, X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

# Metrics summary (also saved as JSON)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
accuracy = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auroc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
auprc = auc(rec, prec)

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "auroc": float(auroc),
    "auprc": float(auprc),
    "best_iteration": int(best_ntree) if best_ntree is not None else None,
    "scale_pos_weight": float(scale_pos_weight),
}
metrics_path = os.path.join(REPORTS_DIR, "metrics_summary.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics JSON saved to {metrics_path}")

# Confusion Matrix (raw and normalized)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"])
plt.title("Confusion Matrix", fontsize=16)
plt.ylabel("Actual")
plt.xlabel("Predicted")
cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
print(f"Confusion matrix saved to {cm_path}")
plt.close()

cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"])
plt.title("Normalized Confusion Matrix", fontsize=16)
plt.ylabel("Actual")
plt.xlabel("Predicted")
cmn_path = os.path.join(REPORTS_DIR, "confusion_matrix_normalized.png")
plt.savefig(cmn_path)
print(f"Normalized confusion matrix saved to {cmn_path}")
plt.close()

# ROC Curve
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUROC = {auroc:.4f}")
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.title("ROC Curve", fontsize=16)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
roc_curve_path = os.path.join(REPORTS_DIR, "roc_curve.png")
plt.savefig(roc_curve_path)
print(f"ROC curve saved to {roc_curve_path}")
plt.close()

# Precision-Recall Curve
plt.figure(figsize=(10, 7))
plt.plot(rec, prec, color="blue", label=f"AUPRC = {auprc:.4f}")
plt.title("Precision-Recall Curve", fontsize=16)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
pr_curve_path = os.path.join(REPORTS_DIR, "precision_recall_curve.png")
plt.savefig(pr_curve_path)
print(f"Precision-Recall curve saved to {pr_curve_path}")
plt.close()

# Optional: choose a decision threshold to maximize F1 on validation and save it
val_proba = _predict_proba_with_best(model, X_val)[:, 1]
val_prec, val_rec, val_thresh = precision_recall_curve(y_val, val_proba)
# Compute F1 for each threshold (avoid division by zero)
val_f1 = np.where((val_prec + val_rec) > 0, 2 * (val_prec * val_rec) / (val_prec + val_rec), 0.0)
best_idx = int(np.argmax(val_f1))
optimal_threshold = float(val_thresh[max(0, best_idx - 1)]) if len(val_thresh) > 0 else 0.5
with open(os.path.join(MODELS_DIR, "decision_threshold.txt"), "w") as f:
    f.write(str(optimal_threshold))
print(f"Saved suggested decision threshold (max F1 on validation): {optimal_threshold:.4f}")

# SHAP Summary (global importance) on a small sample for speed
try:
    import shap
    shap.sample = 500  # small sample if very large
    sample_n = min(1000, len(X_test))
    X_shap = X_test.sample(n=sample_n, random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X_shap)
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_vals, show=False, max_display=15)
    shap_path = os.path.join(REPORTS_DIR, "shap_summary_beeswarm.png")
    plt.title("SHAP Summary (Beeswarm): Top Features")
    plt.savefig(shap_path, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary beeswarm saved to {shap_path}")
except Exception as e:
    print(f"Skipping SHAP summary due to error: {e}")

print("\n--- Train Script Finished ---")
