# cv_evaluate.py
"""
Stratified k-fold evaluation for imbalanced metrics. Reports mean/std for:
- Fraud precision, recall, f1
- AUROC, AUPRC
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import xgboost as xgb

DATA_PATH = Path("data/PS_20174392719_1491204439457_log.csv")
df = pd.read_csv(DATA_PATH)
df = df[(df["type"] == "TRANSFER") | (df["type"] == "CASH_OUT")].copy()
df["errorBalanceOrig"] = (df["newbalanceOrig"] + df["amount"]) - df["oldbalanceOrg"]
df["errorBalanceDest"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
df["isBlackHoleTransaction"] = ((df['oldbalanceDest'] == df['newbalanceDest']) & (df['amount'] > 0)).astype(int)
df["hourOfDay"] = df["step"] % 24
df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=True)

y = df["isFraud"].values
X = df.drop(["isFraud", "nameOrig", "nameDest", "isFlaggedFraud"], axis=1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {"precision": [], "recall": [], "f1": [], "auroc": [], "auprc": []}

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight = max(1.0, neg / max(1, pos))

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        n_estimators=200,
        verbosity=0,
        seed=42,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    results["precision"].append(precision_score(y_test, pred, zero_division=0))
    results["recall"].append(recall_score(y_test, pred, zero_division=0))
    results["f1"].append(f1_score(y_test, pred, zero_division=0))
    results["auroc"].append(roc_auc_score(y_test, proba))
    results["auprc"].append(average_precision_score(y_test, proba))

def summarize(res):
    for k, v in res.items():
        arr = np.array(v)
        print(f"{k:6s} mean={arr.mean():.4f} std={arr.std():.4f}")

if __name__ == "__main__":
    summarize(results)