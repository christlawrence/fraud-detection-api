# Fraud Detection System Runbook

This comprehensive runbook provides end-to-end instructions for training, evaluating, tuning, visualizing, and deploying the fraud detection model used in this project.

## Prerequisites

- Python 3.12.x
- A virtual environment with packages installed from requirements.txt
- Project root with proper directory structure as outlined in README.md

## Environment Setup

### Quick Setup (One-Time)

1. **Create and activate a virtual environment**:
   - macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify directory structure**:
   ```bash
   mkdir -p data models reports/figures reports/gifs reports/gifs/_frames
   ```

## Model Training and Evaluation

### 1. Baseline / Improved Training

- **Purpose**: Produce canonical model artifact and metrics
- **Command**:
  ```bash
  python scripts/train_improved.py
  ```
- **Output**:
  - `models/fraud_detection_model.json`
  - `reports/figures/metrics_summary.json`
  - `reports/figures/shap_summary_beeswarm.png`
  - `reports/figures/confusion_matrix.png`
  - `reports/figures/precision_recall_curve.png`
  - `reports/figures/roc_curve.png`
- **Verification**: Confirm script prints training path, best iteration, and saved paths

### 2. Sanity Metrics Report

- **Purpose**: Generate human-friendly metric checks
- **Command**:
  ```bash
  python scripts/metrics_report.py
  ```
- **Verification**: Inspect `reports/figures/metrics_summary.json` for:
  - AUROC (should be > 0.95)
  - AUPRC (should be > 0.90)
  - Precision/Recall/F1 (F1 should be > 0.85)
  - Best iteration number (should be < 200)

### 3. Stability Evaluation (Stratified Cross-Validation)

- **Purpose**: Estimate mean/std of fraud metrics across folds
- **Command**:
  ```bash
  python scripts/cv_evaluate.py
  ```
- **Verification**:
  - Check mean/std results printed to console
  - Confirm standard deviation of AUPRC across folds is < 0.05 (stable model)
  - Verify no folds show significantly degraded performance

### 4. Hyperparameter Search (Optional, Compute-Intensive)

- **Purpose**: Find optimal hyperparameters for maximizing AUPRC
- **Command**:
  ```bash
  python scripts/hyperparam_search.py --trials 24 --early_rounds 50
  ```
- **Output**: `models/hp_search/` (meta and model files)
- **Next steps**:
  - Option A: Copy best model to `models/fraud_detection_model_tuned.json`
  - Option B: Re-run `train_improved.py` with the chosen hyperparameters
  - Verify improvement over baseline metrics (AUPRC should increase by at least 0.01)

### 5. Final Holdout Evaluation

- **Purpose**: Unbiased evaluation on the reserved test split
- **Command** (only needed if hyperparameters were tuned):
  ```bash
  python scripts/train.py
  ```
- **Verification**: Confirm `metrics_summary.json` updated with final metrics:
  - AUROC/AUPRC should not degrade from validation metrics
  - No signs of overfitting (similar performance on validation and test)

## Explainability and Visualization

### 6. SHAP-Driven Failure Analysis

- **Purpose**: Inspect false positives and false negatives to improve model
- **Command**:
  ```bash
  python scripts/shap_analysis.py
  ```
- **Output**:
  - `reports/figures/shap_waterfall_fraud.png`
  - `reports/figures/shap_waterfall_legit.png`
  - Additional SHAP visualizations
- **Analysis**:
  - Inspect plots to identify common patterns in misclassifications
  - Look for feature importance discrepancies between correct and incorrect predictions
  - Consider feature engineering improvements based on findings

### 7. Generate Visualization Suite

- **Purpose**: Create comprehensive visual analysis artifacts
- **Command**:
  ```bash
  python scripts/gifs_gen.py
  ```
- **Output**:
  - `reports/gifs/1_fraud_contagion.gif`
  - `reports/gifs/3_3D_fraud_clusters.gif`
  - `reports/gifs/4_risk_escalation.gif`
  - `reports/gifs/5_animated_amount_drift.gif`
  - `reports/gifs/6_animated_balance_drift.gif`
  - `reports/gifs/9_temporal_anomalies.gif`
  - `reports/gifs/9_animated_temporal_anomalies.gif`
- **Verification**:
  - Confirm all GIFs are generated correctly
  - Visually inspect each for anomalies or unexpected patterns
  - Use findings to inform potential model improvements

## Deployment Preparation

### 8. Inference Smoke Test (Hybrid Rule + Model)

- **Purpose**: Validate the safety-net rule and model together
- **Command**:
  ```bash
  python scripts/inference_smoke_test.py
  ```
- **Expected results**:
  - Black-hole case: `rule_flag = 1` and `final_pred = 1` regardless of model probability
  - Normal cases: Rule bypassed, model prediction used
  - Verify feature ordering is consistent with training
  - Check prediction response time (should be < 100ms)

### 9. Calibration & Threshold Optimization

- **Purpose**: Adjust threshold for production according to business cost
- **Command**:
  ```bash
  python scripts/select_threshold.py
  ```
- **Actions**:
  - Use validation folds to calibrate probabilities (CalibratedClassifierCV)
  - Choose threshold by maximizing business metric (precision@recall or cost-based)
  - Default is to maximize F1 score, but can be customized
- **Output**: Updated decision threshold in `models/decision_threshold.txt`

### 10. Create Final Deployment Artifacts

- **Purpose**: Package all required model files for production
- **Command**:
  ```bash
  python scripts/package_model.py
  ```
- **Output**:
  - `models/fraud_detection_model_final.json`
  - `models/fraud_detection_model_final.json.features`
  - `models/fraud_detection_model_final.metadata.json`
- **Verification**:
  - Confirm metadata includes:
    - Hyperparameters
    - Random seed
    - Best iteration
    - Training method
    - AUPRC/AUROC
    - Decision threshold
    - Creation timestamp

### 11. API Testing and Validation

- **Purpose**: Ensure the FastAPI prediction endpoint works correctly
- **Setup**:
  ```bash
  uvicorn scripts.predict:app --host 0.0.0.0 --port 8000
  ```
- **Test endpoints**:
  - `/health`: Should return {"status": "ok", "model_loaded": true}
  - `/predict`: Test with examples from `API_Demo_Cases.md`
- **Validation**:
  - Black-hole rule correctly identifies risky transactions
  - Model scoring properly applied to non-rule cases
  - Responses are fast (< 100ms)
  - Error handling works correctly for malformed inputs

## Production Monitoring and Maintenance

### 12. Production Monitoring Checklist

- **Logging**:
  - Configure to log predictions & raw inputs (redact PII)
  - Ensure rotation policy for log files
  - Verify structured logging format

- **Monitoring setup**:
  - Feature distributions (drift detection)
  - Prediction score distribution
  - Incoming fraud labels / feedback loop
  - Response time and service availability
  - Resource utilization (CPU, memory)

- **Alerting**:
  - Configure alerts for distribution drift beyond thresholds
  - Set up performance degradation alerts
  - Create API availability monitoring

### 13. Scheduled Maintenance

- **Regular retraining**:
  - Schedule monthly retraining with fresh data
  - Implement automated A/B testing for new models
  - Maintain version history of all deployed models

- **Drift analysis**:
  - Run `scripts/drift_analysis.py` weekly
  - Compare production distributions to training
  - Trigger retraining when drift exceeds thresholds

## Troubleshooting

### Common Issues and Solutions

- **XGBoost fitting errors**:
  - If early_stopping_rounds or callbacks cause errors, the scripts will automatically fall back to xgb.train approach
  - Verify XGBoost version compatibility with your Python version

- **SHAP errors for large inputs**:
  - Sample the test set (e.g., 500–1000 rows)
  - Use `sample_n = min(1000, len(X_test))` as in train.py

- **Memory issues during visualization**:
  - Reduce DPI in Config class (currently 150)
  - Lower the NUM_FRAMES constant for animations
  - Use a smaller test sample

- **API performance issues**:
  - Check for resource contention
  - Verify model is being loaded correctly at startup
  - Confirm feature list matches training features exactly

### Maintaining Reproducibility

- Always save metadata with each model:
  - Hyperparameters
  - Random seed
  - Training timestamp
  - Dataset version (e.g., CSV checksum)
  - Environment information

- Version control:
  - Tag model artifacts in version control
  - Store artifacts in a dedicated registry
  - Document all non-default parameter choices

### Support

If any script fails:
1. Capture the full console output
2. Note the exact command and environment
3. Submit to the team issue tracker or contact the model owner
4. Include information about input data and any recent changes
