```markdown
# API Test Scenarios & Demo Suite

This suite exercises the hybrid Fraud Detection API end-to-end. All requests are sent to POST `/predict` with a top-level `"payload"` object. The API applies the “black-hole” safety rule first and, if not triggered, uses the XGBoost model with the stored `decision_threshold`.

Base URL (local)
- http://127.0.0.1:8000

Endpoint
- POST /predict

Content-Type
- application/json

General request shape
```
json
{
  "payload": {
    "type": "TRANSFER | CASH_OUT",
    "amount": 0.0,
    "oldbalanceOrg": 0.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "step": 1
  }
}
```
Notes
- `step` is optional; if omitted, `hourOfDay` defaults to `0`.
- The service computes engineered fields (errorBalanceOrig, errorBalanceDest, hourOfDay) and type dummies internally.
- The “black-hole” rule triggers when `newbalanceDest == oldbalanceDest` and `amount > 0`.

---

## 1) Baseline: Legitimate Transactions

1.1 Standard CASH_OUT (arithmetically consistent)
```
json
{
  "payload": {
    "type": "CASH_OUT",
    "amount": 5200.50,
    "oldbalanceOrg": 78000,
    "newbalanceOrig": 72799.50,
    "oldbalanceDest": 25000,
    "newbalanceDest": 30200.50,
    "step": 12
  }
}
```
Expected: `fraud=false`, `model_used=true`

1.2 Large, legitimate TRANSFER (account drain but consistent)
```
json
{
  "payload": {
    "type": "TRANSFER",
    "amount": 500000,
    "oldbalanceOrg": 500000,
    "newbalanceOrig": 0,
    "oldbalanceDest": 25000,
    "newbalanceDest": 525000,
    "step": 3
  }
}
```
Expected: `fraud=false`, `model_used=true`

---

## 2) Rule-Based Safety Net

2.1 Black-hole TRANSFER (recipient unchanged; rule must override)
```
json
{
  "payload": {
    "type": "TRANSFER",
    "amount": 1500,
    "oldbalanceOrg": 20000,
    "newbalanceOrig": 18500,
    "oldbalanceDest": 100,
    "newbalanceDest": 100,
    "step": 5
  }
}
```
Expected: `fraud=true`, `reason="black_hole_rule"`, `model_used=false`

2.2 Black-hole CASH_OUT (also caught by rule)
```
json
{
  "payload": {
    "type": "CASH_OUT",
    "amount": 250,
    "oldbalanceOrg": 1500,
    "newbalanceOrig": 1250,
    "oldbalanceDest": 0,
    "newbalanceDest": 0,
    "step": 7
  }
}
```
Expected: `fraud=true`, `reason="black_hole_rule"`, `model_used=false`

---

## 3) Model-Detected Fraud Patterns

3.1 Anomalous zero-balance CASH_OUT (drain to/from zero; suspicious)
```
json
{
  "payload": {
    "type": "CASH_OUT",
    "amount": 10000,
    "oldbalanceOrg": 10000,
    "newbalanceOrig": 0,
    "oldbalanceDest": 0,
    "newbalanceDest": 0,
    "step": 9
  }
}
```
Expected: `fraud=true`, `model_used=true`

3.2 “Probing” fraud attempt (very small amount to zero-balance dest)
```
json
{
  "payload": {
    "type": "CASH_OUT",
    "amount": 1.00,
    "oldbalanceOrg": 100.00,
    "newbalanceOrig": 99.00,
    "oldbalanceDest": 0,
    "newbalanceDest": 0,
    "step": 10
  }
}
```
Expected: `fraud=true`, `model_used=true`

---

## 4) Near-Threshold Cases (calibration checks)

4.1 Slightly suspicious but likely legitimate
```
json
{
  "payload": {
    "type": "TRANSFER",
    "amount": 120.00,
    "oldbalanceOrg": 1020.00,
    "newbalanceOrig": 900.00,
    "oldbalanceDest": 300.00,
    "newbalanceDest": 420.00,
    "step": 14
  }
}
```
Expected: `fraud` depends on `decision_threshold`; often `false`

4.2 Slightly high-risk combination (borderline)
```
json
{
  "payload": {
    "type": "CASH_OUT",
    "amount": 250.00,
    "oldbalanceOrg": 260.00,
    "newbalanceOrig": 10.00,
    "oldbalanceDest": 0.00,
    "newbalanceDest": 10.00,
    "step": 18
  }
}
```
Expected: `fraud` may be `true` if `model_prob >= decision_threshold`

---

## 5) Robustness & Input Validation

5.1 Negative amount (invalid)
```
json
{
  "payload": {
    "type": "CASH_OUT",
    "amount": -100,
    "oldbalanceOrg": 1000,
    "newbalanceOrig": 1100,
    "oldbalanceDest": 50,
    "newbalanceDest": 50,
    "step": 2
  }
}
```
Expected: `422 Unprocessable Entity` (or `400`), clear validation error

5.2 Missing optional step (should default hourOfDay safely)
```
json
{
  "payload": {
    "type": "TRANSFER",
    "amount": 300,
    "oldbalanceOrg": 2000,
    "newbalanceOrig": 1700,
    "oldbalanceDest": 500,
    "newbalanceDest": 800
  }
}
```
Expected: fraud result consistent; `model_used` depends on pattern

5.3 Non-numeric strings (invalid types)
```
json
{
  "payload": {
    "type": "CASH_OUT",
    "amount": "one hundred",
    "oldbalanceOrg": 1000,
    "newbalanceOrig": 900,
    "oldbalanceDest": 200,
    "newbalanceDest": 300,
    "step": 4
  }
}
```
Expected: `422/400` validation error

---

## 6) Edge Conditions and Stress

6.1 Very large amount but consistent balances
```
json
{
  "payload": {
    "type": "TRANSFER",
    "amount": 2500000,
    "oldbalanceOrg": 5000000,
    "newbalanceOrig": 2500000,
    "oldbalanceDest": 1000000,
    "newbalanceDest": 3500000,
    "step": 21
  }
}
```
Expected: `fraud=false` if arithmetic is consistent

6.2 Micro-amount with tiny origin balance
```
json
{
  "payload": {
    "type": "CASH_OUT",
    "amount": 0.01,
    "oldbalanceOrg": 0.05,
    "newbalanceOrig": 0.04,
    "oldbalanceDest": 0.00,
    "newbalanceDest": 0.01,
    "step": 22
  }
}
```
Expected: borderline; depends on calibration

6.3 Same balances (no movement) with zero amount (should not trigger rule)
```
json
{
  "payload": {
    "type": "TRANSFER",
    "amount": 0,
    "oldbalanceOrg": 1000,
    "newbalanceOrig": 1000,
    "oldbalanceDest": 500,
    "newbalanceDest": 500,
    "step": 6
  }
}
```
Expected: `fraud=false` (rule requires `amount > 0`)

---

## 7) Black-Hole Variations (safety net regression tests)

7.1 Recipient unchanged, non-zero amount (classic rule hit)
```
json
{
  "payload": {
    "type": "TRANSFER",
    "amount": 75,
    "oldbalanceOrg": 1000,
    "newbalanceOrig": 925,
    "oldbalanceDest": 200,
    "newbalanceDest": 200,
    "step": 8
  }
}
```
Expected: `fraud=true`, `reason="black_hole_rule"`

7.2 Recipient unchanged, tiny amount (still a rule hit)
```
json
{
  "payload": {
    "type": "CASH_OUT",
    "amount": 0.49,
    "oldbalanceOrg": 10,
    "newbalanceOrig": 9.51,
    "oldbalanceDest": 55,
    "newbalanceDest": 55,
    "step": 13
  }
}
```
Expected: `fraud=true`, `reason="black_hole_rule"`

---

## 8) cURL Examples

Send a request (replace host/port if needed):
```
bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"payload":{"type":"TRANSFER","amount":1500,"oldbalanceOrg":20000,"newbalanceOrig":18500,"oldbalanceDest":100,"newbalanceDest":100,"step":5}}'
```
Expected response (example):
```
json
{
  "fraud": true,
  "reason": "black_hole_rule",
  "model_used": false,
  "model_prob": null,
  "decision_threshold": 0.95
}
```
---

## 9) Acceptance Checklist

- Black-hole scenarios always return `fraud=true` with `reason="black_hole_rule"`.
- Legitimate scenarios with consistent balances are `fraud=false`.
- Near-threshold cases behave consistently with the stored `decision_threshold`.
- Invalid inputs produce clear 4xx errors (no crashes).
- API returns `decision_threshold` where `model_used=true`, confirming metadata was loaded.

---

## 10) Troubleshooting

- Rule cases don’t override:
  - Ensure request is wrapped in `"payload"`.
  - Verify `newbalanceDest == oldbalanceDest` and `amount > 0`.
- Startup/load issues:
  - Confirm the model, `.features`, and `.metadata.json` exist under `models/`.
- Decisions seem off:
  - Verify the deployed metadata contains `decision_threshold` and the API is reading it.
  - Re-run training to refresh metadata, then restart the API.
```
