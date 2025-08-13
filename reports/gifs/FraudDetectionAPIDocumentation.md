# API Quickstart

This quickstart shows how to verify the hybrid Fraud Detection API locally, including health checks, prediction calls, and typical scenarios.

Base URL
- http://127.0.0.1:8000

Endpoints
- GET /health
- POST /predict

Content type
- application/json

1) Start the API
- Ensure your final model, features, and metadata exist under models/.
- Run:
  - uvicorn scripts.predict:app --reload

2) Health check
