# Heart Rate Abnormality Detector (Wearables): SVM + 1D-CNN + Web UI

**Educational prototype — not a medical device.**

## Quickstart

```bash
# 1) setup
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) train tiny synthetic models (creates ./models/*.joblib and ./models/cnn.pt)
python scripts/train_synthetic.py

# 3) run API
uvicorn api.main:app --reload --port 8000

# 4) open the UI (points to http://localhost:8000)
# double-click web/index.html# hr-abnormal-detector

![Heart Rate Abnormality Detector Screenshot](https://github.com/user-attachments/assets/d3a41bea-64ad-453b-94a5-f024782402fe)
