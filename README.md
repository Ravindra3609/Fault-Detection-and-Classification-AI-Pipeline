# FDC-AI — Equipment Fault Detection & Classification Pipeline

> **FDC-AI Fault Detection Pipeline** — Built end-to-end equipment fault detection system on NASA CMAPSS sensor degradation data. Implemented Isolation Forest + PyTorch Autoencoder (unsupervised anomaly detection) + XGBoost fault classifier with SHAP sensor attribution, deployed via FastAPI with live inference dashboard. Hybrid pipeline achieved F1 >0.90 on fault classification with interpretable per-sensor anomaly contributions.

**Tech:** `fault-detection` `FDC` `semiconductor` `anomaly-detection` `autoencoder` `xgboost` `shap` `fastapi` `pytorch` `cmapss` `predictive-maintenance`


<img width="1462" height="655" alt="Screenshot 2026-03-30 at 5 09 53 PM" src="https://github.com/user-attachments/assets/57d1baca-a66b-41d9-a7d2-a5530a1c2ad4" />
<img width="1462" height="641" alt="Screenshot 2026-03-30 at 5 10 10 PM" src="https://github.com/user-attachments/assets/60414c17-acc0-4956-ba74-a73bf4784829" />
<img width="1464" height="642" alt="Screenshot 2026-03-31 at 10 12 21 AM" src="https://github.com/user-attachments/assets/1cc57ed9-d487-4bba-9bef-a466a8959a07" />
<img width="1464" height="642" alt="Screenshot 2026-03-31 at 10 12 43 AM" src="https://github.com/user-attachments/assets/4d518a5c-0b26-4b1a-8b9c-c0b6070ada16" />
<img width="1464" height="642" alt="Screenshot 2026-03-31 at 10 12 53 AM" src="https://github.com/user-attachments/assets/a68a2b8b-1c3d-433e-9bcb-2e2a0bdcb35f" />
<img width="1106" height="738" alt="Screenshot 2026-03-31 at 10 13 51 AM" src="https://github.com/user-attachments/assets/5522601d-5de8-4e6f-907b-2c7a01fede6f" />




**End-to-end Fault Detection and Classification (FDC) system for semiconductor/industrial equipment**  
Isolation Forest + PyTorch Autoencoder + XGBoost + SHAP → FastAPI + Dashboard

---

## What this project does

| Layer | What | Why |
|---|---|---|
| **Data** | NASA CMAPSS turbofan degradation dataset | Direct analog to semiconductor equipment sensor streams |
| **Anomaly detection** | Isolation Forest (unsupervised) | No fault labels needed — learns normal behavior |
| **Deep anomaly detection** | PyTorch Autoencoder | Learns complex multivariate normal patterns, reconstruction error = anomaly score |
| **Fault classification** | XGBoost | Supervised fault/normal classification using augmented features |
| **Explainability** | SHAP TreeExplainer | Per-sensor attribution — tells engineers which sensors drove each alarm |
| **Deployment** | FastAPI + HTML dashboard | Production-ready API with live inference UI |

---

## Quick start — MacBook Air M4 (3 commands)

```bash
# 1. Place all project files in a folder, then:
cd ~/Desktop/fdc_ai

# 2. Make setup script executable
chmod +x setup_and_run.sh

# 3. Run — installs everything, trains models, launches server
bash setup_and_run.sh
```

**Open browser at: http://localhost:8000**  
**Swagger API docs at: http://localhost:8000/docs**

---

## Manual setup

```bash
cd ~/Desktop/fdc_ai

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train all models (~3–4 min on M4)
python train.py

# Start API server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Project structure

```
fdc_ai/
│
├── data_pipeline.py    Data layer
│                         - NASA CMAPSS loader + caching
│                         - Synthetic data generator (offline fallback)
│                         - Feature extraction (raw + rolling stats)
│                         - Fault labeling (last 30 cycles = fault zone)
│
├── autoencoder.py      PyTorch Autoencoder
│                         - Encoder: input → h1 → h2 → latent (16D)
│                         - Decoder: latent → h2 → h1 → input
│                         - Reconstruction error = anomaly score
│                         - Per-feature error → sensor attribution
│
├── fdc_pipeline.py     Full FDC-AI orchestration
│                         - IsolationForest (unsupervised baseline)
│                         - Autoencoder (deep anomaly detection)
│                         - XGBoost (supervised fault classifier)
│                         - SHAP (global + per-sample attribution)
│                         - save() / load() persistence
│
├── train.py            One-shot training script
│                         - Builds dataset → trains all models → saves
│
├── api.py              FastAPI backend
│                         - GET  /health
│                         - GET  /metrics   (F1, AUC-ROC, AUC-PR)
│                         - POST /predict   (single sample inference)
│                         - POST /batch     (batch inference)
│                         - GET  /importance (global SHAP)
│                         - GET  /demo      (random test sample)
│
├── static/index.html   Dark-theme dashboard UI
│                         - Live anomaly scores (ISO + AE + XGBoost)
│                         - SHAP waterfall bar chart
│                         - Global feature importance chart
│                         - Real-time prediction log
│
├── requirements.txt    Dependencies
├── setup_and_run.sh    One-shot macOS setup + launch
└── models/             Saved model files (auto-created by train.py)
    ├── iso_forest.pkl
    ├── autoencoder.pt
    ├── xgboost.pkl
    ├── shap_explainer.pkl
    ├── scaler.pkl
    └── meta.json
```

---

## API usage examples

### Run demo prediction
```bash
curl http://localhost:8000/demo
curl http://localhost:8000/demo?fault=true
curl http://localhost:8000/demo?fault=false
```

### Custom prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -0.3, 0.8, ...]}'
```

### Model metrics
```bash
curl http://localhost:8000/metrics
```

---

## How the three detection layers work together

```
Raw sensor data (21 sensors, rolling features)
         │
         ├──→ Isolation Forest ──→ iso_score (0–1)
         │                                        │
         ├──→ Autoencoder ──────→ ae_score  ──────┤
         │    (recon error)                        │
         │                                        ↓
         └──────────────────────→ XGBoost ──→ fault_proba (0–1)
                                    ↑              │
                              [iso_score,          ↓
                               ae_score       SHAP attribution
                               appended]      (which sensors?)
```

The key insight: XGBoost sees both the raw features AND the anomaly scores from the two unsupervised detectors. This hybrid approach outperforms either alone.

---

## CMAPSS dataset — what it is

NASA's Commercial Modular Aero-Propulsion System Simulation (CMAPSS):
- 100 turbofan engines (training set), each run to failure
- 21 sensor readings per cycle (temperatures, pressures, speeds, flow rates)
- 3 operating conditions (flight phase, altitude, Mach number)
- Label: Remaining Useful Life (RUL) per cycle

**Why it maps to semiconductor FDC:**
- Equipment sensor streams → fab tool sensor streams (pressure, RF power, gas flow)
- Engine degradation to failure → chamber degradation between PMs
- RUL prediction → PM scheduling / predictive maintenance
- Fault zone (last 30 cycles) → pre-failure warning window

---

## Troubleshooting

**Port 8000 already in use:**
```bash
uvicorn api:app --port 8001
```

**SHAP slow on first call:**
Normal — SHAP builds an internal tree structure on first call. Subsequent calls are fast.

**torch not found on M4:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Models not found error:**
Run `python train.py` first before starting the server.

---
