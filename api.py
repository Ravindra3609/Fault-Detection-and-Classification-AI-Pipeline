"""
api.py
======
FastAPI backend for FDC-AI inference.

Endpoints:
  GET  /              → serves the UI (index.html)
  GET  /health        → health check + model status
  GET  /metrics       → model performance metrics
  POST /predict       → single sample fault prediction + SHAP
  POST /batch         → batch prediction
  GET  /importance    → global feature importance
  GET  /demo          → run prediction on a random test sample
"""

import numpy as np
import joblib
import json
import random
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fdc_pipeline  import FDCPipeline
from data_pipeline import build_dataset, USEFUL_SENSORS

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FDC-AI API",
    description="Fault Detection & Classification for semiconductor equipment using Isolation Forest + Autoencoder + XGBoost + SHAP",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Load models at startup ────────────────────────────────────────────────────
MODELS_DIR = Path("models")
pipeline: Optional[FDCPipeline] = None
scaler = None
test_samples = None      # cache test samples for /demo endpoint


@app.on_event("startup")
async def load_models():
    global pipeline, scaler, test_samples
    try:
        pipeline = FDCPipeline.load(MODELS_DIR)
        scaler   = joblib.load(MODELS_DIR / "scaler.pkl")

        # Cache a few test samples for /demo
        dataset = build_dataset()
        test_samples = {
            "X_test":  dataset["X_test"],
            "y_test":  dataset["y_test"],
        }
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load models — {e}")
        print("Run 'python train.py' first to train models.")


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: List[float]   # scaled feature vector (len = input_dim)

class BatchRequest(BaseModel):
    samples: List[List[float]]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI."""
    html_path = Path("index.html")
    if html_path.exists():
        return html_path.read_text()
    return "<h1>FDC-AI API</h1><p>Visit /docs for API documentation.</p>"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": pipeline is not None,
        "model_dir": str(MODELS_DIR),
        "endpoints": ["/health", "/metrics", "/predict", "/batch", "/importance", "/demo"],
    }


@app.get("/metrics")
async def get_metrics():
    if pipeline is None:
        raise HTTPException(503, "Models not loaded. Run python train.py first.")
    m = pipeline.metrics.copy()
    m.pop("global_importance", None)
    return {"metrics": m}


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    Run full FDC inference on one scaled feature vector.
    Returns: iso_score, ae_score, fault_proba, SHAP attribution.
    """
    if pipeline is None:
        raise HTTPException(503, "Models not loaded. Run python train.py first.")

    x = np.array(req.features, dtype=np.float32)
    if len(x) != len(pipeline.feature_names):
        raise HTTPException(
            400,
            f"Expected {len(pipeline.feature_names)} features, got {len(x)}."
        )

    result = pipeline.predict_single(x)
    return result


@app.post("/batch")
async def batch_predict(req: BatchRequest):
    """Batch prediction — returns list of results."""
    if pipeline is None:
        raise HTTPException(503, "Models not loaded.")

    results = []
    for sample in req.samples:
        x = np.array(sample, dtype=np.float32)
        if len(x) != len(pipeline.feature_names):
            results.append({"error": f"Wrong feature count: {len(x)}"})
        else:
            results.append(pipeline.predict_single(x))
    return {"predictions": results, "count": len(results)}


@app.get("/importance")
async def feature_importance():
    """Global feature importance from SHAP."""
    if pipeline is None:
        raise HTTPException(503, "Models not loaded.")

    if "global_importance" in pipeline.metrics:
        return pipeline.metrics["global_importance"]

    # Recompute if not cached
    if test_samples is None:
        raise HTTPException(503, "Test data not available.")

    from fdc_pipeline import global_feature_importance, isolation_forest_scores, build_classifier_features
    from autoencoder  import anomaly_scores

    X = test_samples["X_test"]
    iso_s = isolation_forest_scores(pipeline.iso, X)
    ae_s  = anomaly_scores(pipeline.ae_model, X)
    X_aug = build_classifier_features(X, iso_s, ae_s)
    imp = global_feature_importance(pipeline.explainer, X_aug, pipeline.aug_feature_names)
    return imp


@app.get("/demo")
async def demo(fault: Optional[bool] = None):
    """
    Run prediction on a random test sample.
    fault=true  → pick a known fault sample
    fault=false → pick a normal sample
    fault=None  → random
    """
    if pipeline is None or test_samples is None:
        raise HTTPException(503, "Models not loaded. Run python train.py first.")

    X = test_samples["X_test"]
    y = test_samples["y_test"]

    if fault is True:
        indices = np.where(y == 1)[0]
    elif fault is False:
        indices = np.where(y == 0)[0]
    else:
        indices = np.arange(len(y))

    if len(indices) == 0:
        raise HTTPException(404, "No samples matching filter.")

    idx = random.choice(indices.tolist())
    x = X[idx]
    true_label = int(y[idx])

    result = pipeline.predict_single(x)
    result["true_label"] = true_label
    result["true_label_str"] = "FAULT" if true_label == 1 else "NORMAL"
    result["sample_index"] = idx
    return result
