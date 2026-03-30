"""
train.py
========
One-shot training script.
Run: python train.py
Trains Isolation Forest + Autoencoder + XGBoost + SHAP.
Saves all models to /models directory.
"""

import numpy as np
import json
from pathlib import Path

from data_pipeline import build_dataset
from fdc_pipeline  import FDCPipeline
import joblib

def main():
    print("=" * 60)
    print("  FDC-AI Pipeline — Training")
    print("=" * 60)

    # ── 1. Build dataset ──────────────────────────────────────────
    dataset = build_dataset()

    X_train = dataset["X_train"]
    X_test  = dataset["X_test"]
    y_train = dataset["y_train"]
    y_test  = dataset["y_test"]
    scaler  = dataset["scaler"]
    feature_names = dataset["feature_names"]

    # Save scaler separately for inference
    Path("models").mkdir(exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print(f"\nFeature count: {len(feature_names)}")

    # ── 2. Train full pipeline ────────────────────────────────────
    pipe = FDCPipeline(device="cpu")
    pipe.fit(
        X_train, y_train,
        X_test,  y_test,
        feature_names=feature_names,
        ae_epochs=60,
    )

    # ── 3. Save ───────────────────────────────────────────────────
    pipe.save()
    print("\nTraining complete. Run: uvicorn api:app --reload")
    print(f"\nFinal metrics:")
    print(f"  F1:       {pipe.metrics.get('f1')}")
    print(f"  AUC-ROC:  {pipe.metrics.get('auc_roc')}")
    print(f"  AUC-PR:   {pipe.metrics.get('auc_pr')}")

if __name__ == "__main__":
    main()
