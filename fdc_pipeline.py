"""
fdc_pipeline.py
===============
Full FDC-AI pipeline:
  1. Isolation Forest  — unsupervised baseline anomaly detection
  2. Autoencoder       — deep anomaly detection (trained on normal data only)
  3. XGBoost Classifier — supervised fault classification
  4. SHAP              — sensor attribution / explainability layer
  5. Model persistence  — save/load all components
"""

import numpy as np
import pandas as pd
import joblib
import shap
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, f1_score,
    roc_auc_score, precision_recall_curve,
    confusion_matrix, average_precision_score,
)
from xgboost import XGBClassifier

import torch
from autoencoder import (
    Autoencoder, train_autoencoder,
    anomaly_scores, anomaly_threshold,
    save_autoencoder, load_autoencoder,
)

MODELS_DIR = Path("models")


# ── Helper ────────────────────────────────────────────────────────────────────

def optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Find decision threshold that maximises F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    return float(thresholds[np.argmax(f1s[:-1])])


# ── 1. Isolation Forest ───────────────────────────────────────────────────────

def train_isolation_forest(
    X_normal: np.ndarray,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> IsolationForest:
    print("Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_normal)
    return iso


def isolation_forest_scores(iso: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Return anomaly scores (higher = more anomalous, range 0–1)."""
    raw = iso.score_samples(X)          # lower = more anomalous in sklearn
    return 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)


# ── 2. Autoencoder Anomaly Detection ─────────────────────────────────────────

def train_ae_detector(
    X_normal: np.ndarray,
    epochs: int = 60,
    latent_dim: int = 16,
    device: str = "cpu",
) -> Tuple[Autoencoder, float]:
    """Train autoencoder, compute threshold from normal data scores."""
    print("Training Autoencoder...")
    model, history = train_autoencoder(
        X_normal, latent_dim=latent_dim, epochs=epochs,
        batch_size=128, device=device, verbose=True,
    )
    scores_normal = anomaly_scores(model, X_normal, device)
    thresh = anomaly_threshold(scores_normal, percentile=95)
    print(f"  AE threshold (95th percentile on normal): {thresh:.6f}")
    return model, thresh


# ── 3. XGBoost Fault Classifier ───────────────────────────────────────────────

def build_classifier_features(
    X: np.ndarray,
    iso_scores: np.ndarray,
    ae_scores: np.ndarray,
) -> np.ndarray:
    """Augment raw features with anomaly scores from both detectors."""
    return np.column_stack([X, iso_scores, ae_scores])


def train_xgboost(
    X_aug: np.ndarray,
    y: np.ndarray,
    scale_pos_weight: Optional[float] = None,
    random_state: int = 42,
) -> XGBClassifier:
    """Train XGBoost with class imbalance handling."""
    if scale_pos_weight is None:
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        scale_pos_weight = neg / (pos + 1e-9)
        print(f"  Auto scale_pos_weight: {scale_pos_weight:.2f}  (neg={neg}, pos={pos})")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_aug, y)
    return model


def evaluate_classifier(
    model: XGBClassifier,
    X_aug: np.ndarray,
    y: np.ndarray,
    label: str = "Test",
) -> Dict:
    """Full evaluation: F1, AUC-ROC, AUC-PR, confusion matrix."""
    proba = model.predict_proba(X_aug)[:, 1]
    thresh = optimal_threshold(y, proba)
    preds  = (proba >= thresh).astype(int)

    metrics = {
        "f1":       round(f1_score(y, preds, zero_division=0), 4),
        "auc_roc":  round(roc_auc_score(y, proba), 4),
        "auc_pr":   round(average_precision_score(y, proba), 4),
        "threshold": round(thresh, 4),
        "confusion": confusion_matrix(y, preds).tolist(),
    }
    print(f"\n── {label} metrics ──────────────────────────────")
    print(f"  F1:       {metrics['f1']}")
    print(f"  AUC-ROC:  {metrics['auc_roc']}")
    print(f"  AUC-PR:   {metrics['auc_pr']}")
    print(f"  Threshold:{metrics['threshold']}")
    print(classification_report(y, preds, target_names=["Normal", "Fault"], zero_division=0))
    return metrics


# ── 4. SHAP Explainer ─────────────────────────────────────────────────────────

def build_shap_explainer(
    model: XGBClassifier,
    X_aug: np.ndarray,
) -> shap.TreeExplainer:
    """Build SHAP TreeExplainer for XGBoost."""
    print("Building SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    return explainer


def explain_sample(
    explainer: shap.TreeExplainer,
    x_single: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> Dict:
    """
    Compute SHAP values for a single sample.
    Returns dict of {feature: shap_value} sorted by absolute importance.
    """
    shap_values = explainer.shap_values(x_single.reshape(1, -1))
    if isinstance(shap_values, list):
        sv = shap_values[1][0]     # fault class SHAP values
    else:
        sv = shap_values[0]

    pairs = sorted(
        zip(feature_names, sv),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    return {
        "features":    [p[0] for p in pairs],
        "shap_values": [round(float(p[1]), 6) for p in pairs],
        "base_value":  round(float(explainer.expected_value
                                    if not isinstance(explainer.expected_value, list)
                                    else explainer.expected_value[1]), 6),
    }


def global_feature_importance(
    explainer: shap.TreeExplainer,
    X_aug: np.ndarray,
    feature_names: List[str],
    top_n: int = 15,
) -> Dict:
    """Mean absolute SHAP values across all samples — global importance."""
    shap_values = explainer.shap_values(X_aug[:200])   # sample for speed
    if isinstance(shap_values, list):
        sv = np.abs(shap_values[1])
    else:
        sv = np.abs(shap_values)

    mean_abs = sv.mean(axis=0)
    pairs = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)[:top_n]
    return {
        "features":    [p[0] for p in pairs],
        "importance":  [round(float(p[1]), 6) for p in pairs],
    }


# ── 5. Full Pipeline: Train + Save ────────────────────────────────────────────

class FDCPipeline:
    """
    Encapsulates the full FDC-AI pipeline:
    IsolationForest + Autoencoder + XGBoost + SHAP.
    """

    def __init__(self, device: str = "cpu"):
        self.device       = device
        self.iso          = None
        self.ae_model     = None
        self.ae_threshold = None
        self.xgb          = None
        self.explainer    = None
        self.scaler       = None
        self.feature_names: List[str] = []
        self.aug_feature_names: List[str] = []
        self.metrics: Dict = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        ae_epochs: int = 60,
    ):
        self.feature_names = feature_names
        self.aug_feature_names = feature_names + ["iso_score", "ae_score"]

        # Split normal data for unsupervised training
        X_normal = X_train[y_train == 0]
        print(f"\nNormal samples for unsupervised training: {len(X_normal)}")

        # 1. Isolation Forest
        self.iso = train_isolation_forest(X_normal)

        # 2. Autoencoder
        self.ae_model, self.ae_threshold = train_ae_detector(
            X_normal, epochs=ae_epochs, device=self.device
        )

        # 3. Augmented features for XGBoost
        print("\nTraining XGBoost classifier...")
        iso_train = isolation_forest_scores(self.iso, X_train)
        ae_train  = anomaly_scores(self.ae_model, X_train, self.device)
        X_aug_train = build_classifier_features(X_train, iso_train, ae_train)

        self.xgb = train_xgboost(X_aug_train, y_train)

        # 4. Evaluate
        iso_test = isolation_forest_scores(self.iso, X_test)
        ae_test  = anomaly_scores(self.ae_model, X_test, self.device)
        X_aug_test = build_classifier_features(X_test, iso_test, ae_test)

        self.metrics = evaluate_classifier(self.xgb, X_aug_test, y_test, "Test")

        # 5. SHAP
        self.explainer = build_shap_explainer(self.xgb, X_aug_train)

        # Global importance
        self.metrics["global_importance"] = global_feature_importance(
            self.explainer, X_aug_train, self.aug_feature_names
        )

        print("\nPipeline training complete.")
        return self

    def predict_single(self, x_raw: np.ndarray) -> Dict:
        """
        Full inference for one sample (already scaled).
        Returns anomaly scores, fault prediction, SHAP attribution.
        """
        iso_s = isolation_forest_scores(self.iso, x_raw.reshape(1, -1))[0]
        ae_s  = anomaly_scores(self.ae_model, x_raw.reshape(1, -1), self.device)[0]
        x_aug = np.append(x_raw, [iso_s, ae_s])

        proba = float(self.xgb.predict_proba(x_aug.reshape(1, -1))[0, 1])
        pred  = int(proba >= self.metrics.get("threshold", 0.5))

        shap_info = explain_sample(
            self.explainer, x_aug, self.aug_feature_names, top_n=10
        )

        return {
            "iso_score":      round(float(iso_s), 4),
            "ae_score":       round(float(ae_s), 6),
            "ae_threshold":   round(float(self.ae_threshold), 6),
            "ae_anomaly":     bool(ae_s > self.ae_threshold),
            "fault_proba":    round(proba, 4),
            "fault_pred":     pred,
            "fault_label":    "FAULT DETECTED" if pred == 1 else "NORMAL",
            "shap":           shap_info,
        }

    def save(self, directory: Path = MODELS_DIR):
        directory.mkdir(exist_ok=True)
        joblib.dump(self.iso, directory / "iso_forest.pkl")
        save_autoencoder(self.ae_model, directory / "autoencoder.pt")
        joblib.dump(self.xgb, directory / "xgboost.pkl")
        joblib.dump(self.explainer, directory / "shap_explainer.pkl")

        meta = {
            "ae_threshold":       self.ae_threshold,
            "feature_names":      self.feature_names,
            "aug_feature_names":  self.aug_feature_names,
            "metrics":            {k: v for k, v in self.metrics.items()
                                   if k != "global_importance"},
            "input_dim":          len(self.feature_names),
        }
        with open(directory / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Models saved to {directory}/")

    @classmethod
    def load(cls, directory: Path = MODELS_DIR, device: str = "cpu") -> "FDCPipeline":
        pipe = cls(device=device)
        pipe.iso      = joblib.load(directory / "iso_forest.pkl")
        pipe.xgb      = joblib.load(directory / "xgboost.pkl")
        pipe.explainer= joblib.load(directory / "shap_explainer.pkl")

        with open(directory / "meta.json") as f:
            meta = json.load(f)

        pipe.ae_threshold     = meta["ae_threshold"]
        pipe.feature_names    = meta["feature_names"]
        pipe.aug_feature_names= meta["aug_feature_names"]
        pipe.metrics          = meta.get("metrics", {})
        input_dim = meta["input_dim"]

        pipe.ae_model = load_autoencoder(
            input_dim=input_dim,   # raw feature count (NOT augmented)
            latent_dim=16,
            path=directory / "autoencoder.pt",
            device=device,
        )
        print(f"Pipeline loaded from {directory}/")
        return pipe
