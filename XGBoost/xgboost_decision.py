"""
xgboost_decision.py — Part 3: XGBoost Decision Layer
======================================================
Fuses outputs from the CNN and LSTM models into a unified decision:

    CNN ──► cnn_embedding (128,) + biomass_tons_per_ha
    LSTM ──► lstm_embedding (64,) + predicted_aqi_lstm
                    │
                    ▼
          Feature Engineering (198 dims)
                    │
                    ▼
      ┌─────────────────────────────────┐
      │   XGBoost Classifier            │ ──► burn_risk_label (0-3)
      │   XGBoost Regressor             │ ──► health_impact_score (0-100)
      └─────────────────────────────────┘
                    │
                    ▼
            DecisionResult
         (label, score, action, confidence)

Usage:
    # Training (one-time, needs labelled historical samples)
    python xgboost_decision.py --mode train

    # Inference (called from pipeline.py per farmer / per day)
    python xgboost_decision.py --mode predict
"""

from __future__ import annotations
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

# ── XGBoost (install: pip install xgboost) ───────────────────────────────────
try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    raise ImportError("Run: pip install xgboost")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, mean_absolute_error,
    mean_squared_error, r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore", category=UserWarning)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs",     exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CNN_EMB_DIM  = 128
LSTM_EMB_DIM = 64
N_SCALARS    = 6                              # engineered scalar features
TOTAL_FEATURES = CNN_EMB_DIM + LSTM_EMB_DIM + N_SCALARS   # 198

RISK_LABELS = {0: "Low", 1: "Moderate", 2: "High", 3: "Critical"}

# AQI + biomass thresholds that define each risk class
# Used for synthetic label generation during training
RISK_THRESHOLDS = [
    # (max_aqi, max_biomass) → class
    (100, 1.5, 0),   # Low
    (200, 3.0, 1),   # Moderate
    (300, 4.5, 2),   # High
    (500, 6.0, 3),   # Critical
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelInputs:
    """
    Inputs received from upstream models.
    Both models must be run BEFORE calling the XGBoost layer.
    """
    cnn_embedding:       np.ndarray    # shape (128,) float32
    biomass_tons_per_ha: float         # CNN scalar output
    lstm_embedding:      np.ndarray    # shape (64,)  float32
    predicted_aqi_lstm:  float         # LSTM scalar output
    farmer_id:           str = ""      # optional, passed through to output
    lat:                 float = 0.0   # optional, for geospatial stage
    lon:                 float = 0.0


@dataclass
class DecisionResult:
    """Full output of the XGBoost Decision Layer."""
    farmer_id:            str
    burn_risk_label:      str          # "Low" | "Moderate" | "High" | "Critical"
    burn_risk_class:      int          # 0-3
    burn_risk_confidence: float        # probability of predicted class (0-1)
    health_impact_score:  float        # 0-100 (higher = more damage if burned)
    intervention_urgency: str          # derived action string
    alert_flag:           bool         # True if class >= 2 (High or Critical)
    biomass_tons_per_ha:  float        # passed through from CNN
    predicted_aqi:        float        # passed through from LSTM
    lat:                  float
    lon:                  float

    def to_dict(self) -> Dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vector(inp: ModelInputs) -> np.ndarray:
    """
    Combines CNN embedding, LSTM embedding, and derived scalar features
    into a single 198-dimensional feature vector for XGBoost.

    Why not feed raw embeddings directly?
      Embeddings capture spatial/temporal patterns; the scalars add explicit
      domain knowledge (risk thresholds, cost proxies) that tree models can
      split on more efficiently than inferred from embedding dimensions alone.

    Feature breakdown:
      [0:128]   CNN embedding   — spatial biomass texture, field size, crop type
      [128:192] LSTM embedding  — recent AQI trend, weather memory
      [192]     biomass_tons_per_ha
      [193]     predicted_aqi_lstm
      [194]     biomass_risk_norm    = biomass / 6.0  (0-1)
      [195]     aqi_risk_norm        = AQI / 500      (0-1)
      [196]     combined_risk        = 0.6*aqi_norm + 0.4*biomass_norm
      [197]     health_cost_proxy    = biomass * AQI * 0.42
    """
    bio  = float(inp.biomass_tons_per_ha)
    aqi  = float(inp.predicted_aqi_lstm)

    bio_norm      = np.clip(bio / 6.0,   0.0, 1.0)
    aqi_norm      = np.clip(aqi / 500.0, 0.0, 1.0)
    combined_risk = 0.6 * aqi_norm + 0.4 * bio_norm
    health_cost   = bio * aqi * 0.42      # economic damage proxy (₹k)

    scalars = np.array([bio, aqi, bio_norm, aqi_norm, combined_risk, health_cost],
                       dtype=np.float32)

    vec = np.concatenate([
        inp.cnn_embedding.astype(np.float32),
        inp.lstm_embedding.astype(np.float32),
        scalars,
    ])
    assert vec.shape == (TOTAL_FEATURES,), f"Expected ({TOTAL_FEATURES},), got {vec.shape}"
    return vec


def build_feature_matrix(inputs: List[ModelInputs]) -> np.ndarray:
    """Vectorised version — builds (N, 198) matrix from a list of inputs."""
    return np.stack([build_feature_vector(inp) for inp in inputs])


# ─────────────────────────────────────────────────────────────────────────────
# LABEL GENERATION (for training from historical data)
# ─────────────────────────────────────────────────────────────────────────────
def assign_risk_label(biomass: float, aqi: float) -> int:
    """
    Rule-based labelling from domain knowledge.
    Used to generate training labels from historical (biomass, AQI) pairs.
    Override with human expert labels where available.

    Class 0 — Low      : AQI ≤ 100  AND biomass ≤ 1.5 t/ha
    Class 1 — Moderate : AQI ≤ 200  AND biomass ≤ 3.0 t/ha
    Class 2 — High     : AQI ≤ 300  AND biomass ≤ 4.5 t/ha
    Class 3 — Critical : anything above
    """
    combined = 0.6 * (aqi / 500) + 0.4 * (biomass / 6)
    if combined < 0.20: return 0
    if combined < 0.45: return 1
    if combined < 0.70: return 2
    return 3


def assign_health_impact(biomass: float, aqi: float) -> float:
    """
    Continuous target: health impact score 0-100.
    Models economic + health cost if the field is burned.
    """
    # Components: AQI severity (0-50), biomass volume (0-30), interaction (0-20)
    aqi_component     = np.clip((aqi - 50) / 450 * 50, 0, 50)
    biomass_component = np.clip(biomass / 6.0 * 30,    0, 30)
    interaction       = np.clip(aqi_component * biomass / 100, 0, 20)
    raw = aqi_component + biomass_component + interaction
    return float(np.clip(raw, 0, 100))


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC TRAINING DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_training_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic synthetic training samples by:
      1. Sampling (biomass, AQI) from realistic Punjab distributions
      2. Generating random CNN/LSTM embeddings with mild signal injection
         (so embeddings correlate with the scalar labels — simulating what
          a real trained CNN/LSTM would produce)
      3. Building feature vectors and labelling

    Replace with real inference outputs from your CNN + LSTM when available.
    """
    rng = np.random.default_rng(seed)

    # ── Sample realistic (biomass, AQI) pairs ─────────────────────────────────
    # Stubble season (Oct-Nov): high biomass, high AQI
    # Off-season: low biomass, lower AQI
    season = rng.choice(["peak", "moderate", "offseason"],
                        size=n_samples, p=[0.25, 0.40, 0.35])

    biomass_params = {"peak": (4.5, 0.8), "moderate": (2.5, 0.7), "offseason": (0.8, 0.4)}
    aqi_params     = {"peak": (320, 80),  "moderate": (175, 60),  "offseason": (90,  40)}

    biomass_arr = np.array([
        np.clip(rng.normal(*biomass_params[s]), 0.1, 6.0) for s in season
    ])
    aqi_arr = np.array([
        np.clip(rng.normal(*aqi_params[s]), 20, 500) for s in season
    ])

    # ── Generate embeddings with signal correlated to (biomass, aqi) ──────────
    # Real embeddings from trained CNN/LSTM will have stronger signal
    risk_signal = (0.6 * aqi_arr / 500 + 0.4 * biomass_arr / 6).reshape(-1, 1)

    cnn_embs  = rng.standard_normal((n_samples, CNN_EMB_DIM)).astype(np.float32)
    lstm_embs = rng.standard_normal((n_samples, LSTM_EMB_DIM)).astype(np.float32)
    # Inject signal into first few dims (simulates what a trained model learns)
    cnn_embs[:, :8]  += (risk_signal * 2.5).astype(np.float32)
    lstm_embs[:, :4] += (risk_signal * 2.0).astype(np.float32)

    # ── Labels ────────────────────────────────────────────────────────────────
    risk_classes = np.array([assign_risk_label(b, a)
                              for b, a in zip(biomass_arr, aqi_arr)])
    health_scores = np.array([assign_health_impact(b, a)
                               for b, a in zip(biomass_arr, aqi_arr)])

    # ── Build feature matrix ───────────────────────────────────────────────────
    inputs = [
        ModelInputs(
            cnn_embedding       = cnn_embs[i],
            biomass_tons_per_ha = biomass_arr[i],
            lstm_embedding      = lstm_embs[i],
            predicted_aqi_lstm  = aqi_arr[i],
        )
        for i in range(n_samples)
    ]
    X = build_feature_matrix(inputs)

    print(f"[data] Generated {n_samples} training samples")
    print(f"       Class distribution: { {k: int((risk_classes==k).sum()) for k in range(4)} }")
    print(f"       Feature matrix : {X.shape}")
    return X, risk_classes, health_scores


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train(n_samples: int = 5000, ckpt_dir: str = "checkpoints"):
    """
    Trains two XGBoost models:
      1. Classifier → burn_risk_class (0-3)
      2. Regressor  → health_impact_score (0-100)

    Both share the same 198-dim feature vector from build_feature_vector().
    """
    print("\n" + "═"*55)
    print("  XGBoost Decision Layer — Training")
    print("═"*55)

    X, y_cls, y_reg = generate_training_data(n_samples)

    # ── Normalise ─────────────────────────────────────────────────────────────
    # XGBoost is largely scale-invariant, but normalising helps the embedding
    # dimensions (which may have very different scales) contribute equally.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f"{ckpt_dir}/xgb_scaler.pkl")

    # ── Class weights (handle class imbalance) ────────────────────────────────
    sample_weights = compute_sample_weight("balanced", y_cls)

    # ── Classifier ────────────────────────────────────────────────────────────
    print("\n[xgb] Training burn risk classifier (4 classes)...")
    clf = XGBClassifier(
        n_estimators      = 400,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.7,
        min_child_weight  = 3,
        reg_alpha         = 0.1,    # L1 — sparsity in 198-dim space
        reg_lambda        = 1.0,    # L2
        objective         = "multi:softprob",
        num_class         = 4,
        eval_metric       = "mlogloss",
        random_state      = 42,
        n_jobs            = -1,
        use_label_encoder = False,
    )

    # 5-fold stratified cross-validation
    cv_scores = cross_val_score(clf, X_scaled, y_cls, cv=5,
                                scoring="f1_weighted", n_jobs=-1)
    print(f"       CV F1-weighted : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    clf.fit(X_scaled, y_cls, sample_weight=sample_weights,
            eval_set=[(X_scaled, y_cls)], verbose=False)
    joblib.dump(clf, f"{ckpt_dir}/xgb_classifier.pkl")

    # Full train report
    y_pred_cls = clf.predict(X_scaled)
    print("\n[xgb] Classifier report:")
    print(classification_report(y_cls, y_pred_cls,
                                target_names=list(RISK_LABELS.values())))

    # ── Regressor ─────────────────────────────────────────────────────────────
    print("[xgb] Training health impact regressor...")
    reg = XGBRegressor(
        n_estimators     = 400,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.7,
        min_child_weight = 3,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        objective        = "reg:squarederror",
        eval_metric      = "rmse",
        random_state     = 42,
        n_jobs           = -1,
    )
    reg.fit(X_scaled, y_reg, eval_set=[(X_scaled, y_reg)], verbose=False)
    joblib.dump(reg, f"{ckpt_dir}/xgb_regressor.pkl")

    y_pred_reg = reg.predict(X_scaled)
    rmse = np.sqrt(mean_squared_error(y_reg, y_pred_reg))
    mae  = mean_absolute_error(y_reg, y_pred_reg)
    r2   = r2_score(y_reg, y_pred_reg)
    print(f"       Regressor RMSE : {rmse:.2f}  MAE: {mae:.2f}  R²: {r2:.4f}")

    # ── Feature importance (top 20) ───────────────────────────────────────────
    feature_names = (
        [f"cnn_{i}"  for i in range(CNN_EMB_DIM)] +
        [f"lstm_{i}" for i in range(LSTM_EMB_DIM)] +
        ["biomass", "aqi", "bio_norm", "aqi_norm", "combined_risk", "health_cost"]
    )
    importances = clf.feature_importances_
    top20_idx   = np.argsort(importances)[::-1][:20]
    print("\n[xgb] Top-20 features (classifier):")
    for rank, idx in enumerate(top20_idx, 1):
        print(f"  {rank:>2}. {feature_names[idx]:<20} {importances[idx]:.4f}")

    print(f"\n✅  Checkpoints saved → {ckpt_dir}/")
    return clf, reg, scaler


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class XGBoostDecisionLayer:
    """
    Stateless inference wrapper — load once, call many times.

    Example:
        layer = XGBoostDecisionLayer()
        result = layer.predict(inp)        # single farmer
        results = layer.predict_batch(inputs)  # list of farmers
    """

    def __init__(self, ckpt_dir: str = "checkpoints"):
        self.clf    = joblib.load(f"{ckpt_dir}/xgb_classifier.pkl")
        self.reg    = joblib.load(f"{ckpt_dir}/xgb_regressor.pkl")
        self.scaler = joblib.load(f"{ckpt_dir}/xgb_scaler.pkl")
        print(f"[XGBoostDecisionLayer] Loaded from {ckpt_dir}/")

    def _intervention_text(self, cls: int, score: float) -> str:
        mapping = {
            0: "No immediate action — monitor weekly",
            1: "Recommend alternative use: offer biomass collection subsidy",
            2: "Urgent: dispatch field officer, offer same-day collection",
            3: "CRITICAL ALERT: notify district authority + emergency biomass pickup",
        }
        return mapping[cls]

    def predict(self, inp: ModelInputs) -> DecisionResult:
        """Predict burn risk for a single farmer."""
        x  = build_feature_vector(inp).reshape(1, -1)
        xs = self.scaler.transform(x)

        cls        = int(self.clf.predict(xs)[0])
        conf       = float(self.clf.predict_proba(xs)[0][cls])
        score      = float(np.clip(self.reg.predict(xs)[0], 0, 100))

        return DecisionResult(
            farmer_id            = inp.farmer_id,
            burn_risk_label      = RISK_LABELS[cls],
            burn_risk_class      = cls,
            burn_risk_confidence = round(conf, 4),
            health_impact_score  = round(score, 2),
            intervention_urgency = self._intervention_text(cls, score),
            alert_flag           = cls >= 2,
            biomass_tons_per_ha  = inp.biomass_tons_per_ha,
            predicted_aqi        = inp.predicted_aqi_lstm,
            lat                  = inp.lat,
            lon                  = inp.lon,
        )

    def predict_batch(self, inputs: List[ModelInputs]) -> List[DecisionResult]:
        """Vectorised batch prediction (much faster than looping predict())."""
        X  = build_feature_matrix(inputs)
        Xs = self.scaler.transform(X)

        classes    = self.clf.predict(Xs)
        proba      = self.clf.predict_proba(Xs)
        scores     = np.clip(self.reg.predict(Xs), 0, 100)

        results = []
        for i, inp in enumerate(inputs):
            cls  = int(classes[i])
            conf = float(proba[i][cls])
            results.append(DecisionResult(
                farmer_id            = inp.farmer_id,
                burn_risk_label      = RISK_LABELS[cls],
                burn_risk_class      = cls,
                burn_risk_confidence = round(conf, 4),
                health_impact_score  = round(float(scores[i]), 2),
                intervention_urgency = self._intervention_text(cls, float(scores[i])),
                alert_flag           = cls >= 2,
                biomass_tons_per_ha  = inp.biomass_tons_per_ha,
                predicted_aqi        = inp.predicted_aqi_lstm,
                lat                  = inp.lat,
                lon                  = inp.lon,
            ))
        return results

    def predict_dataframe(self, inputs: List[ModelInputs]) -> pd.DataFrame:
        """Returns results as a pandas DataFrame (useful for downstream joins)."""
        return pd.DataFrame([r.to_dict() for r in self.predict_batch(inputs)])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["train", "predict"], default="train")
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()

    if args.mode == "train":
        train(n_samples=args.samples)

    else:
        # Demo inference with a fake input
        print("\n[demo] Running single inference...")
        layer = XGBoostDecisionLayer()
        demo_input = ModelInputs(
            cnn_embedding       = np.random.randn(128).astype(np.float32),
            biomass_tons_per_ha = 4.2,
            lstm_embedding      = np.random.randn(64).astype(np.float32),
            predicted_aqi_lstm  = 287.0,
            farmer_id           = "FARMER_PB_001",
            lat                 = 30.901,
            lon                 = 75.857,
        )
        result = layer.predict(demo_input)
        print("\n── Decision Result ──────────────────────────────")
        for k, v in result.to_dict().items():
            print(f"  {k:<25} : {v}")
