"""
main.py
=======
GreenKisan Master Pipeline
Connects CNN → LSTM → XGBoost → Buyer Matching

Exact classes used:
    CNN  : CNNInferenceEngine         (inference.py)
    LSTM : AQIForecastLSTM            (LSTM/model.py)
    XGB  : XGBoostDecisionLayer       (XGboost_decision/xgboost_decision.py)
    BUYER: GeospatialMatcher          (XGboost_decision/buyer_matching.py)

Usage:
    python main.py --lat 30.901 --lon 75.857 --date 2023-10-21
    python main.py --lat 30.901 --lon 75.857 --date 2023-10-21 --farmer-id FARM_001 --crop rice_straw
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch

# ── Add subfolders to path ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "XGBoost"))
sys.path.insert(0, str(Path(__file__).parent / "LSTM"))
sys.path.insert(0, str(Path(__file__).parent / "CNN"))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# STEP 1 — CNN
# =============================================================================
def run_cnn(lat, lon, date_str, farmer_id, ckpt_path="Checkpoints/cnn_biomass.pt"):
    logger.info(f"[Step 1/4] CNN — {farmer_id}")
    from inference import CNNInferenceEngine
    from torch_dataset import compute_dataset_stats

    # Use known training stats directly (avoids loading all .npy files)
    try:
        _, _, lb_mean, lb_std = compute_dataset_stats("Data/metadata.csv", "Data")
    except Exception:
        lb_mean, lb_std = 1.9797, 0.6505   # from your training session
        logger.warning("[CNN] Using hardcoded label stats: mean=1.9797 std=0.6505")
    engine = CNNInferenceEngine(ckpt_path=ckpt_path)
    raw    = engine.predict(lat=lat, lon=lon, date_str=date_str, farmer_id=farmer_id)

    if raw is None:
        logger.error("[CNN] No imagery returned")
        return None

    biomass   = float(raw["biomass_prediction"]) * lb_std + lb_mean
    biomass   = max(0.0, biomass)
    embedding = np.array(raw["cnn_embedding"]).flatten()[:128]

    logger.info(f"  ✅ Biomass   : {biomass:.4f} t/ha")
    logger.info(f"  ✅ NDVI      : {raw['ndvi_mean']:.4f}")

    return {
        "biomass_prediction": round(biomass, 4),
        "cnn_embedding"     : embedding,
        "ndvi_mean"         : float(raw["ndvi_mean"]),
        "date_used"         : raw["date_used"],
        "cloud_pct"         : float(raw["cloud_pct"]),
    }


# =============================================================================
# STEP 2 — LSTM (AQIForecastLSTM — input_size=7)
# =============================================================================
def run_lstm(cnn_result, ckpt_path="LSTM/checkpoints/best_lstm.pt"):
    logger.info("[Step 2/4] LSTM — AQI forecast")

    from model import AQIForecastLSTM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AQIForecastLSTM(input_size=7, hidden_size=128, num_layers=2, output_size=1, dropout=0.2).to(device)

    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        # Try different checkpoint key formats
        state = ckpt.get("model_state") or ckpt.get("model") or ckpt.get("state_dict") or ckpt
        model.load_state_dict(state)
        logger.info(f"  ✅ Loaded from {ckpt_path}")
    else:
        logger.warning(f"  ⚠️  No checkpoint — using untrained LSTM")

    model.eval()

    # Build 30-day sequence with 7 features per timestep
    biomass = cnn_result["biomass_prediction"]
    ndvi    = cnn_result["ndvi_mean"]
    cloud   = cnn_result["cloud_pct"] / 100.0
    rng     = np.random.default_rng(42)

    seq = np.array([[
        biomass + rng.normal(0, 0.05),
        ndvi    + rng.normal(0, 0.02),
        cloud   + rng.normal(0, 0.01),
        25.0    + rng.normal(0, 2.0),
        60.0    + rng.normal(0, 5.0),
        10.0    + rng.normal(0, 1.0),
        150.0   + rng.normal(0, 30.0),
    ] for _ in range(30)], dtype=np.float32)          # (30, 7)

    tensor = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, 30, 7)

    with torch.no_grad():
        aqi_out = model(tensor)

    predicted_aqi = max(0.0, float(aqi_out.squeeze().item()))

    # 64-dim embedding from last timestep features (padded)
    emb64      = np.zeros(64, dtype=np.float32)
    emb64[:7]  = seq[-1]

    trend      = "increasing" if ndvi > 0.4 else "stable" if ndvi > 0.2 else "decreasing"
    confidence = round(min(1.0, (ndvi + (1 - cloud)) / 2), 3)

    logger.info(f"  ✅ AQI       : {predicted_aqi:.2f}")
    logger.info(f"  ✅ Trend     : {trend}")

    return {
        "predicted_aqi" : round(predicted_aqi, 2),
        "lstm_embedding": emb64,
        "confidence"    : confidence,
        "trend"         : trend,
    }


# =============================================================================
# STEP 3 — XGBOOST
# =============================================================================
def run_xgboost(farmer_id, lat, lon, cnn_result, lstm_result,
                field_area_ha=2.0, biomass_type="rice_straw",
                harvest_month=10, ckpt_dir="XGBoost/Checkpoints"):
    logger.info("[Step 3/4] XGBoost — burn risk")

    from xgboost_decision import XGBoostDecisionLayer, ModelInputs

    xgb = XGBoostDecisionLayer(ckpt_dir)
    inp = ModelInputs(
        cnn_embedding       = cnn_result["cnn_embedding"],
        biomass_tons_per_ha = cnn_result["biomass_prediction"],
        lstm_embedding      = lstm_result["lstm_embedding"],
        predicted_aqi_lstm  = lstm_result["predicted_aqi"],
        farmer_id           = farmer_id,
        lat                 = lat,
        lon                 = lon,
    )

    decision = xgb.predict(inp)

    logger.info(f"  ✅ Burn Risk : {decision.burn_risk_label} (class {decision.burn_risk_class})")
    logger.info(f"  ✅ Impact    : {decision.health_impact_score:.1f}/100")
    logger.info(f"  ✅ Alert     : {decision.alert_flag}")

    return {
        "burn_risk_label"     : decision.burn_risk_label,
        "burn_risk_class"     : decision.burn_risk_class,
        "burn_risk_confidence": decision.burn_risk_confidence,
        "health_impact_score" : decision.health_impact_score,
        "alert_flag"          : decision.alert_flag,
        "intervention_urgency": decision.intervention_urgency,
    }


# =============================================================================
# STEP 4 — BUYER MATCHING
# =============================================================================
def run_buyer_matching(farmer_id, lat, lon, cnn_result, xgb_result,
                       biomass_type="rice_straw", harvest_month=10,
                       field_area_ha=2.0, top_k=3, max_dist_km=150):
    logger.info("[Step 4/4] Buyer matching")

    from buyer_matching import GeospatialMatcher, FarmerNeed, load_default_buyers

    matcher     = GeospatialMatcher(load_default_buyers())
    farmer_need = FarmerNeed(
        farmer_id           = farmer_id,
        lat                 = lat,
        lon                 = lon,
        biomass_tons        = round(cnn_result["biomass_prediction"] * field_area_ha, 2),
        biomass_type        = biomass_type,
        harvest_month       = harvest_month,
        burn_risk_class     = xgb_result["burn_risk_class"],
        health_impact_score = xgb_result["health_impact_score"],
    )

    report = matcher.match_farmer(farmer_need, top_k=top_k, max_dist_km=max_dist_km)
    logger.info(f"  ✅ Found {len(report.top_matches)} buyers")

    return [{
        "rank"        : m.recommendation_rank,
        "buyer_name"  : m.buyer_name,
        "buyer_type"  : m.buyer_type,
        "distance_km" : m.distance_km,
        "price_per_ton": m.price_per_ton_inr,
        "net_revenue" : m.net_revenue_inr,
        "match_score" : m.match_score,
        "available"   : m.available_this_month,
    } for m in report.top_matches]


# =============================================================================
# MASTER PIPELINE
# =============================================================================
def run_pipeline(lat, lon, date_str, farmer_id="FARM_001",
                 crop_type="rice_straw", field_area_ha=2.0,
                 harvest_month=10, top_k=3):

    print(f"\n{'='*60}")
    print(f"  GreenKisan Master Pipeline")
    print(f"  Farmer   : {farmer_id}")
    print(f"  Location : {lat}, {lon}")
    print(f"  Date     : {date_str}")
    print(f"{'='*60}\n")

    # Step 1 — CNN
    cnn = run_cnn(lat, lon, date_str, farmer_id)
    if cnn is None:
        print("❌ Pipeline stopped — CNN failed")
        return

    # Step 2 — LSTM
    lstm = run_lstm(cnn)

    # Step 3 — XGBoost
    try:
        xgb = run_xgboost(farmer_id, lat, lon, cnn, lstm,
                           field_area_ha, crop_type, harvest_month)
    except Exception as e:
        logger.warning(f"[XGB] {e} — using fallback")
        xgb = {"burn_risk_label": "Unknown", "burn_risk_class": 1,
               "health_impact_score": 50.0, "alert_flag": False,
               "intervention_urgency": "Normal", "burn_risk_confidence": 0.5}

    # Step 4 — Buyer Matching
    try:
        buyers = run_buyer_matching(farmer_id, lat, lon, cnn, xgb,
                                    crop_type, harvest_month, field_area_ha, top_k)
    except Exception as e:
        logger.warning(f"[Buyer] {e}")
        buyers = []

    # Print results
    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"\n  ── CNN ─────────────────────────────────────")
    print(f"  Biomass        : {cnn['biomass_prediction']:.4f} t/ha")
    print(f"  NDVI           : {cnn['ndvi_mean']:.4f}")
    print(f"  Cloud cover    : {cnn['cloud_pct']:.1f}%")

    print(f"\n  ── LSTM ────────────────────────────────────")
    print(f"  Predicted AQI  : {lstm['predicted_aqi']:.2f}")
    print(f"  Trend          : {lstm['trend']}")
    print(f"  Confidence     : {lstm['confidence']:.3f}")

    print(f"\n  ── XGBoost ─────────────────────────────────")
    print(f"  Burn Risk      : {xgb['burn_risk_label']}")
    print(f"  Health Impact  : {xgb['health_impact_score']:.1f}/100")
    print(f"  Alert          : {'🔴 YES' if xgb['alert_flag'] else '🟢 NO'}")

    if buyers:
        print(f"\n  ── Top {len(buyers)} Buyers ──────────────────────────────")
        for b in buyers:
            print(f"  {b['rank']}. {b['buyer_name']:<32} Score: {b['match_score']:.3f}")
            print(f"     ₹{b['price_per_ton']:,.0f}/ton  |  {b['distance_km']} km  |  Net: ₹{b['net_revenue']:,.0f}")
    else:
        print("\n  ⚠️  No buyer matches (burn risk must be ≥ 1 for matching)")

    print(f"\n{'='*60}\n")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GreenKisan Master Pipeline")
    parser.add_argument("--lat",           type=float, default=30.901)
    parser.add_argument("--lon",           type=float, default=75.857)
    parser.add_argument("--date",          default="2023-10-21")
    parser.add_argument("--farmer-id",     default="FARM_001")
    parser.add_argument("--crop",          default="rice_straw",
                        choices=["rice_straw", "wheat_straw"])
    parser.add_argument("--field-area",    type=float, default=2.0)
    parser.add_argument("--harvest-month", type=int,   default=10)
    parser.add_argument("--top-k",         type=int,   default=3)
    args = parser.parse_args()

    run_pipeline(
        lat           = args.lat,
        lon           = args.lon,
        date_str      = args.date,
        farmer_id     = args.farmer_id,
        crop_type     = args.crop,
        field_area_ha = args.field_area,
        harvest_month = args.harvest_month,
        top_k         = args.top_k,
    )