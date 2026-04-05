"""
pipeline.py — Full End-to-End Pipeline
========================================
CNN → LSTM → XGBoost Decision → Geospatial Buyer Matching

This is the INTEGRATION layer. Your CNN and LSTM are already trained.
This file shows exactly how to plug them in and run the complete system.

                Sentinel-2 Image
                      │
                      ▼
               ┌─────────────┐
               │  CNN Model  │  →  cnn_embedding (128,)
               │  (yours)    │  →  biomass_tons_per_ha (float)
               └─────────────┘
                      │
                      ▼
               ┌─────────────┐
               │ LSTM Model  │  →  lstm_embedding (64,)
               │  (yours)    │  →  predicted_aqi_lstm (float)
               └─────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │ XGBoost Decision Layer │  →  burn_risk_label (0-3)
         │  (xgboost_decision.py) │  →  health_impact_score (0-100)
         └────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │ Geospatial Buyer Match │  →  Top-K buyer matches per farmer
         │  (buyer_matching.py)   │  →  Net revenue, logistics cost
         └────────────────────────┘
                      │
                      ▼
               outputs/pipeline_results.csv
               outputs/pipeline_results.json
               outputs/alert_farmers.csv

Usage:
    # Train XGBoost layer first (one-time):
    python pipeline.py --mode train

    # Run full inference pipeline:
    python pipeline.py --mode run

    # Run with your own CNN/LSTM outputs CSV:
    python pipeline.py --mode run --cnn-lstm-csv path/to/your_outputs.csv
"""

from __future__ import annotations
import os
import json
import argparse
import time
from dataclasses import asdict
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from xgboost_decision import (
    XGBoostDecisionLayer, ModelInputs, DecisionResult, train as train_xgb,
    RISK_LABELS,
)
from buyer_matching import (
    GeospatialMatcher, FarmerNeed, MatchReport,
    load_default_buyers, generate_demo_farmers,
)

os.makedirs("outputs", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CNN / LSTM STUB — Replace with your actual model calls
# ─────────────────────────────────────────────────────────────────────────────
class CNNModelStub:
    """
    REPLACE this class with your actual CNN.
    Your CNN should accept a Sentinel-2 image (numpy array or file path)
    and return the embedding + biomass scalar.

    Expected interface:
        cnn = YourCNNModel.load("checkpoints/cnn_model.pth")
        embedding, biomass = cnn.predict(sentinel2_image_array)
    """
    def predict(self, image_array: np.ndarray):
        """Returns (cnn_embedding: np.ndarray shape (128,), biomass_tons_per_ha: float)"""
        # ── STUB: generates realistic-looking outputs ──────────────────────────
        # Replace this entire method body with your real CNN inference
        embedding = np.random.randn(128).astype(np.float32)
        biomass   = float(np.clip(np.random.normal(3.0, 1.5), 0.1, 6.0))
        return embedding, biomass


class LSTMModelStub:
    """
    REPLACE this class with your actual LSTM.
    Your LSTM should accept a (30, n_features) time-series window
    and return the embedding + AQI scalar.

    Expected interface:
        lstm = YourLSTMModel.load("checkpoints/best_lstm.pt")
        embedding, aqi = lstm.predict(recent_30day_df, biomass)
    """
    def predict(self, recent_df: pd.DataFrame, biomass: float):
        """Returns (lstm_embedding: np.ndarray shape (64,), predicted_aqi: float)"""
        # ── STUB: generates realistic-looking outputs ──────────────────────────
        # Replace this entire method body with your real LSTM inference
        embedding     = np.random.randn(64).astype(np.float32)
        predicted_aqi = float(np.clip(np.random.normal(200, 80), 20, 500))
        return embedding, predicted_aqi


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
class StubbleBurningPipeline:
    """
    Main orchestrator. Wires CNN → LSTM → XGBoost → Buyer Matching.

    Parameters
    ----------
    cnn_model   : your trained CNN (replace CNNModelStub)
    lstm_model  : your trained LSTM (replace LSTMModelStub)
    ckpt_dir    : directory containing XGBoost checkpoints
    buyers      : buyer database (defaults to load_default_buyers())
    """

    def __init__(
        self,
        cnn_model   = None,
        lstm_model  = None,
        ckpt_dir: str = "checkpoints",
        buyers        = None,
    ):
        self.cnn    = cnn_model  or CNNModelStub()
        self.lstm   = lstm_model or LSTMModelStub()
        self.xgb    = XGBoostDecisionLayer(ckpt_dir)
        self.matcher = GeospatialMatcher(buyers or load_default_buyers())
        print("\n[Pipeline] All components loaded ✓")

    def process_single_farmer(
        self,
        farmer_id: str,
        lat: float,
        lon: float,
        satellite_image: np.ndarray,       # H×W×C Sentinel-2 array
        recent_weather_df: pd.DataFrame,   # last 30 days, see train.py FEATURE_COLS
        biomass_type: str  = "rice_straw",
        harvest_month: int = 10,
        field_area_ha: float = 2.0,
        top_k_buyers: int  = 3,
        max_dist_km: float = 150,
    ) -> Dict:
        """
        Full pipeline for ONE farmer field.
        Returns a dict with decision + buyer matches.
        """
        t0 = time.time()

        # ── Stage 1: CNN ──────────────────────────────────────────────────────
        cnn_emb, biomass_ph = self.cnn.predict(satellite_image)

        # ── Stage 2: LSTM ─────────────────────────────────────────────────────
        lstm_emb, pred_aqi = self.lstm.predict(recent_weather_df, biomass_ph)

        # ── Stage 3: XGBoost Decision ─────────────────────────────────────────
        inp    = ModelInputs(
            cnn_embedding       = cnn_emb,
            biomass_tons_per_ha = biomass_ph,
            lstm_embedding      = lstm_emb,
            predicted_aqi_lstm  = pred_aqi,
            farmer_id           = farmer_id,
            lat                 = lat,
            lon                 = lon,
        )
        decision: DecisionResult = self.xgb.predict(inp)

        # ── Stage 4: Buyer Matching (only for risk class ≥ 1) ─────────────────
        buyer_report: Optional[MatchReport] = None
        if decision.burn_risk_class >= 1:
            farmer_need = FarmerNeed(
                farmer_id           = farmer_id,
                lat                 = lat,
                lon                 = lon,
                biomass_tons        = round(biomass_ph * field_area_ha, 2),
                biomass_type        = biomass_type,
                harvest_month       = harvest_month,
                burn_risk_class     = decision.burn_risk_class,
                health_impact_score = decision.health_impact_score,
            )
            buyer_report = self.matcher.match_farmer(
                farmer_need, top_k=top_k_buyers, max_dist_km=max_dist_km
            )

        elapsed = time.time() - t0

        return {
            "farmer_id"          : farmer_id,
            "processing_time_s"  : round(elapsed, 3),
            "cnn_biomass_ph"     : round(biomass_ph, 3),
            "lstm_predicted_aqi" : round(pred_aqi, 1),
            "burn_risk_label"    : decision.burn_risk_label,
            "burn_risk_class"    : decision.burn_risk_class,
            "burn_risk_confidence": decision.burn_risk_confidence,
            "health_impact_score": decision.health_impact_score,
            "intervention_urgency": decision.intervention_urgency,
            "alert_flag"         : decision.alert_flag,
            "buyer_matches"      : (
                [asdict(m) for m in buyer_report.top_matches]
                if buyer_report else []
            ),
            "best_buyer"         : (
                asdict(buyer_report.best_match)
                if buyer_report and buyer_report.best_match else None
            ),
        }

    def process_batch(
        self,
        farmer_records: List[Dict],
        top_k_buyers: int  = 3,
        max_dist_km: float = 150,
    ) -> pd.DataFrame:
        """
        Batch process a list of farmer dicts.

        Each dict must contain:
          farmer_id, lat, lon, satellite_image (np.ndarray),
          recent_weather_df (pd.DataFrame), biomass_type, harvest_month,
          field_area_ha

        Returns a DataFrame with one row per farmer.
        """
        # ── Batch CNN inference ───────────────────────────────────────────────
        print(f"\n[Pipeline] Processing {len(farmer_records)} farmers...")
        print("[Stage 1/4] CNN inference...")
        cnn_results = [
            self.cnn.predict(r["satellite_image"]) for r in farmer_records
        ]
        cnn_embeddings = [c[0] for c in cnn_results]
        biomasses      = [c[1] for c in cnn_results]

        # ── Batch LSTM inference ──────────────────────────────────────────────
        print("[Stage 2/4] LSTM inference...")
        lstm_results = [
            self.lstm.predict(r["recent_weather_df"], biomasses[i])
            for i, r in enumerate(farmer_records)
        ]
        lstm_embeddings = [l[0] for l in lstm_results]
        pred_aqis       = [l[1] for l in lstm_results]

        # ── Batch XGBoost decision ────────────────────────────────────────────
        print("[Stage 3/4] XGBoost decision layer...")
        model_inputs = [
            ModelInputs(
                cnn_embedding       = cnn_embeddings[i],
                biomass_tons_per_ha = biomasses[i],
                lstm_embedding      = lstm_embeddings[i],
                predicted_aqi_lstm  = pred_aqis[i],
                farmer_id           = r["farmer_id"],
                lat                 = r["lat"],
                lon                 = r["lon"],
            )
            for i, r in enumerate(farmer_records)
        ]
        decisions: List[DecisionResult] = self.xgb.predict_batch(model_inputs)

        # ── Batch buyer matching ──────────────────────────────────────────────
        print("[Stage 4/4] Geospatial buyer matching...")
        farmer_needs = [
            FarmerNeed(
                farmer_id           = r["farmer_id"],
                lat                 = r["lat"],
                lon                 = r["lon"],
                biomass_tons        = round(biomasses[i] * r.get("field_area_ha", 2.0), 2),
                biomass_type        = r.get("biomass_type", "rice_straw"),
                harvest_month       = r.get("harvest_month", 10),
                burn_risk_class     = decisions[i].burn_risk_class,
                health_impact_score = decisions[i].health_impact_score,
            )
            for i, r in enumerate(farmer_records)
        ]
        match_reports = self.matcher.match_all(
            farmer_needs, top_k=top_k_buyers, max_dist_km=max_dist_km
        )
        # Index reports by farmer_id for O(1) lookup
        report_map = {rep.farmer_id: rep for rep in match_reports}

        # ── Assemble output DataFrame ─────────────────────────────────────────
        rows = []
        for i, r in enumerate(farmer_records):
            d   = decisions[i]
            rep = report_map.get(r["farmer_id"])
            rows.append({
                "farmer_id"           : d.farmer_id,
                "lat"                 : d.lat,
                "lon"                 : d.lon,
                "biomass_tons_per_ha" : round(biomasses[i], 3),
                "predicted_aqi"       : round(pred_aqis[i], 1),
                "burn_risk_label"     : d.burn_risk_label,
                "burn_risk_class"     : d.burn_risk_class,
                "burn_risk_confidence": d.burn_risk_confidence,
                "health_impact_score" : d.health_impact_score,
                "alert_flag"          : d.alert_flag,
                "intervention_urgency": d.intervention_urgency,
                "best_buyer_name"     : rep.best_match.buyer_name if rep and rep.best_match else "None",
                "best_buyer_distance" : rep.best_match.distance_km if rep and rep.best_match else None,
                "best_buyer_net_rev"  : rep.best_match.net_revenue_inr if rep and rep.best_match else None,
                "n_buyers_available"  : rep.total_buyers_found if rep else 0,
            })

        df = pd.DataFrame(rows).sort_values("health_impact_score", ascending=False)
        return df, match_reports

    def save_outputs(
        self,
        results_df: pd.DataFrame,
        match_reports: List[MatchReport],
        out_dir: str = "outputs",
    ):
        os.makedirs(out_dir, exist_ok=True)

        # Full pipeline results
        csv_path = f"{out_dir}/pipeline_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"[save] Pipeline results → {csv_path}")

        # Alert-only rows
        alerts = results_df[results_df["alert_flag"] == True]
        if not alerts.empty:
            alert_path = f"{out_dir}/alert_farmers.csv"
            alerts.to_csv(alert_path, index=False)
            print(f"[save] Alert farmers ({len(alerts)}) → {alert_path}")

        # Buyer matches JSON
        self.matcher.export_reports(match_reports, out_dir)

        # Summary stats
        print("\n" + "═"*55)
        print("  PIPELINE SUMMARY")
        print("═"*55)
        print(f"  Total farmers processed : {len(results_df)}")
        print(f"  Alert (High+Critical)   : {int(results_df['alert_flag'].sum())}")
        print(f"  Avg health impact score : {results_df['health_impact_score'].mean():.1f}")
        print(f"  Farmers with buyer match: {int((results_df['n_buyers_available'] > 0).sum())}")
        print("═"*55)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO: load CNN/LSTM outputs from CSV (bridge to real models)
# ─────────────────────────────────────────────────────────────────────────────
def load_from_cnn_lstm_csv(csv_path: str) -> List[Dict]:
    """
    If you've already run your CNN and LSTM and saved outputs to a CSV,
    load them here instead of calling the stubs.

    Expected CSV columns:
        farmer_id, lat, lon, field_area_ha, biomass_type, harvest_month,
        biomass_tons_per_ha, predicted_aqi,
        cnn_emb_0 ... cnn_emb_127,        (128 columns)
        lstm_emb_0 ... lstm_emb_63         (64 columns)
    """
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        cnn_emb  = row[[f"cnn_emb_{i}"  for i in range(128)]].values.astype(np.float32)
        lstm_emb = row[[f"lstm_emb_{i}" for i in range(64)]].values.astype(np.float32)
        records.append({
            "farmer_id"       : str(row["farmer_id"]),
            "lat"             : float(row["lat"]),
            "lon"             : float(row["lon"]),
            "field_area_ha"   : float(row.get("field_area_ha", 2.0)),
            "biomass_type"    : str(row.get("biomass_type", "rice_straw")),
            "harvest_month"   : int(row.get("harvest_month", 10)),
            # Pre-computed by your CNN/LSTM — no need to re-run stubs
            "_cnn_embedding"  : cnn_emb,
            "_biomass_ph"     : float(row["biomass_tons_per_ha"]),
            "_lstm_embedding" : lstm_emb,
            "_pred_aqi"       : float(row["predicted_aqi"]),
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _make_demo_records(n: int = 30) -> List[Dict]:
    """Generates synthetic farmer records for demo run."""
    rng     = np.random.default_rng(42)
    farmers = generate_demo_farmers(n)
    return [
        {
            "farmer_id"        : f.farmer_id,
            "lat"              : f.lat,
            "lon"              : f.lon,
            "field_area_ha"    : float(rng.uniform(1, 8)),
            "biomass_type"     : f.biomass_type,
            "harvest_month"    : f.harvest_month,
            # Dummy image and weather (stubs ignore these anyway)
            "satellite_image"  : rng.random((64, 64, 13)).astype(np.float32),
            "recent_weather_df": pd.DataFrame(rng.random((30, 15))),
        }
        for f in farmers
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",         choices=["train", "run"], default="train")
    parser.add_argument("--n-farmers",    type=int,   default=30)
    parser.add_argument("--top-k",        type=int,   default=3)
    parser.add_argument("--max-dist",     type=float, default=150)
    parser.add_argument("--cnn-lstm-csv", type=str,   default="",
                        help="CSV with pre-computed CNN/LSTM outputs")
    args = parser.parse_args()

    if args.mode == "train":
        print("═"*55)
        print("  Training XGBoost Decision Layer")
        print("═"*55)
        train_xgb(n_samples=5000)
        print("\nNext: python pipeline.py --mode run")

    else:
        pipeline = StubbleBurningPipeline()

        if args.cnn_lstm_csv:
            print(f"[pipeline] Loading CNN/LSTM outputs from {args.cnn_lstm_csv}")
            records = load_from_cnn_lstm_csv(args.cnn_lstm_csv)
        else:
            print("[pipeline] Using demo data (replace with real CNN/LSTM calls)")
            records = _make_demo_records(args.n_farmers)

        results_df, match_reports = pipeline.process_batch(
            records, top_k_buyers=args.top_k, max_dist_km=args.max_dist
        )

        pipeline.save_outputs(results_df, match_reports)
        print("\n✅  Pipeline complete. Check outputs/ folder.")
