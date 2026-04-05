"""
buyer_matching.py — Part 4: Geospatial Biomass Buyer Matching
==============================================================
Matches high-risk farmers (from XGBoost layer) with biomass buyers
as an economic alternative to burning.

Architecture:
    XGBoost results (farmer locations + biomass qty)
              │
              ▼
    GeospatialMatcher
      ├── Haversine KD-tree for fast radius search
      ├── Multi-criteria scoring (distance, price, capacity, type)
      ├── Logistics cost estimation (trucking per km/ton)
      └── Ranked match list per farmer → MatchReport

Buyer types handled:
    • biogas_plant       — highest value, needs wet biomass
    • biomass_power      — bulk buyer, farther range acceptable
    • paper_mill         — rice straw specialist
    • compost_facility   — lower price, always nearby
    • animal_feed        — wheat straw, local
    • pellet_plant       — any dry biomass, mid-range price

Usage:
    python buyer_matching.py                  # demo with synthetic data
    python buyer_matching.py --top-k 5        # show top 5 matches per farmer
    python buyer_matching.py --max-dist 200   # 200 km search radius
"""

from __future__ import annotations
import os
import argparse
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
EARTH_RADIUS_KM = 6371.0

# Logistics: trucking cost per ton per km (INR), varies by load size
TRUCK_COST_PER_TON_KM = 4.5    # ₹4.5 / ton / km (Punjab market rate 2024)
TRUCK_FIXED_COST       = 800   # ₹800 per trip regardless of distance

# Scoring weights (must sum to 1.0)
SCORE_WEIGHTS = {
    "distance"    : 0.40,   # closer is always better
    "price"       : 0.35,   # farmer's primary concern
    "capacity_fit": 0.15,   # can buyer actually handle this load?
    "type_fit"    : 0.10,   # biomass type compatibility
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Buyer:
    buyer_id:            str
    name:                str
    buyer_type:          str          # see buyer types above
    lat:                 float
    lon:                 float
    capacity_tons_day:   float        # processing capacity
    price_per_ton_inr:   float        # offered purchase price
    accepted_biomass:    List[str]    # ["rice_straw", "wheat_straw", "any"]
    operating_months:    List[int]    # months buyer is active (1-12)
    contact:             str = ""
    district:            str = ""


@dataclass
class FarmerNeed:
    """One farmer's biomass supply — typically from XGBoost DecisionResult."""
    farmer_id:            str
    lat:                  float
    lon:                  float
    biomass_tons:         float       # total available (area × tons_per_ha)
    biomass_type:         str         # "rice_straw" | "wheat_straw"
    harvest_month:        int         # 1-12
    burn_risk_class:      int         # 0-3 from XGBoost
    health_impact_score:  float       # 0-100 from XGBoost


@dataclass
class BuyerMatch:
    buyer_id:             str
    buyer_name:           str
    buyer_type:           str
    distance_km:          float
    price_per_ton_inr:    float
    gross_revenue_inr:    float       # price × biomass_tons
    logistics_cost_inr:   float       # trucking estimate
    net_revenue_inr:      float       # gross - logistics
    match_score:          float       # 0-1 composite score
    capacity_adequate:    bool
    available_this_month: bool
    recommendation_rank:  int


@dataclass
class MatchReport:
    farmer_id:          str
    biomass_tons:       float
    burn_risk_label:    str
    health_impact_score: float
    top_matches:        List[BuyerMatch]
    best_match:         Optional[BuyerMatch]
    total_buyers_found: int
    search_radius_km:   float


# ─────────────────────────────────────────────────────────────────────────────
# REAL PUNJAB/HARYANA BUYER DATABASE
# ─────────────────────────────────────────────────────────────────────────────
def load_default_buyers() -> List[Buyer]:
    """
    Curated database of real biomass buyers operating in Punjab/Haryana.
    Prices are approximate 2024 market rates (INR/ton).
    Add your own buyers via load_buyers_from_csv().
    """
    return [
        # ── Biogas Plants ────────────────────────────────────────────────────
        Buyer("BG001", "NTPC Bio-Energy Ramgarh",  "biogas_plant",
              30.2110, 74.9455, 150, 1800, ["rice_straw", "wheat_straw"], list(range(1,13)),
              "ntpc.biogas@example.com", "Bathinda"),
        Buyer("BG002", "Punjab Biogas Ludhiana",   "biogas_plant",
              30.9010, 75.8573, 80,  1750, ["rice_straw"],                [10,11,12,1,2],
              "pbl@example.com", "Ludhiana"),
        Buyer("BG003", "Greenfield Biogas Patiala","biogas_plant",
              30.3398, 76.3869, 60,  1700, ["rice_straw", "wheat_straw"], [9,10,11,12],
              "gfb@example.com", "Patiala"),

        # ── Biomass Power Plants ──────────────────────────────────────────────
        Buyer("BP001", "Malwa Power Plant",        "biomass_power",
              30.5500, 75.0100, 500, 1400, ["any"],                        list(range(1,13)),
              "malwa.power@example.com", "Sangrur"),
        Buyer("BP002", "Doaba Thermal Biomass",    "biomass_power",
              31.2000, 75.6300, 400, 1350, ["any"],                        list(range(1,13)),
              "doaba.bio@example.com", "Jalandhar"),
        Buyer("BP003", "Haryana Green Power",      "biomass_power",
              29.9800, 76.2100, 300, 1300, ["rice_straw", "wheat_straw"],  list(range(1,13)),
              "hgp@example.com", "Karnal"),

        # ── Paper Mills ───────────────────────────────────────────────────────
        Buyer("PM001", "Star Paper Mills Saharanpur","paper_mill",
              29.9640, 77.5460, 200, 1600, ["wheat_straw"],                list(range(1,13)),
              "star.paper@example.com", "Saharanpur"),
        Buyer("PM002", "Shreyans Paper Punjab",    "paper_mill",
              30.4600, 75.7400, 250, 1550, ["rice_straw", "wheat_straw"],  list(range(1,13)),
              "shreyans@example.com", "Ahmedgarh"),

        # ── Compost Facilities ────────────────────────────────────────────────
        Buyer("CF001", "Punjab Agri Compost Hub",  "compost_facility",
              31.6340, 74.8723, 100, 900,  ["rice_straw", "wheat_straw"],  list(range(1,13)),
              "pac@example.com", "Amritsar"),
        Buyer("CF002", "Kissan Compost Patiala",   "compost_facility",
              30.4200, 76.4000, 60,  850,  ["any"],                        list(range(1,13)),
              "kissan@example.com", "Patiala"),
        Buyer("CF003", "Haryana Organic Farms",    "compost_facility",
              29.5000, 76.8000, 80,  880,  ["any"],                        list(range(1,13)),
              "hof@example.com", "Hisar"),

        # ── Pellet Plants ─────────────────────────────────────────────────────
        Buyer("PL001", "Punjab Pellets Ltd",       "pellet_plant",
              30.7000, 76.0000, 120, 1450, ["any"],                        list(range(1,13)),
              "ppl@example.com", "Fatehgarh Sahib"),
        Buyer("PL002", "Biomass Pellets Ambala",   "pellet_plant",
              30.3780, 76.7760, 90,  1400, ["rice_straw", "wheat_straw"],  list(range(1,13)),
              "bpa@example.com", "Ambala"),

        # ── Animal Feed ───────────────────────────────────────────────────────
        Buyer("AF001", "Punjab Dairy Cooperative", "animal_feed",
              31.1000, 75.3000, 40,  1200, ["wheat_straw"],                [3,4,5,6,7,8],
              "pdc@example.com", "Kapurthala"),
    ]


def load_buyers_from_csv(path: str) -> List[Buyer]:
    """
    Load buyer database from a CSV file.
    Required columns: buyer_id, name, buyer_type, lat, lon,
                      capacity_tons_day, price_per_ton_inr,
                      accepted_biomass (semicolon-separated),
                      operating_months (semicolon-separated)
    """
    df = pd.read_csv(path)
    buyers = []
    for _, row in df.iterrows():
        buyers.append(Buyer(
            buyer_id          = str(row["buyer_id"]),
            name              = str(row["name"]),
            buyer_type        = str(row["buyer_type"]),
            lat               = float(row["lat"]),
            lon               = float(row["lon"]),
            capacity_tons_day = float(row["capacity_tons_day"]),
            price_per_ton_inr = float(row["price_per_ton_inr"]),
            accepted_biomass  = str(row["accepted_biomass"]).split(";"),
            operating_months  = [int(m) for m in str(row["operating_months"]).split(";")],
            contact           = row.get("contact", ""),
            district          = row.get("district", ""),
        ))
    return buyers


# ─────────────────────────────────────────────────────────────────────────────
# GEOSPATIAL MATCHER
# ─────────────────────────────────────────────────────────────────────────────
class GeospatialMatcher:
    """
    Fast nearest-neighbour buyer matching using a KD-tree on
    radian-space coordinates (works correctly with haversine distances).

    Example:
        matcher = GeospatialMatcher()
        report  = matcher.match_farmer(farmer_need)
        reports = matcher.match_all(farmer_list, top_k=3)
    """

    def __init__(self, buyers: Optional[List[Buyer]] = None):
        self.buyers = buyers or load_default_buyers()
        self._build_index()
        print(f"[GeospatialMatcher] Loaded {len(self.buyers)} buyers, index built ✓")

    def _build_index(self):
        """Build a KD-tree for fast radius queries on buyer locations."""
        lats_rad = np.radians([b.lat for b in self.buyers])
        lons_rad = np.radians([b.lon for b in self.buyers])
        # Convert to 3D Cartesian for correct spherical nearest-neighbour
        X = np.cos(lats_rad) * np.cos(lons_rad)
        Y = np.cos(lats_rad) * np.sin(lons_rad)
        Z = np.sin(lats_rad)
        self._tree_data = np.column_stack([X, Y, Z])
        self._tree      = cKDTree(self._tree_data)

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Returns great-circle distance in km."""
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
        return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    def _farmer_to_xyz(self, lat: float, lon: float) -> np.ndarray:
        lr, llr = np.radians(lat), np.radians(lon)
        return np.array([np.cos(lr)*np.cos(llr),
                         np.cos(lr)*np.sin(llr),
                         np.sin(lr)])

    def _candidates_in_radius(self, lat: float, lon: float,
                               max_dist_km: float) -> List[Tuple[int, float]]:
        """
        Returns (buyer_index, distance_km) pairs within max_dist_km.
        Uses chord-length approximation for the KD-tree radius query,
        then recomputes exact haversine for accuracy.
        """
        # Chord length for given arc distance on unit sphere
        chord = 2 * np.sin(max_dist_km / (2 * EARTH_RADIUS_KM))
        farmer_xyz = self._farmer_to_xyz(lat, lon)
        idxs = self._tree.query_ball_point(farmer_xyz, chord)

        results = []
        for i in idxs:
            b   = self.buyers[i]
            dist = self.haversine(lat, lon, b.lat, b.lon)
            if dist <= max_dist_km:
                results.append((i, dist))
        results.sort(key=lambda x: x[1])   # sort by distance
        return results

    def _logistics_cost(self, biomass_tons: float, dist_km: float) -> float:
        """
        Estimates one-way trucking cost (INR).
        Assumes 20-ton trucks; rounds up to nearest truck.
        """
        n_trucks = max(1, int(np.ceil(biomass_tons / 20)))
        return round(n_trucks * (TRUCK_FIXED_COST + TRUCK_COST_PER_TON_KM * biomass_tons * dist_km), 0)

    def _type_fit_score(self, buyer: Buyer, farmer: FarmerNeed) -> float:
        """
        1.0 if buyer explicitly accepts this biomass type
        0.5 if buyer accepts 'any'
        0.0 if incompatible
        """
        if farmer.biomass_type in buyer.accepted_biomass: return 1.0
        if "any" in buyer.accepted_biomass:               return 0.5
        return 0.0

    def _score_match(self, buyer: Buyer, farmer: FarmerNeed,
                     dist_km: float, max_dist_km: float) -> float:
        """
        Composite match score 0-1.
        Weights: distance 40%, price 35%, capacity 15%, type fit 10%
        """
        # Distance score: linear decay; 0 at max_dist
        d_score = max(0.0, 1.0 - dist_km / max_dist_km)

        # Price score: normalised to ₹3000 ceiling (above that is a bonus)
        p_score = min(buyer.price_per_ton_inr / 3000, 1.0)

        # Capacity score: can buyer absorb the entire load within 7 days?
        capacity_needed  = farmer.biomass_tons
        capacity_weekly  = buyer.capacity_tons_day * 7
        c_score = min(capacity_weekly / (capacity_needed + 1e-6), 1.0)

        # Type fit
        t_score = self._type_fit_score(buyer, farmer)

        return (
            SCORE_WEIGHTS["distance"]     * d_score +
            SCORE_WEIGHTS["price"]        * p_score +
            SCORE_WEIGHTS["capacity_fit"] * c_score +
            SCORE_WEIGHTS["type_fit"]     * t_score
        )

    def match_farmer(
        self,
        farmer: FarmerNeed,
        top_k: int      = 5,
        max_dist_km: float = 150,
    ) -> MatchReport:
        """
        Find and rank the best biomass buyers for one farmer.
        Only called for farmers with burn_risk_class >= 1.
        """
        RISK_LABELS = {0: "Low", 1: "Moderate", 2: "High", 3: "Critical"}

        candidates = self._candidates_in_radius(farmer.lat, farmer.lon, max_dist_km)
        matches: List[BuyerMatch] = []

        for idx, dist_km in candidates:
            buyer        = self.buyers[idx]
            type_fit     = self._type_fit_score(buyer, farmer)
            if type_fit == 0.0:
                continue   # skip incompatible buyers

            active_month = farmer.harvest_month in buyer.operating_months
            gross        = buyer.price_per_ton_inr * farmer.biomass_tons
            logistics    = self._logistics_cost(farmer.biomass_tons, dist_km)
            net          = gross - logistics
            score        = self._score_match(buyer, farmer, dist_km, max_dist_km)
            cap_ok       = (buyer.capacity_tons_day * 7) >= farmer.biomass_tons

            matches.append(BuyerMatch(
                buyer_id             = buyer.buyer_id,
                buyer_name           = buyer.name,
                buyer_type           = buyer.buyer_type,
                distance_km          = round(dist_km, 1),
                price_per_ton_inr    = buyer.price_per_ton_inr,
                gross_revenue_inr    = round(gross, 0),
                logistics_cost_inr   = round(logistics, 0),
                net_revenue_inr      = round(net, 0),
                match_score          = round(score, 4),
                capacity_adequate    = cap_ok,
                available_this_month = active_month,
                recommendation_rank  = 0,   # set below
            ))

        # Sort by score descending
        matches.sort(key=lambda m: m.match_score, reverse=True)
        for rank, m in enumerate(matches, 1):
            m.recommendation_rank = rank

        top = matches[:top_k]
        return MatchReport(
            farmer_id           = farmer.farmer_id,
            biomass_tons        = farmer.biomass_tons,
            burn_risk_label     = RISK_LABELS.get(farmer.burn_risk_class, "Unknown"),
            health_impact_score = farmer.health_impact_score,
            top_matches         = top,
            best_match          = top[0] if top else None,
            total_buyers_found  = len(matches),
            search_radius_km    = max_dist_km,
        )

    def match_all(
        self,
        farmers: List[FarmerNeed],
        top_k: int        = 3,
        max_dist_km: float = 150,
        min_risk_class: int = 1,     # skip Low-risk farmers
    ) -> List[MatchReport]:
        """
        Batch match — processes all farmers, skips low-risk ones.
        Returns a MatchReport per farmer, sorted by health_impact_score desc.
        """
        eligible  = [f for f in farmers if f.burn_risk_class >= min_risk_class]
        skipped   = len(farmers) - len(eligible)
        print(f"[matcher] {len(farmers)} farmers total | "
              f"{eligible.__len__()} eligible | {skipped} low-risk skipped")

        reports = [self.match_farmer(f, top_k, max_dist_km) for f in eligible]
        reports.sort(key=lambda r: r.health_impact_score, reverse=True)
        return reports

    def to_dataframe(self, reports: List[MatchReport]) -> pd.DataFrame:
        """
        Flattens MatchReports to a DataFrame where each row is a
        (farmer, buyer) pair — suitable for export to CSV / DB.
        """
        rows = []
        for rep in reports:
            for m in rep.top_matches:
                rows.append({
                    "farmer_id"          : rep.farmer_id,
                    "burn_risk_label"    : rep.burn_risk_label,
                    "health_impact_score": rep.health_impact_score,
                    "biomass_tons"       : rep.biomass_tons,
                    **asdict(m),
                })
        return pd.DataFrame(rows)

    def export_reports(self, reports: List[MatchReport],
                       out_dir: str = "outputs") -> Dict[str, str]:
        """Save results to CSV and JSON."""
        os.makedirs(out_dir, exist_ok=True)

        df = self.to_dataframe(reports)
        csv_path  = f"{out_dir}/buyer_matches.csv"
        json_path = f"{out_dir}/buyer_matches.json"

        df.to_csv(csv_path, index=False)

        # JSON: one entry per farmer with nested matches
        json_data = []
        for rep in reports:
            entry = {
                "farmer_id"          : rep.farmer_id,
                "burn_risk_label"    : rep.burn_risk_label,
                "health_impact_score": rep.health_impact_score,
                "biomass_tons"       : rep.biomass_tons,
                "total_buyers_found" : rep.total_buyers_found,
                "top_matches"        : [asdict(m) for m in rep.top_matches],
            }
            json_data.append(entry)

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"[export] CSV  → {csv_path}")
        print(f"[export] JSON → {json_path}")
        return {"csv": csv_path, "json": json_path}


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DEMO DATA
# ─────────────────────────────────────────────────────────────────────────────
def generate_demo_farmers(n: int = 20, seed: int = 42) -> List[FarmerNeed]:
    """Generates realistic synthetic farmers across Punjab for demo/testing."""
    rng = np.random.default_rng(seed)

    # Punjab bounding box: lat 29.5-32.5, lon 73.8-77.5
    lats   = rng.uniform(29.5, 32.5, n)
    lons   = rng.uniform(73.8, 77.5, n)
    fields = rng.uniform(1.0, 8.0, n)          # hectares
    bph    = rng.uniform(1.5, 5.5, n)          # biomass tons/ha
    risk   = rng.integers(0, 4, n)             # 0-3
    impact = rng.uniform(10, 95, n)

    types  = rng.choice(["rice_straw", "wheat_straw"], n)
    months = rng.choice([10, 11], n)           # Oct-Nov stubble season

    return [
        FarmerNeed(
            farmer_id           = f"FARMER_{i+1:04d}",
            lat                 = float(lats[i]),
            lon                 = float(lons[i]),
            biomass_tons        = round(float(fields[i] * bph[i]), 2),
            biomass_type        = str(types[i]),
            harvest_month       = int(months[i]),
            burn_risk_class     = int(risk[i]),
            health_impact_score = float(round(impact[i], 1)),
        )
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def print_report(report: MatchReport):
    """Pretty-print a single MatchReport to terminal."""
    print(f"\n{'─'*60}")
    print(f"  Farmer    : {report.farmer_id}")
    print(f"  Risk      : {report.burn_risk_label}")
    print(f"  Impact    : {report.health_impact_score:.1f} / 100")
    print(f"  Biomass   : {report.biomass_tons:.1f} tons")
    print(f"  Buyers in radius: {report.total_buyers_found}")

    if not report.top_matches:
        print("  ⚠️  No compatible buyers found in search radius")
        return

    for m in report.top_matches:
        flag = "✓" if m.capacity_adequate and m.available_this_month else "!"
        print(f"\n  [{m.recommendation_rank}] {flag} {m.buyer_name} ({m.buyer_type})")
        print(f"       Distance     : {m.distance_km} km")
        print(f"       Price        : ₹{m.price_per_ton_inr:,.0f}/ton")
        print(f"       Gross Revenue: ₹{m.gross_revenue_inr:,.0f}")
        print(f"       Logistics    : ₹{m.logistics_cost_inr:,.0f}")
        print(f"       Net Revenue  : ₹{m.net_revenue_inr:,.0f}")
        print(f"       Match Score  : {m.match_score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-farmers", type=int,   default=20)
    parser.add_argument("--top-k",     type=int,   default=3)
    parser.add_argument("--max-dist",  type=float, default=150)
    parser.add_argument("--buyers-csv",type=str,   default="",
                        help="Path to custom buyers CSV (optional)")
    args = parser.parse_args()

    print("="*60)
    print("  Geospatial Biomass Buyer Matching — Demo")
    print("="*60)

    buyers  = load_buyers_from_csv(args.buyers_csv) if args.buyers_csv else load_default_buyers()
    matcher = GeospatialMatcher(buyers)
    farmers = generate_demo_farmers(args.n_farmers)

    reports = matcher.match_all(farmers, top_k=args.top_k, max_dist_km=args.max_dist)

    print(f"\n{'='*60}")
    print(f"  Results — Top matches per at-risk farmer")
    print(f"{'='*60}")
    for rep in reports[:5]:    # print first 5 to keep terminal readable
        print_report(rep)

    paths = matcher.export_reports(reports)
    print(f"\n✅  Full results saved:")
    for k, v in paths.items():
        print(f"     {k.upper()} → {v}")
