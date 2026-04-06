# 🌾 GreenKisan — AI-Powered Biomass Intelligence Platform

> Satellite imagery → Biomass prediction → Burn risk assessment → Buyer matching

GreenKisan is an end-to-end machine learning pipeline that uses Sentinel-2 satellite data to predict crop biomass, forecast air quality impact, assess stubble burning risk, and match farmers with the nearest biomass buyers — all from a single GPS coordinate.

---

## 🚀 Pipeline Overview

```
GPS Coordinates (lat, lon)
         │
         ▼
┌─────────────────────┐
│   CNN Model         │  ← Sentinel-2 satellite patch (5×128×128)
│   ResNet18 backbone │  → Biomass prediction (t/ha)
│   MAE: 0.1383 t/ha  │  → CNN embedding (128,)
│   Accuracy: 91.55%  │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   LSTM Model        │  ← 30-day time-series (7 features)
│   AQIForecastLSTM   │  → Predicted AQI
│   2-layer, 128 units│  → Pollution trend (increasing/stable/decreasing)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   XGBoost           │  ← CNN + LSTM embeddings + tabular features
│   Decision Layer    │  → Burn risk class (0-3)
│                     │  → Health impact score (0-100)
│                     │  → Intervention urgency
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Geospatial        │  ← Farmer location + biomass quantity
│   Buyer Matching    │  → Top-K buyers ranked by distance + price
│   KD-tree search    │  → Net revenue after logistics cost
└─────────────────────┘
```

---

## 📁 Project Structure

```
Green_Kissan/
│
├── main.py                        ← Master pipeline (entry point)
│
├── CNN/
│   ├── cnn_model.py               ← ResNet18 CNN architecture
│   ├── torch_dataset.py           ← BiomassDataset + normalization
│   ├── train_cnn.py               ← Training loop (two-phase)
│   ├── inference.py               ← CNNInferenceEngine
│   ├── sentinel_downloader.py     ← Planetary Computer downloader
│   ├── real_data_pipeline.py      ← Sentinel-2 data generation
│   ├── dataset_builder.py         ← Dataset builder
│   └── test.py                    ← Model evaluation
│
├── LSTM/
│   ├── model.py                   ← AQIForecastLSTM architecture
│   ├── train.py                   ← LSTM training loop
│   ├── evaluate.py                ← LSTM evaluation
│   └── download_data.py           ← AQI data downloader
│
├── XGBoost/
│   ├── xgboost_decision.py        ← XGBoostDecisionLayer
│   ├── pipeline.py                ← StubbleBurningPipeline
│   └── buyer_matching.py          ← GeospatialMatcher + buyer DB
│
├── Checkpoints/                   ← Trained model weights (not in repo)
│   ├── cnn_biomass.pt
│   └── label_stats.npy
│
├── Data/                          ← Satellite patches (not in repo)
│   └── metadata.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Model Performance

| Model | Metric | Value |
|---|---|---|
| CNN | MAE | 0.1383 t/ha |
| CNN | RMSE | 0.1736 t/ha |
| CNN | R² Score | 0.9448 |
| CNN | Accuracy | 91.55% |
| CNN | MSE (normalised) | 0.0712 |
| CNN | Training data | 3000 Sentinel-2 patches |
| CNN | Backbone | ResNet18 (pretrained) |
| CNN | Input channels | 5 (B02, B03, B04, B08, NDVI) |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/aryantripathi0527/Green_Kissan.git
cd Green_Kissan

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Quick Start

### Run the full pipeline
```bash
python main.py --lat 30.901 --lon 75.857 --date 2023-10-21 --farmer-id FARM_001
```

### With custom options
```bash
python main.py \
  --lat 30.901 \
  --lon 75.857 \
  --date 2023-10-21 \
  --farmer-id FARM_001 \
  --crop rice_straw \
  --field-area 2.5 \
  --harvest-month 10 \
  --top-k 3
```

### Expected output
```
============================================================
  GreenKisan Master Pipeline
  Farmer   : FARM_001
  Location : 30.901, 75.857
  Date     : 2023-10-21
============================================================

  ── CNN ─────────────────────────────────────
  Biomass        : 3.2509 t/ha
  NDVI           : 0.1288
  Cloud cover    : 0.0%

  ── LSTM ────────────────────────────────────
  Predicted AQI  : 0.89
  Trend          : decreasing
  Confidence     : 0.564

  ── XGBoost ─────────────────────────────────
  Burn Risk      : Moderate
  Health Impact  : 13.6/100
  Alert          : 🟢 NO

  ── Top 3 Buyers ────────────────────────────
  1. Punjab Biogas Ludhiana    Score: 0.854
     ₹1,750/ton  |  0.0 km  |  Net: ₹10,574
  2. Punjab Pellets Ltd        Score: 0.699
     ₹1,450/ton  |  26.2 km  |  Net: ₹7,859
  3. Shreyans Paper Punjab     Score: 0.697
     ₹1,550/ton  |  50.3 km  |  Net: ₹7,804
============================================================
```

---

## 🧠 Training

### Train CNN
```bash
cd CNN
python train_cnn.py --epochs 80 --lr 5e-4 --patience 20 --batch-size 16
```

### Train LSTM
```bash
cd LSTM
python train.py
```

### Train XGBoost
```bash
cd XGBoost
python pipeline.py --mode train
```

---

## 🔧 CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--lat` | 30.901 | Farm latitude |
| `--lon` | 75.857 | Farm longitude |
| `--date` | 2023-10-21 | Target date (YYYY-MM-DD) |
| `--farmer-id` | FARM_001 | Unique farm identifier |
| `--crop` | rice_straw | Crop type (rice_straw / wheat_straw) |
| `--field-area` | 2.0 | Farm size in hectares |
| `--harvest-month` | 10 | Harvest month (1-12) |
| `--top-k` | 3 | Number of buyer matches |

---

## 🌍 Data Sources

| Source | Usage |
|---|---|
| Sentinel-2 (ESA) | Satellite imagery via Planetary Computer |
| Planetary Computer | Free satellite data API (no key required) |
| Punjab/Haryana Buyer DB | 14 real biomass buyers (hardcoded) |

---

## 🤝 Buyer Types Supported

| Type | Price Range | Accepts |
|---|---|---|
| Biogas Plant | ₹1,700-1,800/ton | Rice + Wheat straw |
| Biomass Power | ₹1,300-1,400/ton | Any biomass |
| Paper Mill | ₹1,550-1,600/ton | Rice + Wheat straw |
| Pellet Plant | ₹1,400-1,450/ton | Any biomass |
| Compost Facility | ₹850-900/ton | Any biomass |
| Animal Feed | ₹1,200/ton | Wheat straw |

---

## 📦 Requirements

- Python 3.10+
- PyTorch 2.0+
- XGBoost 2.0+
- Planetary Computer (free, no API key needed)
- 4GB RAM minimum

---

## 👤 Author

**Aryan Tripathi**
- GitHub: [@aryantripathi0527](https://github.com/aryantripathi0527)

---

## 📄 License

MIT License — free to use and modify.
