"""
train.py — Data Preprocessing, Sequence Creation & Training Loop
=================================================================
Runs the full pipeline:
  1. Generate / load dummy dataset
  2. Scale features
  3. Create sliding-window sequences
  4. Train LSTM with early stopping
  5. Save best checkpoint
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

from model import AQIForecastLSTM

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DUMMY DATASET GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
def generate_dummy_dataset(n_days: int = 730, save_path: str = "data/aqi_data.csv") -> pd.DataFrame:
    """
    Generates a synthetic daily dataset that mimics real stubble-burning
    pollution data in the Indo-Gangetic Plain (Punjab / Haryana region).

    Columns
    -------
    date          : daily timestamps
    aqi           : Air Quality Index (target, 0–500+)
    temperature   : °C  (25–45 summer, 5–20 winter)
    humidity      : %   (20–90)
    wind_speed    : m/s (0–15)
    day_of_year   : 1–365  (cyclical time feature)
    month         : 1–12
    biomass_tons_per_ha : CNN output — predicted biomass available for burning
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    doy   = dates.day_of_year.values          # 1–365
    month = dates.month.values

    # Seasonal temperature (°C)
    temp = 28 + 15 * np.sin(2 * np.pi * (doy - 80) / 365) + np.random.normal(0, 2, n_days)

    # Humidity (%) — inversely correlated with temp, higher in monsoon
    humidity = 55 - 20 * np.sin(2 * np.pi * (doy - 80) / 365) + np.random.normal(0, 5, n_days)
    humidity = np.clip(humidity, 10, 95)

    # Wind speed (m/s)
    wind_speed = 3 + 2 * np.cos(2 * np.pi * doy / 365) + np.random.exponential(1, n_days)
    wind_speed = np.clip(wind_speed, 0, 15)

    # Biomass (tons/ha) — peaks in Oct-Nov stubble burning season
    biomass = (
        1.5
        + 3.5 * np.exp(-((doy - 305) ** 2) / (2 * 20**2))   # Kharif harvest peak ~Nov 1
        + 1.0 * np.exp(-((doy - 130) ** 2) / (2 * 15**2))   # Rabi harvest peak ~May 10
        + np.random.exponential(0.2, n_days)
    )
    biomass = np.clip(biomass, 0.1, 6.0)

    # AQI — driven by biomass, inversely by wind, seasonally elevated in winter
    aqi = (
        80                                          # baseline
        + 40 * biomass                              # biomass burning contribution
        - 8 * wind_speed                            # dispersion by wind
        + 30 * np.sin(2 * np.pi * (doy - 305) / 365)  # stubble season bump
        + np.random.normal(0, 15, n_days)
    )
    # Lag effect: yesterday's pollution lingers
    for i in range(1, n_days):
        aqi[i] += 0.3 * aqi[i - 1]
    aqi = np.clip(aqi, 20, 500)

    df = pd.DataFrame({
        "date"               : dates,
        "aqi"                : aqi.round(1),
        "temperature"        : temp.round(1),
        "humidity"           : humidity.round(1),
        "wind_speed"         : wind_speed.round(2),
        "day_of_year"        : doy,
        "month"              : month,
        "biomass_tons_per_ha": biomass.round(3),
    })

    df.to_csv(save_path, index=False)
    print(f"[data] Dummy dataset saved → {save_path}  ({n_days} rows)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = ["aqi", "temperature", "humidity", "wind_speed",
                "day_of_year", "month", "biomass_tons_per_ha"]
TARGET_COL   = "aqi"


def preprocess(df: pd.DataFrame, scaler_path: str = "checkpoints/scaler.pkl"):
    """
    Scale all features with MinMaxScaler.
    Returns scaled numpy array and fitted scaler.
    """
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURE_COLS].values)
    joblib.dump(scaler, scaler_path)
    print(f"[preprocess] Scaler saved → {scaler_path}")
    return scaled, scaler


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SLIDING WINDOW SEQUENCE CREATOR
# ═══════════════════════════════════════════════════════════════════════════════
def create_sequences(data: np.ndarray, seq_len: int, forecast_horizon: int):
    """
    Sliding window over time-series data.

    Each sample:
      X : (seq_len, n_features)   — past `seq_len` days of all features
      y : scalar or vector        — next `forecast_horizon` AQI values

    The target index (AQI) is column 0 in FEATURE_COLS.

    Example with seq_len=30, forecast_horizon=1:
      X[0] = data[0:30]    →  y[0] = data[30, aqi_col]
      X[1] = data[1:31]    →  y[1] = data[31, aqi_col]
      ...
    """
    AQI_IDX = FEATURE_COLS.index(TARGET_COL)
    X, y = [], []
    total = len(data) - seq_len - forecast_horizon + 1
    for i in range(total):
        X.append(data[i : i + seq_len])
        # Multi-step: collect next `forecast_horizon` AQI values
        y.append(data[i + seq_len : i + seq_len + forecast_horizon, AQI_IDX])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════════════════════
class AQIDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def train(
    seq_len: int          = 30,
    forecast_horizon: int = 1,
    hidden_size: int      = 128,
    num_layers: int       = 2,
    dropout: float        = 0.2,
    lr: float             = 1e-3,
    epochs: int           = 60,
    batch_size: int       = 64,
    patience: int         = 10,
    data_path: str        = "data/aqi_data.csv",
    ckpt_path: str        = "checkpoints/best_lstm.pt",
):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    if not os.path.exists(data_path):
        generate_dummy_dataset(save_path=data_path)
    df = pd.read_csv(data_path, parse_dates=["date"])

    scaled, scaler = preprocess(df)
    X, y = create_sequences(scaled, seq_len, forecast_horizon)
    print(f"[train] Sequences — X: {X.shape}, y: {y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)

    train_loader = DataLoader(AQIDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(AQIDataset(X_val,   y_val),   batch_size=batch_size)

    # ── Model ────────────────────────────────────────────────────────────────
    model = AQIForecastLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=forecast_horizon,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)
    criterion = nn.HuberLoss()   # robust to occasional AQI outliers (smoke events)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'LR':>10}")
    print("─" * 46)

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"{epoch:>6} {train_loss:>12.5f} {val_loss:>12.5f} {current_lr:>10.6f}")

        # ── Early Stopping ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": {
                    "input_size": len(FEATURE_COLS),
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "output_size": forecast_horizon,
                    "dropout": dropout,
                    "seq_len": seq_len,
                },
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[train] Early stopping at epoch {epoch} (best val loss: {best_val_loss:.5f})")
                break

    print(f"\n[train] Best checkpoint saved → {ckpt_path}")
    return history


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    history = train(
        seq_len=30,
        forecast_horizon=1,   # change to 7 for next-week prediction
        epochs=60,
        batch_size=64,
        patience=10,
    )
