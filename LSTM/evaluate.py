"""
evaluate.py — Validation, Metrics & Inference
==============================================
Loads the best checkpoint and computes:
  • RMSE  (Root Mean Squared Error on original AQI scale)
  • R²    (Coefficient of Determination)
  • Plots predicted vs actual AQI (saved to outputs/)

Also exposes `predict_aqi()` for integration into the
CNN → LSTM → impact estimation pipeline.
"""

import os
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from model import AQIForecastLSTM
from train import (
    generate_dummy_dataset,
    preprocess,
    create_sequences,
    AQIDataset,
    FEATURE_COLS,
    TARGET_COL,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def load_model_and_scaler(
    ckpt_path: str = "checkpoints/best_lstm.pt",
    scaler_path: str = "checkpoints/scaler.pkl",
    device: torch.device = None,
):
    """Load trained LSTM and MinMaxScaler from disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]

    model = AQIForecastLSTM(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        output_size=cfg["output_size"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scaler = joblib.load(scaler_path)
    print(f"[evaluate] Loaded checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f})")
    return model, scaler, cfg, device


def inverse_scale_aqi(scaled_values: np.ndarray, scaler) -> np.ndarray:
    """
    The scaler was fit on all features; inverse-transform only the AQI column.
    Uses a dummy matrix where all non-AQI columns are zero.
    """
    AQI_IDX = FEATURE_COLS.index(TARGET_COL)
    dummy   = np.zeros((len(scaled_values), len(FEATURE_COLS)))
    dummy[:, AQI_IDX] = scaled_values.ravel()
    return scaler.inverse_transform(dummy)[:, AQI_IDX]


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(
    data_path: str  = "data/aqi_data.csv",
    ckpt_path: str  = "checkpoints/best_lstm.pt",
    scaler_path: str = "checkpoints/scaler.pkl",
    seq_len: int    = 30,
    test_split: float = 0.15,
    output_dir: str = "outputs/",
):
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(data_path):
        generate_dummy_dataset(save_path=data_path)
    df = pd.read_csv(data_path, parse_dates=["date"])

    model, scaler, cfg, device = load_model_and_scaler(ckpt_path, scaler_path)
    forecast_horizon = cfg["output_size"]
    seq_len = cfg["seq_len"]

    scaled, _ = preprocess(df, scaler_path)

    # Re-use same scaler (don't refit on test data)
    scaled = scaler.transform(df[FEATURE_COLS].values).astype(np.float32)
    X, y   = create_sequences(scaled, seq_len, forecast_horizon)

    # Hold-out test split (last `test_split` of data)
    n_test  = int(len(X) * test_split)
    X_test  = X[-n_test:]
    y_test  = y[-n_test:]

    loader = DataLoader(AQIDataset(X_test, y_test), batch_size=128)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            all_preds.append(pred)
            all_true.append(yb.numpy())

    preds_scaled = np.concatenate(all_preds)   # (N, horizon)
    true_scaled  = np.concatenate(all_true)

    # Inverse transform (take first horizon step for single-step metrics)
    preds_aqi = inverse_scale_aqi(preds_scaled[:, 0], scaler)
    true_aqi  = inverse_scale_aqi(true_scaled[:, 0], scaler)

    # ── Metrics ───────────────────────────────────────────────────────────────
    rmse = np.sqrt(mean_squared_error(true_aqi, preds_aqi))
    r2   = r2_score(true_aqi, preds_aqi)

    print("\n" + "═" * 40)
    print("  TEST SET RESULTS")
    print("═" * 40)
    print(f"  RMSE : {rmse:.2f} AQI units")
    print(f"  R²   : {r2:.4f}")
    print("═" * 40)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = os.path.join(output_dir, "aqi_forecast_vs_actual.png")
    plt.figure(figsize=(14, 5))
    plt.plot(true_aqi,  label="Actual AQI",    linewidth=1.2, color="#2196F3")
    plt.plot(preds_aqi, label="Predicted AQI", linewidth=1.2, color="#F44336", alpha=0.8, linestyle="--")
    plt.title(f"LSTM AQI Forecast — RMSE: {rmse:.1f} | R²: {r2:.3f}")
    plt.xlabel("Test Day Index")
    plt.ylabel("AQI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[evaluate] Plot saved → {plot_path}")

    return {"rmse": rmse, "r2": r2, "preds": preds_aqi, "actual": true_aqi}


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Inference Function
# ═══════════════════════════════════════════════════════════════════════════════
def predict_aqi(
    recent_df: pd.DataFrame,
    biomass_prediction: float,
    ckpt_path: str = "checkpoints/best_lstm.pt",
    scaler_path: str = "checkpoints/scaler.pkl",
) -> dict:
    """
    Integration point for the CNN → LSTM pipeline.

    Args:
        recent_df          : DataFrame with the last `seq_len` days of:
                             [aqi, temperature, humidity, wind_speed,
                              day_of_year, month, biomass_tons_per_ha]
        biomass_prediction : float — latest CNN biomass output (tons/ha).
                             Overrides the last row's biomass column to simulate
                             what would happen if predicted biomass were burned.

    Returns:
        dict with keys:
            'predicted_aqi'    : float  — next-day forecast (AQI units)
            'impact_category'  : str    — Good / Moderate / Unhealthy / ...
            'risk_flag'        : bool   — True if predicted AQI > 200
    """
    model, scaler, cfg, device = load_model_and_scaler(ckpt_path, scaler_path)
    seq_len = cfg["seq_len"]

    # Inject CNN biomass into the most recent timestep
    df_input = recent_df[FEATURE_COLS].copy().tail(seq_len)
    df_input.loc[df_input.index[-1], "biomass_tons_per_ha"] = biomass_prediction

    if len(df_input) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows; got {len(df_input)}")

    scaled = scaler.transform(df_input.values).astype(np.float32)
    x_tensor = torch.tensor(scaled).unsqueeze(0).to(device)  # (1, seq_len, features)

    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()            # (1, horizon)

    predicted_aqi = float(inverse_scale_aqi(pred_scaled[:, 0], scaler)[0])

    # AQI category mapping (US EPA standard)
    def aqi_category(v):
        if v <= 50:   return "Good"
        if v <= 100:  return "Moderate"
        if v <= 150:  return "Unhealthy for Sensitive Groups"
        if v <= 200:  return "Unhealthy"
        if v <= 300:  return "Very Unhealthy"
        return "Hazardous"

    return {
        "predicted_aqi"  : round(predicted_aqi, 1),
        "impact_category": aqi_category(predicted_aqi),
        "risk_flag"      : predicted_aqi > 200,
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    results = evaluate()
    print(f"\nSummary — RMSE: {results['rmse']:.2f},  R²: {results['r2']:.4f}")
