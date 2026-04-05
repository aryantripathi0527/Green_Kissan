"""
model.py — LSTM Architecture for AQI Pollution Forecasting
============================================================
Pipeline: CNN (biomass) → LSTM (AQI forecast) → Impact Estimation

How LSTM uses temporal dependencies:
  - The hidden state (h_t) carries a compressed memory of all past timesteps.
  - The cell state (c_t) acts as a long-term conveyor belt, selectively retaining
    or forgetting information via three learned gates:
      * Forget gate  : decides what past context to erase
      * Input gate   : decides what new information to write
      * Output gate  : decides what part of the cell state to expose as h_t
  - This lets the model learn multi-scale patterns:
      short-term (yesterday's AQI spike) and seasonal (stubble burning peaks in Oct-Nov).
  - The CNN biomass output feeds into each timestep alongside weather/AQI features,
    so the LSTM learns how accumulated biomass interacts with meteorological conditions
    to drive future pollution levels.
"""

import torch
import torch.nn as nn


class AQIForecastLSTM(nn.Module):
    """
    Multi-layer LSTM that maps a sequence of environmental features
    (including CNN-predicted biomass) to next-day or next-week AQI.

    Args:
        input_size   : number of features per timestep (default 7)
        hidden_size  : LSTM hidden units
        num_layers   : stacked LSTM layers
        output_size  : forecast horizon (1 = next day, 7 = next week)
        dropout      : dropout between LSTM layers (applied if num_layers > 1)
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # --- Feature projection (optional but helps with diverse feature scales) ---
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )

        # --- Core LSTM ---
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- Regression head ---
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, seq_len, input_size)
        Returns:
            out : (batch, output_size)  — predicted AQI value(s)
        """
        # Project each timestep independently
        x_proj = self.input_proj(x)          # (batch, seq_len, hidden_size)

        # LSTM forward — we only use the final hidden state
        lstm_out, _ = self.lstm(x_proj)      # (batch, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]     # (batch, hidden_size)

        return self.head(last_hidden)        # (batch, output_size)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    batch, seq_len, n_features = 32, 30, 7
    model = AQIForecastLSTM(input_size=n_features, output_size=1)
    dummy = torch.randn(batch, seq_len, n_features)
    out = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")          # expect (32, 1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
