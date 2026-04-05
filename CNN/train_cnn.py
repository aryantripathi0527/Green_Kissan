"""
train_cnn.py
============
Training pipeline for BiomassCNN (ResNet18 backbone)
Two-phase: freeze backbone → train head → unfreeze → fine-tune
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cnn_model import BiomassCNN, save_checkpoint, load_checkpoint
from torch_dataset import BiomassDataset, compute_dataset_stats

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs",     exist_ok=True)


# -----------------------------------------------------------------------------
# DATA SPLIT
# -----------------------------------------------------------------------------
def temporal_split(dataset, train_ratio=0.8, val_ratio=0.1):
    n    = len(dataset)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    n_te = n - n_tr - n_va
    logger.info(f"[Split] train={n_tr}  val={n_va}  test={n_te}")
    return (
        Subset(dataset, range(0,           n_tr)),
        Subset(dataset, range(n_tr,        n_tr + n_va)),
        Subset(dataset, range(n_tr + n_va, n)),
    )


# -----------------------------------------------------------------------------
# RUN ONE EPOCH
# -----------------------------------------------------------------------------
def run_epoch(
    model, loader, criterion, optimizer,
    device, training, desc="",
    label_mean=0.0, label_std=1.0,
):
    model.train() if training else model.eval()

    total_mse = 0.0
    total_mae = 0.0
    context   = torch.enable_grad() if training else torch.no_grad()

    with context:
        bar = tqdm(loader, desc=desc, leave=False)
        for images, labels_norm in bar:
            images      = images.to(device)
            labels_norm = labels_norm.to(device)

            if torch.isnan(images).any() or torch.isnan(labels_norm).any():
                continue

            biomass, _ = model(images)
            pred_norm  = biomass.squeeze(1)
            loss       = criterion(pred_norm, labels_norm)

            if torch.isnan(loss):
                continue

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            pred_real   = pred_norm.detach()   * label_std + label_mean
            labels_real = labels_norm.detach() * label_std + label_mean
            mae         = torch.abs(pred_real - labels_real).mean().item()

            total_mse += loss.item() * len(images)
            total_mae += mae         * len(images)
            bar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{mae:.4f}")

    n = len(loader.dataset)
    return total_mse / n, total_mae / n


# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
def train(
    metadata_csv = "data/real_metadata.csv",
    data_root    = "data",
    ckpt_path    = "checkpoints/cnn_biomass.pt",
    log_csv      = "outputs/train_log.csv",
    epochs       = 80,
    batch_size   = 8,
    lr           = 5e-4,
    patience     = 20,
    in_channels  = 5,
    num_workers  = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Train] Device: {device}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    logger.info("[Train] Computing dataset statistics...")
    ch_mean, ch_std, lb_mean, lb_std = compute_dataset_stats(
        metadata_csv, data_root
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    def make_dataset(augment):
        return BiomassDataset(
            metadata_csv   = metadata_csv,
            data_root      = data_root,
            augment        = augment,
            require_labels = True,
            channel_mean   = ch_mean,
            channel_std    = ch_std,
            label_mean     = lb_mean,
            label_std      = lb_std,
        )

    full_dataset = make_dataset(augment=False)

    if len(full_dataset) == 0:
        raise RuntimeError(
            "Dataset is empty. Run:\n"
            "  python real_data_pipeline.py --step all --label-source ndvi"
        )

    train_ds, val_ds, test_ds = temporal_split(full_dataset)

    train_full = make_dataset(augment=True)
    train_ds   = Subset(train_full, list(train_ds.indices))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader  = DataLoader(val_ds,  batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BiomassCNN(in_channels=in_channels, dropout_rate=0.6).to(device)
    logger.info(f"[Train] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Phase 1: Freeze backbone, train head only ─────────────────────────────
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=5e-3
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val     = float("inf")
    patience_ctr = 0
    log_rows     = []
    phase        = 1

    print(f"\n{'─'*78}")
    print(f"{'Ep':>4}  {'TrainMSE':>10}  {'TrainMAE':>10}  "
          f"{'ValMSE':>10}  {'ValMAE':>10}  Status")
    print(f"{'─'*78}")

    for epoch in range(1, epochs + 1):

        # ── Phase 2: Unfreeze backbone at epoch 20 ────────────────────────────
        if epoch == 20 and phase == 1:
            logger.info("[Train] Phase 2: Unfreezing backbone...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": 1e-5},
                {"params": model.head.parameters(),     "lr": 1e-4},
            ], weight_decay=5e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            phase = 2

        t0 = time.time()

        tr_mse, tr_mae = run_epoch(
            model, train_loader, criterion, optimizer,
            device, True, f"Ep{epoch:3d} Train",
            label_mean=lb_mean, label_std=lb_std,
        )
        va_mse, va_mae = run_epoch(
            model, val_loader, criterion, None,
            device, False, f"Ep{epoch:3d} Val",
            label_mean=lb_mean, label_std=lb_std,
        )

        scheduler.step(va_mse)
        elapsed  = time.time() - t0
        improved = va_mse < best_val
        status   = "✓ saved" if improved else f"– ({patience_ctr+1}/{patience})"

        print(
            f"{epoch:>4}  {tr_mse:>10.5f}  {tr_mae:>10.4f}  "
            f"{va_mse:>10.5f}  {va_mae:>10.4f}  {status}  [{elapsed:.1f}s]"
            + ("  [Phase 2]" if epoch == 20 else "")
        )

        log_rows.append({
            "epoch":     epoch,
            "train_mse": tr_mse,
            "train_mae": tr_mae,
            "val_mse":   va_mse,
            "val_mae":   va_mae,
        })

        if improved:
            best_val     = va_mse
            patience_ctr = 0
            save_checkpoint(model, optimizer, epoch, va_mse, ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(f"[Train] Early stopping at epoch {epoch}")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    loaded = load_checkpoint(ckpt_path, device)
    if loaded is not None:
        model = loaded
    else:
        logger.warning("[Train] No checkpoint — using last weights.")

    te_mse, te_mae = run_epoch(
        model, test_loader, criterion, None,
        device, False, "Test",
        label_mean=lb_mean, label_std=lb_std,
    )
    te_rmse = np.sqrt(te_mse) * lb_std

    print(f"\n{'═'*50}")
    print("  FINAL TEST RESULTS")
    print(f"{'═'*50}")
    print(f"  MSE  (normalised) : {te_mse:.5f}")
    print(f"  RMSE (t/ha)       : {te_rmse:.4f}")
    print(f"  MAE  (t/ha)       : {te_mae:.4f}")
    print(f"{'═'*50}")

    # ── Save log ──────────────────────────────────────────────────────────────
    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    logger.info(f"[Train] Log saved → {log_csv}")

    return {"test_mse": te_mse, "test_rmse": te_rmse, "test_mae": te_mae}


# -----------------------------------------------------------------------------
# PLOT CURVES
# -----------------------------------------------------------------------------
def plot_training_curves(log_csv="outputs/train_log.csv"):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv(log_csv)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(df["epoch"], df["train_mse"], label="Train MSE", color="#1565C0")
        axes[0].plot(df["epoch"], df["val_mse"],   label="Val MSE",   color="#E53935", linestyle="--")
        axes[0].set_title("MSE Loss")
        axes[0].legend()
        axes[0].set_xlabel("Epoch")

        axes[1].plot(df["epoch"], df["train_mae"], label="Train MAE", color="#2E7D32")
        axes[1].plot(df["epoch"], df["val_mae"],   label="Val MAE",   color="#F57F17", linestyle="--")
        axes[1].set_title("MAE (t/ha)")
        axes[1].legend()
        axes[1].set_xlabel("Epoch")

        plt.tight_layout()
        out = "outputs/training_curves.png"
        plt.savefig(out, dpi=150)
        plt.close()
        logger.info(f"[Plot] Saved → {out}")

    except ImportError:
        logger.warning("[Plot] matplotlib/pandas not installed — skipping.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiomassCNN")
    parser.add_argument("--metadata",    default="data/real_metadata.csv")
    parser.add_argument("--data-root",   default="data")
    parser.add_argument("--epochs",      type=int,   default=80)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--patience",    type=int,   default=20)
    parser.add_argument("--in-channels", type=int,   default=5)
    parser.add_argument("--workers",     type=int,   default=0)
    args = parser.parse_args()

    results = train(
        metadata_csv = args.metadata,
        data_root    = args.data_root,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        patience     = args.patience,
        in_channels  = args.in_channels,
        num_workers  = args.workers,
    )

    plot_training_curves()
    print("\n✅  Training complete. Checkpoint → checkpoints/cnn_biomass.pt")