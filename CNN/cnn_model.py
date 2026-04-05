"""
cnn_model.py
============
BiomassCNN — pretrained ResNet18 backbone with BatchNorm head
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from torchvision import models


class BiomassCNN(nn.Module):

    def __init__(self, in_channels=5, dropout_rate=0.6, **kwargs):
        super().__init__()

        # Pretrained ResNet18
        base     = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        old_conv = base.conv1

        # Adapt first conv: 3 → in_channels
        new_conv = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            for i in range(3, in_channels):
                new_conv.weight[:, i] = old_conv.weight.mean(dim=1)

        base.conv1    = new_conv
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1)

        # Head with BatchNorm to reduce overfitting
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        biomass  = self.head(features)
        return biomass, features


# -----------------------------------------------------------------------------
# CHECKPOINT UTILITIES
# -----------------------------------------------------------------------------
def save_checkpoint(model, optimizer, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss":  val_loss,
    }, path)


def load_checkpoint(path, device):
    if not os.path.exists(path):
        print(f"[Warning] No checkpoint at {path}")
        return None
    ckpt  = torch.load(path, map_location=device)
    model = BiomassCNN().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model