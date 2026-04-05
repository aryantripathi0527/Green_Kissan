"""
dataset_builder.py
==================
Builds a structured on-disk dataset from a CSV of farmer coordinates.

Output structure:
    data/
    ├── images/
    │   ├── FARMER_001_2024-10-20.npy    # shape (5, 128, 128) float32
    │   ├── FARMER_002_2024-10-22.npy
    │   └── ...
    └── metadata.csv

metadata.csv columns:
    farmer_id, lat, lon, date, cloud_pct, source,
    ndvi_mean, ndvi_std,
    biomass_label (if provided in input CSV),
    image_path, valid

Input CSV format (farmers.csv):
    farmer_id, lat, lon, date, field_area_ha, biomass_label (optional)

Usage:
    from dataset_builder import SentinelDatasetBuilder
    builder = SentinelDatasetBuilder()
    builder.build_dataset("data/farmers.csv")
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from sentinel_downloader import (
    SentinelDownloader, SceneResult, add_ndvi_channel, compute_ndvi
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────
class SentinelDatasetBuilder:
    """
    Downloads Sentinel-2 patches for each farmer and builds a
    ready-to-train dataset saved to disk.

    Parameters
    ----------
    output_dir      : root folder for images/ and metadata.csv
    sh_client_id    : SentinelHub client ID (optional; falls back to PC)
    sh_client_secret: SentinelHub secret   (optional)
    add_ndvi        : if True, images are (5, 128, 128); else (4, 128, 128)
    max_cloud_pct   : skip scenes cloudier than this
    start_date      : override per-farmer date for batch (YYYY-MM-DD)
    end_date        : override per-farmer date for batch (YYYY-MM-DD)
    """

    def __init__(
        self,
        output_dir:       str = "data",
        sh_client_id:     str = "",
        sh_client_secret: str = "",
        add_ndvi:         bool = True,
        max_cloud_pct:    float = 20.0,
        start_date:       Optional[str] = None,
        end_date:         Optional[str] = None,
    ):
        self.output_dir    = Path(output_dir)
        self.images_dir    = self.output_dir / "images"
        self.metadata_path = self.output_dir / "metadata.csv"
        self.add_ndvi      = add_ndvi
        self.max_cloud_pct = max_cloud_pct
        self.start_date    = start_date
        self.end_date      = end_date

        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.downloader = SentinelDownloader(
            sh_client_id     = sh_client_id,
            sh_client_secret = sh_client_secret,
            cache_dir        = str(self.output_dir / "image_cache"),
        )

        logger.info(f"[Builder] Output dir : {self.output_dir.resolve()}")
        logger.info(f"[Builder] NDVI channel: {self.add_ndvi}")

    # ── Public API ─────────────────────────────────────────────────────────────
    def fetch_image(
        self,
        farmer_id: str,
        lat:       float,
        lon:       float,
        date_str:  str,
    ) -> Optional[np.ndarray]:
        """
        Fetch one image patch for a single (lat, lon, date).

        Returns:
            numpy array shape (5, 128, 128) float32 if add_ndvi=True
                               (4, 128, 128) float32 if add_ndvi=False
            or None if download failed / cloud cover exceeded threshold.
        """
        if self.start_date and self.end_date:
            scene = self.downloader.fetch_best_in_range(
                farmer_id  = farmer_id,
                lat        = lat,
                lon        = lon,
                start_date = self.start_date,
                end_date   = self.end_date,
            )
        else:
            scene = self.downloader.fetch(
                farmer_id = farmer_id,
                lat       = lat,
                lon       = lon,
                date_str  = date_str,
            )

        if scene is None:
            logger.warning(f"[Builder] {farmer_id}: no valid scene available")
            return None

        img = scene.image   # (4, 128, 128)
        if self.add_ndvi:
            img = add_ndvi_channel(img)   # (5, 128, 128)

        return img

    def build_dataset(
        self,
        csv_path: str,
        skip_existing: bool = True,
    ) -> pd.DataFrame:
        """
        Reads a CSV of farmer records, downloads imagery for each,
        saves .npy files to data/images/, and writes metadata.csv.

        Input CSV required columns: farmer_id, lat, lon, date
        Optional columns: field_area_ha, biomass_label (ground truth)

        Returns the metadata DataFrame.
        """
        df_in = pd.read_csv(csv_path)
        self._validate_input_csv(df_in)

        logger.info(f"[Builder] Processing {len(df_in)} farmers from {csv_path}")

        records: List[Dict] = []
        failed:  List[str]  = []

        for _, row in tqdm(df_in.iterrows(), total=len(df_in),
                           desc="Downloading patches", unit="farmer"):

            fid      = str(row["farmer_id"])
            lat      = float(row["lat"])
            lon      = float(row["lon"])
            date_str = str(row["date"])

            img_path = self.images_dir / f"{fid}_{date_str}.npy"

            # Skip if already downloaded
            if skip_existing and img_path.exists():
                logger.debug(f"[Builder] Skipping {fid} (already exists)")
                meta = self._load_existing_meta(fid, date_str, lat, lon,
                                                img_path, row)
                records.append(meta)
                continue

            img = self.fetch_image(fid, lat, lon, date_str)

            if img is None:
                failed.append(fid)
                records.append(self._failed_meta(fid, lat, lon, date_str, row))
                continue

            # Save image
            np.save(str(img_path), img)

            # Compute NDVI statistics for metadata
            ndvi       = compute_ndvi(img[:4])   # always from first 4 bands
            ndvi_mean  = float(ndvi.mean())
            ndvi_std   = float(ndvi.std())

            records.append({
                "farmer_id"      : fid,
                "lat"            : lat,
                "lon"            : lon,
                "date"           : date_str,
                "cloud_pct"      : 0.0,     # updated below if we have scene info
                "source"         : "downloaded",
                "ndvi_mean"      : round(ndvi_mean, 4),
                "ndvi_std"       : round(ndvi_std, 4),
                "image_path"     : str(img_path.relative_to(self.output_dir)),
                "channels"       : img.shape[0],
                "valid"          : True,
                "biomass_label"  : row.get("biomass_label", np.nan),
                "field_area_ha"  : row.get("field_area_ha", np.nan),
            })

        meta_df = pd.DataFrame(records)
        meta_df.to_csv(str(self.metadata_path), index=False)

        # ── Summary ─────────────────────────────────────────────────────────
        valid   = meta_df["valid"].sum()
        invalid = len(meta_df) - valid
        logger.info(f"[Builder] ✅  Done: {valid} valid | {invalid} failed")
        logger.info(f"[Builder] metadata.csv → {self.metadata_path}")

        if failed:
            logger.warning(f"[Builder] Failed farmer IDs: {failed}")

        return meta_df

    def generate_dummy_dataset(
        self,
        n_samples:   int   = 100,
        output_csv:  str   = "data/farmers_dummy.csv",
        seed:        int   = 42,
    ) -> str:
        """
        Creates a synthetic dataset for testing without downloading anything.
        Generates random (but realistic) image patches and biomass labels
        derived from NDVI, mimicking real Punjab stubble-burning conditions.

        Returns path to generated metadata.csv.
        """
        rng = np.random.default_rng(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for i in range(1, n_samples + 1):
            fid  = f"FARMER_{i:04d}"
            lat  = float(rng.uniform(29.5, 32.5))
            lon  = float(rng.uniform(73.8, 77.5))
            date = f"2024-{rng.integers(10, 12):02d}-{rng.integers(1, 28):02d}"

            # Simulate realistic reflectance values for Punjab agriculture
            # Stubble fields: low NIR, moderate Red; healthy crop: high NIR, low Red
            season_factor = rng.uniform(0.3, 1.0)   # 1.0 = peak biomass season

            img = np.zeros((4, 128, 128), dtype=np.float32)
            img[0] = rng.uniform(0.04, 0.10, (128, 128))  # B02 Blue
            img[1] = rng.uniform(0.06, 0.14, (128, 128))  # B03 Green
            img[2] = rng.uniform(0.05, 0.20, (128, 128))  # B04 Red
            img[3] = rng.uniform(0.15, 0.50, (128, 128)) * season_factor  # B08 NIR

            if self.add_ndvi:
                img = add_ndvi_channel(img)   # → (5, 128, 128)

            img_path = self.images_dir / f"{fid}_{date}.npy"
            np.save(str(img_path), img)

            # Biomass label correlated with NIR (NDVI proxy)
            ndvi_mean      = float(compute_ndvi(img[:4]).mean())
            biomass_label  = float(np.clip(
                ndvi_mean * 8.0 + rng.normal(0, 0.4),   # NDVI → biomass
                0.1, 6.0
            ))

            records.append({
                "farmer_id"    : fid,
                "lat"          : round(lat, 6),
                "lon"          : round(lon, 6),
                "date"         : date,
                "cloud_pct"    : round(float(rng.uniform(0, 15)), 1),
                "source"       : "synthetic",
                "ndvi_mean"    : round(ndvi_mean, 4),
                "ndvi_std"     : round(float(compute_ndvi(img[:4]).std()), 4),
                "image_path"   : str(img_path.relative_to(self.output_dir)),
                "channels"     : img.shape[0],
                "valid"        : True,
                "biomass_label": round(biomass_label, 3),
                "field_area_ha": round(float(rng.uniform(1.0, 8.0)), 2),
            })

        meta_df = pd.DataFrame(records)
        meta_df.to_csv(str(self.metadata_path), index=False)

        # Also save input CSV for reference
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        meta_df[["farmer_id", "lat", "lon", "date",
                 "field_area_ha", "biomass_label"]].to_csv(output_csv, index=False)

        logger.info(f"[Builder] 🧪 Dummy dataset: {n_samples} samples")
        logger.info(f"[Builder]    images   → {self.images_dir}")
        logger.info(f"[Builder]    metadata → {self.metadata_path}")
        return str(self.metadata_path)

    # ── PyTorch Dataset ────────────────────────────────────────────────────────
    def get_torch_dataset(self, require_labels: bool = True):
        """
        Returns a PyTorch Dataset from the built metadata.
        Call build_dataset() or generate_dummy_dataset() first.
        """
        from torch_dataset import BiomassDataset
        return BiomassDataset(str(self.metadata_path),
                              str(self.output_dir),
                              require_labels=require_labels)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _validate_input_csv(df: pd.DataFrame):
        required = {"farmer_id", "lat", "lon", "date"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Input CSV missing required columns: {missing}\n"
                f"Required: farmer_id, lat, lon, date"
            )

    def _load_existing_meta(self, fid, date_str, lat, lon, img_path, row) -> Dict:
        img        = np.load(str(img_path))
        ndvi       = compute_ndvi(img[:4])
        return {
            "farmer_id"    : fid,
            "lat"          : lat,
            "lon"          : lon,
            "date"         : date_str,
            "cloud_pct"    : 0.0,
            "source"       : "cache",
            "ndvi_mean"    : round(float(ndvi.mean()), 4),
            "ndvi_std"     : round(float(ndvi.std()),  4),
            "image_path"   : str(img_path.relative_to(self.output_dir)),
            "channels"     : img.shape[0],
            "valid"        : True,
            "biomass_label": row.get("biomass_label", np.nan),
            "field_area_ha": row.get("field_area_ha", np.nan),
        }

    @staticmethod
    def _failed_meta(fid, lat, lon, date_str, row) -> Dict:
        return {
            "farmer_id"    : fid,
            "lat"          : lat,
            "lon"          : lon,
            "date"         : date_str,
            "cloud_pct"    : 100.0,
            "source"       : "failed",
            "ndvi_mean"    : np.nan,
            "ndvi_std"     : np.nan,
            "image_path"   : "",
            "channels"     : 0,
            "valid"        : False,
            "biomass_label": row.get("biomass_label", np.nan),
            "field_area_ha": row.get("field_area_ha", np.nan),
        }

    def get_stats(self) -> Dict:
        """Returns summary statistics of the built dataset."""
        if not self.metadata_path.exists():
            return {}
        df    = pd.read_csv(str(self.metadata_path))
        valid = df[df["valid"] == True]
        return {
            "total_samples"    : len(df),
            "valid_samples"    : len(valid),
            "failed_samples"   : len(df) - len(valid),
            "date_range"       : f"{df['date'].min()} → {df['date'].max()}",
            "ndvi_mean"        : round(valid["ndvi_mean"].mean(), 4),
            "biomass_mean"     : round(valid["biomass_label"].mean(), 3)
                                 if "biomass_label" in valid.columns else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset (used by train_cnn.py)
# ─────────────────────────────────────────────────────────────────────────────
# Saved in a separate importable file to avoid circular imports
_TORCH_DATASET_CODE = '''
"""torch_dataset.py — PyTorch Dataset for biomass CNN training."""
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset

class BiomassDataset(Dataset):
    """
    Loads (image, biomass_label) pairs from metadata.csv + .npy image files.

    If biomass_label is missing (inference mode), returns image only.
    Optionally applies random horizontal/vertical flips for augmentation.
    """
    def __init__(
        self,
        metadata_csv:   str,
        data_root:      str  = "data",
        augment:        bool = False,
        require_labels: bool = True,
    ):
        df = pd.read_csv(metadata_csv)
        self.df            = df[df["valid"] == True].reset_index(drop=True)
        self.data_root     = Path(data_root)
        self.augment       = augment
        self.require_labels = require_labels

        if require_labels:
            before = len(self.df)
            self.df = self.df.dropna(subset=["biomass_label"])
            dropped = before - len(self.df)
            if dropped:
                import logging
                logging.getLogger(__name__).warning(
                    f"[Dataset] Dropped {dropped} rows with missing biomass_label"
                )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self.data_root / row["image_path"]
        img      = np.load(str(img_path)).astype(np.float32)   # (C, H, W)

        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=2).copy()   # horizontal flip
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()   # vertical flip
            # Random 90° rotation
            k = np.random.randint(0, 4)
            img = np.rot90(img, k=k, axes=(1, 2)).copy()

        x = torch.from_numpy(img)   # (C, 128, 128)

        if self.require_labels:
            y = torch.tensor(float(row["biomass_label"]), dtype=torch.float32)
            return x, y
        return x
'''


def ensure_torch_dataset_file(output_dir: str = "."):
    """Writes torch_dataset.py next to other module files."""
    p = Path(output_dir) / "torch_dataset.py"
    if not p.exists():
        p.write_text(_TORCH_DATASET_CODE)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt = "%H:%M:%S",
    )
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["dummy", "real"], default="dummy")
    parser.add_argument("--csv",     default="data/farmers.csv",
                        help="Input CSV (for --mode real)")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of dummy samples")
    args = parser.parse_args()

    builder = SentinelDatasetBuilder()
    ensure_torch_dataset_file(".")

    if args.mode == "dummy":
        meta_path = builder.generate_dummy_dataset(n_samples=args.samples)
        print(f"\n✅  Dummy dataset ready → {meta_path}")
    else:
        meta_df = builder.build_dataset(args.csv)
        print(f"\n✅  Dataset built: {meta_df['valid'].sum()} valid images")

    print("\nDataset stats:")
    for k, v in builder.get_stats().items():
        print(f"  {k:<22}: {v}")
