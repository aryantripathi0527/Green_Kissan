"""
inference.py
============
Production inference entry point for the CNN module.

Public API:
    extract_cnn_features(lat, lon, date)
    → {"biomass_prediction": float, "cnn_embedding": np.ndarray (128,)}

    extract_cnn_features_batch(farmer_list)
    → List of feature dicts (vectorised, fast)

    These outputs plug directly into pipeline.py:
        CNNModelStub.predict() → replace with extract_cnn_features()

Usage:
    from inference import extract_cnn_features

    result = extract_cnn_features(
        lat      = 30.901,
        lon      = 75.857,
        date_str = "2024-10-20",
    )
    print(result["biomass_prediction"])   # e.g. 3.74 tons/ha
    print(result["cnn_embedding"].shape)  # (128,)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from sentinel_downloader import SentinelDownloader, add_ndvi_channel
from cnn_model import BiomassCNN, load_checkpoint

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class CNNInferenceEngine:
    """
    Load-once, call-many inference wrapper.

    Parameters
    ----------
    ckpt_path        : path to trained checkpoint
    sh_client_id     : SentinelHub credentials (optional; PC is free fallback)
    sh_client_secret :
    device           : "cuda" | "cpu" | None (auto-detect)
    cache_downloads  : cache downloaded patches to avoid re-downloading

    Example:
        engine = CNNInferenceEngine()
        result = engine.predict(lat=30.901, lon=75.857, date_str="2024-10-20")
    """

    def __init__(
        self,
        ckpt_path:        str = "checkpoints/cnn_biomass.pt",
        sh_client_id:     str = "",
        sh_client_secret: str = "",
        device:           Optional[str] = None,
        cache_downloads:  bool = True,
    ):
        # ── Device ────────────────────────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Model ─────────────────────────────────────────────────────────────
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"No checkpoint at {ckpt_path}\n"
                f"Train first: python train_cnn.py"
            )
        self.model = load_checkpoint(ckpt_path, self.device)
        self.model.eval()

        # ── Downloader ────────────────────────────────────────────────────────
        self.downloader = SentinelDownloader(
            sh_client_id     = sh_client_id,
            sh_client_secret = sh_client_secret,
            cache_dir        = "data/image_cache",
        )
        self.cache_downloads = cache_downloads

        logger.info(f"[Engine] Ready | device={self.device} | ckpt={ckpt_path}")

    def predict(
        self,
        lat:       float,
        lon:       float,
        date_str:  str,
        farmer_id: str = "",
    ) -> Optional[Dict]:
        """
        Full inference for one location.

        Returns:
            {
                "farmer_id"           : str,
                "biomass_prediction"  : float (tons/ha),
                "cnn_embedding"       : np.ndarray (128,),
                "ndvi_mean"           : float,
                "date_used"           : str,
                "cloud_pct"           : float,
                "source"              : str,
            }
            or None if imagery unavailable.
        """
        fid = farmer_id or f"loc_{lat:.4f}_{lon:.4f}"

        # ── Download ──────────────────────────────────────────────────────────
        scene = self.downloader.fetch(
            farmer_id = fid,
            lat       = lat,
            lon       = lon,
            date_str  = date_str,
            use_cache = self.cache_downloads,
        )

        if scene is None:
            logger.warning(f"[Engine] No imagery for {fid} on {date_str}")
            return None

        # ── Preprocess ────────────────────────────────────────────────────────
        img   = add_ndvi_channel(scene.image)   # (4,128,128) → (5,128,128)
        ndvi  = img[4]                           # 5th channel is NDVI
        ndvi_mean = float(ndvi.mean())

        # ── Inference ─────────────────────────────────────────────────────────
        self.model.eval()
        with torch.no_grad():
            x   = torch.from_numpy(img).unsqueeze(0).to(self.device)  # (1,5,128,128)
            bio, emb = self.model(x)

        from torch_dataset import compute_dataset_stats
        lb_mean, lb_std = 1.9797, 0.6505
        biomass   = float(bio.item()) * lb_std + lb_mean
        biomass   = max(0.0, biomass)
        embedding = emb.squeeze().cpu().numpy().flatten()[:128]

        result = {
            "farmer_id"          : fid,
            "biomass_prediction" : round(biomass, 4),
            "cnn_embedding"      : embedding,
            "ndvi_mean"          : round(ndvi_mean, 4),
            "date_used"          : scene.date,
            "cloud_pct"          : round(scene.cloud_pct, 1),
            "source"             : scene.source,
        }

        logger.info(
            f"[Engine] {fid} | biomass={biomass:.3f} t/ha | "
            f"NDVI={ndvi_mean:.3f} | cloud={scene.cloud_pct:.1f}%"
        )
        return result

    def predict_batch(
        self,
        farmer_list: List[Dict],
        date_str:    str,
        batch_size:  int = 16,
    ) -> List[Optional[Dict]]:
        """
        Vectorised batch prediction.

        farmer_list items must have: farmer_id, lat, lon
        date_str     : single target date applied to all (or per-item via 'date' key)

        Returns list of result dicts (None where imagery unavailable).
        """
        from tqdm import tqdm

        logger.info(f"[Engine] Batch predict: {len(farmer_list)} farmers")
        results = []

        # ── Download phase ────────────────────────────────────────────────────
        scenes: List[Optional[object]] = []
        for farmer in tqdm(farmer_list, desc="Downloading patches", unit="farmer"):
            d = farmer.get("date", date_str)
            s = self.downloader.fetch(
                farmer_id = farmer["farmer_id"],
                lat       = farmer["lat"],
                lon       = farmer["lon"],
                date_str  = d,
                use_cache = self.cache_downloads,
            )
            scenes.append(s)

        # ── Model inference in batches ─────────────────────────────────────────
        valid_idx   = [i for i, s in enumerate(scenes) if s is not None]
        valid_imgs  = [add_ndvi_channel(scenes[i].image) for i in valid_idx]

        all_bio = np.zeros(len(farmer_list))
        all_emb = np.zeros((len(farmer_list), self.model.embedding_dim))

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(valid_idx), batch_size):
                batch_ids = valid_idx[start : start + batch_size]
                imgs_np   = np.stack([valid_imgs[j]
                                      for j in range(start, min(start+batch_size,
                                                                  len(valid_idx)))])
                x         = torch.from_numpy(imgs_np).to(self.device)
                bio, emb  = self.model(x)

                for k, gi in enumerate(batch_ids):
                    all_bio[gi] = float(bio[k].item())
                    all_emb[gi] = emb[k].cpu().numpy()

        # ── Assemble results ──────────────────────────────────────────────────
        for i, farmer in enumerate(farmer_list):
            if scenes[i] is None:
                logger.warning(f"[Engine] No scene: {farmer['farmer_id']}")
                results.append(None)
            else:
                ndvi_mean = float(add_ndvi_channel(scenes[i].image)[4].mean())
                results.append({
                    "farmer_id"         : farmer["farmer_id"],
                    "biomass_prediction": round(float(all_bio[i]), 4),
                    "cnn_embedding"     : all_emb[i],
                    "ndvi_mean"         : round(ndvi_mean, 4),
                    "date_used"         : scenes[i].date,
                    "cloud_pct"         : round(scenes[i].cloud_pct, 1),
                    "source"            : scenes[i].source,
                })

        n_ok = sum(1 for r in results if r is not None)
        logger.info(f"[Engine] Batch done: {n_ok}/{len(farmer_list)} successful")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONVENIENCE FUNCTION (compatible with pipeline.py stubs)
# ─────────────────────────────────────────────────────────────────────────────
_engine: Optional[CNNInferenceEngine] = None


def _get_engine() -> CNNInferenceEngine:
    """Lazy-load a singleton engine (avoids re-loading model on every call)."""
    global _engine
    if _engine is None:
        _engine = CNNInferenceEngine()
    return _engine


def extract_cnn_features(
    lat:       float,
    lon:       float,
    date_str:  str,
    farmer_id: str = "",
    ckpt_path: str = "checkpoints/cnn_biomass.pt",
) -> Optional[Dict]:
    """
    Top-level inference function — this is what pipeline.py calls.

    Returns:
        {
            "biomass_prediction" : float          (tons/ha)
            "cnn_embedding"      : np.ndarray     (128,) float32
            "ndvi_mean"          : float
            "date_used"          : str
            "cloud_pct"          : float
            "source"             : str
        }
        or None on failure.

    Integration with pipeline.py:
        # In your CNNModelStub.predict():
        from inference import extract_cnn_features
        result = extract_cnn_features(lat, lon, date_str, farmer_id)
        return result["cnn_embedding"], result["biomass_prediction"]
    """
    global _engine
    if _engine is None or not Path(ckpt_path).exists():
        if not Path(ckpt_path).exists():
            logger.error(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Train the model first: python train_cnn.py"
            )
            return None
        _engine = CNNInferenceEngine(ckpt_path=ckpt_path)

    return _engine.predict(lat=lat, lon=lon, date_str=date_str, farmer_id=farmer_id)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE.PY INTEGRATION ADAPTER
# ─────────────────────────────────────────────────────────────────────────────
class RealCNNModel:
    """
    Drop-in replacement for CNNModelStub in pipeline.py.

    Usage in pipeline.py:
        from inference import RealCNNModel
        pipeline = StubbleBurningPipeline(cnn_model=RealCNNModel())

    The pipeline calls:
        cnn_model.predict(image_array)
    which here ignores the raw array and instead re-downloads fresh imagery
    using the farmer's GPS coordinates (passed via the farmer record dict).
    """

    def __init__(self, ckpt_path: str = "checkpoints/cnn_biomass.pt"):
        self.engine = CNNInferenceEngine(ckpt_path=ckpt_path)

    def predict(self, image_array: np.ndarray) -> tuple:
        """
        Called by pipeline.py. image_array is unused — we re-fetch from API.
        The engine was already set up with the farmer's lat/lon in predict_batch().

        For direct integration, prefer predict_from_coords() below.
        """
        # If image already downloaded externally, run inference directly
        img   = add_ndvi_channel(image_array) if image_array.shape[0] == 4 else image_array
        emb, bio = self.engine.model.predict_numpy(img, self.engine.device)
        return emb, bio

    def predict_from_coords(
        self,
        lat:      float,
        lon:      float,
        date_str: str,
        farmer_id: str = "",
    ) -> Optional[tuple]:
        """
        Preferred method: downloads imagery + runs inference.
        Returns (cnn_embedding: np.ndarray, biomass: float) or None.
        """
        result = self.engine.predict(lat, lon, date_str, farmer_id)
        if result is None:
            return None
        return result["cnn_embedding"], result["biomass_prediction"]


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

    parser = argparse.ArgumentParser(description="CNN feature extraction")
    parser.add_argument("--lat",      type=float, default=30.901)
    parser.add_argument("--lon",      type=float, default=75.857)
    parser.add_argument("--date",     default="2024-10-20")
    parser.add_argument("--id",       default="DEMO_FARMER_001")
    parser.add_argument("--ckpt",     default="checkpoints/cnn_biomass.pt")
    args = parser.parse_args()

    print(f"\n[Demo] Extracting CNN features for ({args.lat}, {args.lon}) on {args.date}")
    result = extract_cnn_features(
        lat       = args.lat,
        lon       = args.lon,
        date_str  = args.date,
        farmer_id = args.id,
        ckpt_path = args.ckpt,
    )

    if result:
        print("\n── CNN Inference Result ───────────────────────────────")
        for k, v in result.items():
            if k == "cnn_embedding":
                print(f"  {'cnn_embedding':<22}: shape={v.shape}  "
                      f"mean={v.mean():.4f}  std={v.std():.4f}")
            else:
                print(f"  {k:<22}: {v}")
    else:
        print("❌ Inference failed — check logs above")
