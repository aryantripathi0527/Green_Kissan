"""
sentinel_downloader.py
======================
Downloads Sentinel-2 Level-2A imagery for a given GPS coordinate and date.

Supports TWO backends (auto-selected):
  1. SentinelHub API            — paid, fast  (https://www.sentinel-hub.com)
  2. Microsoft Planetary Computer STAC — FREE, no account needed (fallback)

Bug fixed (v2):
  - odc.stac band extraction now correctly squeezes the time dimension
    before stacking, preventing shape (1,H,W) instead of (4,H,W)
  - Added _validate_image_shape() guard called at every exit point
  - Added rasterio band-by-band fallback if odc.stac parse fails
  - compute_ndvi / add_ndvi_channel now assert input has ≥4 bands

Install:
    pip install pystac-client planetary-computer odc-stac rasterio numpy tqdm
    pip install sentinelhub        # optional, for paid backend
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BAND_ORDER         = ["B02", "B03", "B04", "B08"]   # Blue, Green, Red, NIR
N_BANDS            = len(BAND_ORDER)                  # 4
PATCH_PX           = 128
PIXEL_RESOLUTION   = 10           # metres/pixel (Sentinel-2 10m bands)
MAX_CLOUD_PCT      = 20.0
SEARCH_WINDOW_DAYS = 15
DN_SCALE           = 10_000.0


@dataclass
class SceneResult:
    """One downloaded Sentinel-2 scene."""
    farmer_id : str
    lat       : float
    lon       : float
    date      : str
    cloud_pct : float
    image     : np.ndarray    # shape (4, 128, 128) float32 [0,1]
    source    : str


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
def latlon_to_bbox(
    lat: float,
    lon: float,
    half_km: float = 0.64,
) -> Tuple[float, float, float, float]:
    """(min_lon, min_lat, max_lon, max_lat) centred on lat/lon."""
    deg_lat = half_km / 111.0
    deg_lon = half_km / (111.0 * np.cos(np.radians(lat)))
    return (lon - deg_lon, lat - deg_lat, lon + deg_lon, lat + deg_lat)


def date_window(date_str: str, days: int = SEARCH_WINDOW_DAYS) -> Tuple[str, str]:
    d     = datetime.strptime(date_str, "%Y-%m-%d")
    start = (d - timedelta(days=days)).strftime("%Y-%m-%d")
    end   = (d + timedelta(days=days)).strftime("%Y-%m-%d")
    return start, end


# ─────────────────────────────────────────────────────────────────────────────
# SHAPE VALIDATION  ← centralised guard used everywhere
# ─────────────────────────────────────────────────────────────────────────────
def _validate_image_shape(arr: np.ndarray, context: str = "") -> np.ndarray:
    """
    Ensures arr is (4, 128, 128) float32.
    Raises ValueError with a clear message if anything is wrong.
    Called at every point where an image is produced or consumed.
    """
    tag = f"[{context}] " if context else ""

    if arr.ndim != 3:
        raise ValueError(
            f"{tag}Expected 3D array (C,H,W), got shape {arr.shape}. "
            f"This usually means the band axis was lost during extraction."
        )

    c, h, w = arr.shape

    if c != N_BANDS:
        raise ValueError(
            f"{tag}Expected {N_BANDS} bands (B02,B03,B04,B08), got {c} channels. "
            f"Shape: {arr.shape}. Check odc.stac band extraction."
        )

    # Resize to exact patch size if needed (handles off-by-one from projection)
    if h != PATCH_PX or w != PATCH_PX:
        logger.debug(f"{tag}Resizing {arr.shape} → ({c},{PATCH_PX},{PATCH_PX})")
        arr = _resize_patch(arr, PATCH_PX)

    return arr.astype(np.float32)


def _resize_patch(arr: np.ndarray, target: int) -> np.ndarray:
    """Crop or pad (C, H, W) → (C, target, target). Pure numpy, no cv2/PIL."""
    c, h, w = arr.shape
    # Crop excess
    arr = arr[:, :min(h, target), :min(w, target)]
    # Pad deficit
    _, h2, w2 = arr.shape
    if h2 < target or w2 < target:
        arr = np.pad(
            arr,
            ((0, 0), (0, target - h2), (0, target - w2)),
            mode="reflect",
        )
    return arr[:, :target, :target]


# ─────────────────────────────────────────────────────────────────────────────
# BAND EXTRACTION HELPERS  ← multiple strategies, most robust first
# ─────────────────────────────────────────────────────────────────────────────
def _extract_bands_from_xarray(ds) -> np.ndarray:
    """
    Extract (4, H, W) from an odc.stac xarray Dataset.

    odc.stac.load() returns a Dataset where each band is a variable
    with dimensions (time, y, x).  We take t=0 and stack bands.

    ── Root cause of the original bug ──────────────────────────────
    The broken code was:
        ds[BAND_ORDER].to_array(dim="band").values[0]
        → shape (4, time, H, W)[0] = (time, H, W) = (1, 128, 128)  ← WRONG

    The correct approach is to squeeze the time dimension per band:
        np.stack([ds[b].values.squeeze(0) for b in BAND_ORDER])
        → shape (4, H, W)  ← CORRECT
    """
    bands = []
    for band_name in BAND_ORDER:
        if band_name not in ds:
            raise ValueError(f"Band {band_name} not found in dataset. "
                             f"Available: {list(ds.data_vars)}")
        arr = ds[band_name].values          # shape: (time, H, W) or (H, W)
        if arr.ndim == 3:
            arr = arr[0]                    # take first time slice → (H, W)
        elif arr.ndim != 2:
            raise ValueError(f"Unexpected band shape for {band_name}: {arr.shape}")
        bands.append(arr)

    return np.stack(bands, axis=0)          # (4, H, W)


def _extract_bands_rasterio(href: str) -> np.ndarray:
    """
    Fallback: read bands directly from a GeoTIFF href via rasterio.
    Used when odc.stac parse fails.
    Each 10m Sentinel-2 band is a separate COG — we read them one by one.
    """
    import rasterio
    from rasterio.enums import Resampling

    # href is for a single band; we get the asset dict from the STAC item
    with rasterio.open(href) as src:
        # Read at native resolution, first band
        data = src.read(1, out_shape=(PATCH_PX, PATCH_PX),
                        resampling=Resampling.bilinear)
    return data   # (H, W)


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND 1 — SentinelHub
# ─────────────────────────────────────────────────────────────────────────────
def _download_sentinelhub(
    farmer_id: str,
    lat: float,
    lon: float,
    date_str: str,
    client_id: str,
    client_secret: str,
) -> Optional[SceneResult]:
    try:
        from sentinelhub import (
            SHConfig, SentinelHubRequest, BBox, CRS, DataCollection,
            MimeType,
        )
    except ImportError:
        logger.warning("sentinelhub not installed. Falling back.")
        return None

    cfg = SHConfig()
    cfg.sh_client_id     = client_id
    cfg.sh_client_secret = client_secret
    if not cfg.sh_client_id:
        return None

    start, end = date_window(date_str)
    bbox       = BBox(latlon_to_bbox(lat, lon), CRS.WGS84)

    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input:  [{ bands: ["B02","B03","B04","B08","CLM"], units: "DN" }],
            output: { bands: 5, sampleType: "FLOAT32" }
        };
    }
    function evaluatePixel(s) {
        return [s.B02, s.B03, s.B04, s.B08, s.CLM];
    }
    """

    request = SentinelHubRequest(
        evalscript   = evalscript,
        input_data   = [SentinelHubRequest.input_data(
            data_collection  = DataCollection.SENTINEL2_L2A,
            time_interval    = (start, end),
            mosaicking_order = "leastCC",
        )],
        responses    = [SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox         = bbox,
        size         = (PATCH_PX, PATCH_PX),
        config       = cfg,
    )

    try:
        data = request.get_data()[0]              # (H, W, 5)
    except Exception as e:
        logger.error(f"[SentinelHub] {farmer_id}: {e}")
        return None

    data      = data.astype(np.float32)
    cloud_pct = float(data[:, :, 4].mean() * 100)

    if cloud_pct > MAX_CLOUD_PCT:
        logger.warning(f"[SentinelHub] {farmer_id}: cloud {cloud_pct:.1f}% — skip")
        return None

    # SentinelHub returns (H, W, C) → transpose to (C, H, W)
    bands = np.transpose(data[:, :, :4], (2, 0, 1)) / DN_SCALE  # (4, H, W)
    bands = _validate_image_shape(bands, "SentinelHub")
    bands = np.clip(bands, 0.0, 1.0)

    logger.info(f"[SentinelHub] {farmer_id} OK | cloud={cloud_pct:.1f}%")
    return SceneResult(
        farmer_id=farmer_id, lat=lat, lon=lon,
        date=date_str, cloud_pct=cloud_pct,
        image=bands, source="sentinelhub",
    )


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND 2 — Microsoft Planetary Computer (FREE)
# ─────────────────────────────────────────────────────────────────────────────
def _download_planetary_computer(
    farmer_id: str,
    lat: float,
    lon: float,
    date_str: str,
) -> Optional[SceneResult]:
    """
    Download via Microsoft Planetary Computer STAC API — completely free.

    Strategy order (most reliable first):
      1. odc.stac.load  with correct band-by-band squeeze
      2. rasterio COG   band-by-band fallback
    """
    try:
        import pystac_client
        import planetary_computer
    except ImportError:
        logger.error("Run: pip install pystac-client planetary-computer odc-stac rasterio")
        return None

    start, end = date_window(date_str)
    bbox_tuple = latlon_to_bbox(lat, lon)
    bbox_list  = list(bbox_tuple)

    # ── STAC search ───────────────────────────────────────────────────────────
    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections = ["sentinel-2-l2a"],
            bbox        = bbox_list,
            datetime    = f"{start}/{end}",
            query       = {"eo:cloud_cover": {"lt": MAX_CLOUD_PCT}},
            sortby      = [{"field": "eo:cloud_cover", "direction": "asc"}],
        )
        items = list(search.items())
    except Exception as e:
        logger.error(f"[PC] STAC search failed for {farmer_id}: {e}")
        return None

    if not items:
        logger.warning(f"[PC] {farmer_id}: no scenes in {start}→{end} with cloud<{MAX_CLOUD_PCT}%")
        return None

    best       = items[0]
    cloud_pct  = float(best.properties.get("eo:cloud_cover", 0))
    scene_date = best.datetime.strftime("%Y-%m-%d")
    logger.info(f"[PC] {farmer_id}: scene {scene_date} cloud={cloud_pct:.1f}%")

    # ── Strategy 1: odc.stac ─────────────────────────────────────────────────
    arr = _try_odc_stac(best, bbox_list, farmer_id)

    # ── Strategy 2: rasterio COG fallback ────────────────────────────────────
    if arr is None:
        arr = _try_rasterio_cog(best, bbox_tuple, farmer_id)

    if arr is None:
        logger.error(f"[PC] {farmer_id}: all extraction strategies failed")
        return None

    # Validate, resize, normalise
    arr = _validate_image_shape(arr, "PlanetaryComputer")
    arr = arr.astype(np.float32) / DN_SCALE
    arr = np.clip(arr, 0.0, 1.0)

    logger.info(
        f"[PC] {farmer_id} OK | shape={arr.shape} "
        f"range=[{arr.min():.3f},{arr.max():.3f}]"
    )
    return SceneResult(
        farmer_id=farmer_id, lat=lat, lon=lon,
        date=scene_date, cloud_pct=cloud_pct,
        image=arr, source="planetary_computer",
    )


def _try_odc_stac(item, bbox_list: list, farmer_id: str) -> Optional[np.ndarray]:
    """
    Load bands via odc.stac, correctly squeezing the time dimension.
    Returns (4, H, W) int array or None on failure.
    """
    try:
        import odc.stac
    except ImportError:
        logger.warning("[odc.stac] not installed — skipping strategy 1")
        return None

    try:
        ds = odc.stac.load(
            [item],
            bands      = BAND_ORDER,
            bbox       = bbox_list,
            resolution = PIXEL_RESOLUTION,
        )

        # ── THE FIX ──────────────────────────────────────────────────────────
        # Each ds[band] has shape (time, y, x).  Squeeze time=0 per band,
        # then stack → (4, H, W).
        # Previously broken: ds[BAND_ORDER].to_array("band").values[0]
        #   gave (1, H, W) because [0] indexed the first BAND, not time.
        # ─────────────────────────────────────────────────────────────────────
        arr = _extract_bands_from_xarray(ds)

        logger.debug(f"[odc.stac] {farmer_id}: extracted shape={arr.shape}")

        if arr.shape[0] != N_BANDS:
            logger.warning(
                f"[odc.stac] {farmer_id}: got {arr.shape[0]} bands, "
                f"expected {N_BANDS}. Falling back."
            )
            return None

        return arr   # (4, H, W) raw DN values

    except Exception as e:
        logger.warning(f"[odc.stac] {farmer_id}: failed — {e}")
        return None


def _try_rasterio_cog(item, bbox_tuple: tuple, farmer_id: str) -> Optional[np.ndarray]:
    """
    Band-by-band rasterio COG read — most reliable fallback.
    Returns (4, H, W) int array or None on failure.
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.enums import Resampling
    except ImportError:
        logger.error("rasterio not installed: pip install rasterio")
        return None

    # Map our band names to Sentinel-2 asset keys in the STAC item
    asset_key_map = {
        "B02": "B02", "B03": "B03",
        "B04": "B04", "B08": "B08",
    }

    bands_out = []
    min_lon, min_lat, max_lon, max_lat = bbox_tuple

    for band_name in BAND_ORDER:
        asset_key = asset_key_map.get(band_name)
        if asset_key not in item.assets:
            # Some items use lowercase band keys
            asset_key = band_name.lower()
        if asset_key not in item.assets:
            logger.error(f"[rasterio] Asset '{band_name}' not in item. "
                         f"Available: {list(item.assets.keys())}")
            return None

        href = item.assets[asset_key].href
        try:
            with rasterio.open(href) as src:
                # Convert WGS84 bbox to dataset CRS
                from rasterio.warp import transform_bounds
                bbox_native = transform_bounds(
                    "EPSG:4326", src.crs,
                    min_lon, min_lat, max_lon, max_lat,
                )
                window = from_bounds(*bbox_native, transform=src.transform)
                data   = src.read(
                    1,
                    window    = window,
                    out_shape = (PATCH_PX, PATCH_PX),
                    resampling = Resampling.bilinear,
                )                                    # (H, W)
            bands_out.append(data)
        except Exception as e:
            logger.warning(f"[rasterio] {farmer_id} band {band_name}: {e}")
            return None

    arr = np.stack(bands_out, axis=0)               # (4, H, W)
    logger.debug(f"[rasterio] {farmer_id}: shape={arr.shape}")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# NDVI
# ─────────────────────────────────────────────────────────────────────────────
def compute_ndvi(image: np.ndarray) -> np.ndarray:
    """
    Compute NDVI from (C, H, W) image tensor.
    Expects C ≥ 4 with band order [B02, B03, B04, B08].
    Returns (1, H, W) float32 in [-1, 1].
    """
    if image.ndim != 3:
        raise ValueError(
            f"compute_ndvi expects (C,H,W), got shape {image.shape}. "
            f"Did the downloader return the wrong shape?"
        )
    if image.shape[0] < 4:
        raise ValueError(
            f"compute_ndvi needs ≥4 bands (B02,B03,B04,B08), "
            f"got {image.shape[0]} channels. Shape: {image.shape}"
        )
    nir  = image[3].astype(np.float32)   # B08
    red  = image[2].astype(np.float32)   # B04
    ndvi = (nir - red) / (nir + red + 1e-8)
    return np.clip(ndvi, -1.0, 1.0)[np.newaxis]    # (1, H, W)


def add_ndvi_channel(image: np.ndarray) -> np.ndarray:
    """
    Appends NDVI as 5th channel.
    Input : (4, H, W)  →  Output: (5, H, W)
    Validates shape before processing.
    """
    if image.ndim != 3 or image.shape[0] < 4:
        raise ValueError(
            f"add_ndvi_channel expects (4, H, W), got {image.shape}.\n"
            f"The scene.image from SentinelDownloader.fetch() must be "
            f"shape (4, 128, 128) float32. Check the downloader output."
        )
    ndvi = compute_ndvi(image)
    return np.concatenate([image, ndvi], axis=0)    # (5, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DOWNLOADER CLASS
# ─────────────────────────────────────────────────────────────────────────────
class SentinelDownloader:
    """
    Single entry point for Sentinel-2 downloads.
    Auto-selects: SentinelHub (if credentials) → Planetary Computer (free fallback).

    Example:
        dl    = SentinelDownloader()        # free Planetary Computer
        scene = dl.fetch("F001", 30.901, 75.857, "2024-10-20")
        print(scene.image.shape)            # (4, 128, 128)
        img5  = add_ndvi_channel(scene.image)
        print(img5.shape)                   # (5, 128, 128)
    """

    def __init__(
        self,
        sh_client_id:     str   = "",
        sh_client_secret: str   = "",
        cache_dir:        str   = "data/image_cache",
        retry_attempts:   int   = 3,
        retry_delay_s:    float = 2.0,
    ):
        self.sh_client_id     = sh_client_id     or os.getenv("SH_CLIENT_ID",     "")
        self.sh_client_secret = sh_client_secret or os.getenv("SH_CLIENT_SECRET", "")
        self.cache_dir        = Path(cache_dir)
        self.retry_attempts   = retry_attempts
        self.retry_delay_s    = retry_delay_s
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._has_sh = bool(self.sh_client_id and self.sh_client_secret)
        logger.info(
            f"[Downloader] Backend: "
            f"{'SentinelHub + ' if self._has_sh else ''}Planetary Computer"
        )

    # ── Cache ─────────────────────────────────────────────────────────────────
    def _cache_path(self, farmer_id: str, date_str: str) -> Path:
        return self.cache_dir / f"{farmer_id}_{date_str}.npy"

    def _from_cache(self, farmer_id: str, date_str: str) -> Optional[np.ndarray]:
        p = self._cache_path(farmer_id, date_str)
        if not p.exists():
            return None
        arr = np.load(str(p))
        try:
            arr = _validate_image_shape(arr, "cache")
            logger.info(f"[Downloader] Cache hit: {p.name}")
            return arr
        except ValueError as e:
            logger.warning(f"[Downloader] Corrupt cache file {p.name}: {e} — re-downloading")
            p.unlink(missing_ok=True)
            return None

    def _to_cache(self, farmer_id: str, date_str: str, image: np.ndarray):
        np.save(str(self._cache_path(farmer_id, date_str)), image)

    # ── Public fetch ──────────────────────────────────────────────────────────
    def fetch(
        self,
        farmer_id: str,
        lat:       float,
        lon:       float,
        date_str:  str,
        use_cache: bool = True,
    ) -> Optional[SceneResult]:
        """
        Download one Sentinel-2 patch.
        Returns SceneResult with image shape (4, 128, 128) float32, or None.
        """
        # Cache
        if use_cache:
            cached = self._from_cache(farmer_id, date_str)
            if cached is not None:
                return SceneResult(
                    farmer_id=farmer_id, lat=lat, lon=lon,
                    date=date_str, cloud_pct=0.0,
                    image=cached, source="cache",
                )

        # Retry loop
        for attempt in range(1, self.retry_attempts + 1):
            logger.info(
                f"[Downloader] {farmer_id} | "
                f"attempt {attempt}/{self.retry_attempts} | {date_str}"
            )
            result = None

            if self._has_sh:
                result = _download_sentinelhub(
                    farmer_id, lat, lon, date_str,
                    self.sh_client_id, self.sh_client_secret,
                )

            if result is None:
                result = _download_planetary_computer(
                    farmer_id, lat, lon, date_str,
                )

            if result is not None:
                # Final shape guard before returning to caller
                try:
                    result.image = _validate_image_shape(result.image, "final")
                except ValueError as e:
                    logger.error(f"[Downloader] Shape validation failed: {e}")
                    result = None

            if result is not None:
                if use_cache:
                    self._to_cache(farmer_id, date_str, result.image)
                return result

            if attempt < self.retry_attempts:
                logger.warning(f"[Downloader] Retrying in {self.retry_delay_s}s...")
                time.sleep(self.retry_delay_s)

        logger.error(f"[Downloader] All attempts failed for {farmer_id}")
        return None

    def fetch_best_in_range(
        self,
        farmer_id:  str,
        lat:        float,
        lon:        float,
        start_date: str,
        end_date:   str,
        use_cache:  bool = True,
    ) -> Optional[SceneResult]:
        """Tries ~5 evenly-spaced dates and returns lowest-cloud scene."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")
        delta = (end - start).days
        step  = max(1, delta // 5)
        dates = [
            (start + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(0, delta + 1, step)
        ]
        best: Optional[SceneResult] = None
        for d in dates:
            r = self.fetch(farmer_id, lat, lon, d, use_cache)
            if r and (best is None or r.cloud_pct < best.cloud_pct):
                best = r
                if best.cloud_pct < 5.0:
                    break
        return best


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt = "%H:%M:%S",
    )
    dl = SentinelDownloader()
    print("Testing download: Ludhiana, Punjab, 2024-10-20 ...")
    scene = dl.fetch("TEST_001", 30.901, 75.857, "2024-10-20")
    if scene:
        img5 = add_ndvi_channel(scene.image)
        print(f"✅  image={scene.image.shape}  with NDVI={img5.shape}")
        print(f"   cloud={scene.cloud_pct:.1f}%  source={scene.source}")
    else:
        print("❌  Download failed — check internet connection")