"""
src/preprocessing/register_ohrc.py
=====================================
Image registration: align an OHRC optical tile to the DFSAR spatial grid.

DFSAR and OHRC have different:
  - Ground sample distances (DFSAR ~75m/px, OHRC ~25cm/px — we work with
    processed tiles so actual GSD depends on the downloaded product level).
  - Projection / coordinate reference systems.

Registration approach (Phase 1 implementation — viva-defensible):
  1. Both images are assumed to be georeferenced (rasterio can read the CRS).
  2. We reproject / resample OHRC to DFSAR's grid using rasterio.warp.reproject.
  3. If no rasterio is available (synthetic mode), we simply resize OHRC to
     match DFSAR dimensions using bilinear interpolation (cv2.resize).

This module is used by scripts/train_multimodal.py to build a fused dataset.
"""

import os
import warnings
import numpy as np

try:
    import rasterio
    import rasterio.warp
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def register_ohrc_to_dfsar(
    dfsar_array: np.ndarray,
    ohrc_array:  np.ndarray,
    dfsar_path:  str = None,
    ohrc_path:   str = None,
) -> np.ndarray:
    """
    Align OHRC to DFSAR spatial extent.

    If both paths are provided and rasterio is available, performs a proper
    georeferenced reproject.  Otherwise falls back to plain resize (synthetic mode).

    Args:
        dfsar_array: float32 (H_d, W_d, 2) — reference grid.
        ohrc_array:  float32 (H_o, W_o)   — OHRC to be resampled.
        dfsar_path:  optional path for georef metadata.
        ohrc_path:   optional path for georef metadata.

    Returns:
        ohrc_registered: float32 (H_d, W_d) resampled to DFSAR grid.
    """
    H_d, W_d = dfsar_array.shape[:2]

    if (
        _HAS_RASTERIO
        and dfsar_path and os.path.isfile(dfsar_path)
        and ohrc_path  and os.path.isfile(ohrc_path)
    ):
        # Georeferenced reproject
        with rasterio.open(dfsar_path) as ds_ref:
            dst_crs       = ds_ref.crs
            dst_transform = ds_ref.transform
            dst_width     = ds_ref.width
            dst_height    = ds_ref.height

        with rasterio.open(ohrc_path) as src:
            destination = np.zeros((dst_height, dst_width), dtype=np.float32)
            rasterio.warp.reproject(
                source      = rasterio.band(src, 1),
                destination = destination,
                src_transform = src.transform,
                src_crs       = src.crs,
                dst_transform = dst_transform,
                dst_crs       = dst_crs,
                resampling    = rasterio.warp.Resampling.bilinear,
            )
        return destination.astype(np.float32)

    else:
        # Fallback: simple resize (valid for synthetic / test mode)
        warnings.warn(
            "[SYNTHETIC/FALLBACK] Registering OHRC via resize (no georef available).",
            RuntimeWarning, stacklevel=2
        )
        if _HAS_CV2:
            registered = cv2.resize(ohrc_array, (W_d, H_d), interpolation=cv2.INTER_LINEAR)
        else:
            # Pure numpy bilinear-ish via zoom
            from scipy.ndimage import zoom
            zy = H_d / ohrc_array.shape[0]
            zx = W_d / ohrc_array.shape[1]
            registered = zoom(ohrc_array, (zy, zx), order=1)
        return registered.astype(np.float32)
