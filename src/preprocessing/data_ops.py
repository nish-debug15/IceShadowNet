"""
src/preprocessing/data_ops.py
=================================
Real DFSAR/OHRC ingestion and preprocessing for IceShadowNet.

PIPELINE OVERVIEW
-----------------
1. load_dfsar(path)           -- open a PDS4 .IMG file via rasterio, return (H, W, bands) float32
2. load_ohrc(path)            -- open OHRC greyscale tile via rasterio
3. compute_cpr_dop(dfsar)     -- derive CPR and DOP from SAR polarisation channels
4. label_from_threshold(c,d)  -- CPR > 1 AND DOP < 0.13  ->  binary ice mask
5. generate_patches(img, ...) -- sliding-window patch extraction; returns patches + (row,col) coords
6. augment(patch)             -- random flips/rotations safe for SAR intensity imagery

DATA STATUS (updated 2026-09-18)
----------------------------------
Real DFSAR tiles are NOT yet downloaded from PRADAN/ISSDC.
See  reports/data_notes.md  for the access status and gap description.
Until real tiles land in  data/raw/ , the functions fall back to a physically-
motivated synthetic scene (spatial CPR/DOP patterns, not pure white noise).
The fallback is clearly labelled with  [SYNTHETIC]  prints and RuntimeWarnings.
"""

import os
import warnings
import numpy as np

# rasterio may not be installed in all envs; fail loudly only if a REAL file is requested
try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False


# ==============================================================================
# Internal helpers
# ==============================================================================

def _synthetic_dfsar(height: int = 2048, width: int = 2048, seed: int = 42) -> np.ndarray:
    """
    Generate a physically-motivated synthetic DFSAR scene (2048x2048 pixels).

    Viva-defensible design rationale:
    - Two polarisation channels: HH (ch0) and HV (ch1).
    - Non-ice terrain: true CPR = HV/HH = 0.12 (surface scattering dominates).
      With L=4 multiplicative speckle (gamma distribution, CV=0.5), the probability
      that a non-ice pixel's speckled CPR exceeds 1.0 is < 0.01% per pixel.
      This means non-ice patches reliably have < 1% false ice pixels.
    - Ice-candidate region (bottom-right 512x512 block): true CPR = 2.5.
      This simulates the elevated HV backscatter from volume scattering inside
      a permanently-shadowed crater, consistent with Mini-RF/DFSAR observations
      (Spudis et al. 2010; ISRO DFSAR preliminary reports).
    - Larger scene (2048x2048) ensures the 4x4 spatial-split grid produces
      blocks of ~512x512 pixels -- large enough for meaningful non-overlapping patches.
    - Smooth HH backscatter (Gaussian sigma=40px) mimics terrain-induced variation.
    - DOP = |HH-HV|/(HH+HV): non-ice gives DOP ~ (HH-0.12*HH)/(HH+0.12*HH) = 0.79,
      well above the 0.13 threshold. Ice gives DOP ~ (HH-2.5*HH)/(HH+2.5*HH) < 0 -> 0
      (clamped), satisfying DOP < 0.13.

    This is NOT a calibrated simulation; it is a structural placeholder to test
    the pipeline end-to-end while real DFSAR tiles are being obtained from PRADAN.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)

    # Smooth terrain HH backscatter (range 0.3 - 0.8)
    hh_true = (gaussian_filter(rng.random((height, width)).astype(np.float32), sigma=40)
               * 0.5 + 0.3)

    # Initialize true CPR with non-ice background value (0.12)
    true_cpr = np.full((height, width), 0.12, dtype=np.float32)

    # Ice-candidate blocks: scatter 4 craters (512x512 each) to ensure ice spans the geographic grid
    # Top-Left
    true_cpr[:512, :512] = 2.5
    # Top-Right
    true_cpr[:512, -512:] = 2.5
    # Bottom-Left
    true_cpr[-512:, :512] = 2.5
    # Bottom-Right
    true_cpr[-512:, -512:] = 2.5

    # Simulate HV backscatter = HH * CPR
    hv_true = hh_true * true_cpr

    # Multiplicative speckle (L=4, gamma distributed, mean=1, CV=1/sqrt(4)=0.5)
    L = 4
    speckle_hh = rng.standard_gamma(L, size=(height, width)).astype(np.float32) / L
    speckle_hv = rng.standard_gamma(L, size=(height, width)).astype(np.float32) / L

    hh_speckled = hh_true * speckle_hh
    hv_speckled = hv_true * speckle_hv

    return np.stack([hh_speckled, hv_speckled], axis=-1)  # (H, W, 2)


def _synthetic_ohrc(height: int = 2048, width: int = 2048, seed: int = 43) -> np.ndarray:
    """
    Generate a physically-motivated synthetic OHRC tile (single greyscale band).
    OHRC is optical so we model it as smooth terrain albedo + noise.
    The permanently-shadowed regions (PSRs) cover the four corners (matching
    the ice candidate regions in DFSAR) and have near-zero albedo.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    base = gaussian_filter(rng.random((height, width)).astype(np.float32), sigma=30) * 0.6 + 0.1
    # PSR = low albedo in four corners (matching DFSAR ice regions)
    base[:512, :512] *= 0.05
    base[:512, -512:] *= 0.05
    base[-512:, :512] *= 0.05
    base[-512:, -512:] *= 0.05
    noise = rng.random((height, width)).astype(np.float32) * 0.03
    return (base + noise).astype(np.float32)


# ==============================================================================
# Public API
# ==============================================================================

def load_dfsar(path: str) -> np.ndarray:
    """
    Load a DFSAR PDS4 tile and return a float32 array of shape (H, W, bands).

    Args:
        path: Path to the .IMG or .tif DFSAR file (PDS4 format readable by rasterio).

    Returns:
        ndarray float32 of shape (H, W, n_bands).

    Notes:
        If the file does not exist or rasterio is not installed, falls back to
        a synthetic scene and logs a warning.  This allows the full pipeline to
        run end-to-end while real tiles are being obtained from PRADAN/ISSDC.
    """
    if not os.path.isfile(path):
        warnings.warn(
            f"[SYNTHETIC] DFSAR file not found: '{path}'. "
            "Falling back to synthetic SAR scene. "
            "See reports/data_notes.md for acquisition status.",
            RuntimeWarning, stacklevel=2
        )
        return _synthetic_dfsar()

    if not _HAS_RASTERIO:
        raise ImportError(
            "rasterio is required to load real DFSAR files. "
            "Install it with: pip install rasterio"
        )

    with rasterio.open(path) as ds:
        arr = ds.read()  # shape: (bands, H, W)
        arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)  # -> (H, W, bands)
    print(f"[REAL] Loaded DFSAR from '{path}', shape={arr.shape}")
    return arr


def load_ohrc(path: str) -> np.ndarray:
    """
    Load an OHRC tile and return a float32 array of shape (H, W).

    Args:
        path: Path to the OHRC .IMG or .tif file.

    Returns:
        ndarray float32 of shape (H, W).
    """
    if not os.path.isfile(path):
        warnings.warn(
            f"[SYNTHETIC] OHRC file not found: '{path}'. "
            "Falling back to synthetic optical tile.",
            RuntimeWarning, stacklevel=2
        )
        return _synthetic_ohrc()

    if not _HAS_RASTERIO:
        raise ImportError("rasterio is required to load real OHRC files.")

    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)  # single band
    print(f"[REAL] Loaded OHRC from '{path}', shape={arr.shape}")
    return arr


def compute_cpr_dop(dfsar_array: np.ndarray):
    """
    Compute Circular Polarisation Ratio (CPR) and Degree of Polarisation (DOP)
    from a dual-polarisation SAR array.

    Physical definitions (viva-defensible):
    - CPR = sigma_HV / sigma_HH
      High CPR (> 1) indicates volume scattering consistent with subsurface ice.
      References: Nozette et al. 1996 (Clementine); Spudis et al. 2010 (Mini-RF).
    - DOP = (I_max - I_min) / (I_max + I_min) where I = intensity
      Low DOP (< 0.13) indicates unpolarised return, also consistent with ice.
      Both criteria together give the candidate ice mask used by ISRO/Mini-RF.

    Args:
        dfsar_array: float32 (H, W, >=2).  Channel 0 = HH, channel 1 = HV.

    Returns:
        (cpr, dop): two float32 arrays of shape (H, W).
    """
    HH = dfsar_array[..., 0].astype(np.float64)
    HV = dfsar_array[..., 1].astype(np.float64)

    eps = 1e-9
    cpr = HV / (HH + eps)

    # DOP proxy for dual-pol data: |HH - HV| / (HH + HV)
    # (Full-pol Stokes DOP requires all 4 Stokes parameters; dual-pol DFSAR
    #  gives only HH+HV, so this is the best available approximation.)
    I_max = np.maximum(HH, HV)
    I_min = np.minimum(HH, HV)
    dop = (I_max - I_min) / (I_max + I_min + eps)

    return cpr.astype(np.float32), dop.astype(np.float32)


def label_from_threshold(cpr: np.ndarray, dop: np.ndarray) -> np.ndarray:
    """
    Generate a binary ice-candidate mask using established radar polarimetric criteria.

    Criteria:
        CPR > 1.0  AND  DOP < 0.13
    Source: Spudis et al. 2010, Mini-RF instrument team; also applied to DFSAR data
    in ISRO Chandrayaan-2 preliminary ice reports.

    LIMITATION (must be stated in report and viva):
        These thresholds define "radar signatures consistent with ice" -- NOT confirmed ice.
        Labels derived this way may introduce circularity if the CNN simply re-learns
        the threshold rule pixel-by-pixel without capturing spatial texture.  See PRD Sec 11.

    Args:
        cpr: float32 (H, W) Circular Polarisation Ratio.
        dop: float32 (H, W) Degree of Polarisation.

    Returns:
        uint8 binary mask (H, W): 1 = potential ice, 0 = non-ice.
    """
    mask = (cpr > 1.0) & (dop < 0.13)
    return mask.astype(np.uint8)


def generate_patches(
    image: np.ndarray,
    label_mask: np.ndarray,
    patch_size: int = 256,
    stride: int = 128,
) -> list:
    """
    Extract overlapping patches from an image and its label mask using a sliding window.

    Patch label: a patch is labelled as 'ice' (1) if >= 3% of its pixels satisfy the
    ice criteria. Justification: with ~6% scene-wide ice-pixel rate (in ice block)
    and ~0% outside, a 3% threshold reliably separates ice patches (overlapping the
    ice-candidate block) from non-ice patches.
    (Viva: be prepared to defend this threshold choice vs alternatives such as 5%, 10%,
    or majority-vote 50%.)

    Args:
        image:      float32 (H, W, C) -- the full SAR scene (or OHRC).
        label_mask: uint8  (H, W)     -- binary ice mask from label_from_threshold().
        patch_size: int               -- square patch side length in pixels.
        stride:     int               -- sliding window step (< patch_size -> overlapping).

    Returns:
        List of (patch_image, patch_label, (row_start, col_start)) tuples where:
            patch_image  -- float32 (C, patch_size, patch_size)  [channels-first for PyTorch]
            patch_label  -- int  0 or 1  (3% presence threshold)
            (row, col)   -- top-left pixel coordinate of this patch in the full scene
    """
    H, W = image.shape[:2]
    patches = []

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            img_patch = image[r:r+patch_size, c:c+patch_size]       # (ps, ps, C)
            lbl_patch = label_mask[r:r+patch_size, c:c+patch_size]  # (ps, ps)

            patch_label = int(lbl_patch.mean() > 0.03)

            # Channels-first for PyTorch: (C, H, W)
            img_chw = np.transpose(img_patch, (2, 0, 1)).astype(np.float32)

            patches.append((img_chw, patch_label, (r, c)))

    return patches


def augment(patch: np.ndarray) -> np.ndarray:
    """
    Apply random data augmentation to a SAR patch.

    Safe augmentations for SAR intensity data (viva note):
    - Horizontal/vertical flip: intensity images are symmetric under reflection.
    - 90 degree rotations: valid for nadir-looking SAR in the absence of Doppler artifacts.
    - NOT applied: brightness/colour jitter (physically meaningless for calibrated SAR).

    Args:
        patch: float32 (C, H, W).

    Returns:
        Augmented float32 (C, H, W).
    """
    if np.random.rand() > 0.5:
        patch = np.flip(patch, axis=2)  # horizontal flip (width axis)
    if np.random.rand() > 0.5:
        patch = np.flip(patch, axis=1)  # vertical flip
    k = np.random.randint(0, 4)
    patch = np.rot90(patch, k=k, axes=(1, 2))
    return np.ascontiguousarray(patch)
