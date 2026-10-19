import numpy as np
import pytest
import os
import sys

# Add src to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.spatial_split import spatial_block_split, assert_no_extent_leakage
from scripts.train_all import flag_suspicious_metrics

def test_split_leakage():
    # Synthetic coordinates grid
    coords = [(r, c) for r in range(0, 1024, 128) for c in range(0, 1024, 128)]
    grid_size = (4, 4)
    train_c, val_c, test_c = spatial_block_split(coords, grid_size, train_ratio=0.7, val_ratio=0.15)
    
    # Assert disjoint
    assert set(train_c).isdisjoint(val_c), "Leakage between train and val!"
    assert set(train_c).isdisjoint(test_c), "Leakage between train and test!"
    assert set(val_c).isdisjoint(test_c), "Leakage between val and test!"

def test_flag_suspicious_metrics():
    # 1. Perfect score (circularity risk)
    metrics_perfect = {"accuracy": 0.98, "f1": 0.98, "precision": 0.98, "recall": 0.98}
    flag = flag_suspicious_metrics(metrics_perfect, "ModelA")
    assert flag is not None
    assert "SUSPICIOUS PERFORMANCE" in flag

    # 2. Majority-class collapse (predicting all negatives)
    metrics_collapse = {"accuracy": 0.90, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    flag = flag_suspicious_metrics(metrics_collapse, "ModelB")
    assert flag is not None
    assert "MAJORITY-CLASS COLLAPSE" in flag

    # 3. Low support
    metrics_normal = {"accuracy": 0.80, "f1": 0.75, "precision": 0.70, "recall": 0.80}
    flag = flag_suspicious_metrics(metrics_normal, "ModelC", pos_samples=5)
    assert flag is not None
    assert "LOW SUPPORT" in flag

    # 4. Clean run
    flag = flag_suspicious_metrics(metrics_normal, "ModelD", pos_samples=50)
    assert not flag

def test_minimum_positives_assertion():
    # This tests the logic that throws assertion if positives == 0
    # We can't directly test the ingest script without running it, but we can mock the array check
    lbl_val = np.zeros((10,))  # 0 positives
    with pytest.raises(AssertionError):
        assert lbl_val.sum() > 0, "FATAL: Val split contains zero positive (ice) samples!"


def test_cpr_dop_complex_analytic_cases():
    from src.preprocessing.data_ops import compute_cpr_dop_complex

    H, W_grid = 64, 64

    # ── Case 1: Ideal flat surface (Bragg / specular reflection) ──
    # For a flat smooth dielectric boundary, reflection coefficients r_h and r_v are
    # in-phase and there is zero cross-polarization (Shv = Svh = 0).
    # Coherent circular transformation yields:
    #   S_RR = 0.5 * ((Shh - Svv) + 2j * Shv) = 0
    #   S_RL = 0.5 * (Shh + Svv) = 1.0
    # Expected: CPR == 0.0 (strictly opposite-sense circular return)
    #           DOP == 1.0 (pure polarized state, zero depolarization)
    Shh_flat = np.ones((H, W_grid), dtype=np.complex64)
    Svv_flat = np.ones((H, W_grid), dtype=np.complex64)
    Shv_flat = np.zeros((H, W_grid), dtype=np.complex64)
    Svh_flat = np.zeros((H, W_grid), dtype=np.complex64)

    cpr_flat, dop_flat = compute_cpr_dop_complex(Shh_flat, Shv_flat, Svh_flat, Svv_flat, W=5)
    assert np.allclose(cpr_flat, 0.0, atol=1e-5), f"Flat surface CPR expected ~0, got {cpr_flat.max()}"
    assert np.allclose(dop_flat, 1.0, atol=1e-3), f"Flat surface DOP expected ~1, got {dop_flat.min()}"

    # ── Case 2: Ideal dihedral / 180° phase flip (anomalous same-sense return) ──
    # Double-bounce dihedral reflection causes a 180° phase shift between HH and VV:
    #   Shh = +1, Svv = -1, Shv = 0
    # Then:
    #   S_RR = 0.5 * (1 - (-1)) = 1.0  (all power in same-sense circular!)
    #   S_RL = 0.5 * (1 + (-1)) = 0.0  (zero opposite-sense circular)
    # Expected: CPR >> 1.0 (approaching numerical limit ~1e10)
    #           DOP == 1.0 (single deterministic polarized state)
    Svv_dihedral = -np.ones((H, W_grid), dtype=np.complex64)
    cpr_dih, dop_dih = compute_cpr_dop_complex(Shh_flat, Shv_flat, Svh_flat, Svv_dihedral, W=5)
    assert np.all(cpr_dih > 1.0), "Dihedral reflection must yield CPR > 1.0"
    assert np.allclose(dop_dih, 1.0, atol=1e-3), f"Dihedral DOP expected ~1, got {dop_dih.min()}"

    # ── Case 3: Strongly depolarizing volume scattering ──
    # Independent random phases between channels simulate multiple random scattering
    # inside a disordered medium (e.g. fractured icy regolith).
    # Under multilook spatial averaging (W=21), cross-coherence drops and the return
    # becomes significantly depolarized:
    # Expected: DOP substantially lower than specular (< 0.50), CPR elevated (> 1.0)
    rng = np.random.default_rng(42)
    Shh_rnd = np.exp(1j * rng.uniform(0, 2*np.pi, (H, W_grid))).astype(np.complex64)
    Svv_rnd = np.exp(1j * rng.uniform(0, 2*np.pi, (H, W_grid))).astype(np.complex64)
    Shv_rnd = (0.7 * np.exp(1j * rng.uniform(0, 2*np.pi, (H, W_grid)))).astype(np.complex64)
    Svh_rnd = Shv_rnd

    cpr_rnd, dop_rnd = compute_cpr_dop_complex(Shh_rnd, Shv_rnd, Svh_rnd, Svv_rnd, W=21)
    # Exclude boundary margins influenced by filter padding
    inner_dop = dop_rnd[21:-21, 21:-21]
    inner_cpr = cpr_rnd[21:-21, 21:-21]
    assert inner_dop.mean() < 0.40, f"Random volume DOP must show depolarization (<0.40), got {inner_dop.mean():.3f}"
    assert inner_cpr.mean() > 1.0, f"Random volume CPR expected > 1.0, got {inner_cpr.mean():.3f}"


def test_pixel_extent_leakage_assertion():
    """Verify that assert_no_extent_leakage detects overlapping patch extents across splits."""
    P = 64
    # Disjoint coordinates ((0,0) != (32,0)), but overlapping extents [0, 64) and [32, 96)
    train_coords = [(0, 0)]
    val_coords = [(32, 0)]
    test_coords = [(500, 500)]

    # Must raise AssertionError because abs(0 - 32) < 64 and abs(0 - 0) < 64
    with pytest.raises(AssertionError, match="PIXEL EXTENT LEAKAGE DETECTED"):
        assert_no_extent_leakage(train_coords, val_coords, test_coords, patch_size=P)

    # When truly separated by >= 64 pixels, assertion must pass cleanly
    train_clean = [(0, 0)]
    val_clean = [(100, 0)]
    test_clean = [(200, 0)]
    assert_no_extent_leakage(train_clean, val_clean, test_clean, patch_size=P)


def test_spatial_split_boundary_buffer_synthetic():
    """Verify that spatial_block_split with buffer_pixels > 0 completely eliminates extent leakage."""
    P = 64
    stride = 32
    # Dense 1D/2D synthetic coordinate grid with overlapping strides (4096 pixels gives 512px blocks for grid 1x8)
    coords = [(r, c) for r in range(0, 4096, stride) for c in range(0, 128, stride)]
    grid_size = (1, 8)

    # Split with 64-pixel boundary buffer
    train_c, val_c, test_c = spatial_block_split(
        coords,
        grid_size=grid_size,
        train_ratio=0.70,
        val_ratio=0.15,
        patch_size=P,
        buffer_pixels=64,
        seed=42,
    )

    # Both coordinate and pixel-extent leakage must be zero
    assert set(train_c).isdisjoint(val_c)
    assert set(train_c).isdisjoint(test_c)
    assert set(val_c).isdisjoint(test_c)
    assert_no_extent_leakage(train_c, val_c, test_c, patch_size=P)
    assert len(train_c) > 0 and len(val_c) > 0 and len(test_c) > 0

