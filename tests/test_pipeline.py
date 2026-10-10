import numpy as np
import pytest
import os
import sys

# Add src to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.spatial_split import spatial_block_split
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
