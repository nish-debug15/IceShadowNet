"""
tests/test_scene_registry.py
============================
Unit tests for scene registry, multi-scene split planning, and hard asserts.
"""

import pytest
import numpy as np
from src.datasets.scene_registry import (
    generate_split_plan,
    format_scene_table_markdown,
    parse_pds4_xml_metadata,
)


def test_split_plan_multi_scene_n_ge_3():
    """Verify leave-one-scene-out cross validation plan for N >= 3 scenes."""
    mock_scenes = [
        {"scene_id": f"scene_{i}", "total_patches": 1000, "positive_patches": 200, "positive_rate": "20.0%"}
        for i in range(4)
    ]
    plan = generate_split_plan(mock_scenes)
    assert plan["n_scenes"] == 4
    assert "LEAVE-ONE-SCENE-OUT" in plan["plan_type"]
    assert len(plan["splits"]) == 4
    for fold in plan["splits"]:
        assert fold["buffer_pixels"] == 0
        assert fold["test_scene"] not in fold["train_scenes"]
        assert len(fold["train_scenes"]) == 3


def test_split_plan_two_scenes():
    """Verify inter-scene partition plan for N == 2 scenes."""
    mock_scenes = [
        {"scene_id": "scene_A", "total_patches": 1200, "positive_patches": 250, "positive_rate": "20.8%"},
        {"scene_id": "scene_B", "total_patches": 1100, "positive_patches": 220, "positive_rate": "20.0%"},
    ]
    plan = generate_split_plan(mock_scenes)
    assert plan["n_scenes"] == 2
    assert "INTER-SCENE" in plan["plan_type"]
    assert plan["train_val_scene"] == "scene_A"
    assert plan["test_scene"] == "scene_B"
    assert plan["buffer_pixels"] == 0


def test_split_plan_single_scene_fallback():
    """Verify fallback to deterministic blocked along-track strip CV with 256px guard buffer when N == 1."""
    mock_scenes = [
        {"scene_id": "ch2_sar_ncxl_20260308t062017233", "total_patches": 14282, "positive_patches": 2393, "positive_rate": "16.76%"}
    ]
    plan = generate_split_plan(mock_scenes)
    assert plan["n_scenes"] == 1
    assert "BLOCKED STRIP" in plan["plan_type"]
    assert plan["buffer_pixels"] == 256
    assert "SINGLE SCENE DETECTED" in plan["note"]
    
    # Check disjoint strip allocations
    train_s = set(plan["assigned_splits"]["train_strips"])
    val_s = set(plan["assigned_splits"]["val_strips"])
    test_s = set(plan["assigned_splits"]["test_strips"])
    assert train_s.isdisjoint(val_s)
    assert train_s.isdisjoint(test_s)
    assert val_s.isdisjoint(test_s)

    # Test positive count check
    assert plan["post_buffer_counts"]["test_pos"] >= 30, (
        f"Test positives must be >= 30, got {plan['post_buffer_counts']['test_pos']}"
    )


def test_scene_table_markdown_formatting():
    """Verify scene table markdown output structure."""
    mock_scenes = [
        {
            "scene_id": "ch2_sar_test",
            "instrument": "DFSAR (L-band Quad-Pol SAR)",
            "total_scene_dimensions": "319812 rows x 116 cols",
            "footprint_lat": "-86.35° to -81.90°",
            "footprint_lon": "13.95° to 49.27°",
            "analyzed_swath": "Lines 90,624..319,812",
            "positive_pixels": "13,815 (0.0524%)",
            "positive_patches": 2393,
            "negative_patches": 11889,
            "positive_rate": "16.76%",
        }
    ]
    md = format_scene_table_markdown(mock_scenes)
    assert "| Scene ID |" in md
    assert "`ch2_sar_test`" in md
    assert "16.76%" in md
