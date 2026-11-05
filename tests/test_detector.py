import os
import json
import numpy as np
import pandas as pd
import pytest

from detection.detector import (
    extract_features_from_dict,
    evaluate_heuristic,
    PolymorphicDetector,
    FEATURE_COLUMNS,
)
from detection.train_model import (
    prepare_dataset,
    split_data_group_aware,
    train_model,
)


def test_extract_features_robustness():
    """Verify feature extractor handles missing keys, malformed data, and defaults gracefully."""
    empty_rec = {}
    flat = extract_features_from_dict(empty_rec)
    assert flat["num_sections"] == 0
    assert flat["avg_section_entropy"] == 0.0
    assert flat["total_section_size"] == 0
    assert flat["imported_api_count"] == 0
    assert flat["label"] == "benign"

    # Robust against string numbers and NaN entries
    noisy_rec = {
        "id": "sample-1",
        "num_sections": "6",
        "section_sizes": [1000, "2000", None],
        "section_entropy": [3.5, 4.5, float("nan")],
        "junk_bytes_count": "3000",
        "label": "malware_sim",
    }
    flat_noisy = extract_features_from_dict(noisy_rec)
    assert flat_noisy["id"] == "sample-1"
    assert flat_noisy["origin_id"] == "sample-1"  # Defaults to id when origin_id missing
    assert flat_noisy["num_sections"] == 6
    assert flat_noisy["total_section_size"] == 3000
    assert flat_noisy["avg_section_entropy"] == 4.0
    assert flat_noisy["junk_bytes_count"] == 3000
    assert flat_noisy["label"] == "malware_sim"


def test_evaluate_heuristic():
    """Verify rule-based heuristic fires only when both junk and entropy exceed thresholds."""
    # Low junk, low entropy -> 0
    assert evaluate_heuristic({"junk_bytes_count": 200, "avg_section_entropy": 4.0}) == 0
    # High junk, low entropy -> 0
    assert evaluate_heuristic({"junk_bytes_count": 5000, "avg_section_entropy": 4.0}) == 0
    # Low junk, high entropy -> 0
    assert evaluate_heuristic({"junk_bytes_count": 200, "avg_section_entropy": 7.5}) == 0
    # High junk, high entropy -> 1
    assert evaluate_heuristic({"junk_bytes_count": 5000, "avg_section_entropy": 7.5}) == 1


def test_zero_group_leakage_guarantee(test_data_dir):
    """Verify that group-aware train/test split has ZERO lineage overlap."""
    df = pd.read_csv(test_data_dir["csv_path"])
    X, y, groups = prepare_dataset(df, FEATURE_COLUMNS)

    X_train, X_test, y_train, y_test, train_idx, test_idx = split_data_group_aware(
        X, y, groups, test_size=0.3, random_state=42
    )

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    intersection = train_groups.intersection(test_groups)

    assert len(intersection) == 0, f"Lineage leakage detected! Overlapping groups: {intersection}"
    assert len(train_idx) + len(test_idx) == len(df)


def test_detector_inference(test_data_dir):
    """Verify PolymorphicDetector bundle loading and prediction."""
    detector = PolymorphicDetector.load(test_data_dir["model_file"])

    # Test high-entropy/high-junk malware-like input
    malware_features = {
        "num_sections": 7,
        "avg_section_entropy": 7.2,
        "total_section_size": 450000,
        "imported_api_count": 15,
        "strings_count": 120,
        "junk_bytes_count": 4800,
        "resource_size": 60000,
    }
    pred_m = detector.predict_features(malware_features)
    assert 0.0 <= pred_m["score"] <= 1.0
    assert pred_m["pred"] in ("benign", "malware_sim")
    assert "explanation" in pred_m
    assert pred_m["explanation"]["heuristic_flag"] == 1  # Heuristic should trigger

    # Test clean benign-like input
    benign_features = {
        "num_sections": 4,
        "avg_section_entropy": 3.8,
        "total_section_size": 180000,
        "imported_api_count": 80,
        "strings_count": 850,
        "junk_bytes_count": 150,
        "resource_size": 35000,
    }
    pred_b = detector.predict_features(benign_features)
    assert pred_b["explanation"]["heuristic_flag"] == 0


def test_metrics_integrity(test_data_dir):
    """Verify metrics.json contains tri-evaluation metrics, CV, and provenance."""
    with open(test_data_dir["metrics_file"], "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "ml_only" in metrics
    assert "heuristic_only" in metrics
    assert "ensemble" in metrics
    assert "samples" in metrics
    assert metrics["samples"]["zero_leakage_verified"] is True
    assert "data_provenance" in metrics

