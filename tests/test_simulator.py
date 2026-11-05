import os
import json
import pytest
from artifact_simulator.simulator import generate_single_artifact, generate_base_artifacts


def test_generate_single_artifact_structure():
    """Verify artifact dictionary has all expected fields, types, and self-referencing lineage root."""
    art = generate_single_artifact()
    assert isinstance(art, dict)
    assert "id" in art
    assert art["origin_id"] == art["id"]
    assert art["parent_id"] is None
    assert art["generation"] == 0
    assert art["label"] in ("benign", "malware_sim")
    assert isinstance(art["section_sizes"], list)
    assert isinstance(art["section_entropy"], list)
    assert len(art["section_sizes"]) == len(art["section_entropy"]) == art["num_sections"]
    assert art["num_sections"] >= 3
    assert art["imported_api_count"] >= 0
    assert art["strings_count"] >= 0
    assert art["junk_bytes_count"] >= 0
    assert art["resource_size"] >= 0


def test_class_conditional_distributions():
    """Verify class-conditional separation between benign and simulated malware distributions."""
    benign_art = generate_single_artifact(target_label="benign")
    assert benign_art["label"] == "benign"
    assert benign_art["junk_bytes_count"] <= 1200
    assert max(benign_art["section_entropy"]) <= 6.5

    malware_art = generate_single_artifact(target_label="malware_sim")
    assert malware_art["label"] == "malware_sim"
    assert malware_art["junk_bytes_count"] >= 1500
    assert any(e >= 5.2 for e in malware_art["section_entropy"])


def test_seed_reproducibility(tmp_path):
    """Verify deterministic generation when seed is supplied."""
    out1 = str(tmp_path / "run1")
    out2 = str(tmp_path / "run2")

    paths1 = generate_base_artifacts(n=5, out_dir=out1, seed=123)
    paths2 = generate_base_artifacts(n=5, out_dir=out2, seed=123)

    assert len(paths1) == len(paths2) == 5

    # Check identical contents
    for p1, p2 in zip(paths1, paths2):
        with open(p1, "r", encoding="utf-8") as f1, open(p2, "r", encoding="utf-8") as f2:
            d1 = json.load(f1)
            d2 = json.load(f2)
            # Remove UUID which is randomly generated per call, compare feature values
            d1_features = {k: v for k, v in d1.items() if k not in ("id", "origin_id")}
            d2_features = {k: v for k, v in d2.items() if k not in ("id", "origin_id")}
            assert d1_features == d2_features

