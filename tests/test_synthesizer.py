import os
import json
import pytest
from artifact_simulator.simulator import generate_single_artifact
from variant_synthesizer.synthesize_variants import perturb_artifact, synthesize_variants


def test_perturb_artifact_lineage_and_alignment():
    """Verify variant retains lineage root, increments generation, and keeps sections aligned."""
    parent = generate_single_artifact(target_label="malware_sim")
    variant = perturb_artifact(parent)

    assert variant["id"] != parent["id"]
    assert variant["origin_id"] == parent["origin_id"]
    assert variant["parent_id"] == parent["id"]
    assert variant["generation"] == parent["generation"] + 1
    assert variant["label"] == parent["label"]

    # Structural alignment invariant
    assert len(variant["section_sizes"]) == len(variant["section_entropy"]) == variant["num_sections"]
    assert len(variant["mutations"]) > 0


def test_multi_generation_lineage():
    """Verify origin_id is preserved across multiple evolutionary generations."""
    root = generate_single_artifact(target_label="malware_sim")
    gen1 = perturb_artifact(root)
    gen2 = perturb_artifact(gen1)
    gen3 = perturb_artifact(gen2)

    assert gen3["origin_id"] == root["id"]
    assert gen3["parent_id"] == gen2["id"]
    assert gen3["generation"] == 3
    assert gen3["label"] == "malware_sim"


def test_synthesize_variants_cli(tmp_path):
    """Verify batch variant synthesis creates expected files."""
    in_dir = str(tmp_path / "in")
    out_dir = str(tmp_path / "out")
    os.makedirs(in_dir, exist_ok=True)

    art = generate_single_artifact()
    with open(os.path.join(in_dir, f"{art['id']}.json"), "w", encoding="utf-8") as f:
        json.dump(art, f)

    variants = synthesize_variants(in_dir=in_dir, out_dir=out_dir, variants_per_file=4, seed=42)
    assert len(variants) == 4
    for vp in variants:
        assert os.path.isfile(vp)

